import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import xgboost as xgb
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, accuracy_score, classification_report
from pathlib import Path
from tqdm import tqdm
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, AttentionalAggregation
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

BASE_PATH = r'/path/to/your/dataset'
DATA_DIR = Path(BASE_PATH) / 'data/graph_tensors_rbf'
CHECKPOINT_DIR = Path(BASE_PATH) / 'checkpoints'

BIN_CLASSIFIER_PATH = CHECKPOINT_DIR / 'xgb_bin_refined.pkl'
MULTI_CLASSIFIER_PATH = CHECKPOINT_DIR / 'xgb_multi_refined.pkl'

STATS_PATH = CHECKPOINT_DIR / 'master_stats.pth'
PRETRAINED_REG_PATH = CHECKPOINT_DIR / 'best_regressor.pth'

CONFIG = {
    "seed": 42,
    "hidden_dim": 256,
    "dropout": 0.05,
    "reg_lr": 0.001,   
    "epochs": 200,
    "delta": 0.5,
    "batch_size": 128
}

CLASS_NAMES = ["Non-SC", "Cu-based", "Fe-based", "Other"]

def seed_everything(seed=CONFIG["seed"]):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

class XGBHybridClassifier:
    def __init__(self, bin_model, multi_model):
        self.xgb_bin = bin_model
        self.xgb_multi = multi_model

    def predict(self, X):
        bin_preds = self.xgb_bin.predict(X)
        final_preds = np.zeros(len(X), dtype=int)
        sc_idx = np.where(bin_preds == 1)[0]
        if len(sc_idx) > 0:
            final_preds[sc_idx] = self.xgb_multi.predict(X[sc_idx]) + 1
        return final_preds

    def predict_probs_for_reg(self, X):
        bin_prob = self.xgb_bin.predict_proba(X) 
        multi_prob = np.zeros((len(X), 3))
        
        sc_mask = self.xgb_bin.predict(X) == 1
        sc_idx = np.where(sc_mask)[0]
        if len(sc_idx) > 0:
            multi_prob[sc_idx] = self.xgb_multi.predict_proba(X[sc_idx])

        return np.concatenate([bin_prob, multi_prob], axis=1)

# ================= 1. Physical feature aggregation (88 dimensions) =================
def aggregate_physical_features(data):
    x = data.x.cpu().numpy()
    e = data.edge_attr.cpu().numpy() if hasattr(data, 'edge_attr') else np.zeros((1, 32))
    x_mean, x_max, x_std = np.mean(x, axis=0), np.max(x, axis=0), np.std(x, axis=0)
    e_mean, e_max = np.mean(e, axis=0), np.max(e, axis=0)
    return np.concatenate([x_mean, x_max, x_std, e_mean, e_max])

# ================= 2. Regression model (structure remains unchanged) =================
class Regressor(nn.Module):
    def __init__(self, node_in=13, edge_in=2, hidden_dim=CONFIG["hidden_dim"]):
        super().__init__()
        self.node_emb = nn.Linear(node_in, hidden_dim)
        self.edge_emb = nn.Linear(edge_in, hidden_dim)
        
        def make_conv():
            return GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.SiLU(), nn.Dropout(CONFIG["dropout"]), nn.Linear(hidden_dim, hidden_dim)
            ))
        
        self.conv1, self.conv2, self.conv3, self.conv4 = make_conv(), make_conv(), make_conv(), make_conv()
        
        gate_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))
        self.attn_pool = AttentionalAggregation(gate_nn)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, 1)
        )
        
    def forward(self, data, cls_probs):
        guidance = cls_probs[data.batch]
        combined_node_x = torch.cat([data.x, guidance], dim=-1) 
        
        h = self.node_emb(combined_node_x)
        e = self.edge_emb(data.edge_attr_norm)
        
        h1 = self.conv1(h, data.edge_index, e)
        h2 = self.conv2(h1, data.edge_index, e) + h1
        h3 = self.conv3(h2, data.edge_index, e) + h2
        h4 = self.conv4(h3, data.edge_index, e) + h3
        
        out = torch.cat([
            global_mean_pool(h4, data.batch), 
            global_max_pool(h4, data.batch), 
            self.attn_pool(h4, data.batch)
        ], dim=1)
        
        return (self.head(out).view(-1) * (data.y >= 0.001).float()).clamp(min=0.0)

# ================= 3. Data preprocessor =================
class RefineProcessor:
    def __init__(self, stats_path, device):
        self.device = device
        stats = torch.load(stats_path, map_location=device)
        self.mean, self.std = stats['mean'].to(device), stats['std'].to(device)
        self.e_mean, self.e_std = stats['e_mean'].to(device), stats['e_std'].to(device)

    def preprocess(self, data):
        data.x = (data.x - self.mean) / self.std
        data.edge_attr_norm = (data.edge_attr[:, :2] - self.e_mean) / self.e_std
        data.y_log = torch.log1p(data.y)
        return data

# ================= 4. Fine-tune the main process =================
def run_refine_pipeline():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not BIN_CLASSIFIER_PATH.exists() or not MULTI_CLASSIFIER_PATH.exists():
        print("Error: Unable to find the refined classifier weights(xgb_bin_refined.pkl/xgb_multi_refined.pkl)")
        return
    
    xgb_bin = joblib.load(BIN_CLASSIFIER_PATH)
    xgb_multi = joblib.load(MULTI_CLASSIFIER_PATH)
    hybrid_cls = XGBHybridClassifier(xgb_bin, xgb_multi)
    print("The fine-tuned classifier has been successfully loaded and the freezing strategy has been executed.")

    processor = RefineProcessor(STATS_PATH, DEVICE)

    all_pts = sorted(list(DATA_DIR.glob('*.pt')))
    raw_dataset = [torch.load(pt, weights_only=False) for pt in tqdm(all_pts, desc="Loading Data")]
    
    np.random.seed(42)
    indices = np.random.permutation(len(raw_dataset))
    split = int(0.8 * len(raw_dataset))
    train_raw = [raw_dataset[i] for i in indices[:split]]
    val_raw = [raw_dataset[i] for i in indices[split:]]

    X_val_phys = np.array([aggregate_physical_features(d) for d in val_raw])
    y_val_true = np.array([d.sc_class.item() for d in val_raw])
    
    y_val_pred = hybrid_cls.predict(X_val_phys)
    print("\n" + "="*45)
    print(f"Fine-tuning classifier validation (ACC): {accuracy_score(y_val_true, y_val_pred):.2%}")
    print(classification_report(y_val_true, y_val_pred, target_names=CLASS_NAMES))
    print("="*45)

    reg_model = Regressor().to(DEVICE)
    if PRETRAINED_REG_PATH.exists():
        reg_model.load_state_dict(torch.load(PRETRAINED_REG_PATH, map_location=DEVICE))
        print("The initial pre-trained weights of the regressor have been loaded.")

    optimizer = optim.AdamW(reg_model.parameters(), lr=CONFIG["reg_lr"])
    criterion = nn.HuberLoss(delta=CONFIG["delta"], reduction='none')
    
    train_loader = DataLoader(train_raw, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_raw, batch_size=CONFIG["batch_size"])
    
    best_r2 = -float('inf')
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        reg_model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            data = processor.preprocess(data)
            
            with torch.no_grad():
                phys_feats = np.array([aggregate_physical_features(d) for d in data.to_data_list()])
                cls_probs = torch.FloatTensor(hybrid_cls.predict_probs_for_reg(phys_feats)).to(DEVICE)
            
            optimizer.zero_grad()
            out = reg_model(data, cls_probs)
            
            mask = (data.y >= 0.001).float()
            loss = (criterion(out, data.y_log) * data.weight * mask).sum() / (mask.sum() + 1e-6)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 10 == 0:
            reg_model.eval()
            y_true_all, y_pred_all = [], []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(DEVICE)
                    data = processor.preprocess(data)
                    
                    phys_feats = np.array([aggregate_physical_features(d) for d in data.to_data_list()])
                    cls_probs = torch.FloatTensor(hybrid_cls.predict_probs_for_reg(phys_feats)).to(DEVICE)
                    
                    out = reg_model(data, cls_probs)
                    preds = torch.expm1(out).clamp(min=0.0).cpu().numpy()
                    y_true_all.extend(data.y.cpu().numpy())
                    y_pred_all.extend(preds)
            
            current_r2 = r2_score(y_true_all, y_pred_all)
            if current_r2 > best_r2:
                best_r2 = current_r2
                torch.save(reg_model.state_dict(), CHECKPOINT_DIR / 'v12_refined_regressor_final.pth')
            
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss/len(train_loader):.4f} | R2: {current_r2:.4f} | Best: {best_r2:.4f}")

    print(f"\nR2: {best_r2:.4f}")

if __name__ == "__main__":
    run_refine_pipeline()
