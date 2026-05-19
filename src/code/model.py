import torch
import numpy as np
import mendeleev
import joblib
from pathlib import Path
from pymatgen.core import Structure
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, AttentionalAggregation
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

CONFIG = {
    "seed": 42,
    "hidden_dim": 256,
    "dropout": 0.05,
    "reg_lr": 0.001,   
    "epochs": 200,
    "delta": 0.5,
    "batch_size": 128
}

BASE_PATH = Path(r'/path/to/your/dataset')
CHECKPOINT_DIR = BASE_PATH / 'checkpoints'
STATS_PATH = CHECKPOINT_DIR / 'master_stats.pth'
BIN_CLASSIFIER_PATH = CHECKPOINT_DIR / 'xgb_bin_refined.pkl'
MULTI_CLASSIFIER_PATH = CHECKPOINT_DIR / 'xgb_multi_refined.pkl'
REG_MODEL_PATH = CHECKPOINT_DIR / 'refined_regressor_final.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_single_tensor(cif_path):
    DIST_CENTERS = np.linspace(0.5, 6.0, 16)
    EN_DIFF_CENTERS = np.linspace(0.0, 4.0, 16)
    GAMMA_DIST = 1.0 / (DIST_CENTERS[1] - DIST_CENTERS[0])**2
    GAMMA_EN = 1.0 / (EN_DIFF_CENTERS[1] - EN_DIFF_CENTERS[0])**2

    def get_node_features(atomic_number):
        try:
            el = mendeleev.element(int(atomic_number))
            is_metal = 1 if (el.is_alkali() or el.is_transition_metal() or el.is_lanthanide()) else 0
            en = float(el.en_pauling or 2.0)
            features = [
                float(el.mass or 0), en, float(el.atomic_radius or 1.5),
                float(el.group_id or 0), float(el.period or 0), 
                float(is_metal), float(atomic_number), float(el.mendeleev_number or 0)
            ]
            return features, en
        except:
            return [float(atomic_number*2), 2.0, 1.5, 0.0, 0.0, 1.0, float(atomic_number), float(atomic_number)], 2.0

    struct = Structure.from_file(cif_path)
    node_list, en_list = [], []
    for s in struct:
        feats, en = get_node_features(s.specie.number)
        node_list.append(feats)
        en_list.append(en)
    
    x = torch.tensor(node_list, dtype=torch.float)
    all_neighbors = struct.get_all_neighbors(r=6.0)
    edge_idx, edge_attrs = [], []
    
    for i, neighbors in enumerate(all_neighbors):
        en_i = en_list[i]
        for nbr in neighbors:
            dist = nbr.nn_distance
            d_rbf = np.exp(-GAMMA_DIST * (dist - DIST_CENTERS)**2)
            en_j = en_list[nbr.index]
            e_rbf = np.exp(-GAMMA_EN * (abs(en_i - en_j) - EN_DIFF_CENTERS)**2)
            edge_idx.append([i, nbr.index])
            edge_attrs.append(np.concatenate([d_rbf, e_rbf]))

    edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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
        
        return self.head(out).view(-1)


class RefineProcessor:
    def __init__(self, stats_path, device):
        self.device = device
        stats = torch.load(stats_path, map_location=device)
        self.mean, self.std = stats['mean'].to(device), stats['std'].to(device)
        self.e_mean, self.e_std = stats['e_mean'].to(device), stats['e_std'].to(device)

    def preprocess(self, data):
        data.x = (data.x - self.mean) / self.std
        data.edge_attr_norm = (data.edge_attr[:, :2] - self.e_mean) / self.e_std
        return data

class V12XGBHybridClassifier:
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

def aggregate_physical_features(data):
    x = data.x.cpu().numpy()
    e = data.edge_attr.cpu().numpy() if hasattr(data, 'edge_attr') else np.zeros((1, 32))
    x_mean, x_max, x_std = np.mean(x, axis=0), np.max(x, axis=0), np.std(x, axis=0)
    e_mean, e_max = np.mean(e, axis=0), np.max(e, axis=0)
    return np.concatenate([x_mean, x_max, x_std, e_mean, e_max])

def run_prediction(cif_path):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    xgb_bin = joblib.load(BIN_CLASSIFIER_PATH)
    xgb_multi = joblib.load(MULTI_CLASSIFIER_PATH)
    hybrid_cls = V12XGBHybridClassifier(xgb_bin, xgb_multi)
    
    reg_model = Regressor().to(DEVICE)
    reg_model.load_state_dict(torch.load(REG_MODEL_PATH, map_location=DEVICE))
    reg_model.eval()
    
    stats = torch.load(STATS_PATH, map_location=DEVICE)
    mean, std = stats['mean'], stats['std']
    e_mean, e_std = stats['e_mean'], stats['e_std']

    data = get_single_tensor(cif_path)
    
    x_np = data.x.numpy()
    e_np = data.edge_attr.numpy()
    phys_feats = np.concatenate([
        np.mean(x_np, axis=0), np.max(x_np, axis=0), np.std(x_np, axis=0), 
        np.mean(e_np, axis=0), np.max(e_np, axis=0)
    ]).reshape(1, -1)
    
    with torch.no_grad():
        probs = hybrid_cls.predict_probs_for_reg(phys_feats)
        cls_t = torch.FloatTensor(probs).to(DEVICE)
        
        data = data.to(DEVICE)
        data.x = (data.x - mean) / std
        data.edge_attr_norm = (data.edge_attr[:, :2] - e_mean) / e_std 
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long).to(DEVICE)
        
        out_log = reg_model(data, cls_t)
        tc_pred = torch.expm1(out_log).clamp(min=0.0).item()
        
        class_idx = hybrid_cls.predict(phys_feats)[0]
        
    if probs[0, 0] > 0.80:
        final_tc = 0.0
        display_idx = 0 
    elif tc_pred > 100.0:
        final_tc = tc_pred
        display_idx = 3
    else:
        final_tc = tc_pred
        display_idx = class_idx

    CLASS_LABELS = ["Non-SC", "Cu-based", "Fe-based", "Other"]
    final_type = CLASS_LABELS[display_idx]

    print(f"Start Processing: {Path(cif_path).name}")
    print("-" * 30)
    print(f"Material type discrimination: {final_type}")
    print(f"Original regression $T_c$: {tc_pred:.4f} K")
    print(f"Final prediction $T_c$: {final_tc:.4f} K")

if __name__ == "__main__":
    TEST_CIF = r"/path/to/your/cif"
    run_prediction(TEST_CIF)
