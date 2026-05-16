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

# ============================================================
# ⚙️ 配置区
# ============================================================
PROJ_ROOT = Path(r'D:\works\science\material\paper_now\paper')
DATA_DIR = PROJ_ROOT / 'data/processed/graph_tensors_v12_rbf' 
SAVE_DIR = PROJ_ROOT / 'checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["Non-SC", "Cu-based", "Fe-based", "Other"]

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# ============================================================
# 📊 物理特征聚合 (同步高性能分类器逻辑)
# ============================================================
def aggregate_physical_features(data):
    x = data.x.cpu().numpy()
    e = data.edge_attr.cpu().numpy() if hasattr(data, 'edge_attr') else np.zeros((1, 32))
    
    # 提取物理统计特征 (8*3 + 32*2 = 88维)
    x_mean, x_max, x_std = np.mean(x, axis=0), np.max(x, axis=0), np.std(x, axis=0)
    e_mean, e_max = np.mean(e, axis=0), np.max(e, axis=0)
    
    return np.concatenate([x_mean, x_max, x_std, e_mean, e_max])

# ============================================================
# 🏗️ 模型定义
# ============================================================

class V12XGBHybridClassifier:
    def __init__(self):
        # 使用高性能参数配置
        self.xgb_bin = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=10, tree_method='hist', random_state=42)
        self.xgb_multi = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=10, random_state=42)

    def train(self, X, y):
        print("--- 正在训练 V12 级联分类器 (物理统计特征版) ---")
        y_bin = (y > 0).astype(int)
        self.xgb_bin.fit(X, y_bin, sample_weight=np.where(y_bin == 0, 1.5, 1.0))
        mask = y > 0
        if mask.any():
            self.xgb_multi.fit(X[mask], y[mask] - 1)
            
    def predict(self, X):
        """标准预测逻辑，用于评估 ACC"""
        bin_preds = self.xgb_bin.predict(X)
        final_preds = np.zeros(len(X), dtype=int)
        sc_idx = np.where(bin_preds == 1)[0]
        if len(sc_idx) > 0:
            final_preds[sc_idx] = self.xgb_multi.predict(X[sc_idx]) + 1
        return final_preds

    def predict_probs_for_reg(self, X):
        """为回归器提供级联引导"""
        bin_prob = self.xgb_bin.predict_proba(X)
        multi_prob = np.zeros((len(X), 3))
        sc_idx = np.where(self.xgb_bin.predict(X) == 1)[0]
        if len(sc_idx) > 0:
            multi_prob[sc_idx] = self.xgb_multi.predict_proba(X[sc_idx])
        return np.concatenate([bin_prob, multi_prob], axis=1)

class V11Regressor(nn.Module):
    def __init__(self, node_in=13, edge_in=2, hidden_dim=256): 
        super().__init__()
        self.node_emb = nn.Linear(node_in, hidden_dim)
        self.edge_emb = nn.Linear(edge_in, hidden_dim)
        def make_conv():
            return GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.SiLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim)
            ))
        self.conv1, self.conv2, self.conv3, self.conv4 = make_conv(), make_conv(), make_conv(), make_conv()
        self.attn_pool = AttentionalAggregation(nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)))
        self.head = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, 1))
        
    def forward(self, data, cls_probs):
        guidance = cls_probs[data.batch]
        combined_node_x = torch.cat([data.x, guidance], dim=-1)
        h = self.node_emb(combined_node_x)
        e = self.edge_emb(data.edge_attr[:, :2])
        h1 = self.conv1(h, data.edge_index, e)
        h2 = self.conv2(h1, data.edge_index, e) + h1
        h3 = self.conv3(h2, data.edge_index, e) + h2
        h4 = self.conv4(h3, data.edge_index, e) + h3
        out = torch.cat([global_mean_pool(h4, data.batch), global_max_pool(h4, data.batch), self.attn_pool(h4, data.batch)], dim=1)
        return (self.head(out).view(-1) * (data.y >= 0.001).float()).clamp(min=0.0)

# ============================================================
# 📊 数据预处理
# ============================================================
class MasterProcessor:
    def __init__(self, dataset, device):
        self.device = device
        all_x = torch.cat([d.x for d in dataset], dim=0)
        self.mean, self.std = all_x.mean(dim=0).to(device), all_x.std(dim=0).to(device)
        self.std[self.std < 1e-5] = 1.0
        all_e = torch.cat([d.edge_attr[:, :2] for d in dataset], dim=0)
        self.e_mean, self.e_std = all_e.mean(dim=0).to(device), all_e.std(dim=0).to(device)
        self.e_std[self.e_std < 1e-5] = 1.0

# ============================================================
# 🚀 预训练主流程
# ============================================================
def run_v12_pretrain_pipeline():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 V12 预训练启动 | 设备: {DEVICE}")

    # 1. 加载数据
    pt_files = list(DATA_DIR.rglob('*.pt'))
    raw_dataset = [torch.load(p, weights_only=False) for p in tqdm(pt_files, desc="加载数据")]
    
    # 划分数据集 (固定种子 42 保证一致性)
    np.random.seed(42)
    indices = np.random.permutation(len(raw_dataset))
    split = int(0.8 * len(raw_dataset))
    train_raw = [raw_dataset[i] for i in indices[:split]]
    val_raw = [raw_dataset[i] for i in indices[split:]]
    
    # 2. 统计量保存
    processor = MasterProcessor(train_raw, DEVICE)
    torch.save({
        "mean": processor.mean.cpu(), "std": processor.std.cpu(),
        "e_mean": processor.e_mean.cpu(), "e_std": processor.e_std.cpu()
    }, SAVE_DIR / 'v12_master_stats.pth')

    # 3. 分类器训练与 ACC 矩阵输出
    print("\n--- 正在提取物理特征 ---")
    X_train_phys = np.array([aggregate_physical_features(d) for d in train_raw])
    X_val_phys = np.array([aggregate_physical_features(d) for d in val_raw])
    y_train = np.array([d.sc_class.item() for d in train_raw])
    y_val = np.array([d.sc_class.item() for d in val_raw])

    hybrid_cls = V12XGBHybridClassifier()
    hybrid_cls.train(X_train_phys, y_train)
    
    # ⭐ 评估与打印 ACC 报告
    y_val_pred = hybrid_cls.predict(X_val_phys)
    acc = accuracy_score(y_val, y_val_pred)
    
    print("\n" + "="*45)
    print(f"⭐ V12 分类结果 (ACC): {acc:.2%}")
    print(classification_report(y_val, y_val_pred, target_names=CLASS_NAMES))
    print("="*45)
    print(f"✅ 分类预估完成，模型已就绪。")

    # 4. 保存全部资产
    joblib.dump(hybrid_cls, SAVE_DIR / 'v12_best_classifier.pkl')
    reg_model = V11Regressor().to(DEVICE)
    torch.save(reg_model.state_dict(), SAVE_DIR / 'v12_best_regressor.pth')

    print(f"\n🎉 资产锁存完毕！\n- 统计量: v12_master_stats.pth\n- 分类器: v12_best_classifier.pkl\n- 回归器: v12_best_regressor.pth")

if __name__ == "__main__":
    run_v12_pretrain_pipeline()