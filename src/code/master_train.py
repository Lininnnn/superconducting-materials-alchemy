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

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

# ============================================================
# ⚙️ 路径与精调配置
# ============================================================
BASE_PATH = r'D:\works\science\material\paper_now\paper'
DATA_DIR = Path(BASE_PATH) / 'data/processed/graph_tensors_v12_rbf'
CHECKPOINT_DIR = Path(BASE_PATH) / 'checkpoints'

# 🚀 修改：指向 continue 脚本生成的精修权重
BIN_CLASSIFIER_PATH = CHECKPOINT_DIR / 'v12_xgb_bin_refined.pkl'
MULTI_CLASSIFIER_PATH = CHECKPOINT_DIR / 'v12_xgb_multi_refined.pkl'

STATS_PATH = CHECKPOINT_DIR / 'v12_master_stats.pth'
PRETRAINED_REG_PATH = CHECKPOINT_DIR / 'v12_best_regressor.pth'

CONFIG = {
    "seed": 42,
    "hidden_dim": 256,
    "dropout": 0.05,
    "reg_lr": 0.001,      # 冻结分类器后，使用低学习率微调回归器
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

# ============================================================
# 🛠️ 核心定义：混合分类器 (对接精修后的分体权重)
# ============================================================
class V12XGBHybridClassifier:
    """
    该类负责组合二分类器(bin)和三分类器(multi)，并为回归模型提供引导概率。
    """
    def __init__(self, bin_model, multi_model):
        self.xgb_bin = bin_model
        self.xgb_multi = multi_model

    def predict(self, X):
        bin_preds = self.xgb_bin.predict(X)
        final_preds = np.zeros(len(X), dtype=int)
        sc_idx = np.where(bin_preds == 1)[0]
        if len(sc_idx) > 0:
            # SC 类别：multi 的结果 (0,1,2) + 1 -> (1,2,3)
            final_preds[sc_idx] = self.xgb_multi.predict(X[sc_idx]) + 1
        return final_preds

    def predict_probs_for_reg(self, X):
        # 1. 获取二分类概率 [P(Non-SC), P(SC)]
        bin_prob = self.xgb_bin.predict_proba(X) 
        # 2. 初始化三分类概率 [P(Cu), P(Fe), P(Other)]
        multi_prob = np.zeros((len(X), 3))
        
        # 仅对判定为超导的样本提取细分概率
        sc_mask = self.xgb_bin.predict(X) == 1
        sc_idx = np.where(sc_mask)[0]
        if len(sc_idx) > 0:
            multi_prob[sc_idx] = self.xgb_multi.predict_proba(X[sc_idx])
            
        # 拼接为 5 维特征向量作为 GNN 的先验引导
        return np.concatenate([bin_prob, multi_prob], axis=1)

# ================= 1. 物理特征聚合 (88维) =================
def aggregate_physical_features(data):
    x = data.x.cpu().numpy()
    e = data.edge_attr.cpu().numpy() if hasattr(data, 'edge_attr') else np.zeros((1, 32))
    x_mean, x_max, x_std = np.mean(x, axis=0), np.max(x, axis=0), np.std(x, axis=0)
    e_mean, e_max = np.mean(e, axis=0), np.max(e, axis=0)
    return np.concatenate([x_mean, x_max, x_std, e_mean, e_max])

# ================= 2. 回归模型 (结构保持不变) =================
class V11Regressor(nn.Module):
    def __init__(self, node_in=13, edge_in=2, hidden_dim=CONFIG["hidden_dim"]):
        super().__init__()
        # node_in = 8 (原始) + 5 (分类器概率) = 13
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
        # 将全局图分类概率广播到每个节点
        guidance = cls_probs[data.batch]
        combined_node_x = torch.cat([data.x, guidance], dim=-1) 
        
        h = self.node_emb(combined_node_x)
        e = self.edge_emb(data.edge_attr_norm)
        
        h1 = self.conv1(h, data.edge_index, e)
        h2 = self.conv2(h1, data.edge_index, e) + h1
        h3 = self.conv3(h2, data.edge_index, e) + h2
        h4 = self.conv4(h3, data.edge_index, e) + h3
        
        # 多尺度池化聚合
        out = torch.cat([
            global_mean_pool(h4, data.batch), 
            global_max_pool(h4, data.batch), 
            self.attn_pool(h4, data.batch)
        ], dim=1)
        
        # 最终预测结果限制在非负空间，且对 Non-SC 强行置零
        return (self.head(out).view(-1) * (data.y >= 0.001).float()).clamp(min=0.0)

# ================= 3. 数据预处理器 =================
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

# ================= 4. 精调主流程 =================
def run_v12_refine_pipeline():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 V12-RBF 级联架构精调 (分类器已冻结) | 设备: {DEVICE}")

    # A. 加载精修后的分类器权重
    if not BIN_CLASSIFIER_PATH.exists() or not MULTI_CLASSIFIER_PATH.exists():
        print("❌ 错误：找不到精修后的分类器权重 (v12_xgb_bin_refined.pkl/v12_xgb_multi_refined.pkl)")
        return
    
    xgb_bin = joblib.load(BIN_CLASSIFIER_PATH)
    xgb_multi = joblib.load(MULTI_CLASSIFIER_PATH)
    # 实例化混合分类器
    hybrid_cls = V12XGBHybridClassifier(xgb_bin, xgb_multi)
    print("✅ 已成功加载精修版分类器并执行冻结策略")

    processor = RefineProcessor(STATS_PATH, DEVICE)

    # B. 数据准备 (保持种子 42 与 continue 脚本一致)
    all_pts = sorted(list(DATA_DIR.glob('*.pt')))
    raw_dataset = [torch.load(pt, weights_only=False) for pt in tqdm(all_pts, desc="Loading Data")]
    
    np.random.seed(42)
    indices = np.random.permutation(len(raw_dataset))
    split = int(0.8 * len(raw_dataset))
    train_raw = [raw_dataset[i] for i in indices[:split]]
    val_raw = [raw_dataset[i] for i in indices[split:]]

    # C. 验证分类器在当前划分下的性能
    print("--- 验证精修分类器效能 ---")
    X_val_phys = np.array([aggregate_physical_features(d) for d in val_raw])
    y_val_true = np.array([d.sc_class.item() for d in val_raw])
    
    y_val_pred = hybrid_cls.predict(X_val_phys)
    print("\n" + "="*45)
    print(f"⭐ 精修分类器验证 (ACC): {accuracy_score(y_val_true, y_val_pred):.2%}")
    print(classification_report(y_val_true, y_val_pred, target_names=CLASS_NAMES))
    print("="*45)

    # D. 加载回归器
    reg_model = V11Regressor().to(DEVICE)
    if PRETRAINED_REG_PATH.exists():
        reg_model.load_state_dict(torch.load(PRETRAINED_REG_PATH, map_location=DEVICE))
        print("✅ 已加载回归器初始预训练权重")

    # 仅优化 reg_model 的参数
    optimizer = optim.AdamW(reg_model.parameters(), lr=CONFIG["reg_lr"])
    criterion = nn.HuberLoss(delta=CONFIG["delta"], reduction='none')
    
    train_loader = DataLoader(train_raw, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_raw, batch_size=CONFIG["batch_size"])
    
    best_r2 = -float('inf')
    
    # E. 训练循环
    for epoch in range(1, CONFIG["epochs"] + 1):
        reg_model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            data = processor.preprocess(data)
            
            # 【核心冻结步骤】：分类器仅做推理，不涉及计算图
            with torch.no_grad():
                # 提取物理特征并获取精修分类器的概率引导
                phys_feats = np.array([aggregate_physical_features(d) for d in data.to_data_list()])
                cls_probs = torch.FloatTensor(hybrid_cls.predict_probs_for_reg(phys_feats)).to(DEVICE)
            
            optimizer.zero_grad()
            out = reg_model(data, cls_probs)
            
            # 计算带权重的损失，仅针对超导样本（y>=0.001）
            mask = (data.y >= 0.001).float()
            loss = (criterion(out, data.y_log) * data.weight * mask).sum() / (mask.sum() + 1e-6)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 评估阶段
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
                    # 指数还原
                    preds = torch.expm1(out).clamp(min=0.0).cpu().numpy()
                    y_true_all.extend(data.y.cpu().numpy())
                    y_pred_all.extend(preds)
            
            current_r2 = r2_score(y_true_all, y_pred_all)
            if current_r2 > best_r2:
                best_r2 = current_r2
                torch.save(reg_model.state_dict(), CHECKPOINT_DIR / 'v12_refined_regressor_final.pth')
            
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss/len(train_loader):.4f} | R2: {current_r2:.4f} | Best: {best_r2:.4f}")

    print(f"\n🎉 流程结束！配合精修分类器后的最佳 R2: {best_r2:.4f}")

if __name__ == "__main__":
    run_v12_refine_pipeline()