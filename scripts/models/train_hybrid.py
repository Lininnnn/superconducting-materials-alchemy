import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from gnn_model_advanced import SuperconductorHybridGNN
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

# --- 路径配置 ---
TENSOR_DIR = r'D:\works\science\材料\当前论文\论文文件夹\data\processed\graph_tensors'

def preprocess_features(x):
    """严格的特征缩放"""
    x = x.clone()
    x[:, 0] /= 92.0    # 原子序数
    x[:, 1] /= 4.0     # 电负性
    x[:, 2] /= 3.0     # 原子半径
    x[:, 3] /= 100.0   # 门捷列夫序数
    x[:, 4] /= 18.0    # 族
    x[:, 5] /= 7.0     # 周期
    return x

def allen_dynes_constraint(tc_pred, lambda_sim, mu_star=0.13):
    """物理趋势约束"""
    denominator = lambda_sim - mu_star * (1 + 0.62 * lambda_sim)
    denominator = torch.clamp(denominator, min=0.01) 
    exponent = (1.04 * (1 + lambda_sim)) / denominator
    tc_phys_trend = torch.exp(-exponent)
    # 使用 Huber 防止梯度爆炸
    return F.huber_loss(torch.log(tc_pred + 1.0), torch.log(tc_phys_trend * 100.0 + 1.0))

def train_hybrid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌟 启动稳健版混合训练 (采样降温 + 梯度裁剪)")

    # 1. 加载数据
    dataset_files = glob.glob(os.path.join(TENSOR_DIR, "*.pt"))
    dataset = [torch.load(f) for f in tqdm(dataset_files, desc="装载张量")]
    
    # 2. 【核心修改】平滑采样权重：使用 log1p(Tc) + 1
    # 这样高 Tc 样本会有优势，但不会产生几千倍的差距
    all_tc = np.array([d.y.item() for d in dataset])
    weights = torch.DoubleTensor(np.log1p(all_tc) + 1.0) 
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=64, 
                              sampler=WeightedRandomSampler(weights[train_idx], len(train_idx)))
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=64)

    # 3. 初始化
    model = SuperconductorHybridGNN(node_features=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 4. 训练
    for epoch in range(1, 151):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            data.x = preprocess_features(data.x)
            
            optimizer.zero_grad()
            is_high_logits, tc_pred, lambda_sim = model(data.x, data.edge_index, data.batch, training=True)
            
            # (1) 分类损失：>10K
            target_cls = (data.y > 10.0).float()
            loss_cls = F.binary_cross_entropy_with_logits(is_high_logits, target_cls)
            
            # (2) 回归损失：Log-Huber (delta=1.0 更稳定)
            loss_reg = F.huber_loss(torch.log(tc_pred + 1.0), torch.log(data.y + 1.0), delta=1.0)
            
            # (3) 物理损失
            loss_phys = allen_dynes_constraint(tc_pred, lambda_sim)
            
            # 综合损失平衡：提高分类权重，先立住骨架
            loss = 1.0 * loss_cls + 1.0 * loss_reg + 0.1 * loss_phys
            
            loss.backward()
            # 必须进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step(total_loss)
        if epoch % 10 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | LR: {curr_lr:.6f}")

    evaluate_hybrid(model, test_loader, device)

def evaluate_hybrid(model, loader, device):
    model.eval()
    y_true, y_pred, cls_true, cls_pred_logits = [], [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.x = preprocess_features(data.x)
            is_high_logits, tc_pred, _ = model(data.x, data.edge_index, data.batch, training=False)
            
            y_true.extend(data.y.tolist())
            y_pred.extend(tc_pred.tolist())
            cls_true.extend((data.y > 10.0).int().tolist())
            cls_pred_logits.extend(is_high_logits.tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cls_preds = (np.array(cls_pred_logits) > 0).astype(int)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    acc = accuracy_score(cls_true, cls_preds)
    
    print("\n" + "✅"*15)
    print(f"  [修正版报告]")
    print(f"  分类准确率: {acc:.2%}")
    print(f"  整体 R² Score: {r2:.4f}")
    print(f"  整体 RMSE: {rmse:.3f} K")
    print("✅"*15)

    torch.save(model.state_dict(), 'superconductor_hybrid_v4.pth')

if __name__ == "__main__":
    train_hybrid()