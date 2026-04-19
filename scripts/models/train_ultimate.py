import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from gnn_model_gat import SuperconductorGAT
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error

# --- 路径配置 ---
TENSOR_DIR = r'D:\works\science\材料\当前论文\论文文件夹\data\processed\graph_tensors'

def allen_dynes_constraint(tc_pred, lambda_sim, mu_star=0.13):
    """物理趋势对齐损失"""
    denominator = lambda_sim - mu_star * (1 + 0.62 * lambda_sim)
    denominator = torch.clamp(denominator, min=0.01) 
    exponent = (1.04 * (1 + lambda_sim)) / denominator
    tc_phys_trend = torch.exp(-exponent)
    # 对数空间的趋势对比
    p_loss = F.huber_loss(torch.log(tc_pred + 1.0), torch.log(tc_phys_trend * 100.0 + 1.0))
    return p_loss

def preprocess_features(x):
    """
    对输入特征进行手动缩放 (简单归一化)
    [原子序数, 电负性, 原子半径, 门捷列夫序数, 族, 周期]
    """
    x[:, 0] = x[:, 0] / 92.0    # 原子序数归一化
    x[:, 1] = x[:, 1] / 4.0     # 电负性归一化
    x[:, 2] = x[:, 2] / 3.0     # 原子半径归一化
    x[:, 3] = x[:, 3] / 100.0   # 门捷列夫序数
    x[:, 4] = x[:, 4] / 18.0    # 族归一化
    x[:, 5] = x[:, 5] / 7.0     # 周期归一化
    return x

def train_ultimate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 启动 PINN-GATv2 深度优化训练 (设备: {device})")
    
    # 1. 加载数据
    dataset_files = glob.glob(os.path.join(TENSOR_DIR, "*.pt"))
    dataset = [torch.load(f) for f in tqdm(dataset_files, desc="数据装载")]
    
    # 2. 【核心优化】指数级采样权重
    all_tc = np.array([d.y.item() for d in dataset])
    # 使用 exp(Tc/20) 让 90K 的样本权重远高于 1K 样本，强迫模型关注高温区
    weights = torch.DoubleTensor(np.exp(all_tc / 15.0)) 
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=64, 
                              sampler=WeightedRandomSampler(weights[train_idx], len(train_idx)))
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=64)

    # 3. 初始化
    model = SuperconductorGAT(node_features=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8)

    # 4. 训练
    for epoch in range(1, 151):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            data.x = preprocess_features(data.x) # 特征缩放
            
            optimizer.zero_grad()
            tc_pred, lambda_sim = model(data.x, data.edge_index, data.batch)
            
            # 使用 HuberLoss 配合 Log 处理
            loss_data = F.huber_loss(torch.log(tc_pred + 1.0), torch.log(data.y + 1.0), delta=1.0)
            loss_phys = allen_dynes_constraint(tc_pred, lambda_sim)
            
            # 损失加权：0.8 数据 + 0.2 物理
            loss = loss_data + 0.2 * loss_phys
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step(total_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {total_loss/len(train_loader):.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 5. 验证与 R2 计算
    evaluate_and_save(model, test_loader, device)

def evaluate_and_save(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.x = preprocess_features(data.x)
            pred, _ = model(data.x, data.edge_index, data.batch)
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 评估指标
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "🚀"*15)
    print(f"  预测性能综合报告")
    print(f"  RMSE: {rmse:.3f} K")
    print(f"  MAE:  {mae:.3f} K")
    print(f"  R² Score: {r2:.4f}")
    
    # 区间分析
    for t in [10, 30]:
        mask = y_true > t
        if mask.any():
            res = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
            print(f"  Tc > {t}K 区域 RMSE: {res:.2f} K (样本数: {mask.sum()})")
    print("🚀"*15)

    torch.save(model.state_dict(), 'superconductor_ultimate_v2.pth')
    print("\n✅ 权重已更新至 superconductor_ultimate_v2.pth")

if __name__ == "__main__":
    train_ultimate()