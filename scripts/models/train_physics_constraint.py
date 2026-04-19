import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from gnn_model_gat import SuperconductorGAT # 使用表现更好的 GAT 架构
import glob
import numpy as np
from tqdm import tqdm

TENSOR_DIR = r'D:\works\science\材料\当前论文\论文文件夹\data\processed\graph_tensors'

# --- 物理公式约束逻辑 ---
def allen_dynes_constraint(tc_pred, lambda_sim, mu_star=0.13):
    # 简化版公式约束：Tc 与 exp(-1/lambda) 成比例趋势
    denominator = lambda_sim - mu_star * (1 + 0.62 * lambda_sim)
    denominator = torch.clamp(denominator, min=0.01) 
    exponent = (1.04 * (1 + lambda_sim)) / denominator
    tc_phys_trend = torch.exp(-exponent)
    
    # 物理损失：对数空间的趋势对齐
    p_loss = F.mse_loss(torch.log(tc_pred + 1), torch.log(tc_phys_trend * 100 + 1))
    return p_loss

def train_physics_test(use_physics=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    status = "开启" if use_physics else "关闭"
    print(f"🚀 实验：{status} 物理约束训练 (设备: {device})...")
    
    # 1. 加载数据与平衡采样 (保持前两次实验的优化)
    dataset_files = glob.glob(os.path.join(TENSOR_DIR, "*.pt"))
    dataset = [torch.load(f) for f in tqdm(dataset_files, desc="加载张量")]
    
    all_tc = np.array([d.y.item() for d in dataset])
    weights = torch.DoubleTensor(1.0 + (all_tc / 10.0))
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8 * len(indices))]
    test_idx = indices[int(0.8 * len(indices)):]
    
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=64, 
                              sampler=WeightedRandomSampler(weights[train_idx], len(train_idx)))
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=64)

    # 2. 初始化模型
    model = SuperconductorGAT(node_features=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 3. 训练循环
    for epoch in range(1, 101):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            tc_pred, lambda_sim = model(data.x, data.edge_index, data.batch)
            
            # 数据损失 (Log-MSE)
            loss_data = F.mse_loss(torch.log(tc_pred + 1), torch.log(data.y + 1))
            
            # 物理损失
            if use_physics:
                loss_phys = allen_dynes_constraint(tc_pred, lambda_sim)
                total_loss = loss_data + 0.1 * loss_phys # 赋予 0.1 的权重
            else:
                total_loss = loss_data
                
            total_loss.backward()
            optimizer.step()
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}, Total Loss: {total_loss.item():.4f}")

    # 4. 验证性能
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred, _ = model(data.x, data.edge_index, data.batch)
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())
    
    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))
    print(f"📊 [{status}物理约束] 最终测试集 RMSE: {rmse:.2f}K")
    return rmse

if __name__ == "__main__":
    # 分别测试两种情况
    rmse_with = train_physics_test(use_physics=True)
    rmse_without = train_physics_test(use_physics=False)
    
    print(f"\n💡 结果对比：")
    print(f"开启物理约束 RMSE: {rmse_with:.2f}K")
    print(f"关闭物理约束 RMSE: {rmse_without:.2f}K")