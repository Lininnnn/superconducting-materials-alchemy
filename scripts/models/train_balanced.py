import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from gnn_model_physics import PhysicsInformedGNN # 继续沿用之前的模型架构进行对比
import glob
import numpy as np
from tqdm import tqdm

# --- 路径配置 ---
TENSOR_DIR = r'D:\works\science\材料\当前论文\论文文件夹\data\processed\graph_tensors'

def train_with_balancing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    dataset_files = glob.glob(os.path.join(TENSOR_DIR, "*.pt"))
    dataset = [torch.load(f) for f in tqdm(dataset_files, desc="加载张量")]
    
    # 2. 计算采样权重 (核心步骤)
    # 我们希望 Tc 越高的样本，权重越大
    all_tc = np.array([d.y.item() for d in dataset])
    
    # 定义权重函数：这里使用 1 + Tc/10，意味着 100K 的样本被抽到的概率是 0K 样本的 11 倍
    # 你也可以尝试更激进的平方权重：(tc + 1)**2
    weights = torch.DoubleTensor(1.0 + (all_tc / 10.0))
    
    # 划分训练/测试索引
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8 * len(indices))]
    test_idx = indices[int(0.8 * len(indices)):]
    
    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    # 针对训练集创建采样器
    train_weights = weights[train_idx]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    # 3. 创建 DataLoader
    # 注意：使用 sampler 时，shuffle 必须为 False
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 4. 初始化模型
    # 注意：node_features 现在应该是 6 (如果你更新了特征提取)
    model = PhysicsInformedGNN(node_features=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print(f"🚀 开始平衡采样训练...")

    for epoch in range(1, 101):
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            tc_pred, _ = model(data.x, data.edge_index, data.batch)
            
            # 使用 Log-MSE 进一步平衡量级，防止高 Tc 的绝对误差主导梯度
            loss = F.mse_loss(torch.log(tc_pred + 1), torch.log(data.y + 1))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Log-Loss: {train_loss / len(train_dataset):.4f}")

    # 5. 最终验证 (关键：观察高 Tc 表现)
    validate_performance(model, test_loader, device)

def validate_performance(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred, _ = model(data.x, data.edge_index, data.batch)
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 重点观察 Tc > 30K 的样本
    high_tc_mask = y_true > 30
    if any(high_tc_mask):
        rmse_high = np.sqrt(np.mean((y_true[high_tc_mask] - y_pred[high_tc_mask])**2))
        print(f"\n📊 [高Tc区域 >30K] 样本数: {sum(high_tc_mask)}, RMSE: {rmse_high:.2f}K")
    else:
        print("\n⚠️ 测试集中没有 Tc > 30K 的样本")
    
    overall_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    print(f"📊 [全局] RMSE: {overall_rmse:.2f}K")

if __name__ == "__main__":
    train_with_balancing()