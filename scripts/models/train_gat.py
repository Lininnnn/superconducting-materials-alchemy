import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from gnn_model_gat import SuperconductorGAT
import glob
import numpy as np
from tqdm import tqdm

# --- 路径配置 ---
TENSOR_DIR = r'D:\works\science\材料\当前论文\论文文件夹\data\processed\graph_tensors'

def train_with_gat():
    # 0. 设定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在启动 GATv2 注意力模型训练 (设备: {device})...")
    
    # 1. 加载数据
    dataset_files = glob.glob(os.path.join(TENSOR_DIR, "*.pt"))
    if not dataset_files:
        print(f"❌ 错误：在 {TENSOR_DIR} 没找到 .pt 文件！")
        return
        
    dataset = [torch.load(f) for f in tqdm(dataset_files, desc="加载张量")]
    
    # 2. 平衡采样逻辑 (保留实验 1 的成功经验)
    all_tc = np.array([d.y.item() for d in dataset])
    weights = torch.DoubleTensor(1.0 + (all_tc / 10.0))
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8 * len(indices))]
    test_idx = indices[int(0.8 * len(indices)):]
    
    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    train_weights = weights[train_idx]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 3. 初始化 GAT 模型
    # node_features=6 是因为我们升级了特征提取
    model = SuperconductorGAT(node_features=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 4. 训练循环
    for epoch in range(1, 101):
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # GAT 模型输出
            tc_pred, _ = model(data.x, data.edge_index, data.batch)
            
            # 损失函数继续使用 Log-MSE
            loss = F.mse_loss(torch.log(tc_pred + 1), torch.log(data.y + 1))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Log-Loss: {train_loss / len(train_dataset):.4f}")

    # 5. 验证性能
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
    
    high_tc_mask = y_true > 30
    if any(high_tc_mask):
        rmse_high = np.sqrt(np.mean((y_true[high_tc_mask] - y_pred[high_tc_mask])**2))
        print(f"\n📊 [GATv2 高Tc区域 >30K] 样本数: {sum(high_tc_mask)}, RMSE: {rmse_high:.2f}K")
    
    overall_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    print(f"📊 [GATv2 全局] RMSE: {overall_rmse:.2f}K")

# --- 核心：确保这个入口存在 ---
if __name__ == "__main__":
    train_with_gat()