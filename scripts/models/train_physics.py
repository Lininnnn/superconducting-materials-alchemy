import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from gnn_model_physics import PhysicsInformedGNN
import glob
from tqdm import tqdm

# --- 配置 ---
TENSOR_DIR = r'D:\works\science\材料\当前论文\论文文件夹\data\processed\graph_tensors'
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# --- 物理公式 Loss ---
def allen_dynes_constraint(tc_pred, lambda_sim, mu_star=0.13):
    """
    基于 Allen-Dynes 形式的正则项
    """
    denominator = lambda_sim - mu_star * (1 + 0.62 * lambda_sim)
    denominator = torch.clamp(denominator, min=0.01) 
    exponent = (1.04 * (1 + lambda_sim)) / denominator
    tc_phys_trend = torch.exp(-exponent)
    
    # 约束预测值与物理趋势的一致性
    p_loss = F.mse_loss(torch.log(tc_pred + 1), torch.log(tc_phys_trend * 100 + 1))
    return p_loss

def train():
    # --- 关键修复：定义 device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行设备: {device}")

    # 1. 加载数据
    dataset_files = glob.glob(os.path.join(TENSOR_DIR, "*.pt"))
    if not dataset_files:
        print("❌ 错误：在指定目录下未找到 .pt 文件！")
        return
        
    dataset = [torch.load(f) for f in tqdm(dataset_files, desc="加载张量数据")]
    
    # 2. 划分数据集
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 3. 初始化模型
    model = PhysicsInformedGNN(node_features=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    print(f"🚀 开始物理增强训练...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # 模型现在有两个输出
            tc_pred, lambda_sim = model(data.x, data.edge_index, data.batch)
            
            # 数据驱动损失
            loss_data = criterion(tc_pred, data.y)
            # 物理约束损失
            loss_phys = allen_dynes_constraint(tc_pred, lambda_sim)
            
            # 联合损失
            loss = loss_data + 0.1 * loss_phys
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        avg_loss = total_loss / len(train_loader.dataset)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Total Loss: {avg_loss:.4f}")

    # 4. 保存
    torch.save(model.state_dict(), 'superconductor_physics_model.pth')
    print("✅ 训练完成，模型已保存！")

if __name__ == "__main__":
    train()