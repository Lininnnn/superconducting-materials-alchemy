import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from gnn_model import SuperconductorGNN
import glob
from tqdm import tqdm

# --- 配置 ---
TENSOR_DIR = r'D:\works\science\材料\当前论文\论文文件夹\data\processed\graph_tensors'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

def train():
    # 1. 加载所有生成的 pt 数据
    dataset_files = glob.glob(os.path.join(TENSOR_DIR, "*.pt"))
    dataset = [torch.load(f) for f in tqdm(dataset_files, desc="加载张量数据")]
    
    # 2. 划分数据集 (80% 训练, 20% 测试)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 3. 初始化模型、优化器和损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperconductorGNN(node_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss() # 回归任务使用均方误差

    print(f"🚀 开始训练 (设备: {device})...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        # 每个 Epoch 打印一次误差
        avg_loss = total_loss / len(train_loader.dataset)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}")

    # 4. 保存模型
    torch.save(model.state_dict(), 'superconductor_model.pth')
    print("✅ 训练完成，模型已保存！")

if __name__ == "__main__":
    train()