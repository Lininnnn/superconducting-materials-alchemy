import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from gnn_model_v6 import SuperconductorResGNN # 确保你已经保存了之前的 ResGNN 模型
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error

# --- 配置 ---
TENSOR_DIR = r'D:\works\science\材料\当前论文\论文文件夹\data\processed\graph_tensors_v3'

def preprocess_9d(x):
    """针对 9 维物理特征的专属归一化"""
    x = x.clone()
    factors = [92, 4, 3, 100, 18, 7, 16, 25, 5]
    for i in range(9):
        x[:, i] /= factors[i]
    return x

def log_cosh_loss(pred, target):
    """比 Huber 更平滑的损失函数，有助于捕捉 Tc 的细微波动"""
    return torch.log(torch.cosh(pred - target) + 1e-12).mean()

def train_pro():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 V6-Pro 决战模式启动 | 设备: {device}")

    # 1. 加载 9D 数据
    dataset_files = glob.glob(os.path.join(TENSOR_DIR, "*.pt"))
    if len(dataset_files) == 0:
        print("❌ 错误：未发现 .pt 文件，请等待转换完成！")
        return
    dataset = [torch.load(f) for f in tqdm(dataset_files, desc="装载 9D 深度张量")]
    
    # 2. 补集平滑采样 (Log-Balanced)
    all_tc = np.array([d.y.item() for d in dataset])
    weights = torch.DoubleTensor(np.log1p(all_tc) + 1.0)
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    
    train_loader = DataLoader([dataset[i] for i in indices[:split]], batch_size=64, 
                              sampler=WeightedRandomSampler(weights[indices[:split]], split))
    test_loader = DataLoader([dataset[i] for i in indices[split:]], batch_size=64)

    # 3. 初始化 V6 残差模型
    model = SuperconductorResGNN(node_features=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    
    # OneCycleLR: 模拟“淬火”过程，冲击全局最优解
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, 
                                                    steps_per_epoch=len(train_loader), epochs=200)

    print("开始 200 轮深度拟合...")
    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            data.x = preprocess_9d(data.x)
            optimizer.zero_grad()
            
            is_high, tc_pred, _ = model(data.x, data.edge_index, data.batch, training=True)
            
            # 多任务损失组合
            l_cls = F.binary_cross_entropy_with_logits(is_high, (data.y > 10).float())
            # 在 Log 空间使用 Log-Cosh 损失，极大增强对 HTS 材料的敏感度
            l_reg = log_cosh_loss(torch.log1p(tc_pred), torch.log1p(data.y))
            
            loss = 1.0 * l_cls + 3.0 * l_reg # 调高回归权重
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            model.eval()
            y_t, y_p = [], []
            with torch.no_grad():
                for d in test_loader:
                    d = d.to(device); d.x = preprocess_9d(d.x)
                    _, p, _ = model(d.x, d.edge_index, d.batch, training=False)
                    y_t.extend(d.y.tolist()); y_p.extend(p.tolist())
            
            r2 = r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | R²: {r2:.4f} | MAE: {mae:.2f}K")

    torch.save(model.state_dict(), 'superconductor_v6_pro_final.pth')
    print("✅ 训练完成，模型已存至 superconductor_v6_pro_final.pth")

if __name__ == "__main__":
    train_pro()