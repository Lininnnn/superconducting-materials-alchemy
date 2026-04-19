import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphNorm

class SuperconductorHybridGNN(torch.nn.Module):
    def __init__(self, node_features=6, hidden_channels=128, heads=4):
        super().__init__()
        # 使用多头注意力提取结构特征
        self.conv1 = GATv2Conv(node_features, hidden_channels, heads=heads)
        self.norm1 = GraphNorm(hidden_channels * heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.norm2 = GraphNorm(hidden_channels * heads)

        # 共享特征层：加入 Dropout 和 LayerNorm 稳定深层网络
        self.shared_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        # 多任务分支
        self.classifier = torch.nn.Linear(hidden_channels, 1)  # 分类：是否为高温
        self.regressor = torch.nn.Linear(hidden_channels, 1)   # 回归：Tc
        self.lambda_head = torch.nn.Linear(hidden_channels, 1) # 物理约束

    def forward(self, x, edge_index, batch, training=True):
        # 数据增强：训练时加入极微小噪声 (0.005)
        if training:
            noise = torch.randn_like(x) * 0.005
            x = x + noise

        x = F.elu(self.norm1(self.conv1(x, edge_index)))
        x = F.elu(self.norm2(self.conv2(x, edge_index)))
        
        x_pool = global_mean_pool(x, batch)
        feat = self.shared_fc(x_pool)
        
        is_high_logits = self.classifier(feat).view(-1)
        # 限制 Tc 预测范围，避免数值爆炸
        tc_pred = F.softplus(self.regressor(feat)).view(-1)
        tc_pred = torch.clamp(tc_pred, max=200.0) 
        
        lambda_sim = torch.sigmoid(self.lambda_head(feat)).view(-1) * 3.0
        
        return is_high_logits, tc_pred, lambda_sim