import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphNorm

class SuperconductorGAT(torch.nn.Module):
    def __init__(self, node_features=6, hidden_channels=128, heads=4):
        super(SuperconductorGAT, self).__init__()
        # heads=4 意味着有 4 组独立的注意力机制同时观察晶体结构
        self.conv1 = GATv2Conv(node_features, hidden_channels, heads=heads)
        self.norm1 = GraphNorm(hidden_channels * heads)
        
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.norm2 = GraphNorm(hidden_channels * heads)

        self.post_conv = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        
        self.lambda_head = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, x, edge_index, batch):
        # 注意力卷积层
        x = F.elu(self.norm1(self.conv1(x, edge_index)))
        x = F.elu(self.norm2(self.conv2(x, edge_index)))

        # 全局池化
        x_pool = global_mean_pool(x, batch)
        
        tc_pred = F.softplus(self.post_conv(x_pool)).view(-1)
        lambda_sim = torch.sigmoid(self.lambda_head(x_pool)).view(-1) * 3.0
        
        return tc_pred, lambda_sim