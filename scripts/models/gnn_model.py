import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SuperconductorGNN(torch.nn.Module):
    def __init__(self, node_features=1, hidden_channels=64):
        super(SuperconductorGNN, self).__init__()
        # 1. 图卷积层：学习原子之间的局部环境
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # 2. 全连接输出层：将图特征映射为预测的 Tc 值
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. 节点特征演化
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout: 将所有节点的特征聚合成整个晶体的特征
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 回归预测
        x = self.lin(x)
        return x.view(-1)