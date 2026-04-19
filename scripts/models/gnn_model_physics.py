import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm

class PhysicsInformedGNN(torch.nn.Module):
    def __init__(self, node_features=1, hidden_channels=128):
        super(PhysicsInformedGNN, self).__init__()
        # 加入 GraphNorm 稳定深层网络训练
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # 物理参数分支
        self.fc = torch.nn.Linear(hidden_channels, hidden_channels)
        self.tc_out = torch.nn.Linear(hidden_channels, 1)
        self.lambda_out = torch.nn.Linear(hidden_channels, 1) # 预测耦合常数 lambda

    def forward(self, x, edge_index, batch):
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = F.relu(self.norm2(self.conv2(x, edge_index)))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc(x))
        
        tc_pred = F.softplus(self.tc_out(x)).view(-1) # Tc 必须为正
        # lambda 通常在 0.1 到 3.0 之间
        lambda_sim = torch.sigmoid(self.lambda_out(x)).view(-1) * 3.0 
        
        return tc_pred, lambda_sim