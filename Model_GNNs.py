"""
GNN Models.
"""

# Libraries
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch_geometric.nn import ClusterGCNConv  # noqa




class ClusterGCNDNNModel(nn.Module):
    """
    ClusterGCN Model.
    Cluster-GCN layers + linear layers

    """
    def __init__(self, num_node_features, num_classes, layer_size1, layer_size2):
        super().__init__()

        self.conv_layer1 = ClusterGCNConv(num_node_features, 256)

        self.conv_layer2 = ClusterGCNConv(256, 128)

        self.linear_layer1 = nn.Linear(128, layer_size1)

        self.linear_layer2 = nn.Linear(layer_size1, layer_size2)

        self.linear_layer3 = nn.Linear(layer_size2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # first layer
        x = self.conv_layer1(x, edge_index)
        x = F.leaky_relu(x)

        # second layer
        x = self.conv_layer2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.linear_layer1(x)
        x = F.leaky_relu(x)

        x = self.linear_layer2(x)
        x = F.leaky_relu(x)

        output = self.linear_layer3(x)

        return output

