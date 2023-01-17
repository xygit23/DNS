import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential


class GCN(nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GCN, self).__init__()

        self.input_size, self.output_size = feature, classes

        layers = []
        for i in range(len(hidden)):
            if i == 0:
                layers.append((GCNConv(feature, hidden[i]), 'x, edge_index -> x'), )
            else:
                layers.append((GCNConv(hidden[i-1], hidden[i]), 'x, edge_index -> x'), )
            layers.append(nn.ReLU())
            layers.append((nn.Dropout(0.5), 'x -> x'), )
        layers.append((GCNConv(hidden[-1], classes), 'x, edge_index -> x'), )

        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()
