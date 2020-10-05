import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class affinity_finalNet(torch.nn.Module):
    def __init__(self):
        super(affinity_finalNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 1), #loads features from two nodes and features of their edge (edge of interest)
            nn.ReLU()
        )

    def forward(self, inputs):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # out = torch.cat([x1, x2, x3], 0)
        out = self.mlp(inputs)
        return out
