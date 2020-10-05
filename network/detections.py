from network.encoderCNN import *
import sys
import torch.nn as nn
import os
import numpy as np
from torch_geometric.nn import MetaLayer


class detectionsNet(nn.Module):
    def __init__(self):
        super(detectionsNet, self).__init__()
        self.cnn = EncoderCNN()

    def forward(self, node_attr):
        return self.cnn(node_attr)