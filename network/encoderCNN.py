from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=512):
        super(EncoderCNN, self).__init__()
        # get the pretrained densenet model
        initial_cnn = models.densenet121(pretrained=True)
        self.cnn = torch.nn.Sequential(*(list(initial_cnn.children())[:-1]))
        # Freeze model weights
        for param in self.cnn.parameters():
            param.requires_grad = True

    def forward(self, images):
        out = self.cnn(images)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        return out