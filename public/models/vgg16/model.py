import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self) -> None:
        super(VGG16, self).__init__()



    def forward(self, x):
        return x