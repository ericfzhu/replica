import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self) -> None:
        super(AlexNet, self).__init__()
        
        self.conv1 = 