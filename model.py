import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize weights and biases
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight.data, 0.0, 0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.01)

# Instance Normalization Layer
class InstanceNorm(nn.Module):
    def __init__(self, num_features):
        super(InstanceNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-3

    def forward(self, x):
        mean = x.mean([2, 3], keepdim=True)
        var = x.var([2, 3], keepdim=True)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

# ResNet Generator Model
class ResNetGenerator(nn.Module):
    def __init__(self):
        super(ResNetGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)

        # Residual Block 1
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.in2 = InstanceNorm(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.in3 = InstanceNorm(64)

        # Residual Block 2
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.in4 = InstanceNorm(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.in5 = InstanceNorm(64)

        # Residual Block 3
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.in6 = InstanceNorm(64)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.in7 = InstanceNorm(64)

        # Residual Block 4
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.in8 = InstanceNorm(64)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.in9 = InstanceNorm(64)

        # Convolutional layers
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Final layers
        self.conv12 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))

        # Residual 1
        c2 = F.relu(self.in2(self.conv2(c1)))
        c3 = F.relu(self.in3(self.conv3(c2))) + c1

        # Residual 2
        c4 = F.relu(self.in4(self.conv4(c3)))
        c5 = F.relu(self.in5(self.conv5(c4))) + c3

        # Residual 3
        c6 = F.relu(self.in6(self.conv6(c5)))
        c7 = F.relu(self.in7(self.conv7(c6))) + c5

        # Residual 4
        c8 = F.relu(self.in8(self.conv8(c7)))
        c9 = F.relu(self.in9(self.conv9(c8))) + c7

        # Convolutional
        c10 = F.relu(self.conv10(c9))
        c11 = F.relu(self.conv11(c10))

        # Final
        out = torch.tanh(self.conv12(c11)) * 0.58 + 0.5

        return out

# Adversarial Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        
        x = x.view(x.size(0), -1)  # Flatten

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.softmax(self.fc2(x), dim=1)
        return x

# Initialize models and apply weight initializer
generator = ResNetGenerator()
generator.apply(initialize_weights)

discriminator = Discriminator()
discriminator.apply(initialize_weights)
