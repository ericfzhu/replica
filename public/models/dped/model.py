import torch
import torch.nn as nn

class ResNeXt(nn.Module):
    def __init__(self) -> None:
        super(ResNeXt, self).__init__()
        self.inplanes = 64

        # Initial convolutions
        self.conv1      = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1        = nn.BatchNorm2d(64)
        self.relu       = nn.LeakyReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNeXt blocks
        self.layer1     = self._layer(planes=64, blocks=3)
        self.layer2     = self._layer(planes=128, blocks=4, stride=2)
        self.layer3     = self._layer(planes=256, blocks=6, stride=2)
        self.layer4     = self._layer(planes=512, blocks=3, stride=2)

        # Deconvolutional layers
        self.deconv1    = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2)
        self.deconv2    = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2)
        self.deconv3    = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2)
        self.deconv4    = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.deconv5    = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1)
        self.deconv6    = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


    def _layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=self.inplanes, out_channels=planes * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 4)
        )
        
        layers = []
        layers.append(Block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * 4

        for _ in range(1, blocks):
            layers.append(Block(inplanes=self.inplanes, planes=planes))
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        out = self.conv1(x) # 7x7, 64, stride=2
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out) # 3x3, 64, stride=2

        out = self.layer1(out) # 3x3, 256, stride=1
        out = self.layer2(out) # 3x3, 512, stride=2
        out = self.layer3(out) # 3x3, 1024, stride=2
        out = self.layer4(out) # 3x3, 2048, stride=2

        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)

        return out
    

class Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super(Block, self).__init__()
        width_per_group = 4
        conv_groups = 32
        width = int(planes * (width_per_group / 64)) * conv_groups
        self.conv1      = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(width)
        self.conv2      = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=conv_groups, bias=False)
        self.bn2        = nn.BatchNorm2d(width)
        self.conv3      = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes * 4)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # groups=32
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# create a descriminator model
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.conv1      = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu       = nn.LeakyReLU(inplace=True)
        self.conv2      = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1        = nn.BatchNorm2d(64)
        self.conv3      = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(128)
        self.conv4      = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3        = nn.BatchNorm2d(128)
        self.conv5      = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4        = nn.BatchNorm2d(256)
        self.conv6      = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5        = nn.BatchNorm2d(256)
        self.conv7      = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.fc         = nn.Linear(512, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv7(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        # out = self.sigmoid(out)

        return out