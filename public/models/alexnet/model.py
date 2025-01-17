import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # First layer (96 -> 2x48 in original paper)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second layer (256 -> 2x128 in original paper)
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Third layer
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth layer
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            
            # Fifth layer
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    # Set bias to 1 for 2nd, 4th, and 5th conv layers
                    layer_name = m._get_name()
                    if 'conv2' in layer_name or 'conv4' in layer_name or 'conv5' in layer_name:
                        nn.init.constant_(m.bias, 1)
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use same initialization for all layers
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # Set bias to 1 for hidden layers, 0 for output layer
                if m.out_features != 1000:  # Hidden layer
                    nn.init.constant_(m.bias, 1)
                else:  # Output layer
                    nn.init.constant_(m.bias, 0)