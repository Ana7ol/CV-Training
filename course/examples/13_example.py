# Residual networks and transfer learning
import torch
from torch import nn
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1))
    def forward(self, x):
        return torch.relu(x + self.f(x))
x = torch.randn(2, 16, 32, 32)
print(ResidualBlock(16)(x).shape)
# Transfer: torchvision.models.resnet18(weights="DEFAULT")
# Replace model.fc; use the selected weights' transforms().
