# CNNs: learnable kernels and receptive fields
import torch
from torch import nn
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(32, 4))
x = torch.randn(2, 3, 64, 64)
print("Logits:", model(x).shape)  # [2, 4]
print("Parameters:", sum(p.numel() for p in model.parameters()))
