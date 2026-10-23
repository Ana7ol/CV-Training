# U-Net: context meets precise boundaries
import torch
from torch import nn
import torch.nn.functional as F
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU())
        self.low = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(48,16,3,padding=1), nn.ReLU())
        self.head = nn.Conv2d(16,2,1)
    def forward(self, x):
        skip = self.enc(x)
        low = self.low(F.max_pool2d(skip,2))
        up = F.interpolate(low, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.dec(torch.cat([skip,up],dim=1)))
print(TinyUNet()(torch.randn(2,1,64,64)).shape)
