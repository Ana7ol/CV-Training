# 3D U-Net and patch-based inference
import torch
from torch import nn
import torch.nn.functional as F
class TinyUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv3d(1,8,3,padding=1),nn.ReLU())
        self.low = nn.Sequential(nn.Conv3d(8,16,3,padding=1),nn.ReLU())
        self.dec = nn.Sequential(nn.Conv3d(24,8,3,padding=1),nn.ReLU())
        self.head = nn.Conv3d(8,2,1)
    def forward(self,x):
        skip = self.enc(x)
        low = self.low(F.max_pool3d(skip,2))
        up = F.interpolate(low,size=skip.shape[-3:],mode='trilinear',align_corners=False)
        return self.head(self.dec(torch.cat([skip,up],1)))
x = torch.randn(1,1,16,32,32)
print(TinyUNet3D()(x).shape)
# Tiny teaching variant; see examples/train_3d.py for a complete experiment.
