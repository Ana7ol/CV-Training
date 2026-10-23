# PointNet and point-wise 3D segmentation
import torch
from torch import nn
torch.manual_seed(5)
points = torch.randn(2, 3, 128)  # batch, xyz, points
shared = nn.Sequential(nn.Conv1d(3,32,1),nn.ReLU(),nn.Conv1d(32,64,1))
head = nn.Conv1d(128,4,1)
local = shared(points)
global_feature = local.amax(dim=2,keepdim=True)
logits = head(torch.cat([local,global_feature.expand_as(local)],dim=1))
print(logits.shape)  # [2, 4 classes, 128 points]
permutation = torch.randperm(128)
print(torch.allclose(shared(points[:,:,permutation]).amax(2), local.amax(2)))
