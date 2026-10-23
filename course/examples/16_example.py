# Semantic, instance, and panoptic segmentation
import torch
from torch import nn
x = torch.randn(2, 3, 64, 64)
target = torch.zeros(2, 64, 64, dtype=torch.long)
target[:, 16:48, 20:44] = 1
head = nn.Conv2d(3, 2, kernel_size=1)
logits = head(x)  # untrained: shape demonstration only
loss = nn.functional.cross_entropy(logits, target)
pred = logits.argmax(dim=1)
print(logits.shape, pred.shape, loss.item())
