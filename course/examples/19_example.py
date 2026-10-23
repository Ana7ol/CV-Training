# Vision transformers and attention
import torch
from torch import nn
x = torch.randn(2, 3, 64, 64)
patches = nn.Conv2d(3, 48, kernel_size=8, stride=8)(x)
tokens = patches.flatten(2).transpose(1,2)  # [2,64,48]
position = nn.Parameter(torch.zeros(1,64,48))
attention = nn.MultiheadAttention(48, num_heads=4, batch_first=True)
z, weights = attention(tokens+position, tokens+position, tokens,
                       need_weights=True)
print(z.shape, weights.shape)
# Teaching fragment: a full ViT also has norms, MLPs, residuals and a head.
