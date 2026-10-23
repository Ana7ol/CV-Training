# Learning, losses, and backpropagation
import torch
from torch import nn
torch.manual_seed(7)
x = torch.randn(16, 4)
y = (x[:, 0] > 0).long()
model = nn.Linear(4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for step in range(50):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
print("Training loss:", loss.item())
