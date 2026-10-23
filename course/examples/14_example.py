# A reliable training and validation loop
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(3)
x = torch.randn(40, 4); y = (x[:, 0] > 0).long()
train = DataLoader(TensorDataset(x[:30], y[:30]), batch_size=8, shuffle=True)
model = nn.Linear(4, 2); opt = torch.optim.Adam(model.parameters(), lr=.03)
for epoch in range(10):
    model.train()
    for images, targets in train:
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), targets)
        loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        val_loss = nn.functional.cross_entropy(model(x[30:]), y[30:])
print("Validation loss:", val_loss.item())
