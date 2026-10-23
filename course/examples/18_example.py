# Losses, Dice, IoU, and boundary quality
import torch
logits = torch.tensor([[[[2., -1.], [1., -2.]]]])
y = torch.tensor([[[[1., 0.], [1., 1.]]]])
p = logits.sigmoid()
soft_dice = (2*(p*y).sum()+1e-6)/(p.sum()+y.sum()+1e-6)
pred = p > .5; truth = y.bool()
tp = (pred & truth).sum().float()
fp = (pred & ~truth).sum().float()
fn = (~pred & truth).sum().float()
print("Soft:", soft_dice.item())
print("Hard:", (2*tp/(2*tp+fp+fn)).item())
