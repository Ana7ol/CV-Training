"""A CPU-sized 3D segmentation experiment with independent synthetic volumes.
Run from the course folder: python examples/train_3d.py --epochs 4
Outputs: outputs_3d/{best_model.pt,metrics.json,case_00.png,...}
No external data or weights are downloaded. This is not a clinical benchmark.
"""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def make_data(count, seed, size=24):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[:size, :size, :size]
    images, masks = [], []
    for _ in range(count):
        center = rng.uniform(9, 15, 3)
        radius = rng.uniform(4, 7)
        mask = ((zz-center[0])**2 + (yy-center[1])**2 +
                (xx-center[2])**2) < radius**2
        # Intensity overlap makes thresholding imperfect but still a useful baseline.
        image = .15 + .55*mask + rng.normal(0, .15, mask.shape)
        images.append(np.clip(image, 0, 1).astype(np.float32)[None])
        masks.append(mask.astype(np.int64))
    return torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks))


def block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1),
                         nn.GroupNorm(4, cout), nn.ReLU(),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.ReLU())


class TinyUNet3D(nn.Module):
    """A teaching variant, not a reproduction of the original 3D U-Net."""
    def __init__(self):
        super().__init__()
        self.encoder = block(1, 8)
        self.low = block(8, 16)
        self.decoder = block(24, 8)
        self.head = nn.Conv3d(8, 2, 1)

    def forward(self, x):
        skip = self.encoder(x)
        low = self.low(F.max_pool3d(skip, 2))
        up = F.interpolate(low, size=skip.shape[-3:], mode='trilinear',
                           align_corners=False)
        return self.head(self.decoder(torch.cat([skip, up], dim=1)))


def dice_per_case(pred, target):
    pred, target = pred.bool(), target.bool()
    dims = tuple(range(1, pred.ndim))
    inter = (pred & target).sum(dims).float()
    denom = pred.sum(dims) + target.sum(dims)
    # Explicit empty-mask policy: two empty masks score 1.
    return torch.where(denom > 0, 2*inter/denom.clamp_min(1),
                       torch.ones_like(inter))


def objective(logits, targets):
    ce = F.cross_entropy(logits, targets)
    p = logits.softmax(1)[:, 1]
    y = targets.float()
    dims = (1, 2, 3)
    dice = (2*(p*y).sum(dims)+1e-5)/(p.sum(dims)+y.sum(dims)+1e-5)
    return ce + (1-dice).mean()


def save_views(image, target, pred, path):
    """Three orthogonal central slices; columns input / truth / prediction."""
    rows = []
    center = image.shape[0]//2
    for axis, name in enumerate(['Axial z', 'Coronal y', 'Sagittal x']):
        plane = np.take(image, center, axis=axis)
        truth = np.take(target, center, axis=axis)
        guess = np.take(pred, center, axis=axis)
        raw = cv2.cvtColor((plane*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        views = []
        for label, mask in [('Input', None), ('Truth', truth), ('Prediction', guess)]:
            panel = raw.copy()
            if mask is not None:
                panel[mask.astype(bool)] = (.45*panel[mask.astype(bool)] +
                                            .55*np.array([80, 220, 140])).astype(np.uint8)
            panel = cv2.resize(panel, (240, 240), interpolation=cv2.INTER_NEAREST)
            panel = cv2.copyMakeBorder(panel, 35, 0, 0, 0, cv2.BORDER_CONSTANT,
                                       value=(30, 44, 40))
            cv2.putText(panel, f'{name} / {label}', (8, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, .47, (220, 240, 225), 1)
            views.append(panel)
        rows.append(np.concatenate(views, axis=1))
    if not cv2.imwrite(str(path), np.concatenate(rows, axis=0)):
        raise RuntimeError(f'Could not save {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('outputs_3d'))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_x, train_y = make_data(24, 11)
    val_x, val_y = make_data(4, 22)
    # This test set is only evaluated after checkpoint and threshold selection.
    test_x, test_y = make_data(4, 33)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4, shuffle=True)
    model = TinyUNet3D()
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    best_score, best_state, history = -1., None, []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = objective(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*len(x)
        model.eval()
        with torch.inference_mode():
            score = dice_per_case(model(val_x).argmax(1), val_y).mean().item()
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
        history.append({'epoch': epoch+1, 'train_loss': loss_sum/len(train_x),
                        'validation_dice': score})
        print(f'Epoch {epoch+1}: loss {loss_sum/len(train_x):.4f}, val Dice {score:.4f}')
    # Select threshold baseline on validation only.
    candidates = [.3, .4, .5, .6]
    threshold = max(candidates, key=lambda t:
        dice_per_case(val_x[:, 0] > t, val_y).mean().item())
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pred = model(test_x).argmax(1)
        learned = dice_per_case(pred, test_y).tolist()
        baseline = dice_per_case(test_x[:, 0] > threshold, test_y).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output/'best_model.pt')
    metrics = {'task': 'synthetic 3D spheres; no real-data claim',
               'seeds': {'model':42, 'train':11, 'validation':22, 'test':33},
               'spacing_dhw_mm':[1,1,1], 'shape_dhw':[24,24,24],
               'selected_threshold':threshold, 'validation_best_dice':best_score,
               'test_dice_model_per_case':learned,
               'test_dice_baseline_per_case':baseline,
               'test_model_mean':float(np.mean(learned)),
               'test_baseline_mean':float(np.mean(baseline)), 'history':history}
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    for i in range(len(test_x)):
        save_views(test_x[i, 0].numpy(), test_y[i].numpy(), pred[i].numpy(),
                   args.output/f'case_{i:02d}.png')
    print(json.dumps({k:v for k,v in metrics.items() if k.startswith('test_')}, indent=2))
    print(f'Results saved to {args.output.resolve()}')
    print('A threshold can win on this easy synthetic task. Report either outcome honestly.')


if __name__ == '__main__':
    main()
