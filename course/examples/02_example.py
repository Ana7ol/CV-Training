# Images are arrays
import numpy as np
rgb = np.zeros((64, 96, 3), dtype=np.uint8)
rgb[:, :48] = [255, 80, 20]
rgb[:, 48:] = [20, 160, 240]
x = rgb.astype(np.float32) / 255.0
gray = x @ np.array([0.299, 0.587, 0.114])
batch = x.transpose(2, 0, 1)[None]
print(gray.shape, batch.shape, batch.nbytes)
