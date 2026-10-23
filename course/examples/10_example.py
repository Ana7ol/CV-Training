# Optical flow and temporal reasoning
import cv2
import numpy as np
rng = np.random.default_rng(8)
a = rng.integers(0, 256, (96, 96), dtype=np.uint8)
b = np.roll(a, 2, axis=1)
flow = cv2.calcOpticalFlowFarneback(a, b, None,
    0.5, 3, 15, 3, 5, 1.2, 0)
print("Median interior displacement:",
      np.median(flow[20:-20, 20:-20], axis=(0, 1)))
