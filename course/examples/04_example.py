# Smoothing, noise, and denoising
import cv2
import numpy as np
rng = np.random.default_rng(7)
clean = np.zeros((96, 96), np.float32)
clean[24:72, 24:72] = 0.8
noisy = np.clip(clean + rng.normal(0, .12, clean.shape), 0, 1).astype(np.float32)
smooth = cv2.GaussianBlur(noisy, (7, 7), 1.2)
print("Noisy / smooth MSE:", np.mean((noisy-clean)**2),
      np.mean((smooth-clean)**2))
