# Frequency, sampling, and image pyramids
import cv2
import numpy as np
y, x = np.mgrid[:64, :64]
image = ((x + y) % 2).astype(np.float32)
naive = image[::2, ::2]
safe = cv2.GaussianBlur(image, (5, 5), 1)[::2, ::2]
spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
print(naive.mean(), safe.mean(), spectrum.shape)
