# Kernels, correlation, and convolution
import cv2
import numpy as np
image = np.arange(81, dtype=np.float32).reshape(9, 9)
kernel = np.ones((3, 3), np.float32) / 9
out = cv2.filter2D(image, -1, kernel,
                   borderType=cv2.BORDER_REFLECT_101)
print(out[4, 4])  # 40.0
print(out.shape)
