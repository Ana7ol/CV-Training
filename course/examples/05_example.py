# Derivatives, edges, and Canny
import cv2
import numpy as np
image = np.zeros((96, 96), np.uint8)
image[25:75, 30:65] = 200
gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
magnitude = np.hypot(gx, gy)
edges = cv2.Canny(image, 50, 120)
print(float(magnitude.max()), np.count_nonzero(edges))
