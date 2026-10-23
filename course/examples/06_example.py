# Thresholds, morphology, and components
import cv2
import numpy as np
image = np.zeros((96, 96), np.uint8)
image[20:70, 25:75] = 180
image[5, 5] = 255
_, mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                         np.ones((3, 3), np.uint8))
n, labels, stats, centers = cv2.connectedComponentsWithStats(clean, 8)
print("Foreground objects:", n - 1)
