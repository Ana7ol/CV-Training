# Geometric transformations and interpolation
import cv2
import numpy as np
image = np.zeros((96, 96), np.uint8)
image[25:70, 30:65] = 220
matrix = cv2.getRotationMatrix2D((48, 48), 25, 1.0)
rotated = cv2.warpAffine(image, matrix, (96, 96),
                         flags=cv2.INTER_LINEAR)
mask = cv2.warpAffine((image > 0).astype(np.uint8), matrix,
                      (96, 96), flags=cv2.INTER_NEAREST)
print(rotated.shape, np.unique(mask))
