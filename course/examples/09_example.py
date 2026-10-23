# Features, matching, and robust fitting
import cv2
import numpy as np
rng = np.random.default_rng(2)
image = rng.integers(0, 256, (192, 192), dtype=np.uint8)
shifted = np.roll(image, 5, axis=1)
orb = cv2.ORB_create(nfeatures=400)
k1, d1 = orb.detectAndCompute(image, None)
k2, d2 = orb.detectAndCompute(shifted, None)
if d1 is not None and d2 is not None:
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    print("Matches:", len(pairs))
