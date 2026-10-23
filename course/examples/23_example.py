# Reconstruction, point clouds, and surfaces
import numpy as np
rng = np.random.default_rng(4)
points = rng.uniform(0, 1, (1000,3))
voxel_size = .1
keys = np.floor(points / voxel_size).astype(np.int64)
unique, inverse = np.unique(keys, axis=0, return_inverse=True)
sums = np.zeros((len(unique),3))
np.add.at(sums, inverse, points)
counts = np.bincount(inverse)
centroids = sums / counts[:,None]
print("Before / after:", len(points), len(centroids))
