# Sparse voxels, meshes, and representation choices
import numpy as np
rng = np.random.default_rng(6)
points = rng.uniform(-1,1,(10000,3))
labels = (points[:,2] > 0).astype(np.int64)
size = .05
coords = np.floor((points + 1) / size).astype(np.int32)
unique, inverse = np.unique(coords,axis=0,return_inverse=True)
# Coordinate quantization only; NOT a sparse-convolution implementation.
counts = np.bincount(inverse)
positive = np.bincount(inverse, weights=labels)
voxel_labels = (positive / counts >= .5).astype(np.int64)
print(unique.shape, voxel_labels.shape)
