# Stereo depth and RGB-D backprojection
import numpy as np
fx = fy = 600.; baseline = .1
cx, cy = 2., 1.5
v, u = np.mgrid[:4, :5]
disparity = np.full((4,5), 30.)
valid = disparity > 0
Z = fx * baseline / disparity[valid]
X = (u[valid]-cx) * Z / fx
Y = (v[valid]-cy) * Z / fy
cloud = np.column_stack([X,Y,Z])
print(cloud.shape, cloud[0])
