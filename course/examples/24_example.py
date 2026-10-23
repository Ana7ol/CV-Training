# Volumes, voxels, spacing, and orientation
import numpy as np
D, H, W = 24, 64, 64
z, y, x = np.mgrid[:D, :H, :W]
spacing_dhw = np.array([3., .8, .8])  # mm
radius_mm = 12
mask = (((z-12)*3)**2 + ((y-32)*.8)**2 +
        ((x-32)*.8)**2) < radius_mm**2
volume_mm3 = mask.sum() * spacing_dhw.prod()
print("Shape / physical volume:", mask.shape, volume_mm3)
# Compare against 4*pi*r**3/3; coarse sampling causes approximation error.
