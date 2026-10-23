# Camera geometry and calibration
import numpy as np
K = np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
points = np.array([[.2, 0., 2.], [.2, 0., 4.]])
projected = (K @ points.T).T
uv = projected[:, :2] / projected[:, 2:3]
print(uv)  # [[370,240], [345,240]]
# Projection assumes an ideal distortion-free camera and positive Z.
