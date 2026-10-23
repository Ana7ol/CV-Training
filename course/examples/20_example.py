# Promptable segmentation and foundation models
# Executable toy prompt rule, NOT a SAM implementation or prediction.
import numpy as np
y, x = np.mgrid[:96, :96]
image = np.exp(-((x-48)**2+(y-48)**2)/400)
positive_click = (48, 48)  # row, column
seed_intensity = image[positive_click]
mask = image > 0.5 * seed_intensity
print("Toy prompted foreground pixels:", mask.sum())
# For SAM itself, use the official repository and a model checkpoint.
# Account for original-image coordinates when mapping prompts.
