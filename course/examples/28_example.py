# Capstones: from a baseline to a defensible result
from pathlib import Path
import json
experiment = {
    "task": "3d_binary_segmentation",
    "data": "synthetic spheres; not a clinical benchmark",
    "split": "independent generated volumes with fixed seeds",
    "baseline": "intensity threshold",
    "model": "TinyUNet3D",
    "metrics": ["per_volume_foreground_dice"],
    "next": "python examples/train_3d.py --epochs 4"
}
Path("experiment.json").write_text(json.dumps(experiment,indent=2))
print(experiment)
