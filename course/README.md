# Computer Vision — From pixels to volumes

Open **index.html** in a modern browser. It is self-contained and works offline,
including all lessons, image demonstrations, architecture diagrams, search,
progress tracking, and both interactive labs. Reference links require internet.
No images, fonts, analytics, JavaScript libraries, or models are fetched remotely.

## What is inside

28 sequential lessons in eight stages: foundations; classical vision; deep
learning; dense prediction; modern representations; camera geometry and 3D;
volumetric and point-cloud segmentation; capstones. Each lesson includes intuition,
explicit equations and symbol definitions, a worked calculation, a pipeline or
architecture diagram, a synthetic before/after visualization, a standalone Python
example, an exercise with an answer, and official documentation or original papers.

The reading and exercises are estimated at 40–60 hours, plus 15–30 for projects.
Basic Python and school algebra are the starting prerequisites.

## Python setup

Use Python 3.10 or later, preferably 3.11 or 3.12 for broad wheel compatibility.
From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/03_example.py
```

Windows activation: `.venv\Scripts\activate`.
For lessons using torch or torchvision:

```sh
python -m pip install -r requirements-deep.txt
python examples/12_example.py
python examples/train_3d.py --epochs 4
```

For CUDA or other platform-specific wheels, follow the official selector at
https://pytorch.org/get-started/locally/ . Install matching torch/torchvision builds.
The dependency ranges here are compatibility bounds, not a reproducibility lockfile.
The exact environment used during verification is recorded in VERIFICATION.md.
The PyTorch examples intentionally use tiny synthetic inputs and run on CPU.

Each numbered example stands alone. Most print shapes or scalar results; the
browser provides visual teaching examples. The snippets with randomly initialized
networks demonstrate operations and do not claim useful trained predictions.
Example 28 writes experiment.json in the directory where you run it.

## Complete 3D experiment

`examples/train_3d.py` generates independent train/validation/test volumes, trains
a compact 3D encoder/decoder with skip concatenation, selects the best checkpoint
on validation Dice, selects an intensity threshold baseline on validation, and
then evaluates both once on the held-out synthetic test volumes.

It saves `outputs_3d/best_model.pt`, per-case and aggregate metrics in
`outputs_3d/metrics.json`, and PNGs with three orthogonal slices. Every PNG shows
input, ground truth, and actual learned prediction. The synthetic volumes use
isotropic 1 mm spacing; real-data loaders must retain orientation and spacing.
The classical threshold may outperform a briefly trained model on this easy task.
This is an educational experiment, not a medical or real-world performance claim.

## Reading and printing

Search the sidebar by topic, math notation, or code. Mark complete to store
progress in this browser; browsers that disable local storage still allow reading.
Print course / PDF renders all 28 lessons and computed images, not just the active
lesson. The two interactive labs are intended for the screen and are omitted from
print. References and a glossary have their own navigation entries.

## Visual provenance

All browser images are generated from synthetic data. Filter, grayscale, warp,
threshold, voxel-centroid, projection and backprojection demonstrations are
computed. Model masks, boxes, prompts and matching visuals are explicitly marked
as teaching illustrations or known targets. No neural inference runs in the HTML.
Browser scenes and standalone Python snippets may use different small inputs;
the labels say this explicitly. The capstone PNGs are actual model outputs.

## Editing the course

`content.py` holds original lesson content; `style.css` and `app.js` hold the UI.
Run `python3 build.py` to regenerate the single HTML file, numbered Python examples,
and the ZIP beside this directory. No build dependencies are needed.
