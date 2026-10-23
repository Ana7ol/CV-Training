# Verification — 23 October 2026

- All 28 numbered Python examples executed successfully. The detection example
  was rerun successfully after providing the compatible torchvision wheel.
- The full 3D experiment completed four epochs on CPU and saved the selected model,
  per-case metrics, and four PNGs. Checkpoint selection used validation only.
- Test foreground Dice averaged 0.951116 for the selected model versus 0.858094
  for the validation-tuned intensity threshold, on four independent synthetic
  volumes. Best validation checkpoint: epoch 3. No real-data accuracy claim.
- The actual saved result image was visually inspected and embedded into the HTML.
- Node syntax checking passed for app.js; Python parsing passed for every source.
- DOM + native-canvas verification passed all 28 lesson routes and all 56 lesson
  canvases, lesson search, progress persistence and undo, numeric kernel results,
  volume threshold and axis controls, embedded image presence, internal links,
  unique IDs, and print preparation. No JavaScript or canvas runtime errors.
- The local HTTP server returned the completed HTML successfully.

Execution environment: Python 3.12.5, NumPy 2.1.3, OpenCV 4.11.0, PyTorch 2.6.0,
torchvision 0.21.0; macOS arm64. DOM verification: Node 22.8.0, jsdom 26.1.0,
@napi-rs/canvas. Dependencies for verification stayed outside the course folder.

Full browser layout, mobile rendering, and print pagination remain unverified.
The browser plugin could not initialize because its runtime imports a disallowed
module, and the headless Chrome fallback terminated during startup. The course
was instead checked using a DOM runtime and native canvas rendering. This does
not substitute for inspecting complete browser page screenshots.
