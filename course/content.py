"""Original course content. Run build.py to create the self-contained HTML."""
LESSONS = []
def add(title, group, intro, concepts, math, symbols, worked, code, visual, pipeline, pitfall, exercise, answer, refs):
    LESSONS.append(dict(title=title, group=group, intro=intro, concepts=concepts, math=math, symbols=symbols, worked=worked, code=code.strip(), visual=visual, pipeline=pipeline, pitfall=pitfall, exercise=exercise, answer=answer, refs=refs))
CV = ('OpenCV · image processing', 'https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html')
FILTER = ('OpenCV · filtering', 'https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html')
TORCH = ('PyTorch · learn the basics', 'https://docs.pytorch.org/tutorials/beginner/basics/intro.html')
CONV = ('Stanford CS231n · convolutional networks', 'https://cs231n.github.io/convolutional-networks/')
UNET = ('U-Net · original paper', 'https://arxiv.org/abs/1505.04597')
UNET3 = ('3D U-Net · original paper', 'https://arxiv.org/abs/1606.06650')
GEOM = ('OpenCV · calibration and 3D reconstruction', 'https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html')
POINT = ('PointNet · original paper', 'https://arxiv.org/abs/1612.00593')
OPEN3 = ('Open3D · point-cloud tutorial', 'https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html')

add('The mathematical toolkit', '01 · Foundations',
'Computer vision turns measurements into decisions. The essential language is arrays, dot products, derivatives, and probability. You do not need to master an entire mathematics degree before starting; learn these operations well enough to trace one small example by hand.',
['A scalar is one number; a vector is an ordered list; a matrix has two axes; a tensor can have more. Shape describes the size of every axis, not the meaning of those axes.', 'A dot product multiplies corresponding entries and adds them. Matrix multiplication composes linear maps. A derivative describes local change; the chain rule passes that change through a sequence of operations.', 'A probability lies between zero and one. Conditional probability P(class | image) describes a prediction under a model; it is not a guarantee that the model is calibrated.'],
'z = wᵀx + b = Σᵢ wᵢxᵢ + b\nσ(z) = 1 / (1 + exp(−z))\n∂L/∂wᵢ = (∂L/∂z) xᵢ',
'x: input vector; w: weights; b: bias; z: score; σ: sigmoid; L: scalar loss; Σ: sum; ∂: partial derivative.',
'For x = [2, 3], w = [1, −1], b = 2, the score is 2 − 3 + 2 = 1. The sigmoid is about 0.731. If ∂L/∂z = 0.2, the weight gradient is [0.4, 0.6].',
'''import numpy as np
x = np.array([2., 3.])
w = np.array([1., -1.])
z = w @ x + 2.
p = 1 / (1 + np.exp(-z))
grad_w = 0.2 * x
print(z, p, grad_w)''', 'math', ['Input x · 2 values','Dot product + bias','Scalar z','Sigmoid · probability'],
'Elementwise multiplication (*) and matrix multiplication (@) are different. Broadcasting can produce a valid but unintended shape; print shapes before trusting results.',
'If x = [1, 4] and w = [2, −0.5] with b = 0, what are z and σ(z)?', 'z = 2 − 2 = 0, so σ(z) = 0.5. A zero logit is the midpoint, not a confident negative.', [TORCH])

add('Images are arrays', '01 · Foundations',
'An image is a grid of sampled measurements. A grayscale image is H × W; a color image often stores H × W × 3. Before choosing any algorithm, inspect channel order, data type, range, and axis order.',
['RGB stores red, green, and blue. OpenCV commonly loads BGR. Converting channel order is different from converting to grayscale.', 'uint8 stores integers from 0 to 255. Convert to floating point before subtraction, normalization, or gradients to avoid overflow and lost precision.', 'PyTorch image batches conventionally use N × C × H × W. NumPy display images usually use H × W × C. Transpose changes axis order; reshape only reinterprets the arrangement.'],
'I_float = I_uint8 / 255\nY ≈ 0.299R + 0.587G + 0.114B\nx̂_c = (x_c − μ_c) / σ_c',
'I: image; Y: conventional grayscale intensity; μ and σ: channel mean and standard deviation. The displayed grayscale rule is an approximation for encoded RGB, not linear-light photometry.',
'RGB [255, 0, 0] becomes [1, 0, 0] after scaling and approximately 76 in 8-bit grayscale. A 224 × 224 RGB float32 image uses 224 × 224 × 3 × 4 = 602,112 bytes.',
'''import numpy as np
rgb = np.zeros((64, 96, 3), dtype=np.uint8)
rgb[:, :48] = [255, 80, 20]
rgb[:, 48:] = [20, 160, 240]
x = rgb.astype(np.float32) / 255.0
gray = x @ np.array([0.299, 0.587, 0.114])
batch = x.transpose(2, 0, 1)[None]
print(gray.shape, batch.shape, batch.nbytes)''', 'gray', ['H × W × 3 · uint8','Cast to float / 255','Optional normalization','N × 3 × H × W'],
'Do not apply a pretrained model’s normalization twice. Inspect its documented preprocessing and use the same preprocessing at inference.',
'What is the shape of a batch of eight grayscale images, each 128 × 256, in PyTorch?', '[8, 1, 128, 256]. The channel axis still exists even for grayscale.', [CV, TORCH])

add('Kernels, correlation, and convolution', '01 · Foundations',
'A kernel is a small array of weights slid over an image. At each location, multiply the neighborhood by the weights and add the products. This is the building block that connects classical filters to learned convolutional networks.',
['Cross-correlation applies the kernel as written. Mathematical convolution flips it along both spatial axes first. OpenCV filter2D and PyTorch Conv2d compute cross-correlation; the distinction matters for asymmetric filters.', 'Padding defines behavior at boundaries. Zero padding can darken borders; reflection extends local image structure. Stride moves the window more than one pixel. Dilation spaces the sampled kernel entries apart.', 'A normalized nonnegative kernel averages measurements. A kernel with positive and negative entries can emphasize changes rather than average brightness.'],
'Y[i,j] = ΣᵤΣᵥ K[u,v] X[i+u−p,j+v−p]\nH_out = floor((H + 2p − d(k−1) − 1)/s + 1)\nK_box = (1/9) [[1,1,1],[1,1,1],[1,1,1]]',
'K: k × k kernel; p: symmetric padding; d: dilation; s: stride. The first formula uses stride 1, dilation 1 and an odd centered kernel.',
'For the patch [[1,2,3],[4,5,6],[7,8,9]], a 3 × 3 box filter returns 45/9 = 5. For H = 32, k = 3, p = 1, s = 2, d = 1, H_out = 16.',
'''import cv2
import numpy as np
image = np.arange(81, dtype=np.float32).reshape(9, 9)
kernel = np.ones((3, 3), np.float32) / 9
out = cv2.filter2D(image, -1, kernel,
                   borderType=cv2.BORDER_REFLECT_101)
print(out[4, 4])  # 40.0
print(out.shape)''', 'blur', ['Image neighborhood','Elementwise × kernel','Sum products','One output pixel'],
'Clipping signed outputs into uint8 can erase negative responses. Use float32 for derivatives and visualize signed or absolute values deliberately.',
'What should a brightness-preserving averaging kernel sum to? What does a zero-sum kernel do on a constant image away from boundaries?', 'An averaging kernel sums to 1. A zero-sum kernel outputs 0 on a constant region because all constant contributions cancel.', [FILTER, CONV])

add('Smoothing, noise, and denoising', '01 · Foundations',
'Noise makes neighboring measurements disagree. Smoothing combines nearby measurements to reduce variation, but also removes real detail. Choose a filter based on the noise and the structures that matter.',
['Gaussian filtering uses distance-weighted averaging. A larger standard deviation removes more high-frequency content. Gaussian filtering is separable: two one-dimensional passes replace one two-dimensional pass.', 'A median filter selects a neighborhood’s middle value and works well on isolated salt-and-pepper corruption. It is nonlinear and cannot be represented by one fixed weighted sum.', 'A bilateral filter weights both spatial proximity and intensity similarity, so strong edges can be preserved while nearby similar pixels are averaged. It costs more than a basic Gaussian filter.'],
'G(x,y) = exp(−(x²+y²)/(2σ²)) / (2πσ²)\nI = S + ε,   E[ε] = 0\nVar(mean of n independent noises) = σ_noise²/n',
'G: continuous Gaussian, discretized and renormalized for a kernel; σ: blur scale; S: clean signal; ε: additive noise; Var: variance.',
'Averaging nine independent noise samples with standard deviation 12 gives standard deviation 12/√9 = 4. Real neighborhoods also contain signal changes, so this reduction comes with blur.',
'''import cv2
import numpy as np
rng = np.random.default_rng(7)
clean = np.zeros((96, 96), np.float32)
clean[24:72, 24:72] = 0.8
noisy = np.clip(clean + rng.normal(0, .12, clean.shape), 0, 1).astype(np.float32)
smooth = cv2.GaussianBlur(noisy, (7, 7), 1.2)
print("Noisy / smooth MSE:", np.mean((noisy-clean)**2),
      np.mean((smooth-clean)**2))''', 'denoise', ['Noisy image','Choose noise model','Gaussian / median / bilateral','Estimate clean image'],
'Blur can improve mean squared error while destroying a small object you need to detect. Evaluate the downstream task and inspect edges.',
'Why does a box filter fail on an isolated extreme pixel more visibly than a median filter?', 'The extreme value contributes to every overlapping average. A median rejects it when fewer than half of the neighborhood samples are corrupted.', [FILTER])

add('Derivatives, edges, and Canny', '01 · Foundations',
'An edge is a rapid spatial change in intensity. Derivatives measure that change. Edge maps help reveal boundaries, but a brightness boundary is not necessarily an object boundary.',
['Sobel combines smoothing in one direction with a discrete derivative in the other. Compute horizontal and vertical responses, then combine them into magnitude and orientation.', 'Canny smooths the image, computes gradients, thins responses with nonmaximum suppression, and connects weak edges to strong edges with hysteresis thresholds.', 'First derivatives respond to change; second derivatives respond to changes of change and amplify noise. Smooth before differentiating when the data is noisy.'],
'K_x = [[−1,0,1],[−2,0,2],[−1,0,1]]\nG = √(G_x² + G_y²)\nθ = atan2(G_y,G_x)',
'G_x and G_y: Sobel responses; G: gradient magnitude; θ: gradient direction, perpendicular to the local edge.',
'If G_x = 3 and G_y = 4, the magnitude is 5 and the orientation is about 53.1°. On a left-to-right increasing ramp, the horizontal derivative is positive under this kernel convention.',
'''import cv2
import numpy as np
image = np.zeros((96, 96), np.uint8)
image[25:75, 30:65] = 200
gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
magnitude = np.hypot(gx, gy)
edges = cv2.Canny(image, 50, 120)
print(float(magnitude.max()), np.count_nonzero(edges))''', 'edge', ['Grayscale','Smooth → gradients','Suppress nonmaxima','Hysteresis → thin edges'],
'Shadows and texture produce edges too. Canny thresholds depend on intensity scale; thresholds for uint8 images do not transfer directly to floats in [0,1].',
'Why can a low threshold produce many fragmented or spurious edges?', 'Small intensity fluctuations become eligible edges. Hysteresis keeps weak responses connected to strong ones, but cannot infer which boundaries are semantically meaningful.', [CV])

add('Thresholds, morphology, and components', '01 · Foundations',
'Thresholding assigns a binary label from intensity. Morphology then modifies the mask’s shape. This gives a useful classical segmentation baseline before you train a neural network.',
['Global thresholds use one cutoff. Adaptive thresholds vary across neighborhoods. Otsu chooses a cutoff that separates two intensity groups by a histogram criterion; a two-mode histogram is helpful but not guaranteed.', 'Binary dilation expands foreground wherever the structuring element overlaps it. Erosion keeps locations where the entire structuring element fits inside foreground. Opening erodes then dilates; closing dilates then erodes.', 'Connected components group touching foreground pixels. Four-connectivity and eight-connectivity disagree on diagonally touching objects. Report the chosen connectivity.'],
'M[i,j] = 1 if I[i,j] > τ, else 0\nA ∘ B = (A ⊖ B) ⊕ B\nA • B = (A ⊕ B) ⊖ B',
'M: binary mask; τ: threshold; A: foreground set; B: structuring element; ⊖: erosion; ⊕: dilation; ∘: opening; •: closing.',
'An isolated 1-pixel foreground speck disappears under opening with a 3 × 3 square. A narrow real structure can disappear too. A diagonal pair is one object under 8-connectivity and two under 4-connectivity.',
'''import cv2
import numpy as np
image = np.zeros((96, 96), np.uint8)
image[20:70, 25:75] = 180
image[5, 5] = 255
_, mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                         np.ones((3, 3), np.uint8))
n, labels, stats, centers = cv2.connectedComponentsWithStats(clean, 8)
print("Foreground objects:", n - 1)''', 'mask', ['Intensity image','Threshold','Opening / closing','Connected components'],
'Class IDs are categorical. Resize label masks with nearest-neighbor interpolation, and never average IDs as if class 2 were halfway between classes 1 and 3.',
'Would you use opening or closing to fill small holes? What can go wrong?', 'Closing can fill small holes and bridge gaps. It can also merge adjacent objects or alter topology.', [CV, ('OpenCV · morphology','https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html')])

add('Frequency, sampling, and image pyramids', '02 · Classical vision',
'The same image can be described by spatial patterns or by frequency components. Slow changes are low frequency; fine stripes and sharp boundaries include high frequencies. Sampling decides which of these patterns you can represent.',
['The discrete Fourier transform decomposes an image into sinusoidal components. The magnitude describes strength; phase describes alignment and carries much of the spatial structure.', 'A spatial convolution corresponds to multiplication in frequency under consistent boundary conventions. FFT multiplication without padding gives circular convolution.', 'Downsampling discards samples. Low-pass filtering before decimation prevents high-frequency details from becoming false low-frequency patterns, called aliasing. Pyramids repeat blur and downsample.'],
'F[u,v] = ΣₓΣᵧ I[x,y] exp(−2πi(ux/H + vy/W))\nf_Nyquist = f_sampling / 2\nI_next = downsample₂(G_σ * I)',
'F: complex Fourier coefficients; i: imaginary unit; *: mathematical convolution; f_Nyquist: highest unaliased sinusoidal frequency below the boundary case.',
'A pattern repeating every two source pixels sits at the Nyquist boundary of the source grid. Taking every other pixel can make it look constant; blur first when reducing resolution.',
'''import cv2
import numpy as np
y, x = np.mgrid[:64, :64]
image = ((x + y) % 2).astype(np.float32)
naive = image[::2, ::2]
safe = cv2.GaussianBlur(image, (5, 5), 1)[::2, ::2]
spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
print(naive.mean(), safe.mean(), spectrum.shape)''', 'alias', ['Sampled image','Low-pass filter','Decimate ×2','Repeat → scale pyramid'],
'fftshift moves the zero frequency to the center for display; it is not a filtering operation. Preserve complex values until the inverse transform.',
'Why do thin fence lines sometimes turn into large stripes after resizing?', 'The original frequency exceeds what the output grid can represent and aliases into another frequency. Antialias filtering removes that unsupported detail before resampling.', [CV])

add('Geometric transformations and interpolation', '02 · Classical vision',
'Geometric transforms move coordinates. The output image must then sample the input at usually noninteger locations. Separate the coordinate map from the interpolation rule.',
['Translation, rotation, scaling, and shear can be written as affine transforms. A projective homography also represents perspective changes for a plane or a pure camera rotation under ideal assumptions.', 'Inverse warping visits each output pixel and asks where it came from. Forward splatting can leave holes unless it handles coverage explicitly.', 'Nearest-neighbor interpolation preserves categorical labels. Bilinear interpolation blends four neighboring intensities; it is suitable for images but can soften edges.'],
'[u′,v′,1]ᵀ ∝ H [u,v,1]ᵀ\n[x′,y′]ᵀ = A[x,y]ᵀ + t\nI_out(u,v) = I_in(T⁻¹(u,v))',
'H: 3 × 3 homography up to scale; A: 2 × 2 linear part; t: translation; ∝: equal after homogeneous-coordinate normalization.',
'A translation by (10, −3) maps input point (20,30) to (30,27). To fill output (30,27), inverse sampling reads input (20,30).',
'''import cv2
import numpy as np
image = np.zeros((96, 96), np.uint8)
image[25:70, 30:65] = 220
matrix = cv2.getRotationMatrix2D((48, 48), 25, 1.0)
rotated = cv2.warpAffine(image, matrix, (96, 96),
                         flags=cv2.INTER_LINEAR)
mask = cv2.warpAffine((image > 0).astype(np.uint8), matrix,
                      (96, 96), flags=cv2.INTER_NEAREST)
print(rotated.shape, np.unique(mask))''', 'rotate', ['Output grid','Inverse coordinate map','Interpolate input','Warped image'],
'A single homography generally cannot align a nonplanar scene under camera translation: different depths create parallax.',
'What happens when you bilinearly resize a mask whose classes are 0 and 2?', 'Boundary pixels can receive intermediate values such as 1, inventing a class that was absent. Use nearest-neighbor for discrete masks.', [CV, GEOM])

add('Features, matching, and robust fitting', '02 · Classical vision',
'Local features identify repeatable locations, describe their neighborhoods, and match those descriptions across views. Robust geometry then separates plausible correspondences from accidental similarities.',
['Corner detectors prefer neighborhoods with intensity variation in two directions. Descriptors encode local patterns; ORB uses binary descriptors compared by Hamming distance.', 'A nearest-neighbor ratio test rejects ambiguous descriptors. Mutual matching is another consistency check, but neither guarantees a geometrically correct pair.', 'RANSAC repeatedly fits a model to a minimal random subset, counts inliers under a residual threshold, and retains a supported model. It assumes enough correspondences satisfy the chosen geometry.'],
'N ≥ log(1−p) / log(1−wˢ)\ne_i = ‖x′ᵢ − project(Hxᵢ)‖₂\ninlier ⇔ e_i < τ',
'N: RANSAC trials; p: desired probability of sampling at least one all-inlier set; w: inlier fraction; s: sample size; τ: geometric error threshold.',
'For p = 0.99, w = 0.5, and s = 4, at least 72 independent trials are needed under the simple model. Lower inlier fractions make the required trials grow quickly.',
'''import cv2
import numpy as np
rng = np.random.default_rng(2)
image = rng.integers(0, 256, (192, 192), dtype=np.uint8)
shifted = np.roll(image, 5, axis=1)
orb = cv2.ORB_create(nfeatures=400)
k1, d1 = orb.detectAndCompute(image, None)
k2, d2 = orb.detectAndCompute(shifted, None)
if d1 is not None and d2 is not None:
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    print("Matches:", len(pairs))''', 'match', ['Detect keypoints','Describe patches','Match descriptors','RANSAC → geometry'],
'Repeated windows, blank walls, blur, and moving objects can defeat matching. Inspect inliers across the whole image rather than trusting a large match count.',
'Why is one cluster of matches in a small corner insufficient evidence for a reliable full-image warp?', 'It constrains that local region but can leave the transform unstable elsewhere. Spatially distributed support and residual checks are needed.', [CV, GEOM])

add('Optical flow and temporal reasoning', '02 · Classical vision',
'Optical flow estimates apparent image motion between frames. It represents a displacement field, not automatically the physical 3D velocity of objects.',
['Brightness constancy assumes a moving point keeps approximately the same intensity. A first-order expansion yields one equation for two unknown motion components.', 'The aperture problem means a single local edge constrains motion only normal to that edge. Lucas–Kanade adds a local constant-motion assumption and solves over a patch.', 'Pyramids help with larger motion. Occlusion, illumination changes, motion blur, and textureless regions violate assumptions. Forward–backward consistency can flag unreliable flow.'],
'I_x u + I_y v + I_t = 0\n(AᵀA) [u,v]ᵀ = −Aᵀb\nflow(x,y) = (Δx, Δy)',
'I_x,I_y: spatial derivatives; I_t: temporal derivative; u,v: horizontal and vertical displacement per frame; rows of A: [I_x,I_y]; b: temporal derivatives.',
'If a corner moves from (20,30) to (24,29), its flow is (4, −1) pixels per frame. Converting to meters per second requires camera geometry, depth, and frame timing.',
'''import cv2
import numpy as np
rng = np.random.default_rng(8)
a = rng.integers(0, 256, (96, 96), dtype=np.uint8)
b = np.roll(a, 2, axis=1)
flow = cv2.calcOpticalFlowFarneback(a, b, None,
    0.5, 3, 15, 3, 5, 1.2, 0)
print("Median interior displacement:",
      np.median(flow[20:-20, 20:-20], axis=(0, 1)))''', 'flow', ['Frame t + frame t+1','Spatial / temporal evidence','Estimate displacement','Motion field'],
'Do not confuse image flow with object motion: camera movement alone can generate a dense flow field.',
'Why is flow hard on a uniformly colored wall?', 'Many displacements produce nearly identical patches. There is little spatial gradient to constrain either motion component.', [CV, ('Torchvision · optical flow examples','https://docs.pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html')])

add('Learning, losses, and backpropagation', '03 · Deep learning',
'A learned model replaces hand-selected rules with parameters fitted to examples. Training alternates prediction, loss computation, gradient computation, and parameter updates. The real goal is performance on independent data.',
['Logits are unconstrained scores. Softmax turns them into a categorical probability distribution. Cross-entropy rewards probability assigned to the correct class.', 'Backpropagation applies the chain rule through the computation graph. Gradient descent moves parameters opposite the loss gradient; Adam adds adaptive estimates of gradient moments.', 'Split data before fitting preprocessing. Keep correlated images from the same patient, subject, scene, or acquisition session in the same split to prevent leakage.'],
'p_k = exp(z_k) / Σⱼ exp(z_j)\nL_CE = −log(p_y)\nθ_next = θ − η ∇_θ L',
'z: logits; y: target class index; θ: model parameters; η: learning rate; ∇: gradient. Implement softmax stably by subtracting max(z).',
'For probabilities [0.8,0.2] and true class 0, loss = −log(0.8) ≈ 0.223. For [0.2,0.8], the same target incurs 1.609.',
'''import torch
from torch import nn
torch.manual_seed(7)
x = torch.randn(16, 4)
y = (x[:, 0] > 0).long()
model = nn.Linear(4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for step in range(50):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
print("Training loss:", loss.item())''', 'learning', ['Batch + labels','Forward → logits','Loss → backward','Optimizer → updated weights'],
'PyTorch CrossEntropyLoss expects raw logits and integer class targets. Applying softmax first changes the intended computation.',
'If training loss decreases while validation loss increases, what should you investigate?', 'Overfitting, data leakage or distribution differences, augmentation mismatch, label noise, and training duration. Choose changes using validation data; keep the test set untouched.', [TORCH])

add('CNNs: learnable kernels and receptive fields', '03 · Deep learning',
'A convolutional layer learns multiple kernels and shares each kernel across locations. This makes local pattern detectors economical. Stacking layers combines small patterns into larger structures.',
['Each output channel sums contributions from every input channel. A 3 × 3 RGB kernel has 3 × 3 × 3 weights per output channel, not just nine.', 'ReLU applies max(0,x) and makes a stack nonlinear. Pooling or strided convolution reduces resolution, increases effective context, and can discard fine detail.', 'A receptive field is the input region that can influence an output unit. The theoretical field can exceed the region that contributes strongly in a trained model.'],
'Y_o = b_o + Σ_c K_{o,c} ⋆ X_c\nparameters = C_out (C_in k² + 1)\nr_l = r_{l−1} + (k_l−1)j_{l−1}; j_l = s_l j_{l−1}',
'⋆: cross-correlation; C_in/out: channel counts; r: receptive-field width; j: jump in input pixels; start with r₀ = j₀ = 1 and dilation 1.',
'A 3 → 16 convolution with 3 × 3 kernels and bias has 16(27+1) = 448 parameters. Two stride-1 3 × 3 layers have a theoretical 5 × 5 receptive field.',
'''import torch
from torch import nn
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(32, 4))
x = torch.randn(2, 3, 64, 64)
print("Logits:", model(x).shape)  # [2, 4]
print("Parameters:", sum(p.numel() for p in model.parameters()))''', 'feature', ['N × 3 × 64 × 64','Conv16 → pool · 32²','Conv32 → global average','Linear → N × 4 logits'],
'Convolution is approximately translation equivariant away from boundaries; a whole network is not automatically invariant to translation, scale, or rotation.',
'What is the parameter count of a 16 → 32, 3 × 3 convolution with bias?', '32 × (16 × 9 + 1) = 4,640. Spatial image size does not change this count.', [CONV, ('PyTorch · Conv2d','https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html')])

add('Residual networks and transfer learning', '03 · Deep learning',
'Residual blocks add a learned update to an existing representation. Transfer learning reuses a representation learned on another dataset, often reducing the data and compute required for a new task.',
['An identity skip gives gradients a direct path through a block. When shapes differ, a learned projection aligns channels or resolution before addition.', 'Freezing a backbone trains only a new head; fine-tuning also updates some or all backbone weights. The appropriate choice depends on data quantity and domain difference.', 'Pretrained weights define a contract: input range, color order, resizing, normalization, and class vocabulary. Reuse the weight-specific transform rather than guessing.'],
'y = ReLU(F(x;W) + S(x))\n∂y_pre/∂x = ∂F/∂x + ∂S/∂x\nS(x) = x when shapes match',
'F: learned residual branch; S: identity or projection shortcut; y_pre: sum before the final ReLU. The identity shortcut contributes an identity Jacobian.',
'If a scalar residual branch outputs −0.2 for x = 1.4, the preactivation sum is 1.2. A concatenation skip would instead retain two values; addition and concatenation are different operations.',
'''import torch
from torch import nn
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1))
    def forward(self, x):
        return torch.relu(x + self.f(x))
x = torch.randn(2, 16, 32, 32)
print(ResidualBlock(16)(x).shape)
# Transfer: torchvision.models.resnet18(weights="DEFAULT")
# Replace model.fc; use the selected weights' transforms().''', 'feature', ['Input x · C × H × W','Conv → ReLU → Conv','Add identity x → ReLU','Same shape output'],
'Freezing gradients does not automatically freeze BatchNorm running statistics. Set the frozen backbone to eval mode when appropriate and choose that policy explicitly.',
'Can you add a [N,32,32,32] tensor to a [N,64,16,16] tensor directly?', 'No. Use a projection such as a 1 × 1 convolution with stride 2 to match channels and spatial size before addition.', [('ResNet · original paper','https://arxiv.org/abs/1512.03385'), ('PyTorch · transfer learning','https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html')])

add('A reliable training and validation loop', '03 · Deep learning',
'The training loop is only one part of an experiment. A useful experiment controls data splits, preprocessing, model state, metrics, and checkpoint selection so that another run can be interpreted.',
['Training uses model.train(), gradients, and optimizer updates. Validation uses model.eval() and inference_mode() to avoid parameter updates and switch dropout and normalization behavior.', 'For paired image/mask augmentation, sample geometric transforms once and apply them to both. Apply color jitter only to images. Normalize using training statistics only.', 'Start by overfitting a tiny batch. If the model cannot learn that batch, investigate targets, tensor shapes, loss functions, gradients, and data alignment before scaling up.'],
'L_epoch = (Σ_b n_b L_b) / (Σ_b n_b)\nθ* = argmin_{checkpoints θ} L_validation(θ)\nstandard error of mean ≈ sample std / √n',
'n_b: batch size; L_b: mean batch loss. The standard-error estimate assumes independent observations, which adjacent slices usually are not.',
'If batches contain 8 and 2 examples with mean losses 1 and 3, the epoch mean is (8+6)/10 = 1.4, not the unweighted batch average 2.',
'''import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(3)
x = torch.randn(40, 4); y = (x[:, 0] > 0).long()
train = DataLoader(TensorDataset(x[:30], y[:30]), batch_size=8, shuffle=True)
model = nn.Linear(4, 2); opt = torch.optim.Adam(model.parameters(), lr=.03)
for epoch in range(10):
    model.train()
    for images, targets in train:
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), targets)
        loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        val_loss = nn.functional.cross_entropy(model(x[30:]), y[30:])
print("Validation loss:", val_loss.item())''', 'learning', ['Group-aware split','Train + augmentation','Validate without updates','Select checkpoint → final test'],
'Do not tune hyperparameters or thresholds on the final test set. Fixed random seeds help debugging but do not guarantee identical results across devices or software versions.',
'Why is splitting random slices from the same patient into training and validation misleading?', 'Adjacent slices share anatomy, acquisition properties, and often nearly identical content. The validation set no longer measures generalization to independent patients.', [TORCH])

add('Object detection and nonmaximum suppression', '04 · Dense predictions',
'Detection predicts both object category and location. Unlike classification, it can return a variable number of boxes. Boxes summarize extent but do not describe precise boundaries.',
['A detector combines a feature backbone, often a feature pyramid, and prediction heads. Two-stage models propose regions and refine them; one-stage models predict densely. DETR uses learned queries and set matching.', 'Boxes can be represented as corners (x1,y1,x2,y2) or center plus width and height. Convert explicitly before computing overlaps.', 'Nonmaximum suppression greedily retains high-scoring boxes and suppresses overlapping candidates. Usually apply it per class; it can remove true objects in crowded scenes.'],
'IoU(A,B) = |A ∩ B| / |A ∪ B|\nL_detection = L_class + λ L_box\nkeep b if IoU(b, retained boxes) ≤ τ',
'IoU: intersection over union; |·|: area; λ: weight balancing loss terms; τ: NMS overlap threshold. Actual detectors may add objectness or matching terms.',
'For A = [0,0,10,10] and B = [5,0,15,10], intersection area is 50 and union is 150, so IoU = 1/3.',
'''import torch
from torchvision.ops import box_iou, nms
boxes = torch.tensor([[0.,0.,10.,10.],
                      [1.,1.,11.,11.],
                      [20.,20.,30.,30.]])
scores = torch.tensor([.9, .8, .7])
print("IoU matrix:", box_iou(boxes, boxes))
keep = nms(boxes, scores, iou_threshold=.5)
print("Retained indices:", keep)  # [0, 2]''', 'boxes', ['Image → backbone','Multiscale features','Class + box predictions','Score filter + NMS'],
'A detector score is not the same as localization accuracy. Detection AP depends on score ordering, IoU criteria, dataset protocol, and class aggregation.',
'Why can NMS miss two overlapping people?', 'It reasons using overlap and scores, so one true box can suppress another. Crowded-scene evaluation and alternatives such as soft-NMS can reveal this failure.', [('Torchvision · detection tutorial','https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html')])

add('Semantic, instance, and panoptic segmentation', '04 · Dense predictions',
'Segmentation assigns information to every pixel. Decide what your label should mean before selecting a model or a metric: class identity, object identity, or both.',
['Semantic segmentation assigns one class per pixel and does not separate objects of the same class. Instance segmentation gives each individual object a separate mask, often alongside a category and score.', 'Panoptic segmentation combines semantic categories with instance identities for countable things and region labels for background stuff.', 'A semantic model commonly outputs N × C × H × W logits. Softmax and argmax operate across C. A binary model can instead output one channel trained with binary cross-entropy.'],
'p_{c,h,w} = softmax_c(z_{c,h,w})\nŷ_{h,w} = argmax_c p_{c,h,w}\nL_pixelCE = −(1/HW) Σ_{h,w} log p_{y(h,w),h,w}',
'z: per-class logits; y: integer class label per pixel; ŷ: predicted class. Ignore unlabeled pixels explicitly rather than treating them as background.',
'Two touching cells can form one semantic foreground region while still requiring two instance masks. Binary foreground accuracy does not test whether the instances were separated correctly.',
'''import torch
from torch import nn
x = torch.randn(2, 3, 64, 64)
target = torch.zeros(2, 64, 64, dtype=torch.long)
target[:, 16:48, 20:44] = 1
head = nn.Conv2d(3, 2, kernel_size=1)
logits = head(x)  # untrained: shape demonstration only
loss = nn.functional.cross_entropy(logits, target)
pred = logits.argmax(dim=1)
print(logits.shape, pred.shape, loss.item())''', 'semantic', ['Image','Dense encoder / decoder','C logits at every pixel','Argmax → class map'],
'A low-resolution prediction must be aligned carefully when upsampled. A one-pixel shift in a thin structure can strongly affect overlap.',
'What information does an instance mask need beyond a semantic label?', 'A distinct object identity, so two objects of the same category remain separate. Some pipelines also need confidence scores and explicit overlap resolution.', [UNET, ('Mask R-CNN · original paper','https://arxiv.org/abs/1703.06870')])

add('U-Net: context meets precise boundaries', '04 · Dense predictions',
'U-Net uses a contracting encoder and an expanding decoder. The encoder captures context while the decoder recovers spatial detail. Skip connections bring high-resolution features directly into corresponding decoder stages.',
['Encoder stages increase channels while reducing resolution. Decoder stages upsample and fuse encoder features of matching resolution, often by channel concatenation.', 'A 1 × 1 output convolution mixes channels into class logits independently at each pixel. It does not increase the spatial receptive field.', 'The compact model below is a teaching variant with one downsampling stage and padding. The original paper uses a deeper architecture with valid convolutions and cropping; do not mistake this for an exact reproduction.'],
'E₀ = block(X); E₁ = block(pool(E₀))\nD₀ = block(concat(E₀, up(E₁)))\nZ = Conv₁×₁(D₀)',
'E: encoder features; D: decoder features; concat: concatenate along the channel axis; up: resize or learned upsampling; Z: logits.',
'With 16-channel E₀ and an upsampled 32-channel E₁, concatenation has 48 channels. Addition would require equal channel counts and would still produce only 16 or 32.',
'''import torch
from torch import nn
import torch.nn.functional as F
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU())
        self.low = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(48,16,3,padding=1), nn.ReLU())
        self.head = nn.Conv2d(16,2,1)
    def forward(self, x):
        skip = self.enc(x)
        low = self.low(F.max_pool2d(skip,2))
        up = F.interpolate(low, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.dec(torch.cat([skip,up],dim=1)))
print(TinyUNet()(torch.randn(2,1,64,64)).shape)''', 'semantic', ['1 × 64² → E₀:16 × 64²','Pool → E₁:32 × 32²','Upsample + concat E₀:48 × 64²','Decode16 → logits2 × 64²'],
'Concatenation preserves features but does not guarantee accurate boundaries. Label quality, augmentations, loss, spatial resolution, and independent evaluation still matter.',
'Why do encoder-only classification networks usually lose exact boundary information?', 'Downsampling compresses spatial detail and global pooling discards position. Dense prediction needs a decoder, preserved resolution, or other mechanisms to recover location.', [UNET])

add('Losses, Dice, IoU, and boundary quality', '04 · Dense predictions',
'Segmentation often has severe imbalance: most pixels or voxels may be background. Use losses and metrics that match the error you care about, and inspect cases rather than relying on a single aggregate number.',
['Cross-entropy evaluates probabilities locally. Soft Dice couples predictions across a region and can reduce the dominance of background. Combined objectives are common, but weights and reductions need validation.', 'Hard Dice and IoU compare discrete sets. Soft Dice is a differentiable training surrogate and is not identical to the reported hard metric.', 'Boundary distances measure geometric disagreement. Specify physical units, surface extraction, aggregation, and behavior for empty masks; Hausdorff distance is especially sensitive to outliers.'],
'Dice = 2TP / (2TP + FP + FN)\nIoU = TP / (TP + FP + FN)\nL_softDice = 1 − (2Σᵢpᵢyᵢ+ε)/(Σᵢpᵢ+Σᵢyᵢ+ε)\nDice = 2IoU / (1+IoU)',
'TP/FP/FN: true positives, false positives, false negatives; p: foreground probabilities; y: binary target; ε: stabilization constant. The shown soft loss uses unsquared sums.',
'TP = 40, FP = 10, FN = 20 gives Dice = 80/110 ≈ 0.727 and IoU = 40/70 ≈ 0.571. Pixel accuracy can still look high if the image contains thousands of background pixels.',
'''import torch
logits = torch.tensor([[[[2., -1.], [1., -2.]]]])
y = torch.tensor([[[[1., 0.], [1., 1.]]]])
p = logits.sigmoid()
soft_dice = (2*(p*y).sum()+1e-6)/(p.sum()+y.sum()+1e-6)
pred = p > .5; truth = y.bool()
tp = (pred & truth).sum().float()
fp = (pred & ~truth).sum().float()
fn = (~pred & truth).sum().float()
print("Soft:", soft_dice.item())
print("Hard:", (2*tp/(2*tp+fp+fn)).item())''', 'overlap', ['Logits → probabilities','Training loss on soft outputs','Threshold / argmax','Per-case overlap + boundary metrics'],
'Define empty-mask handling explicitly. Averaging Dice per patient differs from pooling all voxels, which gives larger volumes more influence. Report per-class scores and uncertainty.',
'If both prediction and target are empty, what is Dice?', 'The unsmoothed expression is 0/0 and undefined. Choose a policy, such as reporting a correct empty case separately or assigning 1, and state it consistently.', [UNET3, ('MONAI · metrics','https://docs.monai.io/en/stable/metrics.html')])

add('Vision transformers and attention', '05 · Modern representations',
'Attention lets one token combine information from other tokens based on content. Vision transformers turn image patches into tokens, add position information, and repeatedly mix tokens and channels.',
['A patch embedding maps a flattened P × P × C patch to a D-dimensional vector. Position encodings supply spatial order that pure token mixing would otherwise lack.', 'Queries compare with keys; softmax makes normalized weights; those weights combine values. Multiple heads learn different mixing patterns.', 'Global attention has quadratic pairwise cost in token count. Smaller patches preserve detail but increase compute. Windowed or hierarchical attention changes this tradeoff.'],
'Attention(Q,K,V) = softmax(QKᵀ / √d_k) V\nT = (H/P)(W/P)\npairwise score storage ∝ T²',
'Q,K,V: query, key, value matrices; d_k: query/key dimension per head; T: token count before any class token; P: square patch width.',
'For 224 × 224 images with 16 × 16 patches, T = 196. Halving patch width gives 784 tokens and sixteen times as many pairwise scores.',
'''import torch
from torch import nn
x = torch.randn(2, 3, 64, 64)
patches = nn.Conv2d(3, 48, kernel_size=8, stride=8)(x)
tokens = patches.flatten(2).transpose(1,2)  # [2,64,48]
position = nn.Parameter(torch.zeros(1,64,48))
attention = nn.MultiheadAttention(48, num_heads=4, batch_first=True)
z, weights = attention(tokens+position, tokens+position, tokens,
                       need_weights=True)
print(z.shape, weights.shape)
# Teaching fragment: a full ViT also has norms, MLPs, residuals and a head.''', 'patches', ['Image → patch embedding','Tokens + position','Attention + MLP blocks','Task-specific output head'],
'Attention maps are not automatically faithful explanations of a decision. Positional embedding interpolation and input resolution must be handled when changing image size.',
'Why does doubling image width and height make global attention expensive?', 'At fixed patch size, token count quadruples, so the pairwise attention matrix grows by a factor of sixteen.', [('Vision Transformer · original paper','https://arxiv.org/abs/2010.11929')])

add('Promptable segmentation and foundation models', '05 · Modern representations',
'Promptable segmentation predicts a mask conditioned on a point, box, or other cue. A foundation model can provide useful reusable image features, but adapting it to a new domain still requires careful evaluation.',
['The original Segment Anything architecture separates an image encoder, a prompt encoder, and a mask decoder. Image embeddings can be reused for multiple prompts on the same image.', 'Prompts disambiguate what to segment. A positive click indicates desired foreground; a negative click excludes a region. Several masks can be reasonable for an ambiguous object.', 'Prompt-based masks do not automatically provide semantic class names or robust volumetric consistency. For medical images or unfamiliar sensors, evaluate domain shift and annotation quality directly.'],
'E = f_image(I)\nP = f_prompt(points, boxes, masks)\n(M, q) = f_mask(E,P)',
'E: cached image embedding; P: prompt representation; M: candidate masks; q: predicted mask-quality score. A quality estimate is not a measured ground-truth IoU.',
'For ten prompts on one unchanged image, the expensive image encoder can run once and the prompt/mask pathway ten times. Editing the input image invalidates the cached embedding.',
'''# Executable toy prompt rule, NOT a SAM implementation or prediction.
import numpy as np
y, x = np.mgrid[:96, :96]
image = np.exp(-((x-48)**2+(y-48)**2)/400)
positive_click = (48, 48)  # row, column
seed_intensity = image[positive_click]
mask = image > 0.5 * seed_intensity
print("Toy prompted foreground pixels:", mask.sum())
# For SAM itself, use the official repository and a model checkpoint.
# Account for original-image coordinates when mapping prompts.''', 'prompt', ['Image → cached embedding','Points / boxes → prompt encoder','Embeddings → mask decoder','Candidate masks + quality scores'],
'Automatic pseudo-labels can reproduce systematic model errors. Review labels independently, and avoid evaluating on labels generated by the same model as if they were ground truth.',
'Why does applying a 2D promptable model independently to every CT slice not guarantee good 3D masks?', 'Independent predictions can flicker, split, or change identities across slices, and the model may not understand spacing or 3D anatomy. Evaluate volume-level consistency and physical geometry.', [('Segment Anything · paper','https://arxiv.org/abs/2304.02643'), ('Segment Anything · official code','https://github.com/facebookresearch/segment-anything')])

add('Camera geometry and calibration', '06 · From images to 3D',
'A camera projects 3D coordinates onto a 2D grid. Intrinsics describe the camera’s pixel geometry; extrinsics describe the transformation from a chosen world frame into the camera frame.',
['The pinhole model divides by camera-space depth. Points twice as far away project with half the apparent scale when other conditions are fixed.', 'Calibration estimates focal lengths, principal point, and lens distortion from known geometry or other constraints. Undistortion corrects lens effects before many geometric operations.', 'An extrinsic convention must specify direction. Here X_camera = R X_world + t. The camera center in world coordinates is −Rᵀt, not generally t.'],
'λ [u,v,1]ᵀ = K [R|t] [X,Y,Z,1]ᵀ\nK = [[f_x,0,c_x],[0,f_y,c_y],[0,0,1]]\nu = f_x X_c/Z_c + c_x; v = f_y Y_c/Z_c + c_y',
'K: intrinsics in pixels; R,t: world-to-camera pose; X_c,Y_c,Z_c: camera coordinates; c: principal point; f: focal length; λ = Z_c under this convention.',
'With f_x = 500, c_x = 320, X_c = 0.2 m and Z_c = 2 m, u = 370 pixels. Moving the point to Z_c = 4 m yields u = 345.',
'''import numpy as np
K = np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
points = np.array([[.2, 0., 2.], [.2, 0., 4.]])
projected = (K @ points.T).T
uv = projected[:, :2] / projected[:, 2:3]
print(uv)  # [[370,240], [345,240]]
# Projection assumes an ideal distortion-free camera and positive Z.''', 'projection', ['World point','R,t → camera frame','K → homogeneous pixel','Divide by depth → (u,v)'],
'Cropping or resizing changes intrinsics. Update focal lengths and principal point consistently; do not reuse the original K blindly.',
'If an image is uniformly resized by 0.5, how do f_x, f_y, c_x, and c_y change under the simple pixel-coordinate scaling convention?', 'They all scale by 0.5. Exact resampling implementations may also require accounting for pixel-center conventions.', [GEOM])

add('Stereo depth and RGB-D backprojection', '06 · From images to 3D',
'Two calibrated views can recover depth from parallax. In a rectified stereo pair, corresponding points lie on the same image row. Their horizontal displacement is disparity.',
['For a rectified pair with equal focal lengths and aligned principal points, depth is inversely proportional to disparity. Matching ambiguity is the central problem.', 'Small disparity errors become large depth errors at long range. Textureless areas, reflections, repetitive patterns, and occlusion are difficult.', 'An RGB-D image already supplies a depth value per pixel. Backprojection converts those values into 3D camera coordinates using intrinsics. Verify whether the sensor reports axial Z-depth or radial range.'],
'Z = f_x B / d\nX = (u−c_x)Z/f_x; Y = (v−c_y)Z/f_y\n|δZ| ≈ (f_x B/d²)|δd|',
'B: stereo baseline in meters; d: disparity in pixels; f_x: focal length in pixels. Formula assumes the stated rectified geometry.',
'With f_x = 600 px, B = 0.1 m, and d = 30 px, Z = 2 m. A 1-pixel disparity error gives about 0.067 m depth error near this point.',
'''import numpy as np
fx = fy = 600.; baseline = .1
cx, cy = 2., 1.5
v, u = np.mgrid[:4, :5]
disparity = np.full((4,5), 30.)
valid = disparity > 0
Z = fx * baseline / disparity[valid]
X = (u[valid]-cx) * Z / fx
Y = (v[valid]-cy) * Z / fy
cloud = np.column_stack([X,Y,Z])
print(cloud.shape, cloud[0])''', 'depth', ['Rectified image pair','Correspondence → disparity','Z = fB/d','Backproject → point cloud'],
'Some stereo APIs return fixed-point disparities scaled by 16. Convert to pixel units before using Z = fB/d, and remove invalid disparities.',
'What happens to estimated depth as disparity approaches zero?', 'Depth tends to infinity and uncertainty becomes very large. Treat tiny or invalid disparities as unreliable rather than returning an arbitrary finite distance.', [GEOM])

add('Reconstruction, point clouds, and surfaces', '06 · From images to 3D',
'A point cloud is an unordered set of 3D samples. A mesh adds surface connectivity. Reconstruction combines measurements, poses, and assumptions to estimate a coherent scene.',
['Voxel downsampling groups nearby points on a spatial grid and reduces redundancy. Normals estimate local surface orientation from neighborhood geometry.', 'ICP alternates correspondence estimation and pose fitting. It is a local method: poor initialization, little overlap, repeated structures, or moving objects can lead to failure.', 'Depth fusion can integrate signed distances into a TSDF volume. Marching cubes extracts an isosurface; triangle connectivity is created during extraction rather than present in the raw points.'],
'argmin_{R,t} Σᵢ ‖R pᵢ + t − q_{c(i)}‖²\nRᵀR = I; det(R) = 1\nvoxel index = floor((point − origin)/voxel_size)',
'p: source points; q: target points; c(i): estimated correspondence; R,t: rigid transform. The objective is point-to-point ICP; point-to-plane uses normal-direction residuals.',
'At voxel size 0.02 m, points at x = 0.101 and 0.119 m share x-index 5 when origin is zero. Downsampling sacrifices detail smaller than the chosen grid.',
'''import numpy as np
rng = np.random.default_rng(4)
points = rng.uniform(0, 1, (1000,3))
voxel_size = .1
keys = np.floor(points / voxel_size).astype(np.int64)
unique, inverse = np.unique(keys, axis=0, return_inverse=True)
sums = np.zeros((len(unique),3))
np.add.at(sums, inverse, points)
counts = np.bincount(inverse)
centroids = sums / counts[:,None]
print("Before / after:", len(points), len(centroids))''', 'cloud', ['Depths + calibrated poses','Backproject and align','Fuse / downsample','Point cloud or extracted mesh'],
'Good alignment residuals can hide the wrong alignment in repeated geometry. Check global consistency, independent landmarks, coordinate frames, and physical scale.',
'Why is a point cloud not already a watertight surface?', 'It has samples but no explicit faces, inside/outside convention, or guarantee of complete coverage. Surface reconstruction adds assumptions and can invent unsupported regions.', [OPEN3, ('Open3D · ICP registration','https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html')])

add('Volumes, voxels, spacing, and orientation', '07 · 3D segmentation',
'A volume is a grid with three spatial axes. Unlike a pile of unrelated images, its neighbors have a shared physical geometry. Understanding that geometry comes before model architecture.',
['PyTorch 3D tensors commonly use N × C × D × H × W. Libraries and file formats can store axes differently; record the mapping explicitly rather than guessing from shape.', 'Voxel spacing describes physical distance between sample centers. Anisotropic volumes can have much thicker slices than in-plane pixels, so a 3 × 3 × 3 neighborhood need not be physically cubic.', 'Medical image affine matrices map voxel indices to physical coordinates and encode orientation, spacing, and origin. Resampling changes the grid; keep the transform needed to restore predictions to the source space.'],
'[x,y,z,1]ᵀ = A [i,j,k,1]ᵀ\nvoxel volume = s_x s_y s_z\nphysical object volume = foreground_count × voxel volume',
'A: index-to-world affine for the array convention in use; i,j,k: file-array indices; s: spacing in physical units, commonly millimeters. D,H,W need not equal i,j,k order.',
'A grid with spacing (0.7,0.7,5.0) mm contains 2.45 mm³ per voxel. A 1,000-voxel mask represents 2,450 mm³, or 2.45 mL. Resampling to isotropic voxels does not create missing anatomical detail.',
'''import numpy as np
D, H, W = 24, 64, 64
z, y, x = np.mgrid[:D, :H, :W]
spacing_dhw = np.array([3., .8, .8])  # mm
radius_mm = 12
mask = (((z-12)*3)**2 + ((y-32)*.8)**2 +
        ((x-32)*.8)**2) < radius_mm**2
volume_mm3 = mask.sum() * spacing_dhw.prod()
print("Shape / physical volume:", mask.shape, volume_mm3)
# Compare against 4*pi*r**3/3; coarse sampling causes approximation error.''', 'volume', ['Volume + metadata','Verify axes / affine / spacing','Resample image and mask','N × C × D × H × W'],
'Never resample integer labels with linear interpolation. Never discard orientation metadata and then overlay the prediction on an unverified original image.',
'Why can a sphere in physical space look stretched when displayed as raw array slices?', 'Pixel display usually assumes equal axis spacing. Unequal physical voxel spacing needs aspect-ratio correction to represent geometry faithfully.', [UNET3, ('NiBabel · coordinate systems','https://nipy.org/nibabel/coordinate_systems.html')])

add('3D U-Net and patch-based inference', '07 · 3D segmentation',
'3D U-Net replaces spatial 2D operations with 3D operations, allowing neighboring slices to contribute jointly. Volumetric context helps resolve ambiguity, but activation memory grows rapidly.',
['A 3D convolution sums across input channels and a depth-height-width neighborhood. Downsample and upsample all three axes only when spacing and object sizes support it; anisotropic architectures can begin with 1 × 3 × 3 kernels.', 'Training on patches fits large volumes into memory. Sample foreground patches deliberately, while retaining background examples. Split patients before extracting patches.', 'Sliding-window inference covers a whole volume with overlapping patches and blends their probabilities or logits consistently. Padding, overlap, and normalization affect seams and edge artifacts.'],
'Y_o[d,h,w] = b_o + Σ_cΣₐΣᵦΣ_γ K_{o,c,a,b,γ} X_{c,d+a,h+b,w+γ}\nparameters = C_out(C_in k_D k_H k_W + 1)\np(v) = Σ_t w_t(v)p_t(v) / Σ_t w_t(v)',
't: tile covering voxel v; w_t: blend weight, positive on covered voxels; p_t: tile probability. Memory includes intermediate activations, gradients, optimizer state, and workspaces.',
'One float32 feature tensor of shape [1,32,128,128,128] needs 268,435,456 bytes = 256 MiB. Multiple saved feature maps can exceed GPU memory even when the model parameters are small.',
'''import torch
from torch import nn
import torch.nn.functional as F
class TinyUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv3d(1,8,3,padding=1),nn.ReLU())
        self.low = nn.Sequential(nn.Conv3d(8,16,3,padding=1),nn.ReLU())
        self.dec = nn.Sequential(nn.Conv3d(24,8,3,padding=1),nn.ReLU())
        self.head = nn.Conv3d(8,2,1)
    def forward(self,x):
        skip = self.enc(x)
        low = self.low(F.max_pool3d(skip,2))
        up = F.interpolate(low,size=skip.shape[-3:],mode='trilinear',align_corners=False)
        return self.head(self.dec(torch.cat([skip,up],1)))
x = torch.randn(1,1,16,32,32)
print(TinyUNet3D()(x).shape)
# Tiny teaching variant; see examples/train_3d.py for a complete experiment.''', 'volume', ['1 × D × H × W','3D encoder → low-resolution context','3D upsample + skip concat','2 × D × H × W logits'],
'BatchNorm estimates can be unstable with tiny 3D batches. Consider InstanceNorm or GroupNorm and validate the choice. A high patch score does not replace whole-volume evaluation.',
'Why can non-overlapping tile inference create visible seams?', 'Boundary voxels see less context and artificial padding. Overlap and weighted blending reduce disagreement, but evaluation must still inspect borders and small structures.', [UNET3, ('MONAI · inferers','https://docs.monai.io/en/stable/inferers.html')])

add('PointNet and point-wise 3D segmentation', '07 · 3D segmentation',
'Point clouds do not have a regular image grid. A model should produce the same global description when points are reordered, while per-point segmentation outputs should reorder along with their inputs.',
['PointNet applies a shared MLP to each point and aggregates with a symmetric function such as max. That gives a global feature independent of input ordering.', 'For segmentation, combine each local point feature with the repeated global feature and classify every point. The code below is a small PointNet-style core and omits the original learned alignment networks.', 'PointNet++ adds hierarchical sampling and local neighborhoods so that local geometric structure can be learned at multiple scales. Sampling density and neighborhood radius affect behavior.'],
'f({x₁,…,x_N}) = γ(maxᵢ h(xᵢ))\ng = maxᵢ h(xᵢ)\nzᵢ = ψ(concat(h(xᵢ),g))',
'xᵢ: point coordinate and optional attributes; h: shared point MLP; max: channelwise symmetric pooling; γ: global head; ψ: point classification head.',
'If point features are [1,5], [4,2], and [3,3], channelwise max is [4,5] in any point order. Concatenating this vector back to each point supplies whole-object context.',
'''import torch
from torch import nn
torch.manual_seed(5)
points = torch.randn(2, 3, 128)  # batch, xyz, points
shared = nn.Sequential(nn.Conv1d(3,32,1),nn.ReLU(),nn.Conv1d(32,64,1))
head = nn.Conv1d(128,4,1)
local = shared(points)
global_feature = local.amax(dim=2,keepdim=True)
logits = head(torch.cat([local,global_feature.expand_as(local)],dim=1))
print(logits.shape)  # [2, 4 classes, 128 points]
permutation = torch.randperm(128)
print(torch.allclose(shared(points[:,:,permutation]).amax(2), local.amax(2)))''', 'pointseg', ['N points · xyz (+ features)','Shared per-point MLP','Max pool → global feature','Concat local + global → per-point labels'],
'Permutation invariance does not imply rotation invariance. Units, coordinate frames, missing regions, and sensor density can change predictions.',
'Is a per-point segmentation output invariant or equivariant under point permutation?', 'Equivariant: permuting inputs should permute the corresponding output labels. A pooled whole-cloud class prediction should be invariant.', [POINT, ('PointNet++ · original paper','https://arxiv.org/abs/1706.02413')])

add('Sparse voxels, meshes, and representation choices', '07 · 3D segmentation',
'No single 3D representation is best for every task. Dense volumes preserve a regular neighborhood; point clouds preserve measured samples; sparse grids avoid allocating empty space; meshes represent explicit surfaces.',
['Dense 3D convolution allocates the entire bounding volume. Sparse convolution stores active voxel coordinates and features, computing only selected neighborhoods. Coordinate bookkeeping and active-site rules matter.', 'Mesh networks operate on vertices, edges, or faces. Message passing combines neighboring features, often using geometry such as normals, lengths, or relative coordinates.', 'Choose labels that match the representation. A voxel occupancy mask, a point semantic label, and a triangle part label are different supervision targets, even if they describe the same object.'],
'dense memory ∝ D H W C\nsparse feature memory ∝ N_active C (plus coordinates / indices)\nhᵢ′ = φ(hᵢ, aggregate_{j∈N(i)} ψ(hᵢ,hⱼ,eᵢⱼ))',
'N_active: stored voxels; N(i): graph neighborhood; eᵢⱼ: edge attributes; φ,ψ: learnable functions. Sparse neighbor maps can be a significant overhead.',
'A 256³ grid has 16,777,216 sites. At 1% occupancy, about 167,772 sites contain data, but sparse speedups depend on neighbor structure, hardware, and library implementation.',
'''import numpy as np
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
print(unique.shape, voxel_labels.shape)''', 'pointseg', ['Choose physical representation','Quantize / build neighborhoods','Dense, sparse, point, or mesh model','Map labels back to source geometry'],
'Quantization can merge points from different classes. Preserve a mapping to original points, define a collision policy, and evaluate at the intended physical resolution.',
'When might a dense 3D grid be preferable to a sparse representation?', 'When the volume is substantially occupied, local dense context matters, the grid fits memory, and mature dense kernels simplify implementation. Medical intensity volumes are often good candidates.', [OPEN3, ('Minkowski Engine · sparse tensor documentation','https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html')])

add('Capstones: from a baseline to a defensible result', '08 · Practice and synthesis',
'Finish the course by building systems whose outputs you can explain, reproduce, and evaluate. Start with a simple baseline, introduce one architectural change at a time, and keep an independent final test set.',
['Capstone A: classical microscopy. Generate or collect images, compare thresholding plus morphology with a learned 2D segmenter, and report per-image foreground Dice, counts, and failure cases.', 'Capstone B: volumetric segmentation. Run the included synthetic-sphere training script, then adapt the data loader to a properly licensed 3D dataset. Preserve patient grouping, spacing, orientation, and a source-space export path.', 'Capstone C: point-cloud scenes. Compare a geometric baseline to a PointNet-style model. Report class IoU, sparse-class recall, and sensitivity to rotations, missing points, and sampling density.'],
'mean class IoU = (1/C_eval) Σ_c TP_c/(TP_c+FP_c+FN_c)\nrelative improvement = (score_new − score_base)/|score_base|\nexperiment = data + split + preprocessing + model + optimizer + seed + evaluation',
'C_eval: classes included under a stated absent-class policy. A relative improvement is undefined when the baseline is zero; report absolute changes in that case.',
'A Dice increase from 0.70 to 0.77 is 0.07 absolute, or 10% relative. Report the per-case distribution and uncertainty, not only the average improvement.',
'''from pathlib import Path
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
print(experiment)''', 'volume', ['Define target + independent split','Reproduce simple baseline','Train → select on validation','Test once → visualize + document failures'],
'A synthetic-data result demonstrates the pipeline, not real-world performance. Dataset access, annotation consistency, domain shift, and independent evaluation remain separate work.',
'What should your final project report contain so someone else can assess the claim?', 'Task definition; data provenance and splits; preprocessing and physical geometry; architecture with shapes; losses and optimization; baseline; per-case and per-class metrics; failure visualizations; uncertainty; runtime/memory; reproduction commands and limitations.', [TORCH, UNET3, POINT])
