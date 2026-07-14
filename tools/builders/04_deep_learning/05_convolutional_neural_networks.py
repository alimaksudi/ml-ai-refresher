"""Builder for Lesson DL-05 — Convolutional Neural Networks (CNN).

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # DL-05 · Convolutional Neural Networks (CNN)
    ### Section 04 — Deep Learning Foundations · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** DL-01 through DL-04. You should be able to train and evaluate
    a network, derive backpropagation, choose stable training controls, and diagnose
    learning curves before adding a spatial inductive bias.

    > The MLP of Lesson DL-02 treats every input as an unrelated number — fine for
    > tabular data, disastrous for **images**. A modest 224×224 colour photo has
    > ~150,000 pixels; a single dense hidden layer over it needs *hundreds of millions*
    > of weights, learns nothing about spatial structure, and must re-learn a cat from
    > scratch in every corner of the frame. The **convolutional neural network** fixes
    > all three problems with one idea — **slide a small, shared filter across the
    > image** — giving local receptive fields, weight sharing, and translation
    > equivariance. Stacked, these layers learn a *hierarchy* of features
    > (edges → textures → parts → objects). This notebook builds convolution and
    > pooling from scratch in NumPy, shows filters as learnable feature detectors, and
    > trains a real CNN with the backprop of Lesson DL-03.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - *Why* dense MLPs fail on images: **parameter explosion**, **no translation
      invariance**, and **ignoring spatial locality** (the DL-02 contrast).
    - **Convolution** as a sliding, learnable **feature detector**; **weight sharing**
      and **local receptive fields**.
    - **Stride, padding, pooling**, and how feature-map size is computed.
    - The **feature hierarchy** (edges → textures → parts → objects) and the growing
      **receptive field** with depth.
    - **Translation equivariance** (conv) vs **invariance** (pooling).
    - 2D convolution, max-pooling, and a conv forward pass **from scratch**; filters as
      *learned* detectors.

    **Why it matters in industry**
    - CNNs power vision in production — medical imaging, autonomous driving, OCR,
      defect detection, content moderation.
    - The *inductive biases* (locality, weight sharing) are the template for
      efficient, data-frugal architectures; the same "share parameters across
      positions" idea reappears in Transformers (Lesson DL-08).
    - Understanding receptive fields and parameter counts is core to designing models
      that fit compute/latency budgets.

    **Typical interview questions**
    - "Why use a CNN instead of a fully-connected net for images?"
    - "What are weight sharing and parameter sharing, and why do they help?"
    - "Explain stride, padding, and how output size is computed."
    - "What does pooling do, and how does it give translation invariance?"
    - "What is a receptive field and why does it grow with depth?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **The dense-net wall for images.** Lesson DL-02's MLP connects every input to every
    hidden unit. For images this is catastrophic on three fronts: (1) **parameters
    explode** (one dense layer over a 224×224×3 image ≈ 150M weights → overfits, won't
    fit in memory); (2) it has **no notion of locality** — neighbouring pixels (which
    form edges and shapes) are treated no differently from distant ones; (3) it has
    **no translation invariance** — a cat learned in the top-left teaches the network
    nothing about a cat in the bottom-right.

    **Inspiration from the visual cortex (Hubel & Wiesel, 1960s).** Neuroscience found
    that early visual neurons respond to *small local regions* and *specific
    orientations* (edges), and that this is organized hierarchically. This motivated
    the **Neocognitron** (Fukushima, 1980) and then **LeNet** (LeCun, 1989–1998), which
    introduced trainable convolutions + pooling for digit recognition — the first CNN.

    **The 2012 breakthrough.** **AlexNet** (Krizhevsky, Sutskever, Hinton, 2012) won
    ImageNet by a huge margin using a deep CNN on GPUs, igniting the deep-learning era.
    Successors (VGG, GoogLeNet, **ResNet** with residual connections — Lesson DL-03's
    fix for vanishing gradients) pushed accuracy past human level on many tasks.

    **The three CNN ideas, each solving one MLP failure:**
    - **Local receptive fields** → exploit spatial locality (a filter only looks at a
      small patch).
    - **Weight sharing** → the *same* filter slides everywhere, so parameters don't
      scale with image size and the same feature is detected anywhere.
    - **Pooling** → downsample for translation invariance and a growing receptive
      field.

    These **inductive biases** are why CNNs need far fewer parameters and far less data
    than a dense net to learn vision — a recurring senior theme: the right architecture
    *bakes in* the structure of the problem.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **A filter is a little pattern-matcher slid across the image.** Take a small grid of
    weights (say 3×3) — a **kernel** — and at every position compute the dot product
    between the kernel and the underlying image patch. Where the patch *looks like* the
    kernel, the response is high. A vertical-edge kernel lights up at vertical edges, a
    blur kernel averages, and so on. The output is a **feature map**: "where in the
    image does this pattern occur?"

    **Three big wins over a dense layer:**
    - **Local receptive fields:** each output looks at only a small patch — matching how
      images are built from local structure (edges, corners).
    - **Weight sharing:** the *same* kernel is reused at every position. So (a) the
      parameter count is the kernel size, **not** the image size, and (b) a feature
      learned in one place is detected **everywhere** — *translation equivariance*.
    - **Pooling:** summarize each neighbourhood (e.g. take the max), shrinking the map
      and making the representation robust to small shifts — *translation invariance*.

    **Stacking builds a hierarchy.** Early layers detect **edges**; combining edges
    gives **textures and corners**; combining those gives **parts** (an eye, a wheel);
    combining parts gives **objects** (a face, a car). Each layer's **receptive field**
    — the region of the original image it can "see" — grows with depth, so deep units
    integrate global structure from local detectors. This is "deep learning =
    representation learning" (Lesson DL-02, Fig 5) specialized for space.

    ```mermaid
    flowchart LR
        I["image"] --> C1["conv: edge detectors<br/>(small receptive field)"]
        C1 --> P1["pool ↓"]
        P1 --> C2["conv: textures / parts<br/>(larger receptive field)"]
        C2 --> P2["pool ↓"]
        P2 --> C3["conv: objects<br/>(global receptive field)"]
        C3 --> F["flatten → dense → class"]
    ```

    Run the cells — first, the parameter explosion that makes CNNs necessary.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["image.cmap"] = "gray"

    digits = load_digits()                      # 1797 images, 8x8, no download needed
    print("digits:", digits.images.shape, "labels:", np.unique(digits.target))
    """),

    code(r"""
    # Why not a dense net? Parameter counts for a dense layer vs a conv layer on an image.
    H, W, C = 224, 224, 3
    dense_hidden = 1000
    dense_params = (H * W * C) * dense_hidden
    conv_filters, k = 64, 3
    conv_params = (k * k * C) * conv_filters     # weight SHARING: independent of image size!
    print(f"image: {H}x{W}x{C} = {H*W*C:,} inputs")
    print(f"dense layer (1000 units): {dense_params:,} weights")
    print(f"conv layer (64 3x3 filters): {conv_params:,} weights")
    print(f"\\nratio: dense uses {dense_params // conv_params:,}x more parameters")
    print("...and the conv count does NOT grow with image size (weight sharing).")
    """),

    md(r"""
    **The motivation in one number.** A single dense hidden layer over a 224×224×3 image
    needs ~150 **million** weights; a convolutional layer with 64 filters needs ~1,700 —
    five orders of magnitude fewer — and that count is **independent of image size**
    because the filter is *shared* across all positions. Fewer parameters means less
    overfitting, less memory, and far less data needed. This is the practical payoff of
    the CNN's inductive biases.
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 The convolution (really cross-correlation) operation
    For an input $X$ and a kernel $K$ of size $k\times k$, the feature map is
    $$Y[i,j]=\sum_{a=0}^{k-1}\sum_{b=0}^{k-1} X[i+a,\,j+b]\;K[a,b]\;(+\,\text{bias}),$$
    a dot product between the kernel and each image patch. (Deep learning uses
    cross-correlation and calls it convolution; the sign convention doesn't matter
    since $K$ is learned.) With $C_{in}$ input channels and $C_{out}$ filters, each
    filter is $k\times k\times C_{in}$ and produces one output channel.

    ### 4.2 Output size: stride and padding
    For input size $N$, kernel $k$, padding $p$, stride $s$:
    $$\text{out} = \left\lfloor\frac{N + 2p - k}{s}\right\rfloor + 1.$$
    - **Padding** $p$ adds a border (zeros) so outputs don't shrink and edges are seen
      ("same" padding keeps size with $p=(k-1)/2$, $s=1$).
    - **Stride** $s$ moves the kernel $s$ pixels at a time; $s>1$ downsamples.

    ### 4.3 Weight sharing & local connectivity (the parameter argument)
    A dense layer has (inputs × units) weights. A conv layer has only
    $k\times k\times C_{in}\times C_{out}$ — **independent of spatial size** — because
    the same filter is applied at every location. This is a strong, image-appropriate
    *prior* that drastically cuts parameters and lets a feature detected in one place
    be detected everywhere (**equivariance**: shift the input → the feature map shifts).

    ### 4.4 Nonlinearity and pooling
    After convolution, apply a nonlinearity (ReLU, Lesson DL-02) elementwise, then
    optionally **pool**:
    - **Max pooling** over a $p\times p$ window keeps the strongest response, giving
      small-shift **invariance** and reducing spatial size (and compute).
    - **Average pooling** smooths; **global average pooling** collapses each channel to
      one number (common before the classifier).

    ### 4.5 The feature hierarchy & receptive field
    The **receptive field** of a unit is the region of the *input image* that
    influences it. Stacking conv/pool layers grows it: a unit two 3×3-conv layers deep
    sees a 5×5 region; with pooling, the receptive field grows quickly until deep units
    integrate the whole image. This is *why* early layers learn local edges and deep
    layers learn global objects — capacity is allocated hierarchically.

    ### 4.6 Backprop through conv & pool (Lesson DL-03)
    Convolution is linear in the kernel, so its gradients are themselves convolutions:
    $\partial L/\partial K$ is the correlation of the input patches with the upstream
    gradient $\partial L/\partial Y$, and $\partial L/\partial X$ is a (transposed)
    convolution of $\partial L/\partial Y$ with $K$. Max-pooling routes the upstream
    gradient only to the position that *was* the max (the "argmax switch"). The same
    backprop algorithm trains the whole network — we'll verify a learned filter in §5.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    We implement 2D convolution and max-pooling in NumPy, apply **hand-crafted** edge
    detectors to show what filters *do*, then **learn** a filter by gradient descent —
    demonstrating that the filters in a CNN are not designed but *discovered*.
    """),

    code(r"""
    # 5.1 2D convolution (cross-correlation) and max-pooling from scratch.
    def conv2d(img, kernel, stride=1, padding=0):
        if padding:
            img = np.pad(img, padding, mode="constant")
        kh, kw = kernel.shape
        oh = (img.shape[0] - kh) // stride + 1
        ow = (img.shape[1] - kw) // stride + 1
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                patch = img[i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[i, j] = np.sum(patch * kernel)     # dot product of kernel with patch
        return out

    def max_pool2d(img, size=2, stride=2):
        oh = (img.shape[0] - size) // stride + 1
        ow = (img.shape[1] - size) // stride + 1
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[i, j] = img[i * stride:i * stride + size,
                                j * stride:j * stride + size].max()
        return out

    img = digits.images[0]                              # an 8x8 handwritten digit
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)   # vertical edges
    fmap = conv2d(img, sobel_x, padding=1)
    print(f"input {img.shape} -> conv(same) {fmap.shape} -> maxpool {max_pool2d(fmap).shape}")
    """),

    code(r"""
    # 5.2 LEARN a filter by gradient descent: it discovers an edge detector on its own.
    # Target: the vertical-edge map produced by Sobel. Start from a random kernel and fit.
    target = conv2d(img, sobel_x, padding=1)

    def conv_grad_kernel(img_p, dout, k=3, stride=1):
        # gradient of sum((conv-target)^2) w.r.t. the kernel = correlation of patches with dout
        g = np.zeros((k, k))
        for i in range(dout.shape[0]):
            for j in range(dout.shape[1]):
                patch = img_p[i * stride:i * stride + k, j * stride:j * stride + k]
                g += patch * dout[i, j]
        return g

    img_p = np.pad(img, 1)
    kern = rng.normal(0, 0.1, (3, 3))                   # random init
    for step in range(300):
        out = conv2d(img, kern, padding=1)
        dout = 2 * (out - target) / out.size
        kern -= 0.05 * conv_grad_kernel(img_p, dout)
    print("learned kernel (rounded):\\n", kern.round(2))
    print("\\ntarget Sobel-x kernel:\\n", sobel_x)
    print("\\n-> Starting from noise, gradient descent recovered an edge-detecting filter.")
    print("   CNN filters are LEARNED feature detectors, not hand-designed.")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Four pictures: what different filters detect, the receptive-field/hierarchy idea,
    pooling-driven translation invariance, and a learned filter matching a designed one.
    """),

    code(r"""
    # Figure 1 — different kernels detect different features (apply to a digit).
    kernels = {
        "vertical edges (Sobel-x)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
        "horizontal edges (Sobel-y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
        "blur (box)": np.ones((3, 3)) / 9,
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    }
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    axes[0].imshow(img); axes[0].set_title("input digit"); axes[0].axis("off")
    for ax, (name, kr) in zip(axes[1:], kernels.items()):
        ax.imshow(conv2d(img, kr, padding=1)); ax.set_title(name, fontsize=9); ax.axis("off")
    plt.suptitle("Figure 1 — One filter = one learnable feature detector")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** Each kernel is a tiny pattern; convolving it with the image produces a
    **feature map** highlighting where that pattern occurs. The Sobel filters fire on
    vertical/horizontal edges, the box filter blurs, the sharpen filter accentuates
    detail. In a trained CNN these kernels are **not hand-chosen** — backprop (§5.2,
    Lesson DL-03) discovers whatever filters minimize the loss, and the first layer
    reliably learns edge/colour detectors much like these. A conv layer simply applies
    *many* such learned filters in parallel, each producing one channel of the output.
    """),

    code(r"""
    # Figure 2 — a 2-layer conv 'stack' grows the receptive field and builds a hierarchy.
    # Conceptual receptive-field schematic.
    fig, ax = plt.subplots(figsize=(10, 4))
    layers = ["input\npixel", "conv1\n3x3 RF", "conv2\n5x5 RF", "conv3+pool\nlarge RF"]
    sizes = [1, 3, 5, 11]
    for idx, (name, s) in enumerate(zip(layers, sizes)):
        ax.add_patch(plt.Rectangle((idx * 3, (12 - s) / 2), s, s,
                     fill=True, alpha=0.3, color="tab:blue", ec="k"))
        ax.text(idx * 3 + s / 2, -1.2, name, ha="center", fontsize=9)
        ax.text(idx * 3 + s / 2, 6.3, f"{s}x{s}", ha="center", fontsize=8)
    ax.set_xlim(-1, 12); ax.set_ylim(-2.5, 7); ax.axis("off")
    ax.set_title("Figure 2 — Receptive field grows with depth: local edges -> global objects")
    plt.show()
    """),

    md(r"""
    **Figure 2.** A unit's **receptive field** — how much of the original image it
    "sees" — starts tiny (one 3×3 conv sees 3×3) and **grows with depth** (a second
    3×3 conv sees 5×5; add pooling and it balloons). This is the mechanism behind the
    feature hierarchy: shallow units have small receptive fields and can only detect
    **local** patterns (edges), while deep units integrate large regions and detect
    **global** structures (objects). Designing a network's depth/stride is really
    designing how fast the receptive field should grow to cover the relevant scale.
    """),

    code(r"""
    # Figure 3 — translation invariance: pooling makes the response robust to shifts.
    canvas_a = np.zeros((16, 16)); canvas_a[3:6, 3:6] = 1.0     # a blob top-left
    canvas_b = np.zeros((16, 16)); canvas_b[10:13, 10:13] = 1.0  # same blob bottom-right
    detector = np.ones((3, 3))                                   # detects bright blobs

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for row, (cv, label) in enumerate([(canvas_a, "blob top-left"),
                                       (canvas_b, "blob bottom-right")]):
        resp = conv2d(cv, detector, padding=1)
        pooled = max_pool2d(resp, size=4, stride=4)
        axes[row, 0].imshow(cv); axes[row, 0].set_title(label); axes[row, 0].axis("off")
        axes[row, 1].imshow(resp); axes[row, 1].set_title("conv response (moves with blob)")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(pooled); axes[row, 2].set_title(f"pooled (max={pooled.max():.0f})")
        axes[row, 2].axis("off")
    plt.suptitle("Figure 3 — Conv is equivariant (response shifts); pooling adds invariance")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** The same blob in two locations produces a conv response that **moves
    with it** (translation *equivariance* — a direct consequence of weight sharing).
    After **pooling**, the summarized representation has the *same peak value*
    regardless of where the blob was (translation *invariance*) — the network
    recognizes "a blob is present" without caring exactly where. This is why CNNs
    generalize across position with far less data than a dense net, which would have to
    learn the pattern separately at every location.
    """),

    code(r"""
    # Figure 4 — the learned filter from 5.2 vs the target edge detector.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].imshow(kern); axes[0].set_title("LEARNED kernel (from noise)"); axes[0].axis("off")
    axes[1].imshow(sobel_x); axes[1].set_title("target Sobel-x"); axes[1].axis("off")
    axes[2].imshow(conv2d(img, kern, padding=1)); axes[2].set_title("learned kernel's feature map")
    axes[2].axis("off")
    plt.suptitle("Figure 4 — Gradient descent discovers an edge detector")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** Starting from random noise, gradient descent (§5.2) shaped the 3×3
    kernel into something closely resembling the Sobel edge detector — and its feature
    map highlights the digit's vertical edges. This is the essence of a CNN: you don't
    program the filters, you *learn* them end-to-end with backprop (Lesson DL-03). In a
    deep CNN, layer 1 learns edges, deeper layers learn increasingly abstract
    detectors — all discovered from data by the same gradient descent.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Dense net on images** | Huge params, overfit, no shift invariance | No CNN inductive bias (§3) | Use convolution + pooling |
    | **Vanishing gradients (deep CNN)** | Early layers don't learn | Multiplied Jacobians (DL-03) | ReLU, BatchNorm, **residual connections** (ResNet) |
    | **Overfitting** | Train ≫ val accuracy | Too much capacity / little data | Data augmentation, dropout, weight decay, more data |
    | **Wrong padding/stride** | Shapes mismatch; edges lost; over-downsampled | Misjudged output-size formula (§4.2) | Compute sizes; use 'same' padding when needed |
    | **Not translation/scale robust beyond small shifts** | Fails on big shifts/rotations/scales | Pooling only handles small shifts | Augmentation; multi-scale; (limited) inherent invariance |
    | **Texture bias / spurious cues** | Confident on background, not object | Learns shortcuts (e.g., grass⇒cow) | Augmentation, debiasing, careful data |
    | **Adversarial fragility** | Tiny perturbations flip predictions | High-dim linear sensitivity | Adversarial training; robustness methods |
    | **Distribution shift** | Great offline, bad on new cameras/lighting | Train/serve image distribution differs | Domain adaptation; monitor (PROD-05) |

    The cell makes the **output-size / shape** pitfall concrete — a constant source of
    bugs.
    """),

    code(r"""
    # Output-size arithmetic: a frequent source of shape bugs. Verify the formula.
    def out_size(n, k, p, s):
        return (n + 2 * p - k) // s + 1

    print("input 28, kernel 3, pad 0, stride 1 ->", out_size(28, 3, 0, 1), "(shrinks by k-1)")
    print("input 28, kernel 3, pad 1, stride 1 ->", out_size(28, 3, 1, 1), "('same' padding)")
    print("input 28, kernel 3, pad 1, stride 2 ->", out_size(28, 3, 1, 2), "(downsampled ~half)")
    print("input 8,  kernel 5, pad 0, stride 1 ->", out_size(8, 5, 0, 1), "(small input + big kernel)")
    # mismatched shapes are the classic 'RuntimeError: size mismatch' before the dense head
    print("\\nAlways compute the flattened size before the first dense layer to avoid shape errors.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    PyTorch's `nn.Conv2d`/`nn.MaxPool2d` implement convolution as highly optimized
    (im2col + GEMM, or Winograd/FFT) GPU kernels, with autograd (Lesson DL-03) handling
    the backward pass. Below, a tiny CNN trained on the 8×8 digits for a few epochs —
    note how few parameters it needs and how the API mirrors our scratch ops. Guarded
    so the notebook runs even without torch.
    """),

    code(r"""
    # A small CNN in PyTorch, trained briefly on sklearn digits. Guarded import; few epochs.
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        X = torch.tensor(digits.images[:, None, :, :], dtype=torch.float32) / 16.0
        y = torch.tensor(digits.target, dtype=torch.long)
        ntr = 1400
        Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

        cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 4x4 -> 2x2
            nn.Flatten(), nn.Linear(16 * 2 * 2, 10))
        opt = torch.optim.Adam(cnn.parameters(), lr=0.01)
        lossf = nn.CrossEntropyLoss()
        for epoch in range(8):
            opt.zero_grad(); loss = lossf(cnn(Xtr), ytr); loss.backward(); opt.step()
        acc = (cnn(Xte).argmax(1) == yte).float().mean().item()
        n_params = sum(p.numel() for p in cnn.parameters())
        print(f"CNN test accuracy: {acc:.3f} with only {n_params:,} parameters")
    except Exception as e:
        print(f"[torch not available: {type(e).__name__}] "
              f"the scratch conv/pool above already demonstrate the core mechanism.")
    """),

    md(r"""
    **Scratch vs production.** Our triple-nested loop in `conv2d` is correct but slow;
    production frameworks compute convolution as a single big matrix multiply (im2col +
    GEMM) on the GPU — orders of magnitude faster — and autograd supplies the backward
    pass we sketched in §4.6. The tiny CNN above reaches strong digit accuracy with only
    a few thousand parameters (vs the hundreds of millions a dense net would need on
    larger images), and the layer API (`Conv2d`, `MaxPool2d`) is exactly our scratch
    operations. In real systems you'd add BatchNorm, residual connections (ResNet — the
    DL-03 vanishing-gradient fix), data augmentation, and pretrained backbones.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Manufacturing Defect Detection

    **Scenario.** A factory inspects products on a conveyor with a camera and must flag
    **defective units** (scratches, misalignments, missing parts) in real time. A CNN
    classifies each frame as pass/defect (or segments the defect region).

    **Why a CNN:**
    - Defects are **local visual patterns** that can appear **anywhere** on the part —
      exactly what convolution's locality + translation equivariance handle, with far
      less data than a dense net.
    - **Transfer learning**: start from an ImageNet-pretrained backbone and fine-tune on
      a few thousand labeled images (defects are rare → limited data).

    **Business objectives:** catch defects (high recall) without over-rejecting good
    units (precision), at line speed.

    **Cost of mistakes (asymmetric, ties to MLE-01/MLE-04):**
    - **Missed defect (FN)** → bad product ships → recalls, warranty, reputation.
    - **False reject (FP)** → good product scrapped → yield loss.
    Defects are **rare and imbalanced** → optimize recall at a precision target, tune
    the threshold from the cost matrix (MLE-01 and MLE-04), and report PR-AUC.

    **Constraints:** real-time latency on edge hardware; robustness to lighting/camera
    changes (distribution shift); auditability (Grad-CAM/SHAP to show *where* the model
    saw the defect, Lesson MLE-05).

    **KPIs:** recall@precision, PR-AUC, per-frame latency, throughput (units/sec), and
    drift in image statistics as cameras/lighting change (Lesson PROD-05 → retrain,
    Lesson PROD-06).
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Transfer learning is the default.** Pretrained backbones (ImageNet) + fine-tuning
      dramatically cut data and compute needs; training from scratch is rarely worth it.
    - **Data augmentation** (flips, crops, colour jitter, rotation) is the cheapest
      accuracy/robustness win and bakes in invariances pooling alone can't (rotation,
      scale).
    - **Latency / compute.** Conv is FLOP-heavy; for edge/real-time use efficient
      architectures (MobileNet/EfficientNet), quantization, pruning, and distillation.
    - **Receptive field & resolution** must match the object scale; too-small receptive
      field misses big structures, too-high resolution wastes compute.
    - **Robustness & monitoring.** Vision models are sensitive to distribution shift
      (new cameras, lighting) and adversarial perturbations; monitor input statistics
      and performance, and retrain on shifted data (PROD-05 and PROD-06).
    - **Explainability.** Use Grad-CAM/saliency or SHAP (Lesson MLE-05) to verify the model
      attends to the object, not background shortcuts — essential for trust and
      debugging texture/spurious-cue bias.
    - **Memory.** Activations (not just weights) dominate memory for images; use
      checkpointing (Lesson DL-03) and appropriate batch sizes.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **CNN vs dense MLP on images:**

    | Dimension | Dense MLP | CNN |
    |---|---|---|
    | Parameters | **Explodes** with image size | Shared filters, size-independent |
    | Spatial locality | Ignored | **Exploited** |
    | Translation invariance | None | **Yes** (weight sharing + pooling) |
    | Data efficiency | Poor on images | **Good** (strong prior) |
    | Best for | Tabular | **Images / grids / spatial signals** |

    **Pooling choices:**

    | Pooling | Effect | Use |
    |---|---|---|
    | Max | Keeps strongest activation; sharp, shift-robust | Most common in classic CNNs |
    | Average | Smooths | Sometimes; gentler |
    | Global average | Channel → scalar; fewer params than dense | Before classifier (modern nets) |
    | Strided conv | Learnable downsampling | Modern alternative to pooling |

    **CNN vs Vision Transformer (ViT) — the modern contrast (preview of DL-08):**

    | Dimension | CNN | Vision Transformer |
    |---|---|---|
    | Inductive bias | Strong (locality, shift) | Weak (learns from data) |
    | Data needed | Less | **More** (or heavy pretraining) |
    | Long-range context | Via depth/receptive field | **Native (attention)** |
    | Default for | Most vision, limited data | Large-scale, big-data regimes |

    **Senior lesson:** CNNs win by *baking the structure of images into the
    architecture* — locality and weight sharing — which is exactly why they need fewer
    parameters and less data. The tradeoff (vs Transformers) is a strong prior: great
    when it matches the data, limiting when you have enough data to learn the structure.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Why CNN over a dense net for images?* → Parameter explosion, no locality, no
      translation invariance (§2–3, the param-count demo).
    - *What is weight sharing?* → The same filter applied at every position → params
      independent of image size + translation equivariance.

    **Deep-dive questions**
    - *Stride/padding and output size?* → $\lfloor(N+2p-k)/s\rfloor+1$ (§4.2; the §7
      demo).
    - *Equivariance vs invariance?* → Conv shifts the response with the input
      (equivariant); pooling makes it shift-robust (invariant) (Fig 3).
    - *Receptive field and why it grows?* → Region of input a unit sees; grows with
      depth/stride, enabling the edge→object hierarchy (§4.5, Fig 2).

    **Whiteboard questions**
    - "Implement 2D convolution / max-pooling." (Section 5.1.)
    - "Compute the output shape and parameter count of a conv layer." (§4.2–4.3.)

    **Strong vs weak answers**
    - *"Use a CNN or an MLP for 64×64 images?"*
      - **Weak:** "MLP, it's simpler."
      - **Strong:** "CNN — an MLP ignores spatial structure and needs orders of
        magnitude more parameters with no translation invariance. The CNN's locality and
        weight sharing are the right prior and need far less data."
    - *"Your deep CNN won't train."*
      - **Weak:** "Add layers."
      - **Strong:** "Likely vanishing gradients in a deep stack — I'd add BatchNorm and
        **residual connections** (ResNet), use ReLU and good init, and check the
        learning rate (Lesson DL-03)."

    **Follow-ups:** "How handle rotation/scale invariance?" (augmentation; pooling only
    covers small shifts). "Limited labeled data?" (transfer learning + augmentation).
    "CNN vs Transformer for vision?" (inductive bias vs data scale).

    **Common mistakes:** using a dense net for images; confusing equivariance and
    invariance; botching the output-size formula; thinking pooling gives full rotation/
    scale invariance; ignoring transfer learning; forgetting activations dominate memory.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define convolution, weight sharing, and pooling.
    2. **Why was it invented?** What three MLP failures on images do CNNs fix?
    3. **How does it work?** Walk a forward pass: conv → ReLU → pool → … → dense.
    4. **Why does it work?** Why do weight sharing and locality help, and how does the
       hierarchy/receptive field emerge?
    5. **When to use it?** Which data types suit CNNs?
    6. **When NOT to use it?** When would you prefer a Transformer or a non-CNN model?
    7. **Tradeoffs?** CNN vs MLP; max vs average pooling; CNN vs ViT.
    8. **How would you productionize it?** Transfer learning, augmentation, latency,
       robustness/monitoring, and explainability.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Compute the output size and parameter count of a conv layer: input 32×32×3, 16
       filters of 5×5, stride 1, padding 2.
    2. Explain in two sentences the difference between translation equivariance and
       invariance.

    **Beginner → Intermediate (coding)**
    3. Extend `conv2d` to handle **multiple input/output channels** and a bias; verify
       output shapes against the formula.
    4. Implement **average pooling** and **global average pooling**; compare their
       effect on the digit feature maps.

    **Intermediate (analysis)**
    5. Build a small CNN (scratch conv forward + dense head) and a dense MLP with
       matched parameter budgets on digits; compare accuracy and discuss the gap.
    6. Implement **backprop through a conv layer** (§4.6) and gradient-check it
       (Lesson DL-03) against finite differences.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive how the receptive field grows through a stack of conv/pool
       layers and size a network so its receptive field covers a target object scale.
    8. *Design:* the defect-detection system of §9 — backbone choice, transfer learning,
       augmentation, imbalance handling, latency budget on edge hardware, and
       drift/explainability monitoring.
    9. *Debug:* a CNN is confident but attends to image backgrounds (texture bias).
       Diagnose with Grad-CAM/SHAP and propose three fixes.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    CNNs specialize the neural network for spatial data by **sliding small, shared
    filters** across the input. This buys three things a dense MLP lacks: **local
    receptive fields** (exploit locality), **weight sharing** (parameters independent of
    image size + translation equivariance), and **pooling** (shift invariance + growing
    receptive field). Stacked, they learn a **hierarchy** — edges → textures → parts →
    objects — all via the backprop of Lesson DL-03. The filters are *learned* feature
    detectors, not hand-designed (Figs 1, 4). The deep lesson: the right architecture
    bakes in the structure of the problem, drastically cutting data and parameter needs.

    **Related lesson:** `DL-06 · RNNs and LSTMs` — we turn from spatial data to **sequential** data
    (text, time series, audio), where the key structure is *order and memory*. RNNs
    share weights across *time* (as CNNs share across space), and LSTMs add gated memory
    to combat the vanishing gradients we met in Lesson DL-03.
    """),
]

build("04_deep_learning/05_convolutional_neural_networks.ipynb", cells)
