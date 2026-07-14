"""Builder for Notebook 13A — PyTorch Foundations and Training Loops."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # 13A · PyTorch Foundations and Training Loops
    ### Phase 3 — From validated ML experiments to runnable deep learning

    **Prerequisites:** Notebooks 03, 05, 09–10, and the Classical ML checkpoint.
    You should understand loss, gradients, train/validation/test roles, metrics,
    leakage, and reproducibility. **Estimated time:** 4–6 hours.

    This notebook teaches the framework mechanics needed by every later neural
    module. It deliberately comes before the NumPy derivation: first learn to inspect
    tensors and run a correct experiment; then Notebook 14 opens the black box.
    """),
    md(r"""
    ## 1 · Learning Objectives

    - Reason about tensor dtype, shape, device, and batch dimensions.
    - Build `Dataset` and `DataLoader` objects without mixing data partitions.
    - Use autograd and explain `zero_grad → forward → loss → backward → step`.
    - Write separate training and evaluation loops with correct model modes.
    - Save a `state_dict` checkpoint plus the metadata needed to reproduce it.
    - Diagnose shape, device, gradient, and leakage failures.
    """),
    md(r"""
    ## 2 · Historical Motivation

    NumPy exposes numerical operations but does not automatically differentiate an
    arbitrary program or manage accelerators. Modern frameworks combine tensor
    kernels, reverse-mode autodiff, reusable layers, data loading, and serialization.
    PyTorch's eager execution made the training program inspectable with ordinary
    Python debugging—a useful fit for learning and production research.
    """),
    md(r"""
    ## 3 · Intuition and Visual Understanding

    A tensor is an array plus a contract: **shape, dtype, device, and gradient role**.
    A batch `X` shaped `(32, 64)` means 32 examples, each with 64 features. A linear
    layer with 10 outputs maps it to `(32, 10)`; the first dimension remains the
    batch. Write shapes beside every arrow before writing model code.

    ```text
    Dataset → DataLoader → batch X,y → model → logits → loss
                                      ↑                ↓
                                      └──── optimizer ← gradients
    ```
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    For batch $X\in\mathbb R^{B\times d}$, weights
    $W\in\mathbb R^{d\times c}$, and bias $b\in\mathbb R^c$:
    $$Z=XW+b,\qquad Z\in\mathbb R^{B\times c}.$$

    **Read aloud:** batch inputs times weights plus a broadcast bias gives one logit
    per class for every example. $B$ is batch size, $d$ feature count, and $c$ class
    count. With `(B,d)=(4,3)` and `c=2`, `(4,3)@(3,2)+(2,)` gives `(4,2)`.
    Cross-entropy expects raw logits, not probabilities; it applies a stable log-softmax.
    """),
    code(r"""
    import random
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    def seed_everything(seed=42):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    seed_everything()
    X = torch.randn(12, 4, dtype=torch.float32)
    y = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    print("X", X.shape, X.dtype, X.device, "y", y.shape, y.dtype)
    """),
    md(r"""
    ## 5 · Manual Implementation from Scratch

    Autograd records tensor operations whose inputs require gradients. The scalar
    loss starts the reverse pass; each parameter accumulates its derivative. Gradients
    accumulate by design, so clear them before every update.
    """),
    code(r"""
    w = torch.tensor(2.0, requires_grad=True)
    loss = (w - 5.0) ** 2
    loss.backward()
    print("loss", loss.item(), "gradient", w.grad.item())
    assert w.grad.item() == -6.0
    """),
    md(r"""
    ## 6 · Visualization

    Inspect the learning curve, but never use training loss alone as evidence of
    generalization. Record training and validation metrics separately at every epoch.
    """),
    code(r"""
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(8):
        model.train(); total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(loader.dataset))
    print("loss history", [round(v, 3) for v in history])
    """),
    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Passing one-hot labels to a loss expecting integer class indices.
    - Applying softmax before `CrossEntropyLoss`.
    - Forgetting `zero_grad`, `model.train()`, `model.eval()`, or `torch.no_grad()`.
    - Evaluating on shuffled or transformed data with inconsistent preprocessing.
    - Moving the model to an accelerator but leaving a batch on the CPU.
    - Squeezing away the batch dimension when batch size is one.
    """),
    md(r"""
    ## 8 · Library Implementation

    Evaluation is a different program: disable training-only behavior and gradient
    recording, collect predictions, then calculate the metric over the whole split.
    """),
    code(r"""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        predictions = logits.argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()
    print("evaluation accuracy", round(accuracy, 3))
    """),
    md(r"""
    ## 9 · Realistic Case Study

    In a digit classifier, split by independent image before fitting normalization.
    Track seed, data fingerprint, split indices, architecture, optimizer, learning
    rate, epoch, selected checkpoint, and test-use count. The Deep Learning checkpoint
    later applies this contract to real images.
    """),
    md(r"""
    ## 10 · Production and Learning Considerations

    Save weights and metadata, not an unexplained Python object. A deployable artifact
    must include input shape/dtype, preprocessing, class mapping, framework version,
    code/data version, and selection metric. Load it on CPU in a clean process as a test.
    """),
    md(r"""
    ## 11 · Tradeoff Analysis

    Larger batches improve hardware utilization but change optimization and memory
    use. Accelerators reduce training time but increase environment complexity.
    Framework convenience speeds development, while scratch implementations remain
    valuable for understanding and custom operations.
    """),
    md(r"""
    ## 12 · Readiness and Interview Preparation

    You are ready when you can trace shapes, explain why logits—not softmax outputs—go
    into cross-entropy, write train/eval loops from memory, and diagnose a model that
    changes its validation predictions between identical runs.
    """),
    md(r"""
    ## 13 · Teach-Back

    Explain the lifecycle of one batch without framework jargon. Then explain why
    validation must use `eval()` and `no_grad()`, and why neither replaces a correct
    data split.
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    1. **Guided (15 min):** predict shapes through `Linear(4,8) → ReLU → Linear(8,3)`
       for batch 32. Self-check: `(32,4) → (32,8) → (32,8) → (32,3)`.
    2. **Independent (25 min):** write an evaluation function returning loss and
       accuracy. It must use `eval()` and `no_grad()` and aggregate by example count.
    3. **Challenge (45 min):** save/load a `state_dict` and prove predictions match.
       Record seed, shapes, classes, and PyTorch version beside it.

    Common mistakes: averaging batch averages without weighting, softmax before
    cross-entropy, and changing preprocessing between training and evaluation.
    Readiness threshold: 80% plus a correct train/eval-loop teach-back.
    """),
]

build("phase3_deep_learning/13a_pytorch_foundations.ipynb", cells)
