"""Builder for Notebook 15A — Stable Neural Training."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # 15A · Stable Neural Training
    ### Phase 3 — Make optimization observable, diagnosable, and reproducible

    **Prerequisites:** Notebooks 13A–15. You should be able to write a PyTorch
    training loop, derive a forward pass, explain backpropagation, and gradient-check
    a small computation. **Estimated time:** 5–7 hours.
    """),
    md(r"""
    ## 1 · Learning Objectives

    Diagnose underfitting, overfitting, vanishing/exploding gradients, dead units,
    unstable loss, and nondeterminism; choose initialization, normalization,
    optimizer, schedule, regularization, clipping, and early stopping based on evidence.
    """),
    md(r"""
    ## 2 · Historical Motivation

    Deeper networks were theoretically expressive long before they trained reliably.
    Xavier/He initialization, ReLU, normalization, residual connections, adaptive
    optimizers, and accelerator-aware practice turned fragile demonstrations into
    repeatable systems. Architecture and training procedure cannot be separated.
    """),
    md(r"""
    ## 3 · Intuition and Visual Understanding

    Training is a controlled feedback system. Activations flow forward, gradients
    flow backward, the optimizer changes parameters, and validation measures whether
    those changes generalize. Log distributions and norms—not only final accuracy.

    ```text
    initialization → forward statistics → loss → gradient norms → update size
          ↑                                                        ↓
          └──── checkpoint / schedule / early stop ← validation ───┘
    ```
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    For a ReLU layer with fan-in $d$, He initialization uses
    $$W_{ij}\sim\mathcal N(0,2/d).$$
    **Read aloud:** each weight is sampled with zero mean and variance two divided by
    the number of inputs. For $d=100$, standard deviation is $\sqrt{0.02}\approx0.141$.
    This approximately preserves signal variance through ReLU layers; it is not a
    guarantee against poor data scaling or bad learning rates.

    Gradient clipping rescales $g$ when $\lVert g\rVert_2>c$:
    $$g\leftarrow g\min(1,c/\lVert g\rVert_2).$$
    Here $c$ is the maximum norm. Clipping limits an update; it does not repair a
    systematically broken model.
    """),
    md(r"""
    ## 5 · Manual Implementation from Scratch

    Compare activation variance under naive and He initialization. A stable network
    keeps useful signal neither collapsing to zero nor exploding with depth.
    """),
    code(r"""
    import random
    import numpy as np
    import torch
    from torch import nn

    torch.manual_seed(42)
    x = torch.randn(512, 100)
    naive = torch.randn(100, 100)
    he = torch.randn(100, 100) * (2 / 100) ** 0.5
    print("input var", round(x.var().item(), 3),
          "naive ReLU var", round(torch.relu(x @ naive).var().item(), 3),
          "He ReLU var", round(torch.relu(x @ he).var().item(), 3))
    """),
    md(r"""
    ## 6 · Visualization

    Plot training and validation loss together. Four common shapes matter: both high
    (underfit), train down/validation up (overfit), both noisy/diverging (optimization
    failure), and both falling then flattening (candidate convergence).
    """),
    code(r"""
    history = {"train": [1.0, .72, .51, .37, .28, .22],
               "validation": [1.05, .78, .59, .54, .61, .73]}
    best_epoch = int(np.argmin(history["validation"]))
    print("best validation epoch", best_epoch,
          "restore it instead of the final epoch")
    """),
    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Changing several controls at once, making diagnosis impossible.
    - Selecting on the test set or reporting the last rather than best validation epoch.
    - Using dropout or batch normalization in the wrong model mode.
    - Treating gradient clipping as a cure for an excessive learning rate.
    - Comparing optimizers with unequal budgets or unrecorded seeds.
    - Reporting one lucky run as a robust improvement.
    """),
    md(r"""
    ## 8 · Library Implementation

    Build an explicit recipe: initialize layers, choose optimizer and schedule, log
    losses/metrics/gradient norm, clip only when justified, save the best validation
    checkpoint, and evaluate the test set once.
    """),
    code(r"""
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 10))
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    print(model, optimizer.__class__.__name__, scheduler.__class__.__name__)
    """),
    md(r"""
    ## 9 · Realistic Case Study

    For handwritten digits, compare logistic regression, an MLP, and a CNN under the
    same split. Run a controlled dropout ablation, preserve the best checkpoint, and
    report mean/variation across seeds. The phase checkpoint implements this pattern.
    """),
    md(r"""
    ## 10 · Production and Learning Considerations

    Record seeds, deterministic settings, hardware, framework version, data/split
    fingerprints, full configuration, selected epoch, and checkpoint hash. Exact
    bitwise reproducibility can trade off against speed; declare the chosen level.
    """),
    md(r"""
    ## 11 · Tradeoff Analysis

    SGD often needs more tuning but can generalize strongly; AdamW is forgiving and
    efficient. Batch normalization depends on batch statistics; layer normalization
    does not. Dropout reduces co-adaptation but can slow fitting. Early stopping saves
    compute but makes the stopping rule part of the model-selection procedure.
    """),
    md(r"""
    ## 12 · Readiness and Interview Preparation

    Given training/validation curves and gradient norms, identify the likely failure,
    propose one controlled intervention, predict its effect, and define what evidence
    would falsify the diagnosis.
    """),
    md(r"""
    ## 13 · Teach-Back

    Explain why good initialization matters even though training changes every weight.
    Contrast normalization, weight decay, dropout, clipping, scheduling, and early
    stopping: each addresses a different failure and they are not interchangeable.
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    1. **Guided (20 min):** calculate He standard deviation for fan-in 256.
       Self-check: $\sqrt{2/256}\approx0.0884$.
    2. **Independent (40 min):** log global gradient norm and the update-to-weight
       ratio for every epoch. Flag non-finite values.
    3. **Challenge (90 min):** compare SGD and AdamW across three fixed seeds under
       equal epoch budgets; report mean, range, curves, and practical effect size.

    Readiness threshold: 80%, a correct diagnosis exercise, and no use of test data
    for choosing optimizer, epoch, architecture, or regularization.
    """),
]

build("phase3_deep_learning/15a_stable_neural_training.ipynb", cells)
