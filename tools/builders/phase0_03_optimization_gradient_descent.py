"""Builder for Notebook 03 — Optimization and Gradient Descent.

Run:  python3 tools/builders/phase0_03_optimization_gradient_descent.py
Emits: notebooks/phase0_foundations/03_optimization_and_gradient_descent.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # 03 · Optimization and Gradient Descent
    ### Phase 0 — Mathematical Foundations · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** Notebooks 00A–00D, Notebooks 01–02, 03A, and Notebook 04.
    You should be able to read functions, vectors, derivatives, gradients,
    expectations, and matrix condition numbers, and explain squared loss for a
    linear model. **Estimated time:** 4–6 hours including exercises.

    > Notebook 04 gave us a concrete quantity to minimize: squared loss for linear
    > regression. This notebook generalizes *how* we minimize objectives. Gradient
    > descent is the single algorithm running underneath linear regression,
    > logistic regression, every neural network, and the training of every LLM in
    > this curriculum. Understand it once, deeply, and you understand the engine of
    > all of modern ML. We will also connect it back to Notebook 01: the
    > **condition number** that governed numerical stability *also* governs how
    > fast gradient descent converges.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - Why ML is *optimization*: we minimize a loss surface, almost always
      **iteratively** (closed forms are the exception, not the rule).
    - The gradient as the direction of **steepest ascent**, and the one-line Taylor
      argument for *why* stepping along $-\nabla f$ decreases the loss.
    - **Convexity**, local vs global minima, and why **saddle points** — not local
      minima — are the real obstacle in high dimensions.
    - The **learning rate**: the most important hyperparameter, and what too-small /
      too-large does.
    - **Batch vs stochastic vs mini-batch** gradient descent, and why SGD is what
      actually scales.
    - **Momentum** and **Adam**, derived and implemented from scratch.
    - Why **feature scaling / conditioning** can matter more than the optimizer
      (the Notebook 01 connection).

    **Why it matters in industry**
    - Training cost, stability, and convergence speed are optimization choices.
    - "My model won't train" / "loss is NaN" / "loss plateaus" are optimization
      diagnoses a senior engineer makes in seconds.
    - LR schedules, warmup, and gradient clipping are the difference between a model
      that trains and one that diverges at scale.

    **Typical interview questions**
    - "Derive the gradient-descent update and explain why we move against the
      gradient."
    - "Batch vs SGD vs mini-batch — tradeoffs?"
    - "What does momentum do, and how does Adam differ from plain SGD?"
    - "Your loss explodes to NaN. Walk me through the causes."
    - "Why does feature scaling speed up training?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **Closed-form solutions don't scale or generalize.** Notebook 04 solved linear
    regression *exactly* with the normal equations. But that requires forming and
    factoring $A^\top A$ — $O(nd^2)$ time and $O(d^2)$ memory — and it only exists
    for that one convex, linear problem. Logistic regression, neural nets, and LLMs
    have **no closed form**. We need a general, scalable hammer.

    **Gradient descent (Cauchy, 1847).** The oldest and most general idea: stand on
    the loss surface, feel which way is downhill (the negative gradient), take a
    step, repeat. It needs only that we can compute a gradient — which
    *backpropagation* (Notebook 15) makes cheap for arbitrarily complex models.

    **Stochastic gradient descent (Robbins–Monro, 1951).** Computing the gradient
    over *all* data per step is wasteful when data is huge and redundant. Estimating
    it from a small random sample is noisy but **far cheaper per step**, and the
    noise even helps escape saddle points. SGD is *the* reason deep learning is
    feasible on internet-scale data.

    **Momentum (Polyak, 1964) and adaptive methods (AdaGrad 2011, RMSProp 2012,
    Adam 2014).** Plain SGD crawls through ill-conditioned valleys (it zig-zags —
    Section 6). Momentum adds inertia; adaptive methods give each parameter its own
    effective learning rate. Adam became the default optimizer for deep learning
    because it is robust to bad scaling and needs little tuning.

    The arc: from *exact-but-narrow* (normal equations) to *approximate-but-
    universal-and-scalable* (SGD + Adam) — the same arc as Notebook 01's move from
    exact solving to robust computation.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The mountain-in-fog analogy.** You are on a foggy hillside and want the
    lowest point. You can't see the valley, but you can feel the slope under your
    feet. Strategy: step downhill (against the steepest slope), and repeat. The
    *slope* is the **gradient**; your *step size* is the **learning rate**.

    - **Too small a step:** you inch down forever — slow, expensive training.
    - **Too large a step:** you overshoot the valley and bounce up the far side —
      the loss oscillates or **diverges**.
    - **Just right:** steady, fast descent.

    **The loss surface's shape matters as much as the optimizer.** A round bowl
    (well-conditioned) lets you walk straight to the bottom. A long narrow ravine
    (ill-conditioned — high condition number from Notebook 01) makes naive descent
    *zig-zag* across the walls while creeping along the floor. Feature scaling
    reshapes the ravine into a bowl; momentum/Adam dampen the zig-zag.

    **In high dimensions, local minima are rare; saddle points dominate.** A
    critical point is a local min only if the surface curves up in *every*
    direction — exponentially unlikely with thousands of parameters. Far more
    common are **saddles** (up some ways, down others). Good optimizers slide off
    them; the noise in SGD actively helps.

    ```mermaid
    flowchart LR
        S["Start: random weights"] --> G["Compute gradient<br/>(slope of loss)"]
        G --> U["Step against gradient<br/>w := w - lr * grad"]
        U --> C{"Converged?<br/>(grad ~ 0 / loss flat)"}
        C -->|no| G
        C -->|yes| D["Done: minimum"]
    ```

    Run the cells and watch fog turn into geometry.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    print("NumPy", np.__version__)
    """),

    code(r"""
    # Figure 1 — convex vs non-convex landscapes; minima and saddles.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.linspace(-3, 3, 400)
    axes[0].plot(x, x**2, color="tab:blue")
    axes[0].set_title("Convex: one global min\n(any descent finds it)")

    f = 0.5 * x**4 - 2 * x**2 + 0.3 * x          # non-convex, two minima
    axes[1].plot(x, f, color="tab:orange")
    axes[1].set_title("Non-convex: local + global minima\n(start point matters)")

    # a 2D saddle: up in x, down in y
    xx = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(xx, xx)
    Z = X**2 - Y**2
    cs = axes[2].contour(X, Y, Z, levels=20, cmap="coolwarm")
    axes[2].plot(0, 0, "ko"); axes[2].annotate("saddle", (0.1, 0.1))
    axes[2].set_title("Saddle point (the real enemy\nin high dimensions)")
    axes[2].set_aspect("equal")
    plt.suptitle("Figure 1 — The shapes of loss surfaces")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** A **convex** surface (left) has a single global minimum — gradient
    descent provably finds it regardless of where you start; linear and logistic
    regression live here. A **non-convex** surface (middle) has multiple minima, so
    the starting point and optimizer matter — neural nets live here. The **saddle**
    (right) curves *up* along one axis and *down* along another; its gradient is
    zero at the center, so a naive method can stall there. In thousand-dimensional
    networks, almost every zero-gradient point is a saddle, not a min — which is why
    modern training mostly works despite non-convexity.
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    **Notation bridge.** $f$ names an objective function, $\mathbf w$ is the vector
    of parameters we can change, $t$ is an iteration number, $\nabla f$ is the
    vector of local slopes, and Greek $\eta$ (eta) is the learning rate. Notebook
    00C introduces derivatives and gradients; Notebook 01 introduces vectors,
    norms, eigenvalues, and condition numbers.

    ### 4.1 The gradient and why we step against it
    For $f:\mathbb R^d\to\mathbb R$, the **gradient** $\nabla f(\mathbf w)$ is the
    vector of partial derivatives; it points in the direction of **steepest
    ascent**, with length equal to that steepest slope. A first-order **Taylor
    expansion** around $\mathbf w$ for a small step $\Delta\mathbf w$:
    $$f(\mathbf w+\Delta\mathbf w)\approx f(\mathbf w)+\nabla f(\mathbf w)^\top\Delta\mathbf w.$$
    **Read and symbols:** $f:\mathbb R^d\to\mathbb R$ means the function accepts a
    vector of $d$ real values and returns one real value; $\Delta\mathbf w$ means a
    small parameter change; $\approx$ means approximately equal; transpose $\top$
    turns the dot product into one predicted change in $f$.

    Choosing $\Delta\mathbf w=-\eta\,\nabla f(\mathbf w)$ gives
    $$f(\mathbf w-\eta\nabla f)\approx f(\mathbf w)-\eta\,\lVert\nabla f\rVert_2^2\le f(\mathbf w).$$
    **Symbols and example:** $\eta>0$ is step size; the squared norm is never
    negative; $\le$ means less than or equal. If local slope is `3` and $\eta=0.1$,
    the first-order predicted loss change is `−0.1 × 3² = −0.9`. This approximation
    is trustworthy only for a sufficiently small local step.

    The update rule is:
    $$\boxed{\ \mathbf w_{t+1}=\mathbf w_t-\eta\,\nabla f(\mathbf w_t)\ }$$
    **Read and symbols:** “the next parameter vector equals the current vector minus
    learning rate times current gradient.” Subscripts $t$ and $t+1$ label current
    and next iteration.

    ### 4.2 Convexity
    $f$ is **convex** if its graph lies below its chords:
    $f(\lambda a+(1-\lambda)b)\le\lambda f(a)+(1-\lambda)f(b)$. Here $\lambda$
    is a mixing number between 0 and 1, while $a$ and $b$ are two inputs. For a
    differentiable convex function, any stationary point ($\nabla f=0$) is a global
    minimum. MSE and logistic loss are convex in linear-model weights; neural-network
    losses are not.

    ### 4.3 Convergence and the condition-number link
    For a strongly convex, $L$-smooth quadratic with Hessian eigenvalues in
    $[\mu,L]$, gradient descent using the optimal fixed learning rate contracts the
    **objective error** each step by at most
    $$\Big(\frac{\kappa-1}{\kappa+1}\Big)^2,\qquad \kappa=\frac{L}{\mu}.$$
    **Read and symbols:** $L$ is largest curvature, $\mu$ (mu) is smallest positive
    curvature, and $\kappa$ (kappa) is their ratio. The factor is between 0 and 1;
    closer to zero means faster contraction. This rate assumes a quadratic and the
    optimal constant step, so it is not universal for every optimizer or schedule.

    A round bowl ($\kappa=1$) converges quickly; a ravine ($\kappa\gg1$) crawls.
    For least squares, the Hessian is proportional to $X^\top X$, so
    $\kappa(X^\top X)=\kappa(X)^2$. Poor conditioning in data becomes worse in the
    quadratic objective. Feature scaling often improves both stability and speed,
    but it cannot remove every source of bad curvature.

    ### 4.4 Stochastic and mini-batch gradients
    The training loss is an average over $n$ examples, so its gradient is too:
    $\nabla f=\frac1n\sum_i \nabla f_i$. **Symbols:** $n$ is total example count;
    $i$ selects an example; $f_i$ is that example's loss contribution; $B$ is
    mini-batch size. Batch GD uses all $n$; SGD samples one $i$; mini-batch GD uses
    $B$ examples. “Unbiased” means the stochastic gradient's average over repeated
    sampling equals the full gradient.

    ### 4.5 Momentum
    Momentum accumulates velocity to reduce zig-zagging:
    $$\mathbf v_{t+1}=\beta\mathbf v_t+\nabla f(\mathbf w_t),\qquad
    \mathbf w_{t+1}=\mathbf w_t-\eta\,\mathbf v_{t+1},\quad \beta\approx 0.9.$$
    **Symbols:** $\mathbf v_t$ is accumulated velocity; $\beta$ (beta) controls how
    much previous velocity remains; $\eta$ controls the parameter step. With
    $\beta=0$, this form reduces to plain gradient descent.

    ### 4.6 Adam (adaptive moment estimation)
    Adam tracks averages of the gradient and its square:
    $$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,$$
    $$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
    \mathbf w_{t+1}=\mathbf w_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$
    **Read and symbols:** $g_t$ is current gradient; $m_t$ averages gradients;
    $v_t$ averages squared gradients; $\beta_1,\beta_2$ are decay factors; hats
    mark bias correction; $\epsilon$ is a small positive number preventing division
    by zero. Products, squares, roots, and divisions act component by component.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    We build the optimizers from the bare update rules and watch them on a quadratic
    bowl, then train **linear regression by gradient descent** and confirm it lands
    on the Notebook-01 normal-equations solution.
    """),

    code(r"""
    # 5.1 Generic first-order optimizers — just the update rules from Section 4.
    def gradient_descent(grad, x0, lr, steps):
        x = np.array(x0, float); traj = [x.copy()]
        for _ in range(steps):
            x = x - lr * grad(x); traj.append(x.copy())
        return np.array(traj)

    def momentum(grad, x0, lr, steps, beta=0.9):
        x = np.array(x0, float); v = np.zeros_like(x); traj = [x.copy()]
        for _ in range(steps):
            v = beta * v + grad(x)
            x = x - lr * v; traj.append(x.copy())
        return np.array(traj)

    def adam(grad, x0, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
        x = np.array(x0, float); m = np.zeros_like(x); v = np.zeros_like(x)
        traj = [x.copy()]
        for t in range(1, steps + 1):
            g = grad(x)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1**t); vhat = v / (1 - b2**t)
            x = x - lr * mhat / (np.sqrt(vhat) + eps)
            traj.append(x.copy())
        return np.array(traj)

    # An ILL-CONDITIONED quadratic bowl: f(x) = 0.5 x^T A x, A = diag([1, 20]).
    A = np.diag([1.0, 20.0])
    def f(x):    return 0.5 * x @ A @ x
    def grad(x): return A @ x
    kappa = np.linalg.cond(A)
    print(f"condition number kappa = {kappa:.0f}  (a long, narrow ravine)")
    """),

    code(r"""
    # 5.2 Train LINEAR REGRESSION by gradient descent; verify against the closed form.
    n, d = 500, 3
    X = rng.normal(size=(n, d))
    Xb = np.hstack([np.ones((n, 1)), X])                 # bias column
    true_w = np.array([1.5, -2.0, 0.5, 3.0])
    y = Xb @ true_w + 0.1 * rng.normal(size=n)

    def mse_grad(w):
        return (2 / n) * Xb.T @ (Xb @ w - y)             # gradient of mean squared error

    w_gd = gradient_descent(mse_grad, np.zeros(d + 1), lr=0.1, steps=2000)[-1]
    w_closed = np.linalg.lstsq(Xb, y, rcond=None)[0]     # Notebook 01's solution
    print("gradient descent :", w_gd.round(3))
    print("closed form      :", w_closed.round(3))
    print("true             :", true_w)
    print("GD matches closed form:", np.allclose(w_gd, w_closed, atol=1e-3))
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    These figures are the heart of the notebook: *see* the learning rate, the
    conditioning, the batch/stochastic tradeoff, and why momentum/Adam win.
    """),

    code(r"""
    # Figure 2 — learning rate: too small crawls, too large diverges.
    def quad(x): return x**2
    def quad_grad(x): return 2 * x

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, lr, label in [(axes[0], 0.01, "too small"),
                          (axes[1], 0.6, "good"),
                          (axes[2], 1.01, "too large (diverges)")]:
        xs = np.linspace(-11, 11, 200)
        ax.plot(xs, quad(xs), color="lightgray")
        traj = gradient_descent(lambda v: np.array([quad_grad(v[0])]), [10.0], lr, 12)[:, 0]
        ax.plot(traj, quad(traj), "o-", ms=4, color="tab:red")
        ax.set_title(f"lr={lr} ({label})"); ax.set_ylim(-5, 130)
    plt.suptitle("Figure 2 — The learning rate is the master hyperparameter")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** On the simple bowl $f(x)=x^2$: a tiny LR (left) takes many timid
    steps; a good LR (middle) descends fast and smooth; an LR past the stability
    threshold (right) makes each step overshoot *farther* than the last — the loss
    **diverges**. There is no universally "correct" LR; it depends on the curvature
    of your loss, which is exactly why adaptive methods and LR schedules exist.
    """),

    code(r"""
    # Figure 3 — conditioning: GD zig-zags in a ravine; Adam adapts per-dimension.
    def contour(ax, title):
        u = np.linspace(-10, 10, 200); v = np.linspace(-3, 3, 200)
        U, V = np.meshgrid(u, v)
        ax.contour(U, V, 0.5 * (U**2 + 20 * V**2), levels=30, cmap="Blues", alpha=0.6)
        ax.set_title(title); ax.set_aspect("auto")

    x0 = [-9.0, 2.5]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, traj) in zip(
        axes,
        [("GD (lr=0.09)", gradient_descent(grad, x0, 0.09, 60)),
         ("Momentum (lr=0.02)", momentum(grad, x0, 0.02, 60)),
         ("Adam (lr=0.5)", adam(grad, x0, 0.5, 60))],
    ):
        contour(ax, name)
        ax.plot(traj[:, 0], traj[:, 1], "o-", ms=3, color="tab:red")
        ax.plot(0, 0, "g*", ms=15)
    plt.suptitle("Figure 3 — Same ill-conditioned ravine (kappa=20): optimizer matters")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** All three start at the same point on a ravine 20× steeper across
    than along. **Plain GD** (left) bounces between the steep walls, making little
    progress along the floor — the condition-number penalty from Section 4.3 made
    visible. **Momentum** (middle) averages out the cross-valley bouncing and rolls
    down the floor. **Adam** (right) gives the steep direction a small effective
    step and the flat direction a large one, walking almost straight to the minimum.
    The lesson: when training is slow, suspect *conditioning*, and fix it with
    scaling or an adaptive optimizer — not just by cranking the LR.
    """),

    code(r"""
    # Figure 4 — batch vs mini-batch vs SGD on linear-regression loss.
    def loss(w): return np.mean((Xb @ w - y) ** 2)

    def train(batch_size, lr, epochs=40):
        w = np.zeros(d + 1); history = [loss(w)]
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                g = (2 / len(idx)) * Xb[idx].T @ (Xb[idx] @ w - y[idx])
                w = w - lr * g
            history.append(loss(w))
        return history

    fig, ax = plt.subplots()
    ax.plot(train(n, 0.1), label="batch (n=500)")
    ax.plot(train(32, 0.05), label="mini-batch (B=32)")
    ax.plot(train(1, 0.02), label="SGD (B=1)")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
    ax.set_title("Figure 4 — Batch is smooth; SGD is noisy but cheap per step")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 4.** **Batch GD** gives a smooth, monotone curve but pays for the full
    dataset every step. **SGD** updates after every single example — its curve is
    noisy (each step uses a rough gradient estimate) but it makes far more updates
    per pass and scales to data that doesn't fit in memory. **Mini-batch** sits in
    between and is what everyone actually uses: enough averaging to be stable,
    small enough to be fast, and the right shape for GPU/BLAS throughput.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Divergence** | Loss ↑ or NaN/Inf within a few steps | Learning rate above the stability threshold | Lower LR; LR warmup; gradient clipping |
    | **Crawling / plateau** | Loss barely moves | LR too small, or ill-conditioned surface | Scale features; raise LR; use Adam; LR schedule |
    | **Zig-zagging** | Oscillates across a valley, slow along it | High condition number $\kappa$ | Feature scaling; momentum; adaptive methods |
    | **Vanishing gradients** | Early layers stop learning | Saturating activations / deep chains (Ntbk 15) | ReLU, residual connections, normalization |
    | **Exploding gradients** | Sudden NaN spikes (esp. RNNs) | Repeated multiplication of large Jacobians | Gradient clipping; normalization |
    | **Stuck on a saddle** | Gradient ~0 but not optimal | Flat saddle region | SGD noise; momentum; perturbations |
    | **Overfitting via over-training** | Train loss ↓, val loss ↑ | Optimizing too long without regularization | Early stopping; weight decay (Ntbk 02's prior) |

    The cell below reproduces **divergence** and the **scaling fix** so you can
    recognize both instantly.
    """),

    code(r"""
    # (a) Divergence when LR exceeds the stability threshold (here ~1.0 for f=x^2).
    for lr in [0.5, 1.0, 1.05]:
        traj = gradient_descent(lambda v: np.array([2 * v[0]]), [1.0], lr, 30)[:, 0]
        final = traj[-1]
        tag = "DIVERGED" if not np.isfinite(final) or abs(final) > 1e6 else f"-> {final:.2e}"
        print(f"lr={lr:<5}: {tag}")

    # (b) Feature scaling slashes the condition number, so the SAME optimizer trains faster.
    Xbad = X * np.array([1.0, 100.0, 0.01])              # wildly different feature scales
    Xbad_b = np.hstack([np.ones((n, 1)), Xbad])
    Xstd = (Xbad - Xbad.mean(0)) / Xbad.std(0)           # standardize
    Xstd_b = np.hstack([np.ones((n, 1)), Xstd])
    print("\\ncond(X^T X) unscaled  :", f"{np.linalg.cond(Xbad_b.T @ Xbad_b):.1e}")
    print("cond(X^T X) standardized:", f"{np.linalg.cond(Xstd_b.T @ Xstd_b):.1e}")
    print("=> standardizing turns a near-singular ravine into a trainable bowl.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    In practice you (1) get gradients automatically from **autodiff**
    (`torch.autograd`, Notebook 15) instead of deriving them, and (2) call a tuned
    optimizer (`torch.optim.SGD`, `Adam`, `AdamW`) instead of writing the update.
    What the framework adds: automatic differentiation through arbitrary models,
    fused GPU kernels, **LR schedulers** (warmup, cosine decay), **gradient
    clipping**, weight decay, and mixed-precision. Below we use scikit-learn's
    `SGDRegressor` (which *is* mini-batch/stochastic GD under the hood) and confirm
    it lands on the same solution our scratch code found.
    """),

    code(r"""
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)               # scaling matters for SGD!
    sgd = SGDRegressor(loss="squared_error", penalty=None, learning_rate="invscaling",
                       eta0=0.01, max_iter=2000, tol=1e-6, random_state=0)
    sgd.fit(Xs, y)
    # Map sklearn's standardized-space coefficients back is fiddly; just compare predictions.
    closed_pred = Xb @ w_closed
    sgd_pred = sgd.predict(Xs)
    corr = np.corrcoef(closed_pred, sgd_pred)[0, 1]
    print(f"correlation(closed-form preds, SGDRegressor preds) = {corr:.4f}")
    print("sklearn's SGD reaches essentially the same fit our scratch GD did.")
    """),

    md(r"""
    **Scratch vs production.** Our loop and sklearn's `SGDRegressor` implement the
    *same mathematics*; the library adds learning-rate schedules, convergence
    checks, regularization, and C-level speed. For deep models the bigger win is
    **autodiff**: you specify the forward computation and the framework computes
    $\nabla f$ exactly via backprop, so you never hand-derive gradients for a
    100-layer network. Your job: choose the optimizer, set the LR/schedule, scale
    inputs, and read the loss curve.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Training Under a Compute Budget

    **Scenario.** A team retrains a large recommender (hundreds of millions of
    rows, billions of parameters) **nightly**. The full pipeline must finish inside
    a **6-hour** window on a fixed GPU cluster, or the morning model is stale.

    **Business objectives:** ship a fresh, accurate model every day within the time
    and cost budget.

    **Cost of mistakes**
    - **Diverged run** (LR too high, no clipping): wasted hours of cluster time and
      a missed model — directly costly.
    - **Under-converged run** (LR too low / too few steps): a weaker model →
      measurable drop in engagement/revenue.
    - **Over-training**: wasted compute and overfitting.

    **Optimization decisions that move the needle**
    - **Mini-batch SGD**, batch size tuned to saturate the GPUs (throughput) without
      blowing memory.
    - **Adam/AdamW** for robustness to feature scaling across heterogeneous inputs.
    - **LR warmup + cosine decay** to start stable and finish fine-grained — the
      single biggest stability lever at scale.
    - **Gradient clipping** as insurance against rare exploding-gradient spikes that
      would NaN the run and cost the whole window.
    - **Early stopping** on a validation metric to avoid burning the budget past the
      point of diminishing returns.

    **Constraints:** fixed wall-clock and GPU budget; data too large for memory
    (hence stochastic, streaming gradients).

    **KPIs:** time-to-convergence, final validation loss, run reliability (fraction
    of nightly runs that complete without diverging), and $/training-run.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Learning-rate schedules** (warmup → decay) are usually higher-leverage than
      the optimizer choice. Warmup prevents early divergence; decay sharpens the
      final solution.
    - **Gradient clipping** caps the update norm — cheap insurance against NaN-ing a
      long, expensive run on a rare bad batch (essential for RNNs/Transformers).
    - **Batch size interacts with LR.** Larger batches → less gradient noise →
      typically need a larger LR (linear-scaling rule) and warmup.
    - **Monitor the loss curve, not just the final number.** Spikes, plateaus, and a
      train/val gap each point to a specific cause (LR, conditioning, overfitting).
    - **Reproducibility & checkpointing.** Seed everything; checkpoint often so a
      crash near hour 5 doesn't cost the whole window.
    - **Mixed precision** speeds training and cuts memory but can underflow
      gradients — use loss scaling.
    - **Feature scaling / normalization** (BatchNorm, LayerNorm) is an *optimization*
      tool: it conditions the loss surface so training is faster and more stable.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Batch vs mini-batch vs SGD:**

    | Dimension | Batch GD | Mini-batch | SGD (B=1) |
    |---|---|---|---|
    | Gradient quality | Exact | Low-variance estimate | Noisy estimate |
    | Cost per step | High ($O(n)$) | Moderate | Low |
    | Scales to huge data | No | **Yes** | Yes |
    | Convergence path | Smooth | Slightly noisy | Very noisy |
    | Escapes saddles | Poorly | Well | Best |
    | Hardware fit | Poor | **Best (GPU)** | Underutilizes |

    **Optimizers:**

    | Dimension | SGD | SGD + Momentum | Adam / AdamW |
    |---|---|---|---|
    | Tuning needed | High (LR-sensitive) | Moderate | **Low** (robust) |
    | Handles bad conditioning | Poorly | Better | **Well** (per-dim scaling) |
    | Memory | Lowest | +1 buffer | +2 buffers |
    | Generalization | Often best (vision) | Strong | Sometimes slightly worse* |
    | Default for | Well-conditioned/CNNs | CNNs | **Transformers/LLMs** |

    *AdamW (decoupled weight decay) narrows the generalization gap; it's the de
    facto default for large language models.

    **First-order vs second-order (Newton):**

    | Dimension | First-order (GD/Adam) | Second-order (Newton) |
    |---|---|---|
    | Uses curvature | No (Adam approximates) | Yes (Hessian) |
    | Cost per step | $O(d)$ | $O(d^2)$–$O(d^3)$ |
    | Scales to deep nets | **Yes** | No (Hessian too big) |
    | Convergence (near min) | Linear | Quadratic |
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Why move against the gradient?* → Taylor: $f(\mathbf w-\eta\nabla f)\approx
      f-\eta\lVert\nabla f\rVert^2$, which decreases for small $\eta$ (Section 4.1).
    - *What does the learning rate control?* → Step size; too small = slow, too
      large = overshoot/diverge.

    **Deep-dive questions**
    - *Batch vs SGD vs mini-batch tradeoffs?* (Section 11 table — say *why* mini-
      batch is the default: variance/cost/hardware.)
    - *Derive Adam's update and explain the $\sqrt{\hat v}$ term.* (Section 4.6 —
      per-parameter adaptive step ≈ auto-conditioning.)
    - *Why does feature scaling speed up training?* → Lowers the condition number;
      connect to the $(\kappa-1)/(\kappa+1)$ rate and Notebook 01.

    **Whiteboard questions**
    - "Implement gradient descent for linear regression." (Section 5.2.)
    - "Implement momentum / Adam from the update rules." (Section 5.1.)

    **Strong vs weak answers**
    - *"Your training loss is NaN. What happened?"*
      - **Weak:** "Bad data."
      - **Strong:** "Most likely the LR is above the stability threshold so updates
        diverge; I'd lower it, add warmup and gradient clipping, check for exploding
        gradients and unscaled features, and confirm no inf/NaN in the inputs."
    - *"Adam or SGD?"*
      - **Weak:** "Adam, always."
      - **Strong:** "Adam for fast, robust convergence on ill-conditioned problems
        like Transformers; tuned SGD+momentum often generalizes better on vision.
        I'd default to AdamW for LLMs and benchmark both with a proper LR schedule."

    **Follow-ups:** "Now batch size is 10×—what do you change?" (scale LR, add
    warmup). "Loss plateaus at epoch 20—debug it." (LR decay vs raise, conditioning,
    saddle, data).

    **Common mistakes:** assuming a single best LR; ignoring feature scaling;
    confusing local minima with saddles; claiming GD finds the global min of a
    non-convex loss; forgetting that batch size and LR are coupled.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define gradient descent and its update rule.
    2. **Why was it invented?** Why iterative optimization over closed-form solving?
    3. **How does it work?** Give the Taylor argument for stepping against the
       gradient.
    4. **Why does it work?** Why does it find the global min for convex losses but
       only a local/critical point otherwise — and why is that usually fine?
    5. **When to use it?** Batch vs mini-batch vs SGD — pick one and justify it.
    6. **When NOT to use it?** When would you prefer a closed form or second-order
       method?
    7. **Tradeoffs?** SGD vs momentum vs Adam.
    8. **How would you productionize it?** Describe LR schedule, clipping, batch
       size, monitoring, and checkpointing for a large nightly training job.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Estimated time:** 90–150 minutes. The first two derivations now have explicit
    self-checks; use them only after making your own attempt.

    **Beginner (conceptual)**
    1. Derive the gradient-descent update from a first-order Taylor expansion and
       state the condition on $\eta$ for the loss to decrease.
    2. For $f(x)=x^2$, find the largest stable learning rate analytically, then
       verify it numerically with the divergence cell.

    **Beginner → Intermediate (coding)**
    3. Add **RMSProp** to the Section 5.1 optimizers and compare its trajectory to
       Adam's on the ill-conditioned bowl.
    4. Train **logistic regression** by gradient descent on a 2-class synthetic
       dataset (you'll derive its gradient in Notebook 05); plot the loss curve.

    **Intermediate (analysis)**
    5. Empirically measure GD's convergence rate on quadratics with $\kappa\in\{1,
       5,20,100\}$ and compare to the $((\kappa-1)/(\kappa+1))^2$ prediction.
    6. Implement an LR **warmup + cosine-decay** schedule and show it trains a
       poorly-scaled problem more reliably than a constant LR.

    **Senior (interview + production design)**
    7. *Whiteboard:* explain, with the condition number, why standardizing features
       can speed training more than switching optimizers — and when it can't.
    8. *Design:* a nightly training job must finish in 6 hours or ship a stale
       model. Specify optimizer, batch size, LR schedule, clipping, early-stopping
       criterion, checkpointing cadence, and the metrics you'd alert on.
    9. *Debug:* given a loss curve that descends, then spikes to NaN at step 4000,
       list the three most likely causes in order and the experiment you'd run to
       confirm each.

    <details>
    <summary><strong>Hints, expected results, and scoring rubric</strong></summary>

    1. Substitute $\Delta\mathbf w=-\eta\nabla f$ into the Taylor approximation.
       The predicted change is $-\eta\lVert\nabla f\rVert^2$, negative for positive
       $\eta$ and non-zero gradient when the local approximation is valid.
    2. For $x_{t+1}=(1-2\eta)x_t$, shrinking requires $|1-2\eta|<1$, giving
       $0<\eta<1$. At exactly 1 the values oscillate without shrinking.
    3. RMSProp keeps the squared-gradient moving average but not momentum's first
       moment. Compare methods using the same start and step budget.
    4. Loss should decrease and accuracy should exceed chance. Verify the gradient
       shape before training.
    5. Use the optimal fixed step for each quadratic before comparing measured
       objective-error ratios with theory.
    6. Plot learning rate and loss together and compare multiple random seeds.
    7. Award points for connecting scaling to Hessian eigenvalues, noting remaining
       correlation/non-convexity, and naming a case where adaptation still helps.
    8. Full credit includes a measurable deadline policy, recovery from checkpoint,
       and alert thresholds—not only optimizer names.
    9. Strong diagnoses test learning-rate instability, gradient overflow, and bad
       input batches independently instead of changing every variable at once.

    A score of 12/15 across Questions 7–9 indicates senior-level readiness.
    </details>
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Gradient descent is the universal engine of ML: estimate the downhill direction
    ($-\nabla f$), take a step sized by the learning rate, repeat. Conditioning
    (Notebook 01) sets the speed; stochasticity (Notebook 02's sampling) makes it
    scale; momentum and Adam tame ill-conditioned ravines; and schedules, clipping,
    and scaling are what keep big training runs alive.

    **Phase 0 is complete** — you now hold the three pillars: the *geometry* of data
    (01), *reasoning under uncertainty* (02), and the *optimization* that turns a
    loss into a trained model (03).

    **Next:** `04 · Linear Regression` — Phase 1 begins. We assemble all three
    pillars into the first complete learning algorithm, end to end.
    """),
]

build("phase0_foundations/03_optimization_and_gradient_descent.ipynb", cells)
