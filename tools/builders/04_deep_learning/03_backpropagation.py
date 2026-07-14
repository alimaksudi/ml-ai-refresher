"""Builder for Lesson DL-03 — Backpropagation.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # DL-03 · Backpropagation
    ### Section 04 — Deep Learning Foundations · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** FND-04, DL-01, and DL-02. You should understand a PyTorch
    training loop, a NumPy network forward pass, loss, and the chain rule.

    > In Lesson DL-02 we derived the gradients of a 2-layer network **by hand**. That
    > doesn't scale — modern models have *billions* of parameters and arbitrary
    > architectures. **Backpropagation** is the algorithm that computes the gradient of
    > a scalar loss with respect to *every* parameter, for *any* differentiable
    > computation, in a single backward pass — by applying the chain rule
    > systematically over a **computation graph**. It is, without exaggeration, the
    > algorithm that makes deep learning possible, and it is exactly what PyTorch's
    > `loss.backward()` does. We'll build a tiny autodiff engine from scratch, derive
    > the general layer-wise rules, and verify everything with **gradient checking**.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The **chain rule on a computation graph**: every model is a graph of simple ops,
      and backprop is the chain rule applied node-by-node.
    - **Reverse-mode automatic differentiation** and why it (not forward mode, not
      numerical differentiation) is the right tool for training neural nets.
    - The **forward pass caches, backward pass propagates** structure: "upstream
      gradient × local gradient" at each node.
    - The **general layer-wise backprop equations** (generalizing Lesson DL-02 to any
      depth).
    - **Gradient checking** with finite differences — the essential correctness tool.
    - Why **vanishing/exploding gradients** are a *backprop* phenomenon, and the fixes.

    **Why it matters in industry**
    - Backprop is the computational engine of *all* deep learning; understanding it is
      what lets you debug training (NaNs, dead layers, vanishing gradients) that a
      framework hides.
    - **Gradient checking** catches subtle bugs in custom layers/losses before they
      waste a training run.
    - The **memory–compute tradeoff** (activation caching, checkpointing) is a real
      production concern for large models.

    **Typical interview questions**
    - "Explain backpropagation. Why not just use numerical gradients?"
    - "What's the difference between forward-mode and reverse-mode autodiff?"
    - "Walk me through the backward pass of a linear layer."
    - "What causes vanishing/exploding gradients and how do you fix them?"
    - "How would you verify your gradient implementation is correct?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **The training bottleneck.** Lesson DL-02 showed a hidden layer can solve XOR — but
    Minsky & Papert's 1969 critique stood partly because nobody had an efficient way to
    *train* multilayer networks. How do you compute the gradient of the loss with
    respect to a weight buried two layers deep?

    **The naive options don't scale.**
    - **Numerical differentiation**: perturb each parameter, re-run the forward pass,
      measure the loss change. Correct, but costs *one forward pass per parameter* —
      hopeless for millions of weights, and numerically noisy.
    - **Symbolic differentiation**: expand the whole expression and differentiate —
      the formulas explode in size ("expression swell").

    **Backpropagation (Werbos 1974; popularized by Rumelhart, Hinton & Williams,
    1986).** The breakthrough: organize the computation as a **graph**, do **one
    forward pass** caching intermediate values, then **one backward pass** that applies
    the chain rule locally at each node, reusing shared sub-results. This computes the
    gradient w.r.t. *all* parameters in time comparable to a *single* forward pass —
    independent of the number of parameters. That efficiency is what made training
    deep networks feasible and ignited the connectionist revival.

    **Automatic differentiation, generalized.** Backprop is a special case of
    **reverse-mode autodiff** — a general technique for differentiating any program
    expressed as a graph of differentiable primitives. Modern frameworks (PyTorch,
    JAX, TensorFlow) implement exactly this: you write the forward computation, they
    record the graph, and `.backward()` (or `grad`) runs reverse-mode AD. Understanding
    backprop *is* understanding what these frameworks do under the hood.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Everything is a graph of simple operations.** A neural network — however
    complex — is ultimately a composition of elementary ops: add, multiply, matmul,
    tanh, exp. Draw it as a **computation graph**: inputs and parameters at the leaves,
    the scalar loss at the root, operations as nodes.

    **The chain rule, locally.** To know how the loss changes when you nudge a deep
    weight, you multiply together the sensitivities along the path from that weight to
    the loss. Backprop does this **once, for all weights simultaneously**, by walking
    the graph backward and applying one rule at every node:
    $$\text{(gradient at a node's inputs)} = \text{(upstream gradient)} \times \text{(local gradient of this op)}.$$
    Each node only needs to know how to turn the gradient flowing *into* it (from the
    loss side) into the gradient flowing to *its* inputs — a purely local computation.

    **Blame assignment flowing backward.** Intuitively, the loss sends "blame" back
    through the graph. A node receives how much it's blamed (upstream gradient),
    multiplies by how sensitive its output is to each input (local gradient), and
    passes the appropriate blame to each input. Forward pass = compute outputs and
    **cache** them; backward pass = distribute blame using those cached values.

    **Why *reverse* mode.** The loss is a single scalar; the parameters are many.
    Reverse mode computes the gradient of *one output* w.r.t. *all inputs* in one
    backward sweep — perfect for "one loss, millions of weights." (Forward mode does
    the opposite and would need one pass per parameter.)

    ```mermaid
    flowchart LR
        x["x"] --> mul["* (w·x)"]
        w["w"] --> mul
        mul --> add["+ b"]
        b["b"] --> add
        add --> act["tanh"]
        act --> loss["loss L"]
        loss -. "grad=1" .-> act
        act -. "× local grad" .-> add
        add -. "×" .-> mul
        mul -. "× → dL/dw, dL/dx" .-> w
    ```

    Run the cells: we'll build an autodiff engine that does this automatically.
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

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 The chain rule, the whole story
    For a composition $L=f(g(h(x)))$, the chain rule gives
    $\frac{dL}{dx}=\frac{dL}{df}\frac{df}{dg}\frac{dg}{dh}\frac{dh}{dx}$. Backprop is
    nothing more than this, applied to a graph and **organized to reuse shared terms**.
    Define the **upstream gradient** at a node's output as $\bar v=\partial L/\partial
    v$. Then for an op $v=\phi(u)$, the gradient at its input is the **vector–Jacobian
    product**
    $$\bar u = \Big(\frac{\partial v}{\partial u}\Big)^{\!\top}\bar v.$$
    Each primitive op only needs to implement this local rule.

    ### 4.2 Forward pass (cache) + backward pass (propagate)
    1. **Forward:** evaluate the graph leaf-to-root, storing each node's output (needed
       for local gradients — e.g. $\tanh'(z)=1-\tanh^2 z$ reuses the output).
    2. **Backward:** set $\bar L=1$ at the root, then visit nodes in **reverse
       topological order**, each accumulating its inputs' gradients via §4.1.
    Cost: one forward + one backward ≈ **2× a forward pass**, regardless of parameter
    count. That is backprop's defining efficiency.

    ### 4.3 Local gradients of common ops
    | Op | forward | local backward (given upstream $\bar v$) |
    |---|---|---|
    | add $v=a+b$ | $a+b$ | $\bar a{+}{=}\bar v,\ \bar b{+}{=}\bar v$ |
    | mul $v=ab$ | $ab$ | $\bar a{+}{=}b\,\bar v,\ \bar b{+}{=}a\,\bar v$ |
    | matmul $V=AW$ | $AW$ | $\bar A{=}\bar V W^\top,\ \bar W{=}A^\top\bar V$ |
    | tanh $v=\tanh u$ | $\tanh u$ | $\bar u{=}(1-v^2)\bar v$ |
    | relu $v=\max(0,u)$ | $\max(0,u)$ | $\bar u{=}\mathbb 1[u>0]\,\bar v$ |

    Note **add distributes** the gradient (used by bias and skip connections) and a node
    used multiple times **accumulates** gradients (sum over its outgoing edges).

    ### 4.4 General layer-wise backprop (generalizing Lesson DL-02)
    For an $L$-layer MLP with $Z_l=A_{l-1}W_l+b_l,\ A_l=g(Z_l)$, and loss $J$, the
    backward recursion is:
    $$dZ_L = \nabla_{Z_L} J\ \ (\text{e.g. } \hat y-y),\qquad
    dW_l = \tfrac1n A_{l-1}^\top dZ_l,\quad db_l=\text{mean}(dZ_l),$$
    $$dA_{l-1}=dZ_l W_l^\top,\qquad dZ_{l-1}=dA_{l-1}\odot g'(Z_{l-1}).$$
    Start at the output, walk to the input. Lesson DL-02 was the $L=2$ case; this is the
    same chain rule for any depth.

    ### 4.5 Reverse vs forward mode
    For $f:\mathbb R^n\to\mathbb R^m$: **forward mode** computes one column of the
    Jacobian per pass (cost $\propto n$ inputs); **reverse mode** computes one row per
    pass (cost $\propto m$ outputs). Training has $m=1$ (scalar loss) and $n$ huge
    (parameters), so **reverse mode wins overwhelmingly** — one backward pass yields the
    full gradient. (Forward mode is preferable when outputs ≫ inputs.)

    ### 4.6 Gradient checking
    To verify an analytic gradient, compare it to the **central finite difference**
    $$\frac{\partial J}{\partial\theta}\approx\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon},
    \quad \epsilon\sim10^{-5},$$
    and check the **relative error** $\frac{|g_{\text{analytic}}-g_{\text{num}}|}{|g_{\text{analytic}}|+|g_{\text{num}}|}$
    is tiny (≲$10^{-6}$). This catches almost every backprop bug — indispensable when
    writing custom layers/losses.

    ### 4.7 Vanishing/exploding gradients
    The backward recursion **multiplies** Jacobians layer after layer. If their
    magnitudes are consistently <1 (e.g. saturated sigmoid, $g'\le0.25$), the product
    shrinks geometrically → **vanishing gradients** (early layers barely learn). If >1,
    it grows → **exploding gradients** (NaNs). Fixes: non-saturating activations
    (ReLU), careful init (Xavier/He), normalization, residual connections (Notebook
    16), and gradient clipping (Lesson FND-04).
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    Two implementations. First a **tiny scalar autodiff engine** (a `Value` that builds
    a graph and back-propagates) — this *is* a miniature PyTorch autograd, and the
    clearest way to see "what `.backward()` does." Then a **general L-layer MLP**
    backprop, verified by **gradient checking**.
    """),

    code(r"""
    # 5.1 A minimal reverse-mode autodiff engine (micrograd-style).
    class Value:
        def __init__(self, data, _children=(), _op=""):
            self.data = float(data)
            self.grad = 0.0
            self._backward = lambda: None           # how to push grad to inputs
            self._prev = set(_children)
            self._op = _op

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), "+")
            def _backward():
                self.grad += out.grad                # add distributes the upstream grad
                other.grad += out.grad
            out._backward = _backward
            return out

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), "*")
            def _backward():
                self.grad += other.data * out.grad   # local grad of a*b w.r.t a is b
                other.grad += self.data * out.grad
            out._backward = _backward
            return out

        def tanh(self):
            t = np.tanh(self.data)
            out = Value(t, (self,), "tanh")
            def _backward():
                self.grad += (1 - t ** 2) * out.grad  # tanh'(x) = 1 - tanh^2
            out._backward = _backward
            return out

        def __pow__(self, p):
            out = Value(self.data ** p, (self,), f"**{p}")
            def _backward():
                self.grad += p * self.data ** (p - 1) * out.grad
            out._backward = _backward
            return out

        def __neg__(self): return self * -1
        def __sub__(self, other): return self + (-other)
        def __radd__(self, other): return self + other
        def __rmul__(self, other): return self * other

        def backward(self):
            # reverse topological order, then apply each node's local rule
            topo, visited = [], set()
            def build(v):
                if v not in visited:
                    visited.add(v)
                    for c in v._prev:
                        build(c)
                    topo.append(v)
            build(self)
            self.grad = 1.0
            for v in reversed(topo):
                v._backward()

    # Demo: f = (w*x + b) passed through tanh, a single neuron.
    w, x, b = Value(0.5), Value(2.0), Value(-1.0)
    out = (w * x + b).tanh()
    out.backward()
    print(f"output = {out.data:.4f}")
    print(f"dout/dw = {w.grad:.4f}, dout/dx = {x.grad:.4f}, dout/db = {b.grad:.4f}")
    """),

    code(r"""
    # 5.2 Verify the autodiff engine against numerical gradients (gradient check).
    def f_numeric(wv, xv, bv):
        return np.tanh(wv * xv + bv)

    eps = 1e-6
    w0, x0, b0 = 0.5, 2.0, -1.0
    num_dw = (f_numeric(w0 + eps, x0, b0) - f_numeric(w0 - eps, x0, b0)) / (2 * eps)
    num_db = (f_numeric(w0, x0, b0 + eps) - f_numeric(w0, x0, b0 - eps)) / (2 * eps)
    print(f"autodiff dw {w.grad:.6f}  vs numerical {num_dw:.6f}  -> match {np.isclose(w.grad, num_dw)}")
    print(f"autodiff db {b.grad:.6f}  vs numerical {num_db:.6f}  -> match {np.isclose(b.grad, num_db)}")
    print("\\nOur 40-line engine computes exact gradients -- this is what loss.backward() does.")
    """),

    code(r"""
    # 5.3 General L-layer MLP backprop (the Section 4.4 recursion), vectorized in NumPy.
    def sigmoid(z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    class DeepMLP:
        def __init__(self, sizes, seed=0):
            r = np.random.default_rng(seed)
            self.W = [r.normal(0, 1, (a, b)) * np.sqrt(2.0 / a)
                      for a, b in zip(sizes[:-1], sizes[1:])]
            self.b = [np.zeros(b) for b in sizes[1:]]

        def forward(self, X):
            self.Z, self.A = [], [X]                 # cache for the backward pass
            a = X
            for i, (W, b) in enumerate(zip(self.W, self.b)):
                z = a @ W + b
                a = sigmoid(z) if i == len(self.W) - 1 else np.tanh(z)
                self.Z.append(z); self.A.append(a)
            return a

        def backward(self, y):
            n = len(y); y = y.reshape(-1, 1)
            grads_W = [None] * len(self.W); grads_b = [None] * len(self.b)
            dZ = (self.A[-1] - y) / n                 # BCE+sigmoid output error
            for l in reversed(range(len(self.W))):
                grads_W[l] = self.A[l].T @ dZ
                grads_b[l] = dZ.sum(0)
                if l > 0:
                    dA = dZ @ self.W[l].T
                    dZ = dA * (1 - self.A[l] ** 2)    # tanh'
            return grads_W, grads_b

    # gradient-check the whole network against finite differences
    Xc = rng.normal(size=(20, 3)); yc = (rng.random(20) > 0.5).astype(float)
    net = DeepMLP([3, 5, 4, 1], seed=1)
    def loss_fn(net, X, y):
        p = np.clip(net.forward(X), 1e-12, 1 - 1e-12); y = y.reshape(-1, 1)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    net.forward(Xc); gW, gb = net.backward(yc)
    # numerical grad for a few entries of W[0]
    eps = 1e-5; errs = []
    for (i, j) in [(0, 0), (1, 2), (2, 3)]:
        net.W[0][i, j] += eps; lp = loss_fn(net, Xc, yc)
        net.W[0][i, j] -= 2 * eps; lm = loss_fn(net, Xc, yc)
        net.W[0][i, j] += eps
        num = (lp - lm) / (2 * eps)
        rel = abs(num - gW[0][i, j]) / (abs(num) + abs(gW[0][i, j]) + 1e-12)
        errs.append(rel)
        print(f"W0[{i},{j}]: analytic {gW[0][i, j]:+.6f}  numerical {num:+.6f}  rel.err {rel:.2e}")
    print(f"\\nmax relative error: {max(errs):.2e}  (< 1e-6 => backprop is correct)")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Three pictures: gradient-check agreement (analytic vs numerical), gradients
    actually training a deep net, and the **vanishing-gradient** phenomenon across
    depth for sigmoid vs ReLU.
    """),

    code(r"""
    # Figure 1 — gradient check: analytic gradients lie exactly on the numerical ones.
    net2 = DeepMLP([3, 6, 1], seed=2)
    net2.forward(Xc); gW2, _ = net2.backward(yc)
    analytic, numerical = [], []
    for i in range(3):
        for j in range(6):
            analytic.append(gW2[0][i, j])
            net2.W[0][i, j] += 1e-5; lp = loss_fn(net2, Xc, yc)
            net2.W[0][i, j] -= 2e-5; lm = loss_fn(net2, Xc, yc)
            net2.W[0][i, j] += 1e-5
            numerical.append((lp - lm) / (2e-5))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(numerical, analytic, s=30)
    lim = [min(numerical), max(numerical)]
    ax.plot(lim, lim, "r--", label="y = x (perfect agreement)")
    ax.set_xlabel("numerical gradient"); ax.set_ylabel("analytic (backprop) gradient")
    ax.set_title("Figure 1 — Gradient check: backprop matches finite differences")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 1.** Every analytic gradient from backprop lands on the $y=x$ line against
    its finite-difference estimate — the visual signature of a correct implementation.
    This is the **first thing to do** when you write a custom layer or loss: if the
    points scatter off the line, you have a bug *before* you waste GPU-hours training.
    Finite differences are too slow to train with (one forward pass per parameter,
    §2) but perfect for *checking* a handful of parameters.
    """),

    code(r"""
    # Figure 2 — gradients training the deep net: loss falls as backprop drives GD.
    Xm = rng.normal(size=(300, 3))
    ym = (Xm[:, 0] * Xm[:, 1] - Xm[:, 2] + 0.3 * rng.normal(size=300) > 0).astype(float)
    net3 = DeepMLP([3, 16, 16, 1], seed=3)
    hist = []
    for _ in range(2000):
        net3.forward(Xm)
        gW, gb = net3.backward(ym)
        for l in range(len(net3.W)):
            net3.W[l] -= 0.5 * gW[l]; net3.b[l] -= 0.5 * gb[l]
        hist.append(loss_fn(net3, Xm, ym))
    fig, ax = plt.subplots()
    ax.plot(hist); ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy")
    acc = ((net3.forward(Xm) > 0.5).ravel() == ym).mean()
    ax.set_title(f"Figure 2 — Backprop trains a 3-hidden-layer net (acc {acc:.2f})")
    plt.show()
    """),

    md(r"""
    **Figure 2.** Backprop computes the gradients; gradient descent (Lesson FND-04) takes
    the steps. A 3-hidden-layer network learns a nonlinear (interaction-driven)
    boundary — the same loop scales to billions of parameters because backprop's cost
    is independent of depth-times-width beyond the forward pass itself. Every deep
    model you'll meet (DL-05 through DL-08) trains with this exact inner loop.
    """),

    code(r"""
    # Figure 3 — VANISHING gradients: in a deep sigmoid net, early-layer grads shrink.
    def grad_norms(activation):
        sizes = [4] + [16] * 8 + [1]                 # 8 hidden layers (deep)
        r = np.random.default_rng(0)
        W = [r.normal(0, 1, (a, b)) * np.sqrt(1.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        bs = [np.zeros(b) for b in sizes[1:]]
        X = r.normal(size=(64, 4)); y = (r.random(64) > 0.5).astype(float).reshape(-1, 1)
        Z, A = [], [X]; a = X
        for i, (w, b) in enumerate(zip(W, bs)):
            z = a @ w + b
            if i == len(W) - 1:
                a = sigmoid(z)
            else:
                a = sigmoid(z) if activation == "sigmoid" else np.maximum(0, z)
            Z.append(z); A.append(a)
        dZ = (A[-1] - y) / len(y); norms = []
        for l in reversed(range(len(W))):
            norms.append(np.linalg.norm(A[l].T @ dZ))
            if l > 0:
                dA = dZ @ W[l].T
                if activation == "sigmoid":
                    s = sigmoid(Z[l - 1]); dZ = dA * s * (1 - s)
                else:
                    dZ = dA * (Z[l - 1] > 0)
        return norms[::-1]                            # layer 1 .. L

    fig, ax = plt.subplots()
    ax.plot(range(1, 10), grad_norms("sigmoid"), "o-", label="sigmoid (vanishes)")
    ax.plot(range(1, 10), grad_norms("relu"), "s-", label="ReLU (sustained)")
    ax.set_yscale("log"); ax.set_xlabel("layer (1 = closest to input)")
    ax.set_ylabel("||gradient|| (log)")
    ax.set_title("Figure 3 — Vanishing gradients: sigmoid decays toward early layers")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 3.** Backprop multiplies a Jacobian at every layer. With **sigmoid**
    (derivative $\le 0.25$, and smaller when saturated), these factors compound and the
    gradient reaching the **early layers** is orders of magnitude smaller than at the
    output — those layers barely learn (the **vanishing-gradient** problem that long
    stalled deep networks). **ReLU** (derivative 1 on the active side) keeps gradients
    far healthier across depth. This is *why* ReLU, careful initialization, batch/layer
    normalization, and **residual connections** (Lesson DL-05) were invented — all of
    them are really fixes to backprop's multiplicative gradient flow.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Vanishing gradients** | Early layers don't learn; loss plateaus | Jacobian factors <1 compound (Fig 3) | ReLU/GELU; Xavier/He init; normalization; residuals |
    | **Exploding gradients** | Loss → NaN/Inf; spikes | Jacobian factors >1 compound | Gradient clipping (FND-04); init; normalization |
    | **Wrong gradient (bug)** | Trains poorly / weirdly | Backprop math error in a custom op | **Gradient checking** (§4.6, Fig 1) |
    | **Forgot to zero grads** | Gradients accumulate across steps | Frameworks *accumulate* by default | `optimizer.zero_grad()` each step |
    | **In-place op breaks graph** | Autograd error / wrong grad | Overwrote a value needed for backward | Avoid in-place ops on graph tensors |
    | **No-grad where grad needed** | Parameter never updates | Accidental `detach`/`stop_gradient` | Check `requires_grad`/graph connectivity |
    | **Memory blowup** | OOM on deep/long models | Caching all activations for backward | Gradient checkpointing (recompute vs store) |
    | **Numerical instability** | NaN in log/exp/div | Unstable op (e.g. log of 0) | Stable softmax/log-sum-exp; clip |

    The cell shows the **"forgot to zero gradients"** trap that frameworks make easy
    to hit (gradients accumulate by default).
    """),

    code(r"""
    # Why frameworks need zero_grad: gradients ACCUMULATE unless reset. Demonstrate with our engine.
    a = Value(3.0)
    L1 = a * a                     # dL1/da = 2a = 6
    L1.backward()
    print(f"after 1st backward: a.grad = {a.grad:.1f}  (correct: 2a = 6)")
    L2 = a * a
    L2.backward()                  # WITHOUT resetting a.grad, it accumulates
    print(f"after 2nd backward (no reset): a.grad = {a.grad:.1f}  (WRONG: 6+6=12, double-counted)")
    a.grad = 0.0                   # the fix: zero gradients between steps
    L3 = a * a; L3.backward()
    print(f"after reset + backward: a.grad = {a.grad:.1f}  (correct again)")
    print("\\nPyTorch accumulates grads by design (useful for RNNs/grad accumulation),")
    print("which is why you must call optimizer.zero_grad() every training step.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    PyTorch's **autograd** is exactly our `Value` engine, scaled to tensors on GPU:
    each operation records itself on a dynamic graph, and `loss.backward()` runs
    reverse-mode AD to populate every parameter's `.grad`. We confirm PyTorch's
    gradients match our hand-derived/numerical ones. The import is guarded so the
    notebook runs without torch.
    """),

    code(r"""
    # PyTorch autograd reproduces our scratch gradients exactly. Guarded import.
    try:
        import torch
        wt = torch.tensor(0.5, requires_grad=True)
        xt = torch.tensor(2.0, requires_grad=True)
        bt = torch.tensor(-1.0, requires_grad=True)
        out_t = torch.tanh(wt * xt + bt)
        out_t.backward()                              # reverse-mode AD over the recorded graph
        print(f"PyTorch  dw={wt.grad.item():.6f}  dx={xt.grad.item():.6f}  db={bt.grad.item():.6f}")
        print(f"scratch  dw={w.grad:.6f}  dx={x.grad:.6f}  db={b.grad:.6f}")
        print("identical -> our engine implements the same algorithm as torch.autograd")
    except Exception as e:
        print(f"[torch not available: {type(e).__name__}] "
              f"the scratch Value engine already demonstrates reverse-mode autodiff.")
    """),

    md(r"""
    **Scratch vs production.** Our `Value` class and `torch.autograd` implement the
    *same* algorithm — build a graph on the forward pass, traverse it in reverse,
    apply local gradient rules — and produce identical gradients. What the framework
    adds: tensor/GPU ops, hundreds of differentiable primitives, dynamic graphs,
    memory management (and **gradient checkpointing** to trade compute for memory on
    huge models), and `.backward()` in one line. The reason to build it from scratch:
    when training breaks (NaNs, vanishing gradients, a custom layer that won't learn),
    you debug it by reasoning about the graph and gradient flow — and you reach for
    **gradient checking** to localize the bug.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Why Backprop Made Modern AI Economically Possible

    **Scenario.** A team trains a 100-million-parameter model. The choice of gradient
    method isn't academic — it determines whether training is feasible *at all*.

    **The economics of the gradient.**
    - **Numerical differentiation:** ~1 forward pass *per parameter* → $10^8$ forward
      passes *per gradient step*, times thousands of steps. Completely infeasible —
      training would take longer than the age of the universe.
    - **Backpropagation:** **one** forward + **one** backward pass per step (≈2× a
      forward pass) gives the gradient for *all* $10^8$ parameters at once. Training
      becomes a matter of hours/days on GPUs instead of impossible.
    This ~$10^8$× efficiency gap is *why* deep learning is a business reality and not a
    theoretical curiosity. Every LLM, recommender, and vision model in production is
    economically viable because of this one algorithm.

    **Business objectives:** train large models within a fixed time/GPU budget;
    catch implementation bugs before they burn that budget.

    **Cost of mistakes:**
    - A **wrong custom-layer gradient** silently degrades the model — caught cheaply by
      **gradient checking** before a multi-day run, or discovered expensively after.
    - **Exploding gradients** NaN-out a run mid-training, wasting the entire budget —
      hence clipping/normalization as standard insurance (Lesson FND-04).
    - **Memory mismanagement** (caching all activations) OOMs large models —
      gradient checkpointing trades recompute for memory.

    **Constraints:** GPU memory (activation caching scales with depth × batch × width),
    wall-clock budget, numerical stability at scale.

    **KPIs:** training throughput (steps/sec), peak memory, run reliability (no NaNs),
    and gradient-check pass-rate for any custom component before launch.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Gradient checking is a CI step for custom ops.** Before shipping a new layer or
      loss, verify analytic vs numerical gradients (Fig 1). Cheap; prevents
      catastrophically wasted training runs.
    - **Memory ↔ compute (checkpointing).** Backward needs cached forward activations;
      memory scales with depth × batch × width. **Gradient checkpointing** stores only
      some activations and *recomputes* the rest in the backward pass — trading ~33%
      more compute for large memory savings, essential for big models.
    - **Numerical stability.** Use stable primitives (log-sum-exp softmax, BCE-with-
      logits) so backprop doesn't propagate NaNs from `log(0)`/overflow.
    - **Gradient clipping & normalization** keep the multiplicative gradient flow in a
      healthy range (vanishing/exploding, §4.7) — standard for deep/recurrent models
      (FND-04 and DL-06).
    - **Zero gradients each step.** Frameworks accumulate by default; forgetting
      `zero_grad()` silently corrupts training (§7).
    - **Mixed precision** speeds training but can underflow small gradients — use loss
      scaling.
    - **Don't reimplement autograd** in production — use the framework; *do* understand
      it so you can debug gradient-flow pathologies the framework surfaces.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Gradient computation methods:**

    | Method | Cost for $P$ params | Accuracy | Use for |
    |---|---|---|---|
    | Numerical (finite diff) | $O(P)$ forward passes | Approximate, noisy | **Checking** only, never training |
    | Symbolic | Expression swell | Exact | Tiny expressions |
    | **Reverse-mode AD (backprop)** | ~2× one forward pass | Exact | **Training** (scalar loss, many params) |
    | Forward-mode AD | $O(\text{inputs})$ passes | Exact | Few inputs, many outputs |

    **Reverse vs forward mode:**

    | | Reverse (backprop) | Forward |
    |---|---|---|
    | Best when | outputs ≪ inputs (loss, params) | inputs ≪ outputs |
    | One pass gives | grad of 1 output w.r.t. all inputs | derivs of all outputs w.r.t. 1 input |
    | Memory | caches activations (high) | low |
    | Deep learning | **the choice** | rare |

    **Memory–compute tradeoff (checkpointing):**

    | Strategy | Memory | Compute |
    |---|---|---|
    | Store all activations | High | 1× backward |
    | Gradient checkpointing | **Low** | ~1.33× (recompute) |

    **Senior lesson:** backprop = reverse-mode AD, chosen because the loss is scalar and
    parameters are many. Its costs (activation memory, multiplicative gradient flow) are
    exactly what production techniques — checkpointing, clipping, normalization,
    gradient checking — exist to manage.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Explain backprop.* → Forward pass caches activations; backward pass applies the
      chain rule node-by-node in reverse topological order, computing all gradients in
      ~one extra forward-pass of cost (§3, §4.2).
    - *Why not numerical gradients?* → $O(P)$ forward passes per step and noisy — fine
      for *checking*, infeasible for *training* (§9).

    **Deep-dive questions**
    - *Forward vs reverse mode?* → Reverse computes grad of one scalar output w.r.t.
      many inputs in one pass — ideal for training (§4.5).
    - *Backward pass of a linear layer?* → $dW=A^\top dZ$, $dA_{prev}=dZ\,W^\top$
      (§4.3–4.4).
    - *Vanishing/exploding gradients — cause and fixes?* → Multiplied Jacobians compound
      (§4.7, Fig 3); ReLU, init, normalization, residuals, clipping.

    **Whiteboard questions**
    - "Implement a scalar autograd `Value` with +, *, and tanh." (Section 5.1.)
    - "How would you gradient-check a custom layer?" (Central differences; §4.6, Fig 1.)

    **Strong vs weak answers**
    - *"Your custom loss trains badly."*
      - **Weak:** "Tune the learning rate."
      - **Strong:** "First **gradient-check** the custom loss/layer against finite
        differences — a wrong analytic gradient is the most likely cause. If it
        passes, then I look at LR, init, and gradient flow."
    - *"Why did adding more layers stop training?"*
      - **Weak:** "Too complex."
      - **Strong:** "Likely vanishing gradients — backprop multiplies Jacobians, so deep
        sigmoid stacks shrink early-layer gradients. I'd switch to ReLU/GELU, use He
        init, add normalization or residual connections."

    **Follow-ups:** "Memory cost of backward?" (caches activations → checkpointing).
    "Why zero_grad?" (frameworks accumulate). "When forward-mode?" (outputs ≫ inputs,
    e.g. Jacobian-vector products).

    **Common mistakes:** confusing backprop (the algorithm) with gradient descent (the
    optimizer); thinking numerical gradients are used for training; forgetting
    activation caching/zeroing; not knowing reverse vs forward mode; assuming deeper is
    always better (vanishing gradients).
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define backpropagation as reverse-mode AD on a computation graph.
    2. **Why was it invented?** Why are numerical/symbolic gradients inadequate for
       training?
    3. **How does it work?** Describe the forward-cache / backward-propagate structure
       and the per-node rule.
    4. **Why does it work?** Why does reverse mode give all parameter gradients in ~one
       backward pass?
    5. **When to use it?** Reverse vs forward mode — when each.
    6. **When NOT to use it?** When do numerical gradients have a (limited) role?
    7. **Tradeoffs?** Memory vs compute (checkpointing); accuracy vs cost of gradient
       methods.
    8. **How would you productionize it?** Gradient checking, clipping, checkpointing,
       stable ops, and zeroing gradients.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. For $L=(wx+b-y)^2$, derive $\partial L/\partial w$ and $\partial L/\partial b$ by
       hand, then confirm with the `Value` engine.
    2. Explain why a node used in two places must *accumulate* (sum) its gradients.

    **Beginner → Intermediate (coding)**
    3. Add `exp`, `relu`, and division to the `Value` engine and gradient-check each.
    4. Use the engine to build a 2-input/1-hidden-neuron network and train it on AND/OR
       by manual gradient descent.

    **Intermediate (analysis)**
    5. Reproduce Figure 3 and quantify the per-layer gradient-decay factor for sigmoid;
       show He-initialized ReLU keeps it near 1.
    6. Implement **gradient checkpointing** for the `DeepMLP`: don't cache all
       activations, recompute them in `backward`, and verify gradients still match.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive the general layer-wise backprop recursion (§4.4) from the
       chain rule and state the per-step compute/memory cost.
    8. *Design:* a CI pipeline that gradient-checks every custom layer/loss before a
       training run; specify tolerances, which params to check, and how you'd catch a
       regression.
    9. *Debug:* training NaNs at step 500. Walk through how you'd localize it
       (clipping off? unstable softmax? exploding gradients?) and the fix for each.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Backpropagation is **reverse-mode automatic differentiation**: represent the model
    as a computation graph, run a forward pass that **caches** activations, then a
    backward pass that applies the chain rule **locally at each node** in reverse,
    computing the gradient w.r.t. *every* parameter in roughly one extra forward-pass of
    cost. That efficiency (vs $O(P)$ for numerical gradients) is what makes training
    large models possible. Its byproducts — activation memory and multiplicative
    gradient flow — drive the production toolkit: **gradient checking**, clipping,
    normalization, residuals, and checkpointing. Our 40-line `Value` engine is, in
    miniature, exactly what `loss.backward()` runs.

    **Related lesson:** `DL-05 · CNNs` — we specialize the network's architecture for spatial data
    (images): weight-sharing convolutions and pooling that build translation-invariant,
    hierarchical features — trained, of course, by the backprop we just built.
    """),
]

build("04_deep_learning/03_backpropagation.ipynb", cells)
