"""Builder for Lesson DL-02 — Neural Networks from Scratch.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # DL-02 · Neural Networks from Scratch
    ### Section 04 — Deep Learning Foundations · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** FND-04, CML-02, and DL-01. You should be able to explain
    loss and gradient descent, trace tensor shapes, and write a correct framework
    training/evaluation loop. This notebook now opens that loop and derives its math.

    > Section 02's models drew straight boundaries (linear/logistic) or axis-aligned boxes
    > (trees). Neural networks do something qualitatively new: they **learn their own
    > features**. A network is just **logistic regressions stacked in layers**
    > (Lesson CML-02), with a **nonlinearity** between them — and that single addition
    > lets it bend space into whatever shape the data needs, approximating *any*
    > function. This notebook builds a multilayer perceptron entirely in NumPy: the
    > forward pass as matrix multiplies (Lesson FND-01), the cross-entropy loss
    > (Lesson FND-02), and a manual training loop (Lesson FND-04). We derive the gradients
    > for our specific 2-layer net by hand here; Lesson DL-03 generalizes that into the
    > **backpropagation** algorithm that powers all of deep learning.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The neuron as a **logistic-regression unit**, and a network as **stacked layers**
      of them with nonlinear activations between (the CML-02 link).
    - Why **nonlinearity is essential**: stacking linear layers collapses to one linear
      layer; the activation is what buys expressiveness.
    - The **XOR problem** that killed the perceptron, and how a hidden layer solves it
      by learning a **new representation** where the classes become linearly separable.
    - **Universal approximation**: enough hidden units can fit any continuous function.
    - Activations (**sigmoid, tanh, ReLU**), their derivatives, and the saturation /
      vanishing-gradient story.
    - A complete **from-scratch NumPy MLP** — forward pass, manual gradients, training
      loop — fit to a nonlinear dataset.

    **Why it matters in industry**
    - Every deep model (CNNs, RNNs, Transformers, LLMs) is this same machinery scaled
      and specialized; the MLP is the atom.
    - "Deep learning = representation learning" is the paradigm shift from hand-crafted
      features (Lesson MLE-03) to learned ones.
    - Understanding the forward pass + gradients from scratch is what lets you debug
      training failures that a framework hides.

    **Typical interview questions**
    - "Why does a neural network need nonlinear activations?"
    - "What is the XOR problem and why did it matter historically?"
    - "Explain universal approximation — does it mean NNs always work?"
    - "Sigmoid vs tanh vs ReLU — tradeoffs?"
    - "Walk me through the forward pass and loss of an MLP."
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **The perceptron (Rosenblatt, 1958).** The first trainable neuron: a weighted sum
    passed through a step function — essentially Lesson CML-02's logistic unit with a
    hard threshold. It learned linearly separable patterns and sparked huge optimism
    about "thinking machines."

    **The XOR winter (Minsky & Papert, 1969).** Minsky and Papert proved a single
    perceptron **cannot** learn XOR — a trivially simple function that isn't linearly
    separable (no single line separates the classes). This devastating result (a single
    linear unit can only draw a line, Lesson CML-02) helped trigger the first "AI winter":
    funding and interest collapsed for over a decade.

    **The fix was always there: hidden layers.** Stacking a *hidden* layer of neurons
    between input and output lets the network first transform the data into a new space
    where the classes *are* linearly separable — then a final linear unit finishes the
    job. The missing piece was *how to train* such a network, solved by the
    rediscovery and popularization of **backpropagation** (Rumelhart, Hinton &
    Williams, 1986 — our Lesson DL-03).

    **Universal approximation (Cybenko 1989, Hornik 1991).** It was then proven that a
    network with a single hidden layer and enough units can approximate *any*
    continuous function to arbitrary accuracy. Neural networks are, in principle,
    universal function approximators — which is *why* they can fit the nonlinear
    boundaries that defeated linear models and the staircases that limited trees.

    **Why over linear models / trees (the Phase-1 contrast).** Linear models need you to
    *engineer* nonlinearity (Lesson MLE-03); trees carve axis-aligned boxes (Lesson CML-03).
    A neural network **learns a smooth, nonlinear representation directly from data** —
    no feature engineering, no axis alignment. The price (which the rest of Section 04
    addresses): they need more data, more compute, careful optimization, and they're
    far less interpretable.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **A neuron is a logistic regression.** Recall Lesson CML-02: compute a weighted sum
    $z=\mathbf w^\top\mathbf x+b$, squash it through a nonlinearity $a=g(z)$. That's one
    neuron. A **layer** is many neurons sharing the same inputs (a matrix multiply,
    Lesson FND-01). A **network** stacks layers: each layer's outputs feed the next.

    **Why the nonlinearity is non-negotiable.** If $g$ were the identity (no
    nonlinearity), then $W_2(W_1\mathbf x)=(W_2W_1)\mathbf x$ — a product of matrices is
    just *another* matrix. A hundred linear layers collapse into a single linear layer:
    no more expressive than logistic regression. The **activation** breaks this
    collapse; it's the source of all the network's power to bend space.

    **The key idea: hidden layers learn a new representation.** Take XOR — four points
    that no line can separate. A hidden layer *folds and stretches* the input space so
    that, in the **hidden-layer coordinates**, the four points *become* linearly
    separable, and the output neuron (a plain logistic unit) finishes easily. "Deep
    learning is representation learning": the early layers learn features so the last
    layer's job is simple.

    **Universal approximation, intuitively.** Each hidden unit contributes a soft
    "bump"/"ridge" (a sigmoid step or ReLU hinge). Add enough of them with the right
    weights and you can paint any shape — like building a curve out of many small tiles.

    ```mermaid
    flowchart LR
        X["input x"] --> H1["hidden layer<br/>z1 = xW1+b1, a1 = g(z1)"]
        H1 --> H2["(more layers...)"]
        H2 --> O["output<br/>z = aW+b, p = softmax/sigmoid"]
        O --> L["loss (cross-entropy, FND-02)"]
        L -.->|"gradients (DL-03)"| H1
    ```

    Run the cells — first, watch a linear model fail XOR and an MLP solve it.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    def sigmoid(z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    print("NumPy", np.__version__)
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 A layer is an affine map + nonlinearity
    For inputs $X\in\mathbb R^{n\times d}$, a layer with $h$ units has weights
    $W\in\mathbb R^{d\times h}$ and bias $\mathbf b\in\mathbb R^h$:
    $$Z = XW + \mathbf b,\qquad A = g(Z),$$
    where $g$ is applied elementwise. The matrix multiply (Lesson FND-01) computes all
    neurons for all examples at once.

    ### 4.2 A 2-layer MLP (one hidden layer)
    $$Z_1 = XW_1+\mathbf b_1,\quad A_1=g(Z_1),\quad
    Z_2 = A_1W_2+\mathbf b_2,\quad \hat y=\sigma(Z_2).$$
    Hidden activation $g$ (tanh/ReLU) provides nonlinearity; the output $\sigma$
    (sigmoid for binary, softmax for multiclass) produces probabilities. This is
    *exactly* logistic regression (Lesson CML-02) applied to the **learned features**
    $A_1$ instead of the raw inputs.

    ### 4.3 Why nonlinearity is essential (the collapse argument)
    With $g=\text{identity}$: $\hat y = (A_1)W_2 = (XW_1)W_2 = X(W_1W_2)=XW'$. The whole
    network is a single linear map $W'$ — no gain over one layer. A nonlinear $g$ makes
    the composition strictly more expressive; this is the mathematical reason hidden
    layers help.

    ### 4.4 Activation functions and their derivatives
    | Activation | $g(z)$ | $g'(z)$ | Range | Notes |
    |---|---|---|---|---|
    | **Sigmoid** | $\frac{1}{1+e^{-z}}$ | $g(1-g)$ | $(0,1)$ | Saturates → vanishing gradients |
    | **Tanh** | $\tanh z$ | $1-\tanh^2 z$ | $(-1,1)$ | Zero-centered; still saturates |
    | **ReLU** | $\max(0,z)$ | $\mathbb 1[z>0]$ | $[0,\infty)$ | No saturation for $z>0$; can "die" |

    ReLU's constant gradient for positive inputs is *why* it enabled training very deep
    networks (no vanishing on the active side) — the default for hidden layers.

    ### 4.5 Loss (Lesson FND-02)
    Binary: cross-entropy $J=-\frac1n\sum[y\log\hat y+(1-y)\log(1-\hat y)]$ (Bernoulli
    NLL). Multiclass: softmax + categorical cross-entropy. Regression: MSE (Gaussian
    NLL). Same loss-design logic as Section 02 — choosing the loss = choosing the noise
    model.

    ### 4.6 Gradients for our 2-layer net (chain rule, by hand)
    With binary cross-entropy + sigmoid output, the output error simplifies beautifully
    (as in Lesson CML-02): $\;dZ_2=\hat y - y$. Then propagate backward through the layer:
    $$dW_2 = \tfrac1n A_1^\top dZ_2,\quad dZ_1 = (dZ_2\,W_2^\top)\odot g'(Z_1),\quad
    dW_1 = \tfrac1n X^\top dZ_1.$$
    This *is* backpropagation for two layers — repeated application of the chain rule,
    reusing each layer's stored activations. **Lesson DL-03 turns this hand-derivation
    into the general algorithm** for arbitrary depth. Here we code these exact formulas.

    ### 4.7 Universal approximation (statement)
    *A feed-forward network with a single hidden layer containing finitely many neurons
    and a non-polynomial activation can approximate any continuous function on a compact
    set to arbitrary accuracy.* Caveat: it guarantees *existence* of weights, not that
    gradient descent will *find* them, nor that the width needed is practical — depth
    often achieves the same with exponentially fewer units.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    A complete one-hidden-layer MLP in NumPy: forward pass, the hand-derived gradients
    from §4.6, and a gradient-descent training loop (Lesson FND-04). We first prove the
    *motivation* — a linear model cannot learn XOR — then watch the MLP succeed.
    """),

    code(r"""
    # 5.1 The MLP: forward pass + manual backward (the §4.6 chain-rule formulas).
    class MLP:
        def __init__(self, n_in, n_hidden, seed=0, activation="tanh"):
            r = np.random.default_rng(seed)
            # Xavier/He-style init: scale by fan-in to keep activations well-scaled (FND-04 conditioning)
            self.W1 = r.normal(0, 1, (n_in, n_hidden)) * np.sqrt(2.0 / n_in)
            self.b1 = np.zeros(n_hidden)
            self.W2 = r.normal(0, 1, (n_hidden, 1)) * np.sqrt(1.0 / n_hidden)
            self.b2 = np.zeros(1)
            self.act = activation

        def _g(self, z):
            return np.tanh(z) if self.act == "tanh" else np.maximum(0, z)

        def _gprime(self, z, a):
            return (1 - a ** 2) if self.act == "tanh" else (z > 0).astype(float)

        def forward(self, X):
            self.X = X
            self.z1 = X @ self.W1 + self.b1
            self.a1 = self._g(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
            self.p = sigmoid(self.z2)
            return self.p

        def backward(self, y):
            n = len(self.X); y = y.reshape(-1, 1)
            dz2 = (self.p - y) / n                       # BCE + sigmoid -> (p - y)
            dW2 = self.a1.T @ dz2
            db2 = dz2.sum(0)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * self._gprime(self.z1, self.a1)   # chain rule through activation
            dW1 = self.X.T @ dz1
            db1 = dz1.sum(0)
            return dW1, db1, dW2, db2

        def step(self, grads, lr):
            dW1, db1, dW2, db2 = grads
            self.W1 -= lr * dW1; self.b1 -= lr * db1
            self.W2 -= lr * dW2; self.b2 -= lr * db2

    def bce(y, p, eps=1e-12):
        p = np.clip(p, eps, 1 - eps); y = y.reshape(-1, 1)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def train(model, X, y, lr=0.5, epochs=3000):
        hist = []
        for _ in range(epochs):
            model.forward(X)
            model.step(model.backward(y), lr)
            hist.append(bce(y, model.p))
        return hist
    """),

    code(r"""
    # 5.2 Motivation: a LINEAR model cannot learn XOR; a 2-layer MLP can.
    # noisy XOR: 4 clusters, class = (x>0) XOR (y>0)
    def make_xor(n=400, seed=0):
        r = np.random.default_rng(seed)
        X = r.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
        X += r.normal(0, 0.08, X.shape)
        return X, y

    Xx, yx = make_xor()
    from sklearn.linear_model import LogisticRegression
    lin = LogisticRegression().fit(Xx, yx)
    mlp = MLP(2, 8, seed=1, activation="tanh")
    _ = train(mlp, Xx, yx, lr=0.5, epochs=4000)

    lin_acc = (lin.predict(Xx) == yx).mean()
    mlp_acc = ((mlp.forward(Xx) > 0.5).ravel() == yx).mean()
    print(f"linear model XOR accuracy : {lin_acc:.2f}  (~chance: no line separates XOR)")
    print(f"2-layer MLP XOR accuracy  : {mlp_acc:.2f}  (hidden layer solves it)")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Five pictures: the XOR boundaries, the activation functions and their gradients,
    the MLP learning a nonlinear boundary, the universal-approximation width effect,
    and — the key insight — the **learned representation** that makes classes
    separable.
    """),

    code(r"""
    # Figure 1 — XOR: linear boundary fails, MLP boundary succeeds.
    def plot_boundary(ax, predict_fn, X, y, title):
        xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 200), np.linspace(-1.3, 1.3, 200))
        Z = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=12)
        ax.set_title(title); ax.set_aspect("equal")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_boundary(axes[0], lambda G: lin.predict(G), Xx, yx, "Linear model: cannot split XOR")
    plot_boundary(axes[1], lambda G: (mlp.forward(G) > 0.5).astype(int).ravel(), Xx, yx,
                  "2-layer MLP: bends space to solve XOR")
    plt.suptitle("Figure 1 — One hidden layer defeats the problem that ended the perceptron era")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** A linear model (left) can only draw one straight line, so on XOR it's
    stuck near chance — the 1969 result that triggered the AI winter. The MLP (right)
    uses its hidden layer to carve a nonlinear boundary that isolates both diagonal
    clusters. Same data, same logistic *output* unit — the difference is the hidden
    layer transforming the inputs first. This one picture is the entire reason deep
    learning exists.
    """),

    code(r"""
    # Figure 2 — activation functions and their derivatives (the vanishing-gradient story).
    z = np.linspace(-5, 5, 300)
    acts = {"sigmoid": (sigmoid(z), sigmoid(z) * (1 - sigmoid(z))),
            "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2),
            "ReLU": (np.maximum(0, z), (z > 0).astype(float))}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (a, d) in acts.items():
        axes[0].plot(z, a, label=name)
        axes[1].plot(z, d, label=f"{name}'")
    axes[0].set_title("Activations g(z)"); axes[0].legend()
    axes[1].set_title("Derivatives g'(z) -- note sigmoid/tanh -> 0 at the extremes")
    axes[1].legend()
    plt.suptitle("Figure 2 — Activations and gradients: why ReLU avoids vanishing gradients")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** The activation injects the nonlinearity that prevents layer collapse
    (§4.3). Their **derivatives** matter for training (Lesson DL-03): sigmoid and tanh
    **saturate** — their gradient $\to 0$ for large $|z|$, so a saturated neuron stops
    learning and, stacked deep, gradients **vanish**. **ReLU**'s derivative is a
    constant 1 for $z>0$, so it doesn't saturate on the active side — the key reason it
    enabled training deep networks. Its weakness: units with $z<0$ have zero gradient
    and can "die" (§7).
    """),

    code(r"""
    # Figure 3 — train an MLP on 'moons' (nonlinear) and watch loss fall + boundary form.
    from sklearn.datasets import make_moons
    Xm, ym = make_moons(n_samples=400, noise=0.2, random_state=0)
    Xm = (Xm - Xm.mean(0)) / Xm.std(0)               # standardize (FND-04/MLE-03)
    net = MLP(2, 16, seed=2, activation="tanh")
    hist = train(net, Xm, ym.astype(float), lr=0.3, epochs=4000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(hist); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Training loss")
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
    P = net.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[1].contourf(xx, yy, P, levels=20, cmap="RdBu", alpha=0.7)
    axes[1].contour(xx, yy, P, levels=[0.5], colors="k")
    axes[1].scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="RdBu", edgecolor="k", s=12)
    acc = ((net.forward(Xm) > 0.5).ravel() == ym).mean()
    axes[1].set_title(f"Learned boundary (acc {acc:.2f})"); axes[1].set_aspect("equal")
    plt.suptitle("Figure 3 — A from-scratch MLP fits a curved boundary")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** The loss falls smoothly as gradient descent (Lesson FND-04) updates the
    hand-derived gradients (§4.6), and the network carves a **smooth, curved** decision
    boundary around the interleaving moons — something neither a linear model nor a
    shallow tree manages cleanly. Note we **standardized** the inputs first: like every
    gradient-based model (Lessons FND-04, CML-01, and CML-02), NNs train far better on well-scaled data.
    """),

    code(r"""
    # Figure 4 — universal approximation: more hidden units -> more boundary complexity.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, h in zip(axes, [2, 8, 64]):
        net_h = MLP(2, h, seed=3, activation="tanh")
        train(net_h, Xm, ym.astype(float), lr=0.3, epochs=3000)
        xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 150), np.linspace(-2.5, 2.5, 150))
        P = net_h.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, P, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.4)
        ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="RdBu", edgecolor="k", s=8)
        acc = ((net_h.forward(Xm) > 0.5).ravel() == ym).mean()
        ax.set_title(f"{h} hidden units (acc {acc:.2f})"); ax.set_aspect("equal")
    plt.suptitle("Figure 4 — Width and expressiveness (universal approximation in action)")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** With only **2 hidden units** the network can bend space a little —
    underfit. **8 units** capture the moons well. **64 units** can express very wiggly
    boundaries (and would overfit on noisier/smaller data — the bias–variance tradeoff
    of Lesson CML-01 returns, now controlled by width/depth and regularization). This is
    universal approximation made visible: more units → more "tiles" to paint the
    function — but expressiveness must be matched to data size, or you overfit.
    """),

    code(r"""
    # Figure 5 — THE key idea: the hidden layer learns a representation where classes
    # become linearly separable. Train with 2 hidden ReLU units and plot the hidden space.
    rep = MLP(2, 2, seed=5, activation="relu")
    train(rep, Xx, yx, lr=0.4, epochs=6000)            # back to XOR (clearest demo)
    H = rep._g(Xx @ rep.W1 + rep.b1)                   # hidden activations (n x 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(Xx[:, 0], Xx[:, 1], c=yx, cmap="RdBu", edgecolor="k", s=15)
    axes[0].set_title("Input space: XOR is NOT linearly separable"); axes[0].set_aspect("equal")
    axes[1].scatter(H[:, 0], H[:, 1], c=yx, cmap="RdBu", edgecolor="k", s=15)
    axes[1].set_title("Hidden-layer space: classes ARE linearly separable")
    axes[1].set_xlabel("hidden unit 1"); axes[1].set_ylabel("hidden unit 2")
    plt.suptitle("Figure 5 — Deep learning = representation learning")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 5.** This is the most important figure in the notebook. In the **input
    space** (left) the two XOR classes are tangled — no line separates them. But the
    network's **hidden layer** transforms the points into a new 2D space (right) where
    the classes fall on opposite sides of a line — now the output neuron (a plain
    logistic unit) separates them trivially. The network *learned features* that make
    the problem easy. This is what "deep learning is representation learning" means, and
    it's the principle behind CNNs learning edges→shapes (Lesson DL-05) and Transformers
    learning contextual meaning (Lesson DL-08).
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **No nonlinearity** | Network no better than linear | Identity activation → layer collapse (§4.3) | Use a nonlinear activation |
    | **Symmetric init (all zeros)** | All hidden units identical; can't learn | Broken symmetry never broken | Random init (Xavier/He) |
    | **Vanishing gradients** | Deep net's early layers don't learn | Sigmoid/tanh saturate, $g'\to0$ (Fig 2) | ReLU; normalization; residuals (DL-05) |
    | **Dead ReLUs** | Many units stuck at 0 output | Large negative pre-activations → zero gradient | Lower LR; LeakyReLU; better init |
    | **Unscaled inputs** | Slow/unstable training | Ill-conditioned loss (FND-04) | Standardize features |
    | **Bad learning rate** | Diverges or crawls | Step too big/small (FND-04) | Tune LR; schedules; Adam |
    | **Overfitting** | Train ≫ test | Too many params for the data | Regularization, dropout, more data, early stop |
    | **Non-convexity** | Different runs, different solutions | Loss surface has many minima/saddles (FND-04) | Good init, SGD noise, multiple runs |

    The cell demonstrates the **zero-init symmetry** failure — a classic interview gotcha.
    """),

    code(r"""
    # Zero initialization breaks learning: all hidden units stay identical (no symmetry breaking).
    net_zero = MLP(2, 8, seed=0, activation="tanh")
    net_zero.W1[:] = 0.0; net_zero.W2[:] = 0.0          # symmetric init
    h_zero = train(net_zero, Xm, ym.astype(float), lr=0.3, epochs=2000)

    net_rand = MLP(2, 8, seed=0, activation="tanh")     # proper random init
    h_rand = train(net_rand, Xm, ym.astype(float), lr=0.3, epochs=2000)

    print(f"final loss, ZERO init   : {h_zero[-1]:.4f}  (stuck -- all units identical)")
    print(f"final loss, RANDOM init : {h_rand[-1]:.4f}  (learns)")
    print("With zero weights every hidden unit computes the same thing and receives the")
    print("same gradient forever -- they never differentiate. Random init breaks the symmetry.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    In practice you never hand-derive gradients — **autograd** (PyTorch) builds the
    computation graph and computes them for you (the general backprop of Lesson DL-03),
    runs on GPU, and gives optimizers, layers, and data loaders. Below, the *same*
    MLP in PyTorch; we verify it reaches comparable accuracy. The import is wrapped so
    the notebook runs even without torch.
    """),

    code(r"""
    # Same 2-layer MLP in PyTorch (autograd replaces our manual backward). Guarded import.
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        Xt = torch.tensor(Xm, dtype=torch.float32)
        yt = torch.tensor(ym, dtype=torch.float32).view(-1, 1)
        model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid())
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        lossf = nn.BCELoss()
        for _ in range(2000):
            opt.zero_grad()
            loss = lossf(model(Xt), yt)
            loss.backward()                              # autograd computes ALL gradients
            opt.step()
        acc = ((model(Xt).detach().numpy() > 0.5).ravel() == ym).mean()
        print(f"PyTorch MLP accuracy: {acc:.3f}  (autograd + Adam; no manual gradients)")
        print(f"our scratch MLP accuracy: {((net.forward(Xm) > 0.5).ravel() == ym).mean():.3f}")
    except Exception as e:
        print(f"[torch not available: {type(e).__name__}] "
              f"scratch NumPy MLP above already demonstrates the full mechanism.")
    """),

    md(r"""
    **Scratch vs production.** Our NumPy MLP and the PyTorch version implement the same
    math; the framework's value-add is enormous at scale: **autograd** computes
    gradients for *any* architecture automatically (so you never hand-derive §4.6
    again), plus GPU acceleration, optimizers (Adam, Lesson FND-04), layers, batching,
    and mixed precision. But the framework hides exactly the machinery we just built —
    which is why understanding the forward pass and gradients from scratch is what lets
    you diagnose vanishing gradients, dead ReLUs, and bad initialization when training
    misbehaves. Lesson DL-03 derives the general autograd/backprop algorithm.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — When (and When Not) to Use a Neural Net

    **Scenario.** A team must choose a model for a customer-facing prediction problem.
    The senior question isn't "can a neural net do it?" (universal approximation says
    yes) but "**should** we?"

    **Where neural nets win:**
    - **Unstructured signals** — images, audio, text, raw sensor streams — where
      features must be *learned*, not engineered (Sections 04–05). Here NNs dominate
      decisively.
    - **Huge datasets with complex interactions** and a need for representation transfer
      (pretraining/fine-tuning, Lesson NLP-03).

    **Where they usually lose (the honest senior take):**
    - **Plain tabular data** — gradient boosting (Lesson CML-05) typically matches or beats
      a neural net with far less tuning, less data, no scaling, and better
      interpretability (Lesson MLE-05). Reaching for deep learning on a 50-column CSV is a
      common junior mistake.

    **Business objectives:** maximize accuracy *per unit of engineering cost,
    latency, and interpretability*, not accuracy in a vacuum.

    **Cost of mistakes:** choosing a neural net for tabular data → months of tuning,
    GPU bills, an opaque model, and often *worse* accuracy than an afternoon with
    XGBoost. Choosing a tree for images → it simply can't compete.

    **Constraints:** data modality and volume, latency/compute budget, interpretability
    and regulatory needs (Lesson MLE-05), and team expertise.

    **KPIs:** accuracy vs a strong GBM baseline, training/inference cost, time-to-ship,
    and explainability — decided *before* committing to deep learning.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Initialization & scaling.** Use Xavier/He init and **standardize inputs** —
      both directly affect whether the net trains at all (§7, Lesson FND-04).
    - **Compute & cost.** NNs are matrix-multiply heavy → GPUs/accelerators; training
      and serving cost far exceed classical models. Budget accordingly.
    - **Latency.** Inference is a few matrix multiplies — fast on GPU, but heavier than
      a linear model or small tree; quantize/distill for tight budgets.
    - **Overfitting controls** are first-class in production NNs: dropout, weight decay
      (L2, Lesson CML-01), early stopping, data augmentation, more data.
    - **Reproducibility.** Seed everything; non-convexity (Lesson FND-04) means runs
      differ — pin seeds and checkpoint.
    - **Interpretability.** NNs are opaque; use SHAP / gradient attributions
      (Lesson MLE-05) and monitor behavior, especially in regulated settings.
    - **Don't over-reach.** For tabular problems, baseline against gradient boosting
      first (§9); only escalate to deep learning when the data modality demands it.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Neural network vs the Phase-1 models:**

    | Dimension | Linear/Logistic | Gradient Boosting | Neural Network |
    |---|---|---|---|
    | Nonlinearity | Manual features | **Automatic** | **Automatic (learned)** |
    | Unstructured data (image/text/audio) | No | No | **Yes (the whole point)** |
    | Tabular accuracy | Lower | **Usually best** | Competitive only with effort |
    | Data needed | Low | Moderate | **High** |
    | Preprocessing | Scaling | Minimal | Scaling + careful init |
    | Training cost | Low | Low–moderate | **High (GPU)** |
    | Interpretability | **High** | Low (SHAP) | **Lowest** |
    | Tuning burden | Low | Moderate | **High** |

    **Activation functions:**

    | Activation | Pros | Cons | Use for |
    |---|---|---|---|
    | Sigmoid | Probabilistic output | Saturates, vanishing grad, not zero-centered | **Output** (binary) only |
    | Tanh | Zero-centered | Saturates | Sometimes hidden (RNNs, DL-06) |
    | ReLU | No positive-side saturation, cheap | Dead units, not zero-centered | **Default hidden** |
    | (Leaky/GELU/…) | Fix dead ReLU / smoother | Slightly more compute | Modern deep nets, Transformers |

    **Width vs depth:** wider layers add capacity linearly; **depth** composes features
    hierarchically and is often exponentially more parameter-efficient for structured
    problems — the rationale for "deep" learning (DL-05 and DL-08).

    **Senior lesson:** universal approximation guarantees a net *can* fit anything, but
    *should* you? Match the model to the **data modality** and the **engineering budget**
    — neural nets for learned representations on unstructured data, boosting for tabular.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Why do NNs need nonlinear activations?* → Without them, stacked linear layers
      collapse to one linear map (§4.3); no gain over logistic regression.
    - *What is the XOR problem?* → A non-linearly-separable function a single perceptron
      can't learn; a hidden layer solves it by re-representing the inputs (Figs 1, 5).

    **Deep-dive questions**
    - *Explain universal approximation and its caveat.* → Enough hidden units approximate
      any continuous function; but existence ≠ trainability, and width may be impractical
      (§4.7).
    - *Sigmoid vs tanh vs ReLU?* → Saturation/vanishing vs ReLU's constant positive
      gradient and dead-unit risk (Fig 2, §11).
    - *Walk through an MLP forward pass and its gradients.* → §4.2, §4.6; the output
      error is $(\hat y - y)$ as in logistic regression.

    **Whiteboard questions**
    - "Implement a 2-layer MLP forward pass and the gradient for $W_2$." (Section 5.1.)
    - "Why does zero-initialization fail?" (Symmetry; §7 demo.)

    **Strong vs weak answers**
    - *"Should we use a neural net for this 40-column tabular dataset?"*
      - **Weak:** "Yes, neural nets are state of the art."
      - **Strong:** "Probably not — gradient boosting usually beats NNs on tabular data
        with less data, tuning, and cost, and it's more interpretable. I'd baseline with
        XGBoost; reserve deep learning for unstructured signals where features must be
        learned."
    - *"Your deep net isn't learning."*
      - **Weak:** "Train longer."
      - **Strong:** "I'd check initialization (not zeros), input scaling, learning rate,
        and activation saturation/dead ReLUs — vanishing gradients are the classic
        culprit, addressed by ReLU, normalization, and good init."

    **Follow-ups:** "Width vs depth?" (capacity vs hierarchical efficiency). "Why ReLU
    over sigmoid in hidden layers?" (no positive-side saturation). "How prevent
    overfitting?" (dropout, weight decay, early stopping, data).

    **Common mistakes:** forgetting nonlinearity is what gives power; zero-initializing
    weights; using sigmoid in deep hidden layers; not scaling inputs; assuming universal
    approximation means NNs always win; defaulting to deep learning for tabular data.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Describe an MLP as stacked logistic units with nonlinearities.
    2. **Why was it invented?** What did hidden layers solve (XOR) that perceptrons
       couldn't?
    3. **How does it work?** Walk the forward pass: affine → activation → … → output →
       loss.
    4. **Why does it work?** Why is the nonlinearity essential, and what does
       universal approximation guarantee (and not)?
    5. **When to use it?** Which data modalities favor NNs over boosting?
    6. **When NOT to use it?** Why is a neural net often the wrong call for tabular data?
    7. **Tradeoffs?** Activation choices; width vs depth; NN vs GBM.
    8. **How would you productionize it?** Init, scaling, regularization, compute,
       interpretability, and the baseline-first discipline.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Prove that a 2-layer network with identity activations is equivalent to a single
       linear layer.
    2. Explain in two sentences why zero-initializing all weights prevents learning.

    **Beginner → Intermediate (coding)**
    3. Swap the hidden activation to **ReLU** in the scratch MLP and compare convergence
       and the learned boundary on moons.
    4. Add **L2 weight decay** to the training loop and show it smooths the degree-64
       boundary (Fig 4) and reduces overfitting.

    **Intermediate (analysis)**
    5. Extend the MLP to **multiclass softmax + categorical cross-entropy** and train it
       on a 3-class spiral dataset; visualize the boundary.
    6. Reproduce the **vanishing-gradient** problem: build a deep (6-layer) sigmoid
       network and measure how gradient magnitude shrinks toward the input layers;
       show ReLU mitigates it.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive the gradients for the 2-layer net (§4.6) from the chain
       rule, then explain how this generalizes to arbitrary depth (preview of DL-03).
    8. *Design:* you're given a 60-feature tabular dataset and a folder of product
       images. Decide which model family fits each, justify with cost/accuracy/
       interpretability tradeoffs, and specify baselines.
    9. *Debug:* a teammate's MLP loss is flat from epoch 0. List the four most likely
       causes (zero init, dead ReLUs, LR, unscaled inputs) and the check for each.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    A neural network is **logistic regression stacked in layers** (Lesson CML-02) with a
    **nonlinearity** between them — and that nonlinearity is everything: it prevents the
    layers from collapsing into one and lets the network **learn a representation** in
    which the problem becomes easy (Fig 5). Hidden layers solved the XOR problem that
    ended the perceptron era, and **universal approximation** says enough units can fit
    any function. We built the whole thing in NumPy — forward pass (matrix multiplies,
    Lesson FND-01), cross-entropy loss (Lesson FND-02), and hand-derived gradients trained
    by gradient descent (Lesson FND-04).

    **Related lesson:** `DL-03 · Backpropagation` — we generalize the hand-derived 2-layer gradients
    into *the* algorithm of deep learning: the chain rule applied systematically over a
    computation graph (reverse-mode autodiff), which trains networks of *any* depth and
    is exactly what PyTorch's `loss.backward()` does.
    """),
]

build("04_deep_learning/02_neural_networks_from_scratch.ipynb", cells)
