"""Builder for Lesson DL-06 — RNNs and LSTMs.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # DL-06 · Recurrent Neural Networks and LSTMs
    ### Section 04 — Deep Learning Foundations · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** DL-04 and NLP-01. You should understand stable neural
    training, tokens, vocabularies, sparse text baselines, and word embeddings.

    > CNNs (Lesson DL-05) exploited *spatial* structure by sharing weights across
    > **space**. Sequences — text, time series, audio, sensor streams — have a
    > different structure: **order and memory**. A recurrent neural network shares
    > weights across **time**: it reads one element at a time, maintaining a hidden
    > **state** that carries information forward. This is the first architecture that
    > can process variable-length sequences and remember the past. But training it
    > runs straight into the **vanishing-gradient** problem from Lesson DL-03, now
    > stretched across time — which is exactly what the **LSTM**'s gated memory was
    > invented to solve. RNNs/LSTMs dominated NLP and forecasting for two decades and
    > are the conceptual bridge to **attention** (Lesson DL-07) and **Transformers**
    > (Lesson DL-08), which replaced them by removing the sequential bottleneck.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - Why dense nets and CNNs are poor fits for sequences, and how an RNN's **hidden
      state recurrence** shares weights across **time**.
    - **Backpropagation through time (BPTT)** and why it causes **vanishing/exploding
      gradients** over long sequences (the DL-03 problem, in time).
    - The **LSTM**: cell state, **forget/input/output gates**, and the *constant error
      carousel* that lets gradients flow across many steps; **GRU** as a lighter
      variant.
    - A vanilla RNN cell + BPTT and an LSTM cell **from scratch** in NumPy.
    - The **sequential bottleneck** that motivates attention/Transformers.

    **Why it matters in industry**
    - RNNs/LSTMs power(ed) time-series forecasting, anomaly detection, speech, and
      pre-Transformer NLP; they remain strong for **streaming/low-latency** and
      **small-data** sequence tasks.
    - Understanding why they fail on long dependencies — and why Transformers replaced
      them — is essential context for modern LLMs.

    **Typical interview questions**
    - "How does an RNN process a sequence, and what does the hidden state represent?"
    - "What is BPTT and why do RNNs suffer from vanishing gradients?"
    - "How does an LSTM solve the long-term dependency problem?"
    - "RNN vs LSTM vs GRU — tradeoffs?"
    - "Why did Transformers replace RNNs?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **Why feed-forward nets can't handle sequences.** An MLP (Lesson DL-02) expects a
    fixed-size input and treats positions independently; a CNN (Lesson DL-05) captures
    *local* patterns but has a fixed receptive field and no notion of an evolving
    *state*. Sequences are **variable length** and **order-dependent**, and often
    require **memory** of arbitrarily distant past events ("the verb agrees with a
    subject 20 words back").

    **The RNN (Elman, 1990; Jordan, 1986).** The idea: process the sequence one step at
    a time, and feed the network's own hidden state back in as an input at the next
    step. The **same weights** are reused at every time step (weight sharing across
    *time*, the temporal analogue of the CNN's spatial sharing), so the model handles
    any length and can, in principle, carry information forward indefinitely.

    **The long-term dependency problem (Bengio et al., 1994).** In practice vanilla RNNs
    couldn't learn long-range dependencies. Training by **backpropagation through time**
    multiplies the same Jacobian once per step; over many steps the gradient either
    **vanishes** (→ early steps are forgotten) or **explodes** (→ NaNs) — the
    DL-03 phenomenon, amplified by depth-in-time.

    **The LSTM (Hochreiter & Schmidhuber, 1997).** The fix: add a separate **cell
    state** with a near-linear path through time, controlled by **gates** that decide
    what to keep, add, and output. The cell state's gradient is modulated by the forget
    gate (often ≈1), creating a *constant error carousel* that lets gradients survive
    across hundreds of steps. The **GRU** (Cho, 2014) simplified this to two gates.

    **The Transformer disruption (2017).** RNNs/LSTMs are inherently **sequential** —
    step $t$ needs step $t{-}1$ — so they can't parallelize across time and still
    struggle with very long contexts. **Attention** (Lesson DL-07) lets every position
    directly access every other in parallel, and **Transformers** (Lesson DL-08) built
    on it now dominate NLP. Understanding RNNs/LSTMs is how you understand *what
    problem* attention solved.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Reading with a running memory.** Imagine reading a sentence word by word, keeping
    a mental summary that you update with each word. That running summary is the
    **hidden state** $h_t$; the update rule (the same at every word) is the RNN cell.
    The final state is a fixed-size summary of the whole sequence — usable for
    classification, or you can emit an output at each step (tagging, generation).

    **Weight sharing across time.** Just as a CNN slides *one* filter across all
    positions, an RNN applies *one* cell (one set of weights) at all time steps. This is
    why it handles variable length and why "unrolling" the RNN over $T$ steps looks like
    a very deep network with **tied** weights — and that depth-in-time is the source of
    the gradient problem.

    **The vanishing-memory problem.** Each step, the old state is squashed through a
    nonlinearity and mixed with new input. Information from step 1 has to survive being
    re-processed $T$ times to influence step $T$ — and it usually decays away (the
    gradient signal back to step 1 vanishes). So a vanilla RNN effectively has a **short
    memory**.

    **The LSTM's gated memory (the key idea).** Add a **cell state** $c_t$ that acts
    like a conveyor belt running straight through time with only minor, *gated*
    edits — multiply by a **forget gate** (keep/erase), add via an **input gate**
    (write), and read out through an **output gate**. Because the belt is near-linear
    (mostly addition, not repeated squashing), gradients flow along it largely intact —
    so the LSTM can remember things from far back.

    ```mermaid
    flowchart LR
        x1["x_1"] --> c1["cell @t1"]
        h0["h_0"] --> c1
        c1 --> h1["h_1"]
        x2["x_2"] --> c2["cell @t2 (same weights)"]
        h1 --> c2
        c2 --> h2["h_2"]
        x3["x_3"] --> c3["...cell @t3..."]
        h2 --> c3
        c3 --> hT["h_T → output"]
    ```

    Run the cells — first an RNN cell, then a measurement of its vanishing memory.
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

    ### 4.1 The vanilla RNN recurrence
    At each step $t$, combine the new input $x_t$ with the previous state $h_{t-1}$:
    $$h_t = \tanh(x_t W_x + h_{t-1} W_h + b),\qquad h_0=\mathbf 0.$$
    The **same** $W_x, W_h, b$ are used at every step (weight sharing across time). For
    a many-to-one task, the output uses the last state: $\hat y=\sigma(h_T W_y + b_y)$.
    Many-to-many tasks emit $\hat y_t$ at each step.

    ### 4.2 Backpropagation through time (BPTT)
    "Unroll" the RNN into a $T$-step feed-forward graph with tied weights and apply
    backprop (Lesson DL-03). The gradient w.r.t. $h_t$ flows backward through the
    recurrence:
    $$\frac{\partial L}{\partial h_{t-1}} = \frac{\partial L}{\partial h_t}\,\frac{\partial h_t}{\partial h_{t-1}},
    \qquad \frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}(1-h_t^2)\,W_h^\top.$$
    Reaching step 1 from step $T$ multiplies $T-1$ such Jacobians.

    ### 4.3 Why gradients vanish/explode over time
    That product behaves like $(\text{something})^{T}$. If the relevant factor (tied to
    $\|W_h\|$ and $\tanh'\le1$) is $<1$, the gradient **vanishes** exponentially in $T$
    — step 1 receives essentially no learning signal, so the RNN can't learn long
    dependencies. If $>1$, it **explodes** (NaNs). Because the *same* $W_h$ is reused,
    this is far more severe than ordinary depth (Lesson DL-03). Exploding gradients are
    patched by **gradient clipping** (Lesson FND-04); vanishing needs an architectural
    fix — the LSTM.

    ### 4.4 The LSTM
    Maintain a **cell state** $c_t$ alongside the hidden state $h_t$, governed by gates
    (each a sigmoid in $[0,1]$ acting as a soft switch):
    $$f_t=\sigma(x_tW_f+h_{t-1}U_f+b_f)\ \text{(forget)},\quad
    i_t=\sigma(\cdots)\ \text{(input)},\quad o_t=\sigma(\cdots)\ \text{(output)},$$
    $$\tilde c_t=\tanh(x_tW_c+h_{t-1}U_c+b_c),\quad
    c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\quad h_t=o_t\odot\tanh(c_t).$$
    The cell update is **additive** (gated): $c_t=f_t\odot c_{t-1}+\dots$.

    ### 4.5 The constant error carousel (why LSTMs remember)
    The gradient along the cell-state path is
    $\;\partial c_t/\partial c_{t-1}=f_t$ (elementwise). So back-propagating across time
    multiplies **forget gates** rather than the same squashing Jacobian. When the model
    *wants* to remember ($f_t\approx1$), the product stays near 1 and the gradient flows
    **undiminished** across many steps — no vanishing. The gates *learn* what to keep,
    add, and read, so memory is selective. (GRU merges cell/hidden state and uses
    update + reset gates — fewer parameters, similar performance.)

    ### 4.6 The sequential bottleneck (toward Transformers)
    Even a perfect LSTM is **sequential**: $h_t$ depends on $h_{t-1}$, so the $T$ steps
    can't be computed in parallel — slow to train on long sequences and on modern
    hardware. And information still passes through a fixed-size state bottleneck. **Self-
    attention** (Lesson DL-07) removes both limits: every position attends to every other
    *directly and in parallel*. That is the leap from RNNs to Transformers.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    We implement a vanilla RNN with **full BPTT** and train it on a memory task
    ("output the first bit you saw" — pure long-term dependency), then implement an
    **LSTM cell** forward pass and show its cell-state gradient is preserved where the
    RNN's vanishes.
    """),

    code(r"""
    # 5.1 Vanilla RNN (many-to-one) with BPTT. Task: remember the FIRST element.
    def make_first_bit(n, T, seed=0):
        r = np.random.default_rng(seed)
        X = r.integers(0, 2, (n, T, 1)).astype(float)
        y = X[:, 0, :].copy()                          # label = the first bit -> needs memory of step 1
        return X, y

    class VanillaRNN:
        def __init__(self, d, h, seed=0):
            r = np.random.default_rng(seed)
            self.Wx = r.normal(0, 0.1, (d, h))
            self.Wh = r.normal(0, 0.1, (h, h))
            self.b = np.zeros(h)
            self.Wy = r.normal(0, 0.1, (h, 1))
            self.by = np.zeros(1)
            self.h = h

        def forward(self, X):
            n, T, d = X.shape
            self.X = X; self.H = [np.zeros((n, self.h))]
            for t in range(T):
                ht = np.tanh(X[:, t, :] @ self.Wx + self.H[-1] @ self.Wh + self.b)
                self.H.append(ht)
            self.logit = self.H[-1] @ self.Wy + self.by
            self.p = sigmoid(self.logit)
            return self.p

        def backward(self, y, record_dh=False):
            n, T, d = self.X.shape
            dlogit = (self.p - y) / n
            dWy = self.H[-1].T @ dlogit; dby = dlogit.sum(0)
            dh = dlogit @ self.Wy.T
            dWx = np.zeros_like(self.Wx); dWh = np.zeros_like(self.Wh); db = np.zeros_like(self.b)
            dh_norms = []
            for t in reversed(range(T)):
                dz = dh * (1 - self.H[t + 1] ** 2)     # tanh'
                dWx += self.X[:, t, :].T @ dz
                dWh += self.H[t].T @ dz
                db += dz.sum(0)
                dh = dz @ self.Wh.T                     # propagate to previous step
                dh_norms.append(np.linalg.norm(dh))
            self.grads = (dWx, dWh, db, dWy, dby)
            return dh_norms[::-1] if record_dh else None

        def step(self, lr):
            dWx, dWh, db, dWy, dby = self.grads
            for p, g in [(self.Wx, dWx), (self.Wh, dWh), (self.b, db),
                         (self.Wy, dWy), (self.by, dby)]:
                p -= lr * g

    def bce(y, p, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def train_rnn(T, epochs=400, lr=0.5, seed=0):
        X, y = make_first_bit(256, T, seed)
        net = VanillaRNN(1, 24, seed=1)
        for _ in range(epochs):
            net.forward(X); net.backward(y); net.step(lr)
        acc = ((net.forward(X) > 0.5) == y).mean()
        return net, acc

    for T in [3, 10, 25]:
        _, acc = train_rnn(T)
        print(f"vanilla RNN, sequence length {T:2d}: train accuracy {acc:.2f}")
    print("\\nMemory degrades as the gap between the first bit and the output grows.")
    """),

    code(r"""
    # 5.2 LSTM cell forward pass from scratch (the Section 4.4 equations).
    class LSTMCell:
        def __init__(self, d, h, seed=0):
            r = np.random.default_rng(seed)
            s = 0.1
            # gates: forget f, input i, output o, candidate g
            self.W = {k: r.normal(0, s, (d, h)) for k in "fiog"}
            self.U = {k: r.normal(0, s, (h, h)) for k in "fiog"}
            self.b = {k: np.zeros(h) for k in "fiog"}
            self.b["f"] += 1.0                          # forget-gate bias 1 -> remember by default
            self.h = h

        def step(self, x, h_prev, c_prev):
            f = sigmoid(x @ self.W["f"] + h_prev @ self.U["f"] + self.b["f"])
            i = sigmoid(x @ self.W["i"] + h_prev @ self.U["i"] + self.b["i"])
            o = sigmoid(x @ self.W["o"] + h_prev @ self.U["o"] + self.b["o"])
            g = np.tanh(x @ self.W["g"] + h_prev @ self.U["g"] + self.b["g"])
            c = f * c_prev + i * g                      # ADDITIVE, gated update
            h = o * np.tanh(c)
            return h, c, f

        def forward(self, X):
            n, T, d = X.shape
            h = np.zeros((n, self.h)); c = np.zeros((n, self.h)); forgets = []
            for t in range(T):
                h, c, f = self.step(X[:, t, :], h, c)
                forgets.append(f.mean())
            return h, c, forgets

    lstm = LSTMCell(1, 24)
    Xd, _ = make_first_bit(64, 20)
    h, c, forgets = lstm.forward(Xd)
    print(f"LSTM ran a length-20 sequence; mean forget-gate per step ~ {np.mean(forgets):.2f}")
    print("Forget gates near 1 keep the cell-state 'conveyor belt' open -> long memory.")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Three pictures: the vanishing gradient through time, RNN memory degrading with
    sequence length, and the LSTM cell-state path that preserves gradient (constant
    error carousel).
    """),

    code(r"""
    # Figure 1 — gradient vanishes as it propagates back through time in a vanilla RNN.
    X, y = make_first_bit(256, 30, 0)
    net = VanillaRNN(1, 24, seed=1)
    for _ in range(50):
        net.forward(X); net.backward(y); net.step(0.5)
    net.forward(X)
    dh_norms = net.backward(y, record_dh=True)          # ||dL/dh_t|| at each step

    fig, ax = plt.subplots()
    ax.plot(range(1, len(dh_norms) + 1), dh_norms, "o-")
    ax.set_yscale("log"); ax.set_xlabel("time step t (1 = earliest)")
    ax.set_ylabel("||gradient w.r.t. h_t|| (log)")
    ax.set_title("Figure 1 — BPTT gradient decays toward early steps (vanishing memory)")
    plt.show()
    """),

    md(r"""
    **Figure 1.** The gradient of the loss with respect to the hidden state shrinks
    geometrically as we propagate it back toward the **earliest** time steps (note the
    log scale). By step 1 it is orders of magnitude smaller than at the end — so the
    weights barely receive a signal about how to use early inputs. This is the
    **vanishing-gradient problem in time** (§4.3): the same $W_h$ Jacobian multiplied
    $T$ times. It's why a vanilla RNN effectively forgets the distant past, and why the
    "remember the first bit" task gets harder as the sequence lengthens.
    """),

    code(r"""
    # Figure 2 — vanilla RNN accuracy on the memory task degrades with sequence length.
    lengths = [3, 6, 10, 15, 20, 30]
    accs = [train_rnn(T, epochs=400)[1] for T in lengths]
    fig, ax = plt.subplots()
    ax.plot(lengths, accs, "o-", color="tab:red")
    ax.axhline(0.5, color="k", ls="--", label="chance")
    ax.set_xlabel("sequence length (gap to remember)"); ax.set_ylabel("train accuracy")
    ax.set_title("Figure 2 — RNN long-term memory collapses as the gap grows")
    ax.legend(); ax.set_ylim(0.4, 1.05)
    plt.show()
    print("Short gaps: solved. Long gaps: accuracy falls toward chance -- the LSTM fixes this.")
    """),

    md(r"""
    **Figure 2.** On the pure-memory task, the vanilla RNN solves short sequences but
    its accuracy decays toward chance as the gap between the informative first bit and
    the output grows — a direct consequence of the vanishing gradient in Figure 1. No
    amount of training fixes this; it's an *architectural* limitation. The LSTM's gated
    cell state is the architectural fix, and the production section shows it holding
    accuracy where the RNN fails.
    """),

    code(r"""
    # Figure 3 — the constant error carousel: cell-state gradient is a product of forget gates.
    # Compare the RNN's backward factor (decays) to the LSTM's (stays ~1 when gates are open).
    T = 40
    rnn_factor = np.cumprod(np.full(T, 0.7))            # illustrative |Wh*tanh'| < 1 per step
    open_gate = 0.95                                    # a trained 'remember' forget gate
    lstm_factor = np.cumprod(np.full(T, open_gate))     # product of forget gates
    fig, ax = plt.subplots()
    ax.plot(range(1, T + 1), rnn_factor, "o-", color="tab:red", label="RNN: prod of squashing Jacobians")
    ax.plot(range(1, T + 1), lstm_factor, "s-", color="tab:green",
            label=f"LSTM cell: prod of forget gates (~{open_gate})")
    ax.set_yscale("log"); ax.set_xlabel("steps back through time")
    ax.set_ylabel("gradient-preservation factor (log)")
    ax.set_title("Figure 3 — Why LSTMs remember: the cell-state gradient barely decays")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 3.** Back-propagating across $T$ steps multiplies a per-step factor. For the
    **vanilla RNN** that factor is the squashing Jacobian ($<1$), so the product decays
    exponentially (red). For the **LSTM cell state**, the factor is the **forget gate**
    $f_t$ (§4.5); when the network has learned to remember ($f_t\approx1$), the product
    stays near 1 (green) and the gradient survives across dozens of steps — the
    *constant error carousel*. The gates *learn* when to hold vs erase, giving
    selective, long-range memory. This single mechanism is the whole reason LSTMs
    displaced vanilla RNNs.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Vanishing gradients** | Can't learn long dependencies (Figs 1–2) | Repeated squashing Jacobian in BPTT | **LSTM/GRU**; skip connections; shorter sequences |
    | **Exploding gradients** | Loss → NaN; spikes | Repeated multiplication >1 | **Gradient clipping** (FND-04); careful init |
    | **No parallelism** | Slow training on long sequences | Inherently sequential ($h_t$ needs $h_{t-1}$) | Truncated BPTT; or switch to Transformers (DL-08) |
    | **Limited context even with LSTM** | Forgets very distant info | Fixed-size state bottleneck | Attention; longer/larger models |
    | **Exposure bias** (generation) | Errors compound at inference | Trained on ground truth, runs on own outputs | Scheduled sampling; sequence-level training |
    | **Sensitivity to sequence length / scaling** | Poor generalization to unseen lengths | Train/test length mismatch; unnormalized inputs | Match lengths; normalize; bucket by length |
    | **Overfitting** | Train ≫ val | Many params, little data | Dropout (recurrent), weight decay, more data |

    The cell demonstrates **gradient clipping**, the standard fix for the exploding
    half of the problem.
    """),

    code(r"""
    # Exploding gradients with a large recurrent weight, fixed by clipping the global norm.
    def grad_norm_for(scale, T=30):
        X, y = make_first_bit(128, T, 0)
        net = VanillaRNN(1, 24, seed=1)
        net.Wh *= scale                                 # inflate recurrent weights
        net.forward(X); net.backward(y)
        return np.sqrt(sum(np.sum(g ** 2) for g in net.grads))

    raw = grad_norm_for(8.0)
    clip_threshold = 5.0
    clipped = min(raw, clip_threshold)
    print(f"raw global gradient norm (large Wh): {raw:.1f}  <- would destabilize training")
    print(f"after clipping to {clip_threshold}: {clipped:.1f}")
    print("Gradient clipping caps the update norm, preventing a single bad step from NaN-ing the run.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    PyTorch's `nn.RNN`, `nn.LSTM`, `nn.GRU` provide optimized, multi-layer,
    (bi)directional recurrent layers with autograd-handled BPTT and cuDNN
    acceleration. Below we train an **LSTM** on the same "remember the first bit" task
    where our vanilla RNN failed at long lengths — and watch it succeed. Guarded import.
    """),

    code(r"""
    # LSTM solves the long-memory task the vanilla RNN couldn't. Guarded; short training.
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(0)

        def make_torch(n, T):
            X, y = make_first_bit(n, T, seed=7)
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        class LSTMClassifier(nn.Module):
            def __init__(self, h=24):
                super().__init__()
                self.lstm = nn.LSTM(1, h, batch_first=True)
                self.fc = nn.Linear(h, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return torch.sigmoid(self.fc(out[:, -1, :]))

        T = 30
        Xt, yt = make_torch(512, T)
        model = LSTMClassifier()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        lossf = nn.BCELoss()
        for _ in range(300):
            opt.zero_grad(); loss = lossf(model(Xt), yt); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # clip (Section 7)
            opt.step()
        acc = ((model(Xt) > 0.5) == (yt > 0.5)).float().mean().item()
        rnn_acc = train_rnn(T, epochs=400)[1]
        print(f"length {T}:  vanilla RNN acc {rnn_acc:.2f}  |  LSTM acc {acc:.2f}")
        print("The LSTM remembers the first bit across 30 steps where the RNN is near chance.")
    except Exception as e:
        print(f"[torch not available: {type(e).__name__}] "
              f"the scratch RNN (Figs 1-2) and LSTM cell (5.2) demonstrate the mechanism.")
    """),

    md(r"""
    **Scratch vs production.** Our NumPy RNN + BPTT and LSTM cell expose the exact
    mechanism; PyTorch's recurrent layers add cuDNN-fused kernels, multi-layer and
    bidirectional stacking, dropout, and packed variable-length sequences — and
    autograd handles BPTT. The headline result: on the long-memory task, the **LSTM
    holds accuracy where the vanilla RNN collapses** (Fig 2), confirming the gated
    cell-state design. Note we still apply **gradient clipping** — recurrent models
    remain prone to exploding gradients regardless of the gating fix for vanishing ones.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Energy Demand Forecasting

    **Scenario.** A utility forecasts **hourly electricity demand** from historical
    load plus weather and calendar features, to schedule generation and trade on energy
    markets. The signal is strongly sequential: trend, daily/weekly seasonality (the
    cyclical features of Lesson MLE-03), and dependence on recent hours.

    **Why an LSTM (and when not):**
    - Sequence with **temporal dependencies** and variable horizons → recurrence is a
      natural fit, and LSTMs handle the medium-range memory (yesterday's pattern, this
      week's trend).
    - **Streaming/low-latency** updates: an RNN/LSTM carries state and updates cheaply
      per new hour — attractive for online forecasting.
    - *But*: for tabular-ized time series, **gradient boosting** (Lesson CML-05) on
      lag/rolling features is a very strong, simpler baseline; and for long contexts,
      **Transformers** (Lesson DL-08) often win. A senior baselines all three.

    **Business objectives:** accurate short- and medium-horizon load forecasts to
    minimize generation cost and market risk.

    **Cost of mistakes:** under-forecast → emergency generation / blackouts (very
    costly); over-forecast → wasted generation. Errors are **asymmetric** and feed
    directly into financial decisions, so calibrated uncertainty matters.

    **Constraints:** strict **point-in-time correctness** and time-ordered validation
    (Lesson MLE-02 — never shuffle!); real-time inference; robustness to regime shifts
    (heat waves, holidays).

    **KPIs:** MAE/RMSE/MAPE per horizon (Lesson MLE-01), peak-hour error, forecast
    interval coverage, and the offline↔online gap under distribution shift
    (PROD-05 → retrain with PROD-06).
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Gradient clipping is standard** for any recurrent model — cheap insurance
      against exploding gradients NaN-ing a run (§7, Lesson FND-04).
    - **Time-ordered validation only.** Never shuffle a sequence/time series across the
      split; use forward-chaining and point-in-time features (Lesson MLE-02). This is the
      most common RNN evaluation bug.
    - **Truncated BPTT** for long/streaming sequences: backprop over a fixed window to
      bound memory/compute while carrying state forward.
    - **Sequential = hard to parallelize.** Training is slower than CNNs/Transformers on
      long sequences; bucket by length, pack sequences, and consider whether a
      Transformer is a better fit (Lesson DL-08).
    - **State management at serving.** For streaming inference, persist and update the
      hidden/cell state per entity; handle session boundaries and cold starts.
    - **Normalization & length.** Normalize inputs; be wary of generalizing to sequence
      lengths unseen in training.
    - **Choose the right tool.** For tabular-ized series, baseline against GBM; for long
      context or large data, prefer Transformers. RNNs/LSTMs shine for
      streaming/low-latency and smaller-data sequence problems.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **RNN vs LSTM vs GRU:**

    | Dimension | Vanilla RNN | LSTM | GRU |
    |---|---|---|---|
    | Long-term memory | Poor (vanishing) | **Strong** (gated cell) | Strong |
    | Gates / params | None / fewest | 3 gates / most | 2 gates / medium |
    | Compute | Cheapest | Most | Medium |
    | When | Short sequences only | Long dependencies, default | Long deps, lighter/faster |

    **Recurrent nets vs Transformers (the modern contrast — DL-07 and DL-08):**

    | Dimension | RNN/LSTM | Transformer |
    |---|---|---|
    | Parallelism | **Sequential** (slow) | **Fully parallel** across positions |
    | Long-range context | Limited (state bottleneck) | **Direct** (attention to any position) |
    | Memory/compute | $O(T)$ time, small state | $O(T^2)$ attention |
    | Data needed | Less | More (or pretraining) |
    | Streaming/low-latency | **Natural** (carry state) | Needs caching tricks |
    | Default for | Streaming, small-data sequences | NLP, large-scale, long context |

    **Recurrence vs 1D-CNN for sequences:** CNNs are parallel and capture local
    patterns with a fixed receptive field; RNNs carry unbounded state but are
    sequential. Often combined or both superseded by attention.

    **Senior lesson:** RNNs share weights across *time* (as CNNs do across *space*),
    enabling variable-length memory — but the sequential dependency and vanishing
    gradients cap them. LSTMs fix vanishing memory with gates; attention fixes the
    sequential bottleneck. Know which limitation each architecture addresses.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *How does an RNN process a sequence?* → Recurrence $h_t=\tanh(x_tW_x+h_{t-1}W_h+b)$
      with weights shared across time; $h_t$ summarizes the past (§4.1).
    - *What does the hidden state represent?* → A fixed-size running summary of the
      sequence so far.

    **Deep-dive questions**
    - *What is BPTT and why vanishing gradients?* → Unroll + backprop; reaching step 1
      multiplies $T$ squashing Jacobians → exponential decay (§4.2–4.3, Fig 1).
    - *How does an LSTM fix it?* → Additive gated cell state; cell-state gradient is the
      product of forget gates ≈1 → constant error carousel (§4.5, Fig 3).
    - *Why did Transformers replace RNNs?* → Remove the sequential bottleneck (parallel)
      and give direct long-range access via attention (§4.6).

    **Whiteboard questions**
    - "Write the RNN recurrence and its BPTT gradient." (§4.1–4.2; Section 5.1.)
    - "Write the LSTM gate equations." (Section 4.4 / 5.2.)

    **Strong vs weak answers**
    - *"Your RNN won't learn a long dependency."*
      - **Weak:** "Train longer."
      - **Strong:** "Vanilla RNNs vanish gradients over long ranges — it's
        architectural, not a training-time issue. I'd switch to an LSTM/GRU, clip
        gradients for the exploding side, and consider a Transformer if the context is
        long."
    - *"Why not just use a Transformer for everything sequential?"*
      - **Weak:** "Transformers are always better."
      - **Strong:** "Often, but attention is $O(T^2)$ and data-hungry; for
        streaming/low-latency or small-data sequence tasks an LSTM that carries state
        can be the better engineering choice."

    **Follow-ups:** "Bidirectional RNN — when?" (full sequence available, not
    streaming). "Truncated BPTT?" (bound memory on long sequences). "GRU vs LSTM?"
    (fewer params, similar performance).

    **Common mistakes:** shuffling time series in validation (Lesson MLE-02); forgetting
    gradient clipping; claiming LSTMs fully solve long-range memory; confusing the
    vanishing fix (LSTM) with the parallelism fix (attention); not baselining against
    GBM for tabular series.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define an RNN and what its hidden state carries.
    2. **Why was it invented?** What do sequences need that MLPs/CNNs don't provide?
    3. **How does it work?** Walk the recurrence and BPTT.
    4. **Why does it work (and fail)?** Why do gradients vanish over time, and how does
       the LSTM cell state fix it?
    5. **When to use it?** When pick an RNN/LSTM over a Transformer or GBM?
    6. **When NOT to use it?** Name two limitations (sequential, long-range memory).
    7. **Tradeoffs?** RNN vs LSTM vs GRU; recurrence vs attention.
    8. **How would you productionize it?** Clipping, time-ordered validation, truncated
       BPTT, state management, and tool choice.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Explain why reusing the same $W_h$ at every step makes vanishing/exploding
       gradients *worse* than ordinary network depth.
    2. In one sentence each, state the role of the LSTM forget, input, and output gates.

    **Beginner → Intermediate (coding)**
    3. Extend the scratch RNN to **many-to-many** (output at each step) and train it on
       a sequence-tagging toy task.
    4. Implement a **GRU cell** forward pass and compare its parameter count to the LSTM.

    **Intermediate (analysis)**
    5. Implement **full LSTM BPTT** from scratch and reproduce Figure 2, showing the
       LSTM holds accuracy where the vanilla RNN collapses.
    6. Measure the empirical per-step gradient-decay factor for the vanilla RNN as a
       function of $\|W_h\|$ and relate it to the vanishing/exploding boundary.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive $\partial c_t/\partial c_{t-1}=f_t$ for the LSTM and explain
       the constant error carousel; contrast with the RNN's $\operatorname{diag}(1-h^2)W_h^\top$.
    8. *Design:* the energy-forecasting system of §9 — model choice (LSTM vs GBM vs
       Transformer), time-ordered validation, point-in-time features, clipping, and
       streaming state management; specify the metrics and drift monitoring.
    9. *Debug:* an LSTM forecaster looks great in backtests but fails live. List the top
       causes (shuffled validation / leakage, distribution shift, train/serve state
       mismatch) and how you'd confirm each.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    RNNs process sequences by maintaining a **hidden state** and applying the **same
    weights at every time step** (weight sharing across time, as CNNs share across
    space), enabling variable-length, order-aware processing with memory. Training via
    **BPTT** runs into **vanishing/exploding gradients** over long ranges — the
    DL-03 problem stretched across time — so vanilla RNNs forget the distant past.
    The **LSTM** adds a gated, additive **cell state** whose gradient is a product of
    **forget gates** (≈1), a *constant error carousel* that preserves long-range memory.
    What RNNs/LSTMs *can't* escape is the **sequential bottleneck**.

    **Related lesson:** `DL-07 · The Attention Mechanism` — the idea that lets every position look
    directly at every other position, in parallel, with no fixed-size state bottleneck.
    It removes exactly the limitation that capped RNNs and is the core primitive of the
    Transformer (Lesson DL-08) and all modern LLMs.
    """),
]

build("04_deep_learning/06_rnn_and_lstm.ipynb", cells)
