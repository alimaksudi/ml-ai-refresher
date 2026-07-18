"""Builder for Lesson CML-05 — Gradient Boosting and XGBoost.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # CML-05 · Gradient Boosting and XGBoost
    ### Section 02 — Classical Machine Learning · *ML/AI Senior Mastery Curriculum*

    > Lesson CML-04 cut **variance** by *averaging* many independent deep trees.
    > Gradient boosting cuts **bias** by the opposite move: add many *shallow* trees
    > **sequentially**, each one trained to fix the errors the current ensemble still
    > makes. The deep idea — due to Friedman — is that boosting is literally
    > **gradient descent (Lesson FND-04) performed in the space of functions**: each
    > new tree is a step along the negative gradient of the loss. XGBoost/LightGBM
    > add second-order information and serious systems engineering, and the result is
    > the model that **wins most tabular ML competitions** and powers countless
    > fraud, credit, and ranking systems in production. This notebook closes Section 02
    > by uniting trees (CML-03 and CML-04) with the optimization of FND-04.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Forward stagewise additive modeling**: build $F_M=\sum_m \nu\,h_m$ one tree
      at a time.
    - The key reframing: **gradient boosting = gradient descent in function space**,
      where each tree fits the **pseudo-residuals** (negative gradient of the loss).
    - Why for squared loss the pseudo-residuals are just the **ordinary residuals**,
      and for log loss they are $y-p$.
    - The **shrinkage (learning rate) ↔ n_estimators** tradeoff and why boosting
      **overfits with too many rounds** (the opposite of Random Forest) → **early
      stopping**.
    - **XGBoost's second-order** objective: optimal leaf weight $-G/(H+\lambda)$ and
      the split-gain formula — prime interview material.
    - Building a gradient-boosted regressor **from scratch**.

    **Why it matters in industry**
    - **State-of-the-art on tabular data** — the default winning model for structured
      problems (fraud, credit, CTR, ranking, demand).
    - Mastering the **bias-vs-variance** contrast with bagging (CML-04) is exactly the
      judgment senior engineers are hired for.
    - The hyperparameters (LR, depth, rounds, subsample, regularization) are a
      practical optimization problem you must reason about, not grid-search blindly.

    **Typical interview questions**
    - "Explain gradient boosting. In what sense is it 'gradient descent'?"
    - "Boosting vs bagging — what does each reduce, and how do they behave with more
      trees?"
    - "Why does a low learning rate usually generalize better?"
    - "Derive XGBoost's optimal leaf weight and gain."
    - "Your boosted model overfits — what knobs do you turn?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **AdaBoost (Freund & Schapire, 1997).** The first practical boosting algorithm:
    train a weak learner, **up-weight the examples it got wrong**, train the next
    learner on the reweighted data, and combine them by weighted vote. It famously
    drove error down by combining "weak" learners (barely better than chance) into a
    strong one — a theoretical surprise at the time.

    **Gradient Boosting (Friedman, 1999–2001).** Friedman's reinterpretation was the
    breakthrough: AdaBoost is a special case of minimizing a loss by **functional
    gradient descent**. At each step, compute the negative gradient of the loss with
    respect to the current model's *predictions* — the **pseudo-residuals** — and fit
    a regression tree to them. This generalized boosting to *any* differentiable loss
    (squared, logistic, Poisson, ranking) and made trees the natural base learner. He
    also introduced **shrinkage** (a learning rate) and **stochastic** boosting
    (row/column subsampling) for regularization.

    **XGBoost (Chen & Guestrin, 2016)** and **LightGBM (2017)**, **CatBoost (2018)**.
    These took Friedman's algorithm and added (a) a **second-order Taylor expansion**
    of the loss (using gradients *and* Hessians) for better, more stable steps, (b)
    explicit **regularization** on tree complexity and leaf weights, and (c) heavy
    **systems engineering** — cache-aware histograms, sparsity-aware splits, parallel
    split-finding, out-of-core training. The result dominated Kaggle and became the
    production default for tabular ML.

    **Why boosting, given we already have Random Forests?** A forest reduces variance
    but **cannot reduce the bias** shared by its trees (Lesson CML-04, §4.1). Boosting
    attacks bias directly — each tree explicitly targets the current errors — so it
    typically reaches **higher accuracy**, at the cost of more careful tuning and the
    risk of overfitting if you boost too long. They are complementary tools, and
    knowing *which to reach for* is the senior skill this notebook builds.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **A team that learns from its mistakes.** Imagine forecasters working in
    sequence. The first makes a rough prediction. The second doesn't redo the work —
    it studies *where the first was wrong* and predicts the **error**. The third
    corrects what's left, and so on. Add up their contributions (each scaled down by
    a cautious **learning rate**) and the combined forecast steadily improves. That
    is boosting: **each new model predicts the residual errors of the running
    ensemble.**

    **The crucial contrast with Random Forest:**
    - **Random Forest (bagging):** many *independent* deep trees, trained in
      **parallel**, **averaged**. Fixes **variance**. More trees → safe (plateaus).
    - **Boosting:** many *dependent* shallow trees, trained **sequentially**, each
      correcting the last, **summed**. Fixes **bias**. More trees → eventually
      **overfits** (you start fitting noise), so you need **early stopping**.

    **Why shallow trees ("stumps")?** Each tree only needs to capture a *small* piece
    of the remaining error. A deep tree per round would overcorrect and overfit fast;
    many shallow trees with a small learning rate make gentle, steady progress — the
    function-space analogue of small gradient-descent steps (Lesson FND-04).

    ```mermaid
    flowchart LR
        F0["F0 = mean(y)<br/>(or log-odds)"] --> R1["residuals r1 = y - F0"]
        R1 --> H1["fit small tree h1 to r1"]
        H1 --> F1["F1 = F0 + nu*h1"]
        F1 --> R2["residuals r2 = y - F1"]
        R2 --> H2["fit h2 to r2"]
        H2 --> Fm["... F_M = F0 + nu*sum(h_m) ..."]
    ```

    Run the cells: watch a boosted ensemble build a function out of residual
    corrections — then watch it overfit if we don't stop.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # A 1D regression target with clear structure + noise, to visualize residual fitting.
    x = np.sort(rng.uniform(-4, 4, 200))
    f_true = np.sin(x) + 0.3 * x
    y = f_true + rng.normal(0, 0.35, len(x))
    X = x.reshape(-1, 1)
    print("data:", X.shape)
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Additive model & forward stagewise fitting
    Boosting builds an additive model of $M$ trees, scaled by a learning rate $\nu$:
    $$F_M(\mathbf x)=F_0(\mathbf x)+\nu\sum_{m=1}^{M}h_m(\mathbf x).$$
    We can't optimize all trees jointly, so we fit them **one at a time** (forward
    stagewise): having $F_{m-1}$, choose the next tree $h_m$ to most reduce the loss,
    then set $F_m=F_{m-1}+\nu h_m$. Earlier trees are frozen.

    ### 4.2 Gradient boosting = gradient descent in function space
    We minimize $\sum_i L(y_i, F(\mathbf x_i))$ over the *function* $F$. Treat the
    vector of predictions $F(\mathbf x_i)$ as the parameters. Gradient descent says
    step along the negative gradient of the loss w.r.t. those predictions — the
    **pseudo-residuals**:
    $$r_{im}=-\left.\frac{\partial L(y_i,F)}{\partial F}\right|_{F=F_{m-1}(\mathbf x_i)}.$$
    But we can't move each prediction independently and still generalize, so we fit a
    **regression tree $h_m$ to the pseudo-residuals** and take a step in that
    direction: $F_m=F_{m-1}+\nu h_m$. **That is the whole algorithm** — gradient
    descent (Lesson FND-04) where each "step" is a tree approximating the gradient.

    ### 4.3 The pseudo-residuals for common losses
    - **Squared loss** $L=\tfrac12(y-F)^2$: $\;r_i=-\partial L/\partial F = y_i-F_i$ —
      the **ordinary residuals**. (So "fit the residuals" is *exactly* gradient
      boosting with squared loss.) Init $F_0=\bar y$.
    - **Logistic / log loss** (binary): working in log-odds $F$ with $p=\sigma(F)$,
      $\;r_i=y_i-p_i$ — the same elegant $(\text{target}-\text{prediction})$ form we
      saw for logistic regression (Lesson CML-02). Init $F_0=\log\frac{\bar y}{1-\bar y}$.

    ### 4.4 Shrinkage (learning rate) and the rounds tradeoff
    The learning rate $\nu\in(0,1]$ scales each tree's contribution. A **small $\nu$**
    means each tree corrects only a little, so you need **more rounds $M$**, but the
    ensemble generalizes better (more, smaller steps explore the loss surface more
    carefully — same intuition as a small GD step size). $\nu$ and $M$ trade off:
    halve $\nu$, roughly double $M$. Unlike Random Forest, **increasing $M$ eventually
    overfits** (you begin fitting noise), so $M$ is chosen by **early stopping** on a
    validation set.

    ### 4.5 Advanced extension — XGBoost's second-order view
    XGBoost expands the loss to **second order** (Taylor) around $F_{m-1}$, using the
    gradient $g_i=\partial_F L$ and Hessian $h_i=\partial_F^2 L$:
    $$\mathcal L^{(m)}\approx\sum_i\Big[g_i\,h_m(\mathbf x_i)+\tfrac12 h_i\,h_m(\mathbf x_i)^2\Big]+\Omega(h_m),\quad
    \Omega=\gamma T+\tfrac12\lambda\sum_j w_j^2.$$
    For a tree whose leaf $j$ has weight $w_j$ and contains instance set $I_j$, define
    $G_j=\sum_{i\in I_j} g_i$ and $H_j=\sum_{i\in I_j} h_i$. Minimizing over $w_j$
    gives the **optimal leaf weight** and the **structure score**:
    $$\boxed{\,w_j^\*=-\frac{G_j}{H_j+\lambda}\,},\qquad
    \mathcal L^\*=-\tfrac12\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma T.$$
    A candidate split is scored by the **gain** it produces:
    $$\text{Gain}=\tfrac12\Big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma.$$
    The $\lambda$ (L2 on leaf weights) and $\gamma$ (penalty per leaf) are explicit
    **regularization** — a major reason XGBoost generalizes better than vanilla GBM.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    We build gradient boosting for **regression** (squared loss), where pseudo-
    residuals are ordinary residuals — the cleanest way to *see* the mechanism. We
    need a small regression tree (variance-reduction splits, leaf = mean), then the
    boosting loop. We use an efficient prefix-sum split finder so it runs fast.
    """),

    code(r"""
    # 5.1 A fast depth-limited regression tree (variance reduction via prefix sums).
    def best_split_reg(X, y):
        n, d = X.shape
        total = y.sum(); total2 = (y ** 2).sum()
        parent_sse = total2 - total * total / n
        best_red, best = 0.0, None
        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys = X[order, f], y[order]
            csum, csum2 = np.cumsum(ys), np.cumsum(ys ** 2)
            for i in range(1, n):
                if xs[i] == xs[i - 1]:
                    continue
                nl, nr = i, n - i
                sl, sl2 = csum[i - 1], csum2[i - 1]
                sr, sr2 = total - sl, total2 - sl2
                sse = (sl2 - sl * sl / nl) + (sr2 - sr * sr / nr)
                red = parent_sse - sse
                if red > best_red:
                    best_red, best = red, (f, (xs[i] + xs[i - 1]) / 2)
        return best, best_red

    def build_reg(X, y, depth, max_depth, min_samples=5):
        if depth >= max_depth or len(y) < min_samples or np.ptp(y) < 1e-9:
            return {"leaf": True, "val": float(y.mean())}
        split, red = best_split_reg(X, y)
        if split is None or red <= 1e-12:
            return {"leaf": True, "val": float(y.mean())}
        f, t = split; left = X[:, f] <= t
        return {"leaf": False, "feature": f, "threshold": t,
                "left": build_reg(X[left], y[left], depth + 1, max_depth, min_samples),
                "right": build_reg(X[~left], y[~left], depth + 1, max_depth, min_samples)}

    def reg_predict(node, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            nd = node
            while not nd["leaf"]:
                nd = nd["left"] if x[nd["feature"]] <= nd["threshold"] else nd["right"]
            out[i] = nd["val"]
        return out
    """),

    code(r"""
    # 5.2 Gradient boosting for regression: each tree fits the residuals (= neg. gradient).
    class GradientBoostingScratch:
        def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=2):
            self.M, self.nu, self.max_depth = n_estimators, learning_rate, max_depth

        def fit(self, X, y):
            self.F0 = float(y.mean())                 # squared-loss init = mean
            F = np.full(len(y), self.F0)
            self.trees = []
            for _ in range(self.M):
                residual = y - F                       # pseudo-residuals for squared loss
                tree = build_reg(X, residual, 0, self.max_depth)
                F += self.nu * reg_predict(tree, X)    # take a shrunk step
                self.trees.append(tree)
            return self

        def staged_predict(self, X):
            F = np.full(len(X), self.F0)
            preds = []
            for tree in self.trees:
                F = F + self.nu * reg_predict(tree, X)
                preds.append(F.copy())
            return preds                               # prediction after each added tree

        def predict(self, X):
            return self.staged_predict(X)[-1]

    gb = GradientBoostingScratch(n_estimators=150, learning_rate=0.1, max_depth=2).fit(X, y)
    train_mse = np.mean((gb.predict(X) - y) ** 2)
    print(f"scratch GB train MSE: {train_mse:.4f}  (noise floor ~ {0.35**2:.4f})")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Four pictures: the ensemble *building up* a function from residual corrections,
    the **overfitting-with-rounds** curve that distinguishes boosting from bagging,
    and the **learning-rate** tradeoff.
    """),

    code(r"""
    # Figure 1 — the boosted prediction assembles itself from residual corrections.
    grid = np.linspace(-4, 4, 400).reshape(-1, 1)
    F = np.full(len(grid), gb.F0)
    stages = {0: F.copy()}
    snapshot = [1, 5, 30, 150]
    for m, tree in enumerate(gb.trees, 1):
        F = F + gb.nu * reg_predict(tree, grid)
        if m in snapshot:
            stages[m] = F.copy()

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, m in zip(axes, snapshot):
        ax.scatter(x, y, s=8, color="lightgray")
        ax.plot(grid[:, 0], np.sin(grid[:, 0]) + 0.3 * grid[:, 0], "g--", lw=1.5, label="true")
        ax.plot(grid[:, 0], stages[m], "r", lw=2, label=f"{m} trees")
        ax.set_title(f"after {m} trees"); ax.set_ylim(-3, 3); ax.legend(fontsize=8)
    plt.suptitle("Figure 1 — Boosting builds the function step-by-step from residuals")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** After **1 tree** the model is a crude step (a single shallow stump's
    correction to the mean). By **5 trees** the broad shape appears; by **30** it
    closely tracks the true curve (green); by **150** it's tight — but look at the
    wiggles starting to chase individual noisy points. Each tree only ever fits
    what's *left over*, so the ensemble greedily reduces bias round by round. The
    danger is visible at 150: keep going and it fits noise — the next figure makes
    that precise.
    """),

    code(r"""
    # Figure 2 — boosting OVERFITS with too many rounds (contrast: RF plateaus, CML-04).
    # multi-feature noisy data so overfitting is pronounced
    n = 300; d = 5
    Xm = rng.normal(size=(n, d))
    ym = (Xm[:, 0] ** 2 + np.sin(2 * Xm[:, 1]) - Xm[:, 2] + rng.normal(0, 0.5, n))
    Xtr, ytr, Xte, yte = Xm[:200], ym[:200], Xm[200:], ym[200:]

    gb2 = GradientBoostingScratch(n_estimators=200, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
    tr = [np.mean((p - ytr) ** 2) for p in gb2.staged_predict(Xtr)]
    te = [np.mean((p - yte) ** 2) for p in gb2.staged_predict(Xte)]
    best_m = int(np.argmin(te)) + 1

    fig, ax = plt.subplots()
    ax.plot(range(1, 201), tr, label="train MSE")
    ax.plot(range(1, 201), te, label="test MSE")
    ax.axvline(best_m, color="k", ls="--", label=f"early stop @ {best_m}")
    ax.set_xlabel("number of boosting rounds"); ax.set_ylabel("MSE")
    ax.set_title("Figure 2 — Train MSE -> 0, but test MSE bottoms then RISES (overfitting)")
    ax.legend()
    plt.show()
    print(f"Best test MSE at {best_m} rounds; training past that memorizes noise.")
    """),

    md(r"""
    **Figure 2.** This is the single most important difference from Random Forest.
    Training error marches toward zero as we add rounds — but **test error bottoms out
    and then climbs**: beyond the optimum, new trees fit residual *noise*, not signal.
    Compare Lesson CML-04, Fig 2, where adding trees only ever *plateaus*. The practical
    consequence: **the number of rounds is a regularization hyperparameter**, set by
    **early stopping** at the validation-error minimum (dashed line), not "as many as
    compute allows."
    """),

    code(r"""
    # Figure 3 — learning rate: small nu generalizes better but needs more rounds.
    fig, ax = plt.subplots()
    for lr, color in [(0.5, "tab:red"), (0.1, "tab:blue"), (0.03, "tab:green")]:
        g = GradientBoostingScratch(n_estimators=200, learning_rate=lr, max_depth=3).fit(Xtr, ytr)
        te_lr = [np.mean((p - yte) ** 2) for p in g.staged_predict(Xte)]
        ax.plot(range(1, 201), te_lr, color=color, label=f"lr={lr} (min {min(te_lr):.2f})")
    ax.set_xlabel("rounds"); ax.set_ylabel("test MSE"); ax.set_ylim(0, max(te) * 1.1)
    ax.set_title("Figure 3 — Lower learning rate: slower but lower test error")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 3.** A **high** learning rate (red, 0.5) drops fast but overfits early
    and bottoms out at a *higher* test error — big, greedy steps overshoot. A **low**
    rate (green, 0.03) needs many more rounds but reaches a **lower** minimum and
    overfits more slowly — small, careful steps. This is the function-space echo of
    Lesson FND-04's learning-rate figure. The standard recipe: pick a **small learning
    rate** (0.01–0.1) and let **early stopping** choose the number of rounds.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Overfitting (too many rounds)** | Train error → 0, test error rises (Fig 2) | Late trees fit noise | **Early stopping**; lower LR; regularize ($\lambda,\gamma$); subsample |
    | **Sensitive to noisy labels/outliers** | Chases mislabeled points hard | Each round targets the largest residuals | Robust loss (Huber); clean labels; cap depth; subsample |
    | **LR / rounds mis-set** | Underfits (LR too low, too few rounds) or overfits | The $\nu$–$M$ tradeoff mis-tuned | Tune jointly; small LR + early stopping |
    | **Slow training** | Long wall-clock vs RF | Trees are **sequential**, not parallel | Histogram methods (LightGBM/HistGB); fewer/shallower trees |
    | **Over-deep trees** | Overfits fast; high variance per round | Each tree captures too much | Keep depth small (3–8); that's the point of *weak* learners |
    | **Extrapolation** | Flat outside training range | Tree leaves are constants | Don't extrapolate; use a trend model if needed |
    | **Leakage amplification** | Unbelievably good CV, bad live | Boosting exploits any leaky feature hard | Rigorous validation (Lesson MLE-02) |

    The cell shows boosting's **outlier sensitivity** — it will bend hard toward a few
    corrupted labels because they dominate the residuals.
    """),

    code(r"""
    # Boosting chases corrupted labels because they produce the biggest residuals.
    xc = np.sort(rng.uniform(-4, 4, 150)); Xc = xc.reshape(-1, 1)
    yc = np.sin(xc) + rng.normal(0, 0.2, 150)
    yc_bad = yc.copy(); yc_bad[[40, 75, 110]] += 5.0     # 3 corrupted labels

    g_clean = GradientBoostingScratch(120, 0.1, 3).fit(Xc, yc)
    g_dirty = GradientBoostingScratch(120, 0.1, 3).fit(Xc, yc_bad)
    gg = np.linspace(-4, 4, 400).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.scatter(xc, yc_bad, s=12, color="lightgray", label="data (3 corrupted)")
    ax.plot(gg[:, 0], g_clean.predict(gg), "g-", lw=2, label="trained on clean")
    ax.plot(gg[:, 0], g_dirty.predict(gg), "r--", lw=2, label="trained on corrupted")
    ax.set_title("Figure 4 — Boosting bends toward corrupted labels (outlier sensitivity)")
    ax.legend(); ax.set_ylim(-2.5, 3)
    plt.show()
    print("Spikes appear where the corrupted points are: large residuals dominate the fit.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    Production gradient boosting comes in three industrial flavors: sklearn's
    `GradientBoosting*` (faithful Friedman GBM), sklearn's **`HistGradientBoosting*`**
    (LightGBM-style histogram binning — fast, handles big data and missing values),
    and **XGBoost/LightGBM/CatBoost** (second-order, regularized, heavily optimized,
    with native early stopping). We compare on a classification task and verify our
    scratch regressor against sklearn. The XGBoost import is wrapped defensively so
    this runs even if the package is absent.
    """),

    code(r"""
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss

    # verify scratch regressor vs sklearn on the 1D problem
    skgb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                     max_depth=2).fit(X, y)
    print(f"scratch GB train MSE : {np.mean((gb.predict(X) - y) ** 2):.4f}")
    print(f"sklearn GB train MSE : {np.mean((skgb.predict(X) - y) ** 2):.4f}")

    # classification benchmark: RF (bagging) vs HistGB (boosting)
    Xc, yc = make_classification(n_samples=4000, n_features=20, n_informative=8,
                                 class_sep=0.8, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300,
                                         early_stopping=True, random_state=0).fit(Xtr, ytr)
    print(f"\\nRandom Forest test acc      : {accuracy_score(yte, rf.predict(Xte)):.4f}")
    print(f"HistGradientBoosting test acc: {accuracy_score(yte, hgb.predict(Xte)):.4f}")
    print(f"HistGB stopped at {hgb.n_iter_} iterations (early stopping).")
    """),

    code(r"""
    # XGBoost (optional): wrapped so the notebook runs whether or not it's installed.
    try:
        import xgboost as xgb
        dtr = xgb.DMatrix(Xtr, label=ytr); dte = xgb.DMatrix(Xte, label=yte)
        params = {"objective": "binary:logistic", "eta": 0.1, "max_depth": 4,
                  "lambda": 1.0, "gamma": 0.0, "eval_metric": "logloss"}
        booster = xgb.train(params, dtr, num_boost_round=500,
                            evals=[(dte, "test")], early_stopping_rounds=20,
                            verbose_eval=False)
        pred = (booster.predict(dte) > 0.5).astype(int)
        print(f"XGBoost test acc: {accuracy_score(yte, pred):.4f} "
              f"(best round {booster.best_iteration}, with L2 lambda + early stopping)")
    except Exception as e:
        print(f"[xgboost not available: {type(e).__name__}] "
              f"HistGradientBoosting above is the sklearn-native stand-in.")
    """),

    md(r"""
    **Scratch vs production.** Our hand-written booster matches sklearn's GBM on the
    1D problem — the mechanism is identical (fit residuals, shrink, repeat). What the
    libraries add is decisive at scale: **histogram binning** (HistGB/LightGBM) turns
    the $O(n)$ split search into $O(\text{bins})$, giving 10–100× speedups; **native
    early stopping** picks the round count for you; **second-order + L1/L2/γ
    regularization** (XGBoost, §4.5) improve accuracy and stability; and missing
    values, categorical handling, and multi-core/GPU training come built in. On this
    tabular benchmark boosting edges out the forest — the usual outcome — at the price
    of tuning and sequential training.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Real-Time Fraud Detection

    **Scenario.** A payments company scores each transaction for **fraud** in
    real-time from hundreds of tabular features (amount, merchant, velocity, device,
    geo). Gradient boosting is the industry-standard model here.

    **Why gradient boosting?**
    - **Top accuracy on tabular data** with complex feature interactions — directly
      reduces fraud losses and false declines.
    - **Probability outputs** feed a cost-sensitive **threshold** (Lesson CML-02): the
      cost of a missed fraud vs a wrongly-declined good customer is very asymmetric.
    - **Fast inference** (shallow trees) meets the real-time latency budget.
    - **SHAP** (Lesson MLE-05) gives per-transaction reason codes for analysts and
      (partial) regulatory explainability.

    **Business objectives:** catch fraud (recall on the rare positive class) while
    keeping false declines low (precision / customer experience), under tight latency.

    **Cost of mistakes (extreme imbalance + asymmetry)**
    - **Missed fraud (FN):** direct monetary loss + chargebacks.
    - **False decline (FP):** lost sale, angry customer, churn.
    Fraud is often <1% of transactions, so **accuracy is useless** — optimize
    PR-AUC / recall at a precision target and tune the threshold to the cost matrix
    (MLE-01 and MLE-04).

    **Constraints:** millisecond scoring; adversaries **adapt** (rapid concept
    drift); models and reason codes must be auditable.

    **KPIs:** PR-AUC / recall@fixed-precision, fraud-dollars caught vs false-decline
    rate, p99 scoring latency, and **drift** metrics (fraudsters change tactics →
    frequent retraining, Lesson PROD-06).
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Training cost.** Sequential by nature (each tree needs the last), so slower to
      train than a parallel forest — mitigated by **histogram** methods and GPU. Plan
      retraining cadence accordingly.
    - **Inference latency.** Many but **shallow** trees → fast, predictable scoring;
      well-suited to real-time. Compile/quantize (e.g. Treelite) for the tightest
      budgets.
    - **Early stopping is mandatory.** Always hold out a validation set and stop at its
      loss minimum; never fix `n_estimators` by guess (Fig 2). This is the #1
      production guardrail against overfitting.
    - **Hyperparameter tuning.** The high-leverage knobs: `learning_rate`, `max_depth`/
      `num_leaves`, `n_estimators` (via early stopping), `subsample`/`colsample`, and
      regularization `lambda`/`gamma`/`min_child_weight`. Tune with CV / Bayesian
      search, not blind grids.
    - **Explainability.** No coefficients; **SHAP** is the standard for per-prediction
      and global explanations (Lesson MLE-05) and is fast for trees (TreeSHAP).
    - **Monitoring & drift.** Tabular targets like fraud/CTR drift quickly; monitor
      feature/score distributions and performance, and **retrain frequently**
      (PROD-05 and PROD-06). Watch for train/serve skew in feature pipelines.
    - **Reproducibility.** Pin seeds, library versions, and the early-stopping round;
      boosting results vary with subsample seeds.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Bagging (Random Forest) vs Boosting — the defining contrast:**

    | Dimension | Random Forest (bagging) | Gradient Boosting |
    |---|---|---|
    | Reduces | **Variance** | **Bias** |
    | Trees | Deep, independent, **parallel** | Shallow, sequential, dependent |
    | More trees | Safe (plateaus) | **Overfits** → early stopping |
    | Accuracy (tabular) | High | **Usually highest** |
    | Tuning effort | Low | Higher (LR, depth, rounds, reg) |
    | Training speed | Fast (parallel) | Slower (sequential) |
    | Robust to noise/outliers | **More** | Less (chases residuals) |
    | Out-of-the-box | **Yes** | Needs care |

    **XGBoost vs LightGBM vs CatBoost:**

    | Dimension | XGBoost | LightGBM | CatBoost |
    |---|---|---|---|
    | Split finding | Pre-sorted / histogram | **Leaf-wise histogram (fast)** | Symmetric trees |
    | Best at | Robust default | **Large data, speed** | **Categorical features** |
    | Overfit risk | Low (regularized) | Higher (leaf-wise) — tune leaves | Low |

    **Tree ensembles vs deep learning on tabular data:**

    | Dimension | Gradient Boosting | Deep Learning |
    |---|---|---|
    | Tabular accuracy | **Usually wins** | Competitive only with effort |
    | Data needed | Moderate | Large |
    | Preprocessing | Minimal (no scaling) | Heavy (encoding, scaling) |
    | Training cost | Low–moderate | High |
    | When to prefer DL | Text/image/audio, huge data | — |

    **Senior lesson:** for tabular problems, **start with gradient boosting** (or RF
    as a no-tuning baseline) before any neural net — it usually wins with far less
    effort. Reach for the other based on whether your error is dominated by **bias**
    (boost) or **variance** (bag).
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Explain gradient boosting.* → Sequential additive trees, each fit to the
      pseudo-residuals (negative gradient) of the loss; sum with shrinkage.
    - *Why is it called "gradient"?* → It's gradient descent in function space
      (Section 4.2); each tree approximates the negative-gradient step.

    **Deep-dive questions**
    - *Bagging vs boosting — what does each reduce?* → Variance vs bias; behavior with
      more trees (plateau vs overfit). Be crisp.
    - *Why does low LR generalize better?* → Smaller, more numerous steps; less
      overshoot (Section 4.4, Fig 3).
    - *Derive XGBoost's leaf weight and gain.* → $w^*=-G/(H+\lambda)$,
      $\text{Gain}=\frac12[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}]-\gamma$
      (Section 4.5).

    **Whiteboard questions**
    - "Implement gradient boosting for regression." (Section 5 — residual loop.)
    - "What are the pseudo-residuals for squared and log loss?" ($y-F$; $y-p$.)

    **Strong vs weak answers**
    - *"How many boosting rounds should you use?"*
      - **Weak:** "More is better, like Random Forest."
      - **Strong:** "Unlike a forest, more rounds *overfit* — train error → 0 while
        test error turns back up. I use a small learning rate and **early stop** at
        the validation minimum; the round count is effectively a regularizer."
    - *"RF or XGBoost for this tabular problem?"*
      - **Weak:** "XGBoost, it's best."
      - **Strong:** "Depends on the error source and budget. RF is a robust,
        low-tuning baseline that won't overfit; XGBoost usually wins accuracy by
        attacking bias but needs tuning and early stopping. I'd baseline with RF, then
        tune XGBoost and compare under proper CV."

    **Follow-ups:** "What if labels are noisy?" (boosting chases them — robust loss,
    subsample, depth cap). "Why second-order in XGBoost?" (better steps + built-in
    regularization). "Make training faster?" (histogram/LightGBM, GPU, fewer rounds).

    **Common mistakes:** confusing bagging and boosting; thinking more rounds is
    always safe; ignoring early stopping; setting LR and rounds independently; using a
    too-deep base tree; reporting accuracy on imbalanced fraud data.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define gradient boosting as forward stagewise additive modeling.
    2. **Why was it invented?** What did it offer over AdaBoost and over Random Forest?
    3. **How does it work?** Explain pseudo-residuals and the residual-fitting loop.
    4. **Why does it work?** In what precise sense is it gradient descent, and why does
       fitting residuals reduce *bias*?
    5. **When to use it?** When is boosting the right tabular choice over a forest?
    6. **When NOT to use it?** Name two situations (very noisy labels, no early-stopping
       budget) where you'd hesitate.
    7. **Tradeoffs?** Bagging vs boosting; LR vs rounds; XGBoost vs LightGBM.
    8. **How would you productionize it?** Early stopping, tuning, SHAP, drift
       monitoring, and retraining for a fraud/CTR system.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. For squared loss, show the pseudo-residual equals $y-F$. For log loss, show it
       equals $y-p$. Why is this the same $(\text{target}-\text{prediction})$ form?
    2. Explain in two sentences why a Random Forest can't overfit by adding trees but
       a boosted model can.

    **Beginner → Intermediate (coding)**
    3. Add **early stopping** to the scratch `GradientBoostingScratch`: track
       validation MSE and stop when it hasn't improved for `patience` rounds.
    4. Implement **binary classification** boosting: work in log-odds, init
       $F_0=\log\frac{\bar y}{1-\bar y}$, fit trees to $y-\sigma(F)$, and report
       log-loss/accuracy.

    **Intermediate (analysis)**
    5. Reproduce Figure 3 and find, for two learning rates, the $(\nu, M)$ pairs that
       reach the same test error — empirically verify the "halve $\nu$, double $M$"
       rule.
    6. Add **row subsampling** (stochastic gradient boosting) to the scratch model and
       show it reduces overfitting (later test-error minimum, lower variance).

    **Senior (interview + production design)**
    7. *Whiteboard:* derive XGBoost's optimal leaf weight $-G/(H+\lambda)$ and the
       split-gain formula from the second-order objective; explain the role of
       $\lambda$ and $\gamma$.
    8. *Design:* build the fraud-detection system of §9 — feature pipeline, class
       imbalance handling, threshold from the cost matrix, early stopping, SHAP reason
       codes, latency budget, and a drift-triggered retraining loop.
    9. *Diagnose:* a boosted model's offline AUC is 0.99 but live performance is poor.
       Enumerate the top causes (leakage exploited by boosting, drift, train/serve
       skew, no early stopping) and how you'd confirm each.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Gradient boosting builds an additive ensemble **one shallow tree at a time**, each
    tree fitting the **pseudo-residuals** (negative gradient) of the loss — i.e.
    **gradient descent in function space**. Shrinkage + early stopping control the
    bias–variance tradeoff; it **reduces bias** (complement to Random Forest's
    variance reduction) and typically wins on tabular data. XGBoost adds a
    second-order objective and explicit regularization ($w^*=-G/(H+\lambda)$), and the
    libraries add histograms, early stopping, and systems speed.

    **Section 02 is complete.** You now command the classical-ML toolkit — linear and
    logistic regression, single trees, and both ensemble paradigms (bagging vs
    boosting) — plus the judgment of *which* to use and *why*.

    **Related lesson:** `MLE-01 · Evaluation Metrics` — Section 03 begins. Every model so far has been
    scored loosely; now we get rigorous about **measuring** them: accuracy's lies,
    precision/recall, ROC vs PR curves, calibration, and choosing the metric that
    matches the business cost.
    """),
]

build("02_classical_ml/05_gradient_boosting_and_xgboost.ipynb", cells)
