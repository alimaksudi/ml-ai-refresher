"""Builder for Notebook 06 — Decision Trees.

Run:  python3 tools/builders/phase1_06_decision_trees.py
Emits: notebooks/phase1_classical_ml/06_decision_trees.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # 06 · Decision Trees
    ### Phase 1 — Classical Machine Learning · *ML/AI Senior Mastery Curriculum*

    > Notebooks 04–05 drew **one straight boundary** through feature space. Real
    > relationships are rarely linear, and forcing them to be requires hand-crafted
    > features. Decision trees take the opposite approach: **recursively chop feature
    > space into axis-aligned boxes**, each box getting its own simple prediction.
    > No equations to fit, no feature scaling, native handling of nonlinearity and
    > interactions, and a model you can literally read as a flowchart. Their fatal
    > flaw — **high variance** — is exactly the flaw that Random Forests (07) and
    > Gradient Boosting (08) are built to cure, so this notebook is the keystone of
    > the most important model family in tabular ML.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The tree as **recursive partitioning**: greedy, axis-aligned splits that carve
      feature space into pure regions.
    - **Impurity** measures — **Gini**, **entropy** (classification), **variance**
      (regression) — and **information gain** as the splitting criterion.
    - Why splitting is **greedy** (finding the optimal tree is NP-hard) and what that
      costs you.
    - How trees **overfit**, and the levers that control it: max depth, min samples,
      and **cost-complexity pruning**.
    - Why a single tree is **high-variance and unstable** — the precise motivation
      for ensembles (07, 08).
    - Building a CART classifier **from scratch** and reading its rules.

    **Why it matters in industry**
    - The building block of **XGBoost/LightGBM**, which win most tabular ML problems
      and Kaggle competitions.
    - **Interpretable as rules** — "if income < X and debt > Y then decline" — which
      domain experts and (sometimes) regulators accept directly.
    - Handles mixed types, missing values, and unscaled features with **almost no
      preprocessing** — a huge practical advantage over linear models and NNs.

    **Typical interview questions**
    - "How does a decision tree choose a split? Define Gini and entropy."
    - "Why are trees prone to overfitting, and how do you prevent it?"
    - "Gini vs entropy — does it matter?"
    - "Why is a single tree unstable, and how do ensembles fix it?"
    - "Trees need no feature scaling — why, and when is that an advantage?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **Before trees: rules by hand.** Early "expert systems" encoded human-written
    if-then rules. They were interpretable but brittle, expensive to build, and
    couldn't learn from data. The question became: can we *learn* the rules
    automatically?

    **ID3 / C4.5 (Quinlan, 1986/1993)** and **CART (Breiman et al., 1984)** answered
    yes. Both greedily grow a tree by repeatedly choosing the split that most
    reduces impurity; CART (Classification And Regression Trees) is the variant
    sklearn implements — binary splits, Gini or variance, and cost-complexity
    pruning. This was a turning point: a *non-parametric* model that makes **no
    assumption of linearity**, handles numeric and categorical features together,
    is invariant to monotonic feature transforms (so no scaling), and emits a
    human-readable model.

    **Why trees over linear models (the Notebook 04–05 contrast).** Linear models
    assume a global linear relationship and need you to *engineer* nonlinearity and
    interactions. A tree discovers interactions automatically ("the effect of income
    depends on whether you're a homeowner" is just a split-within-a-split) and bends
    to any shape given enough depth. The price: a single tree **overfits and is
    unstable**.

    **Why this motivates everything after it.** Breiman's own response to tree
    instability was **bagging** and then **Random Forests** (1996–2001); Friedman's
    was **Gradient Boosting** (1999). Both are *ensembles of trees*. So Decision
    Trees are not just a model — they are the unit cell of the dominant tabular-ML
    paradigm. Understand the single tree's strengths and (especially) its variance
    problem, and 07–08 become obvious.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The 20-questions game.** A tree is a sequence of yes/no questions about
    features that narrows down the answer. "Is income > \$50k?" → "Is debt ratio >
    0.4?" → … Each question **splits the data into two purer groups**. Keep asking
    until each group is (nearly) all one class, then predict that class.

    **Geometrically: axis-aligned boxes.** Each split is a threshold on *one*
    feature — a vertical or horizontal cut in feature space. Stacking splits tiles
    the space into rectangles; every rectangle (a **leaf**) gets one prediction (the
    majority class, or the mean for regression). The decision boundary is therefore
    a **staircase**, never a smooth diagonal — a key limitation (§7).

    **What "good split" means.** A split is good if the two resulting groups are
    **purer** (more single-class) than the parent. We measure mess with an
    **impurity** score and pick the split that reduces it most — a greedy, local
    choice repeated all the way down.

    ```mermaid
    flowchart TD
        R["All data<br/>(mixed classes)"] -->|"income > 50k?"| A["yes"]
        R -->|"no"| B["no: mostly decline"]
        A -->|"debt ratio > 0.4?"| C["yes: mostly decline"]
        A -->|"no"| D["no: mostly approve"]
    ```

    Run the cells: build a real tree, see the boxes, and watch it overfit.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # A nonlinear (circular-boundary) 2D dataset — impossible for one straight line.
    def make_circle_data(n=400):
        X = rng.uniform(-3, 3, (n, 2))
        r = X[:, 0] ** 2 + X[:, 1] ** 2
        y = (r < 4.0).astype(int)                    # inside a circle of radius 2
        flip = rng.random(n) < 0.05                  # 5% label noise
        y[flip] = 1 - y[flip]
        return X, y

    X, y = make_circle_data()
    print("data:", X.shape, "| class balance:", np.bincount(y))
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Impurity — measuring "mess" in a node
    For a node with class proportions $p_k$:
    - **Gini impurity:** $G=1-\sum_k p_k^2$ — the probability of misclassifying a
      random sample if you label it by the node's class distribution. 0 = pure.
    - **Entropy:** $H=-\sum_k p_k\log_2 p_k$ — bits of uncertainty. 0 = pure, max at
      uniform. Information gain uses entropy.
    - **Regression (variance/MSE):** $\frac1n\sum_i (y_i-\bar y)^2$ — spread of the
      target; the leaf predicts $\bar y$.

    Gini and entropy are numerically close and almost always pick the same splits;
    Gini is slightly cheaper (no log), which is why it's CART's default.

    ### 4.2 The splitting criterion: impurity decrease (information gain)
    A split sends $n_L$ samples left and $n_R$ right. Its quality is the **weighted
    impurity decrease**:
    $$\Delta = I(\text{parent}) - \frac{n_L}{n}I(\text{left}) - \frac{n_R}{n}I(\text{right}).$$
    With entropy, $\Delta$ is the **information gain**. We choose the
    $(\text{feature},\text{threshold})$ pair maximizing $\Delta$.

    ### 4.3 Greedy recursive construction (CART)
    ```
    build(node):
        if stopping condition: make a leaf (predict majority class / mean)
        else:
            for each feature f, for each candidate threshold t:
                compute weighted impurity decrease of splitting on (f, t)
            pick the best (f*, t*); split the data; recurse on both children
    ```
    Candidate thresholds are the **midpoints between consecutive sorted feature
    values**. This is **greedy**: each split is locally optimal, never reconsidered.

    ### 4.4 Why greedy? Because optimal trees are NP-hard
    Finding the globally smallest/most-accurate tree is **NP-complete** — the number
    of possible trees is astronomical. Greedy CART is a heuristic that runs in
    roughly $O(\text{features}\times n\log n)$ per level and works well in practice,
    at the cost of sometimes missing a better tree that needs a "bad-looking" split
    first.

    ### 4.5 Overfitting, stopping, and pruning
    Grown unrestricted, a tree keeps splitting until **every leaf is pure** — it
    memorizes the training set (including noise), giving 100% train accuracy and poor
    test accuracy. Controls:
    - **Pre-pruning (stopping):** `max_depth`, `min_samples_split`,
      `min_samples_leaf`, `min_impurity_decrease`.
    - **Post-pruning (cost-complexity, CCP):** grow fully, then prune back the
      subtrees that add complexity without enough accuracy, by minimizing
      $R_\alpha(T)=R(T)+\alpha|T|$ (error + $\alpha\times$#leaves). $\alpha$ is
      chosen by cross-validation — the same bias–variance dial as Ridge's $\lambda$
      (Notebook 04).

    ### 4.6 The variance problem (the bridge to ensembles)
    A tree's structure is **unstable**: change a few training rows and an early split
    can flip, cascading into a completely different tree. This is **high variance** —
    low bias (it can fit anything) but high sensitivity to the data sample. The cure
    is to **average many decorrelated trees** (bagging → Random Forest, Notebook 07)
    or to **add trees that correct each other's errors** (boosting, Notebook 08).
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    A complete CART classifier in pure NumPy: Gini impurity, exhaustive best-split
    search, recursive growth with stopping rules, prediction, and a readable rule
    dump. This *is* the algorithm sklearn runs (minus the C-level speed).
    """),

    code(r"""
    # 5.1 Impurity and best-split search.
    def gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def best_split(X, y):
        n, d = X.shape
        parent = gini(y)
        best_gain, best = 0.0, None
        for f in range(d):
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2          # midpoints
            for t in thresholds:
                left = X[:, f] <= t
                nl, nr = left.sum(), (~left).sum()
                if nl == 0 or nr == 0:
                    continue
                weighted = (nl * gini(y[left]) + nr * gini(y[~left])) / n
                gain = parent - weighted
                if gain > best_gain:
                    best_gain, best = gain, (f, t)
        return best, best_gain
    """),

    code(r"""
    # 5.2 Recursive tree growth (a node is a dict) + prediction + rule dump.
    def build_tree(X, y, depth=0, max_depth=None, min_samples=2):
        # leaf if: depth cap hit, too few samples, or already pure
        if (max_depth is not None and depth >= max_depth) or len(y) < min_samples \
                or len(np.unique(y)) == 1:
            return {"leaf": True, "pred": int(np.round(y.mean())), "n": len(y),
                    "p1": float(y.mean())}
        split, gain = best_split(X, y)
        if split is None or gain <= 0:
            return {"leaf": True, "pred": int(np.round(y.mean())), "n": len(y),
                    "p1": float(y.mean())}
        f, t = split
        left = X[:, f] <= t
        return {"leaf": False, "feature": f, "threshold": t,
                "left": build_tree(X[left], y[left], depth + 1, max_depth, min_samples),
                "right": build_tree(X[~left], y[~left], depth + 1, max_depth, min_samples)}

    def predict_one(node, x):
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["pred"]

    def predict(tree, X):
        return np.array([predict_one(tree, x) for x in X])

    def print_rules(node, depth=0, name=("x0", "x1")):
        pad = "  " * depth
        if node["leaf"]:
            print(f"{pad}-> predict {node['pred']}  (n={node['n']}, P(1)={node['p1']:.2f})")
        else:
            print(f"{pad}if {name[node['feature']]} <= {node['threshold']:.2f}:")
            print_rules(node["left"], depth + 1, name)
            print(f"{pad}else:")
            print_rules(node["right"], depth + 1, name)

    tree = build_tree(X, y, max_depth=3)
    acc = np.mean(predict(tree, X) == y)
    print(f"depth-3 tree training accuracy: {acc:.3f}\\n")
    print_rules(tree)
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Four pictures: the axis-aligned regions a tree learns, the impurity curve that
    drives a split, the overfitting-with-depth story, and the instability that
    motivates ensembles.
    """),

    code(r"""
    # Figure 1 — decision regions: the tree tiles space into axis-aligned boxes.
    def plot_regions(ax, tree, X, y, title):
        xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = predict(tree, grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=12)
        ax.set_title(title); ax.set_aspect("equal")
        # overlay the true circular boundary for reference
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(2 * np.cos(th), 2 * np.sin(th), "g--", lw=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_regions(ax, tree, X, y, "Figure 1 — depth-3 tree (green = true circle)")
    plt.show()
    """),

    md(r"""
    **Figure 1.** The true boundary is a smooth **circle** (green dashed), but the
    tree can only cut along the axes, so it approximates the circle with a
    **staircase of rectangles**. A depth-3 tree captures the gist with a handful of
    boxes. Deeper trees add finer steps — better fit, but soon they start fencing off
    individual noisy points (overfitting, next figure). This staircase nature is the
    tree's signature strength (any shape, given depth) *and* weakness (no smooth or
    diagonal boundaries).
    """),

    code(r"""
    # Figure 2 — the impurity curve that selects the first split on feature x0.
    f = 0
    vals = np.unique(X[:, f]); thr = (vals[:-1] + vals[1:]) / 2
    parent = gini(y); gains = []
    for t in thr:
        left = X[:, f] <= t
        nl, nr = left.sum(), (~left).sum()
        weighted = (nl * gini(y[left]) + nr * gini(y[~left])) / len(y)
        gains.append(parent - weighted)
    gains = np.array(thr), np.array(gains)

    fig, ax = plt.subplots()
    ax.plot(gains[0], gains[1], color="tab:purple")
    best_t = gains[0][np.argmax(gains[1])]
    ax.axvline(best_t, color="r", ls="--", label=f"best threshold = {best_t:.2f}")
    ax.set_xlabel("split threshold on x0"); ax.set_ylabel("impurity decrease (gain)")
    ax.set_title("Figure 2 — CART picks the threshold of maximum impurity decrease")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 2.** For one feature, we sweep every candidate threshold and compute the
    weighted Gini decrease. The algorithm picks the peak (red line). Note the curve
    is **symmetric-ish around the data's structure**: thresholds near $\pm$ the
    circle's edge separate inside/outside best. CART does this for *every* feature
    and takes the global best — that single greedy choice becomes the root, and the
    process recurses. This is the entire learning algorithm, made visible.
    """),

    code(r"""
    # Figure 3 — overfitting: deeper trees fit training noise, hurting test accuracy.
    Xtr, ytr = X[:300], y[:300]
    Xte, yte = X[300:], y[300:]
    depths = range(1, 16)
    tr_acc, te_acc = [], []
    for dmax in depths:
        t = build_tree(Xtr, ytr, max_depth=dmax)
        tr_acc.append(np.mean(predict(t, Xtr) == ytr))
        te_acc.append(np.mean(predict(t, Xte) == yte))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(list(depths), tr_acc, "o-", label="train")
    axes[0].plot(list(depths), te_acc, "s-", label="test")
    axes[0].set_xlabel("max depth"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Train accuracy -> 100%; test accuracy peaks then drops")
    axes[0].legend()

    plot_regions(axes[1], build_tree(Xtr, ytr, max_depth=None), Xtr, ytr,
                 "Fully grown tree: jagged, noise-fitting boundary")
    plt.suptitle("Figure 3 — The depth dial is the bias-variance dial")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** **Left:** training accuracy climbs to 100% as depth grows (a deep
    tree can isolate every point), but test accuracy is **U-shaped inverted** — it
    peaks at a moderate depth and then *falls* as the tree starts memorizing noise.
    Same bias–variance story as Notebook 04's polynomial degree, different model.
    **Right:** the fully-grown tree's boundary is a jagged mess that fences off
    individual noisy points — textbook overfitting. The fix: cap depth / prune, or
    (better) ensemble many trees.
    """),

    code(r"""
    # Figure 4 — INSTABILITY: two bootstrap samples -> two different trees.
    # This high variance is the single fact that motivates Random Forests (Notebook 07).
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, seed in zip(axes, [1, 2]):
        boot = np.random.default_rng(seed).integers(0, len(X), len(X))
        t = build_tree(X[boot], y[boot], max_depth=5)
        plot_regions(ax, t, X[boot], y[boot], f"tree on bootstrap sample #{seed}")
    plt.suptitle("Figure 4 — Same distribution, resampled data -> very different trees (high variance)")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** Both trees were trained on resamples of the *same* dataset, yet
    their boundaries differ noticeably — an early split landed differently and the
    whole structure cascaded. This is **high variance**: the model depends heavily on
    the particular sample. Individually unreliable, but here's the key insight that
    powers Notebook 07: if you **average many such decorrelated trees**, the errors
    cancel and the variance collapses while the low bias remains. That is a Random
    Forest in one sentence.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Overfitting** | 100% train acc, poor test acc; jagged boundary | Grown until leaves are pure | `max_depth`, `min_samples_leaf`, CCP pruning |
    | **Instability / high variance** | Tiny data change → different tree | Greedy early splits cascade | **Ensembles** (bagging/RF, boosting) |
    | **Diagonal / smooth boundaries** | Staircase artifacts; needs huge depth | Axis-aligned splits only | Feature engineering (rotations); ensembles; or linear/SVM |
    | **Extrapolation** | Flat, constant prediction outside training range | Leaves are constants | Don't extrapolate; use models with trend (linear) if needed |
    | **High-cardinality bias** | Splits favor IDs/many-valued features | More thresholds → more chances to fit noise | Limit candidate splits; target/impact encoding; RF importance caveats |
    | **Class imbalance** | Minority class ignored | Impurity dominated by majority | `class_weight`, resampling, careful metrics (Ntbk 12) |

    The cell shows the **axis-aligned limitation**: a simple **diagonal** boundary
    forces the tree into a clumsy staircase.
    """),

    code(r"""
    # A diagonal boundary (y = x) is trivial for a line but awkward for a tree.
    Xd = rng.uniform(-3, 3, (400, 2))
    yd = (Xd[:, 1] > Xd[:, 0]).astype(int)               # linear diagonal boundary
    td = build_tree(Xd, yd, max_depth=4)
    fig, ax = plt.subplots(figsize=(6, 6))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    Z = predict(td, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.4)
    ax.scatter(Xd[:, 0], Xd[:, 1], c=yd, cmap="RdBu", edgecolor="k", s=10)
    ax.plot([-3, 3], [-3, 3], "g--", lw=2, label="true boundary (y=x)")
    ax.set_title("Figure 5 — A diagonal boundary becomes a staircase"); ax.legend()
    plt.show()
    print("A logistic regression would nail y=x with one line; the tree approximates "
          "it with steps and needs ever more depth to refine them.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    sklearn's `DecisionTreeClassifier`/`Regressor` implement optimized CART with all
    the pruning knobs, plus `export_text`/`plot_tree` for inspection. What the
    library adds over our scratch code: C-level split search, presorting,
    cost-complexity pruning (`ccp_alpha`), `class_weight`, and feature-importance
    accounting. We confirm sklearn and our scratch tree agree on a depth-limited fit.
    """),

    code(r"""
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import accuracy_score

    skt = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    print(f"sklearn depth-3 train acc : {accuracy_score(y, skt.predict(X)):.3f}")
    print(f"scratch depth-3 train acc : {np.mean(predict(tree, X) == y):.3f}")
    print("\\nsklearn's learned rules:")
    print(export_text(skt, feature_names=["x0", "x1"], max_depth=3))

    # cost-complexity pruning path: alpha trades accuracy vs tree size
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X, y)
    print("a few ccp_alpha values:", path.ccp_alphas[:5].round(4),
          "...  (larger alpha -> smaller pruned tree)")
    """),

    md(r"""
    **Scratch vs production.** Our hand-grown tree reaches essentially the same
    training accuracy and the same kinds of splits as sklearn at matched depth — the
    algorithm is identical; sklearn is just faster and adds pruning, importances, and
    sample weighting. The `cost_complexity_pruning_path` exposes the $\alpha$ knob
    from §4.5: cross-validate over `ccp_alpha` to pick the right tree size instead of
    guessing `max_depth`. For production tabular work, though, you almost never ship
    a *single* tree — you ship the ensemble built from them (07, 08).
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Loan Approval Triage Rules

    **Scenario.** A lender wants a **transparent first-pass triage**: a small set of
    if-then rules that auto-approve clearly-good and auto-decline clearly-bad
    applicants, routing the ambiguous middle to human underwriters.

    **Why a (shallow) tree here?**
    - **Direct interpretability:** the model *is* the rulebook — "if FICO > 720 and
      DTI < 0.35 → auto-approve." Underwriters and compliance can read and sign off
      on every path, which a coefficient vector or a forest cannot offer as cleanly.
    - **No preprocessing:** mixes categorical (employment type) and numeric (income)
      features without scaling or encoding gymnastics.
    - **Cheap, fast, auditable** decisions.

    **Business objectives:** cut manual review volume while keeping risk controlled
    and decisions explainable.

    **Cost of mistakes**
    - **Auto-approve a bad applicant** → loan loss (expensive).
    - **Auto-decline a good applicant** → lost revenue + customer friction.
    - **Unexplainable decision** → regulatory and reputational risk.
    These costs argue for a **shallow** tree (few, defensible rules) plus a human in
    the loop for the uncertain middle — not a deep, opaque tree.

    **Constraints:** protected attributes handled per law; monotonicity often
    required; rules must be stable enough to publish.

    **The catch (and the lead-in to 07–08):** a single tree's **instability** means
    the published rules can change substantially on retrain. So the triage tree is
    for *interpretable policy*, while the lender's actual *risk score* is produced by
    a **gradient-boosted ensemble** (Notebook 08) and explained post-hoc with SHAP
    (Notebook 13). Right tool, right job.

    **KPIs:** auto-decision rate, default rate within auto-approved, approval rate,
    rule stability across retrains, and fairness across protected groups.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Latency / cost.** Inference is a handful of comparisons down a path —
      nanoseconds, trivially cacheable, no matrix math. Among the cheapest models to
      serve. (Ensembles multiply this by the number of trees but it's still fast.)
    - **Interpretability.** A shallow tree is self-documenting; export the rules and
      version them. Deep trees lose this — past ~depth 4–5 nobody can hold the logic.
    - **Stability / monitoring.** Because trees are unstable, **monitor rule drift**
      across retrains; large structural changes usually signal data-pipeline issues
      or genuine distribution shift (Notebook 45). This instability is the main
      operational reason to prefer ensembles for the production scorer.
    - **No scaling required**, and **monotonic constraints** are supported by modern
      libraries (LightGBM/XGBoost) when regulation demands "more debt never lowers
      risk."
    - **Missing values.** CART can handle them via surrogate splits or default
      directions (XGBoost learns a default branch) — far less preprocessing than
      linear models.
    - **Retraining.** Cheap; but pin `random_state` and pruning params so published
      rules don't churn arbitrarily.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Single tree vs linear models vs tree ensembles:**

    | Dimension | Decision Tree | Linear/Logistic | Random Forest / XGBoost |
    |---|---|---|---|
    | Accuracy (tabular) | Moderate (overfits/unstable) | Low–moderate | **High** |
    | Interpretability | **High (shallow)** / low (deep) | High (coefficients) | Low (needs SHAP) |
    | Nonlinearity & interactions | **Automatic** | Manual | **Automatic** |
    | Feature scaling needed | **No** | Yes (esp. regularized) | No |
    | Variance / stability | **High / unstable** | Low | **Low (ensemble)** |
    | Latency / cost | **Lowest** | Lowest | Higher (many trees) |
    | Extrapolation | Poor (constant) | Good (linear trend) | Poor |
    | Handles mixed/missing data | **Yes** | Needs encoding/imputation | Yes |

    **Gini vs entropy:**

    | Dimension | Gini | Entropy |
    |---|---|---|
    | Formula | $1-\sum p_k^2$ | $-\sum p_k\log p_k$ |
    | Cost | Cheaper (no log) | Slightly higher |
    | Splits chosen | Nearly identical | Nearly identical |
    | Default in | CART/sklearn | ID3/C4.5 |

    **Senior lesson:** a single tree is best understood as the **interpretable unit**
    and the **ensemble building block** — rarely the final production model on its
    own, because its variance is a liability that bagging and boosting exist to fix.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *How does a tree pick a split?* → Maximize weighted impurity decrease over all
      (feature, threshold) pairs (Sections 4.1–4.3).
    - *Define Gini and entropy.* → $1-\sum p_k^2$ vs $-\sum p_k\log p_k$; both measure
      node purity.

    **Deep-dive questions**
    - *Why do trees overfit and how do you stop it?* → They grow until leaves are
      pure; control with depth/min-samples/pruning (CCP $\alpha$ via CV).
    - *Why is finding the optimal tree hard?* → NP-hard; CART is a greedy heuristic.
    - *Why no feature scaling?* → Splits depend only on the *order* of values, so any
      monotonic transform leaves the tree unchanged.

    **Whiteboard questions**
    - "Implement Gini and a best-split search." (Section 5.1.)
    - "Pseudocode recursive tree growth with a stopping rule." (Section 4.3 / 5.2.)

    **Strong vs weak answers**
    - *"Your tree gets 100% train accuracy."*
      - **Weak:** "Great, it learned the data."
      - **Strong:** "That's a red flag — it almost certainly memorized noise. I'd
        check the train/test gap, limit depth or prune via cross-validated
        `ccp_alpha`, and probably move to an ensemble for the production model."
    - *"Single tree vs random forest?"*
      - **Weak:** "Forest is just better."
      - **Strong:** "A single tree is low-bias but high-variance/unstable. A forest
        averages many decorrelated trees to cut variance with little bias cost — at
        the price of interpretability, which I'd recover with SHAP."

    **Follow-ups:** "How do you choose depth?" (CV / pruning path). "Trees for
    regression?" (variance impurity, predict the mean). "High-cardinality categorical
    — what happens?" (split bias; encode carefully).

    **Common mistakes:** shipping a deep single tree to prod; equating train accuracy
    with quality; thinking Gini vs entropy matters much; forgetting tree instability;
    claiming trees extrapolate.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Describe a decision tree as recursive partitioning.
    2. **Why was it invented?** What did it offer over hand-written rules and over
       linear models?
    3. **How does it work?** Explain impurity, information gain, and greedy splitting.
    4. **Why does it work?** Why does reducing impurity at each split produce a good
       classifier — and why only locally optimal?
    5. **When to use it?** When is a single shallow tree the right production choice?
    6. **When NOT to use it?** Name three weaknesses (variance, diagonal boundaries,
       extrapolation) and their symptoms.
    7. **Tradeoffs?** Tree vs linear vs ensemble; Gini vs entropy; depth vs pruning.
    8. **How would you productionize it?** Discuss rule export, stability monitoring,
       and why you'd often ship an ensemble instead.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Compute Gini and entropy by hand for a node with class counts [8, 2]. Which is
       larger, and why are they "close"?
    2. Explain in two sentences why a decision tree needs no feature scaling.

    **Beginner → Intermediate (coding)**
    3. Add **entropy** as an alternative impurity to the scratch tree and verify it
       picks nearly the same splits as Gini.
    4. Implement a **regression tree** (variance impurity, leaf predicts the mean)
       and fit it to a noisy 1D sine; visualize the staircase prediction.

    **Intermediate (analysis)**
    5. Reproduce Figure 3's overfitting curve with **k-fold cross-validation** and
       pick the best `max_depth`. Then prune a full tree via `ccp_alpha` and compare.
    6. Quantify **instability**: train 50 trees on bootstrap samples and measure how
       often the *root* split feature/threshold changes. Relate this to variance.

    **Senior (interview + production design)**
    7. *Whiteboard:* explain why optimal tree induction is NP-hard and what greedy
       CART sacrifices; give a concrete example where greedy misses a better tree.
    8. *Design:* build the loan-triage system of §9 — shallow interpretable tree for
       auto-decisions, human-in-the-loop for the middle, plus a separate boosted
       ensemble for risk scoring; specify rule export, stability monitoring, and
       fairness checks.
    9. *Diagnose:* a published decision-tree rulebook changes drastically every
       monthly retrain, alarming stakeholders. Explain the cause from first
       principles and propose three fixes (constraints, ensembling, monitoring).
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    A decision tree learns a flowchart: greedily split feature space on the
    (feature, threshold) that most reduces **impurity** (Gini/entropy/variance),
    recurse, and predict the majority/mean in each leaf box. It captures nonlinearity
    and interactions for free, needs no scaling, and is interpretable when shallow —
    but it **overfits** and is **high-variance/unstable**. That variance is not a
    footnote; it is the entire reason the next two notebooks exist.

    **Next:** `07 · Random Forest` — take this exact tree, train hundreds of them on
    bootstrap samples with randomized features, and **average** them. Variance
    collapses, accuracy jumps, and we get the first truly production-grade tabular
    model.
    """),
]

build("phase1_classical_ml/06_decision_trees.ipynb", cells)
