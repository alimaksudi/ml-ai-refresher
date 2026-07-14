"""Builder for Notebook 07 — Random Forest.

Run:  python3 tools/builders/phase1_07_random_forest.py
Emits: notebooks/phase1_classical_ml/07_random_forest.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # 07 · Random Forest
    ### Phase 1 — Classical Machine Learning · *ML/AI Senior Mastery Curriculum*

    > Notebook 06 ended on a problem: a single decision tree is **low-bias but
    > high-variance** — resample the data and you get a wildly different tree. A
    > Random Forest is the fix, and it's almost embarrassingly simple: **train
    > hundreds of decorrelated trees and average them.** The averaging cancels the
    > variance while preserving the low bias, turning the shakiest classifier in
    > Phase 1 into one of the most reliable, lowest-maintenance models in all of
    > tabular ML. This notebook is really about one deep idea — the **bias–variance
    > arithmetic of ensembles** — which also explains why *boosting* (08) attacks the
    > opposite term.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Bagging** (bootstrap aggregating) and the extra trick that makes it a
      *forest*: **random feature subsets at every split**.
    - The variance arithmetic: averaging $M$ estimators with correlation $\rho$ gives
      variance $\rho\sigma^2+\frac{1-\rho}{M}\sigma^2$ — so *decorrelation* (lowering
      $\rho$), not just more trees, is the lever.
    - **Out-of-bag (OOB) error**: a free cross-validation estimate baked into
      training.
    - **Feature importance** (impurity-based vs permutation) and its traps.
    - Why a forest **reduces variance but not bias** — and why that points straight
      to boosting.
    - Building a forest **from scratch** on top of a CART tree.

    **Why it matters in industry**
    - A superb **default model** for tabular data: strong accuracy, minimal tuning,
      no scaling, robust to outliers and noise, **embarrassingly parallel**.
    - **OOB** gives validation for free; **importances** give a first read on what
      drives predictions.
    - The conceptual partner to XGBoost (08); knowing *bagging vs boosting* is a
      staple senior interview distinction.

    **Typical interview questions**
    - "How does a Random Forest reduce variance? Why bootstrap *and* feature
      randomness?"
    - "What is OOB error and why is it useful?"
    - "Bagging vs boosting — what's the fundamental difference?"
    - "Why doesn't adding more trees overfit?"
    - "What are the pitfalls of impurity-based feature importance?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **The variance problem (recap of Notebook 06).** A fully-grown tree fits the
    training data almost perfectly (low bias) but its structure is unstable: small
    data changes cascade into very different trees (high variance). High-variance
    models generalize poorly.

    **Bagging (Breiman, 1996).** Breiman's insight from statistics: if you average
    many independent estimates of the same quantity, the average has *lower variance*
    than any single estimate. So: draw $M$ **bootstrap samples** (sample $n$ rows
    *with replacement*), fit a tree to each, and average their predictions. Variance
    drops; bias stays roughly the same.

    **The correlation ceiling, and Random Forests (Breiman, 2001; Ho's random
    subspaces, 1998).** Bagging alone has a limit: the trees are **correlated**
    because they see overlapping data and tend to pick the same strong features
    first. Averaging correlated estimators only helps so much. The Random Forest adds
    a second source of randomness — **at each split, consider only a random subset of
    features** (typically $\sqrt{d}$ for classification). This *forces* trees to be
    different, lowering their correlation $\rho$ and thus the variance floor.

    **Why it became the default.** Random Forests need almost no tuning, no feature
    scaling, handle nonlinearity/interactions/mixed types natively, resist
    overfitting (more trees never hurts test error, only compute), give free OOB
    validation, and parallelize trivially. For two decades they were *the* go-to
    tabular model — until gradient boosting (08) edged them out on raw accuracy by
    attacking **bias** instead of variance. The two are complementary, and the
    contrast is the whole point.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Wisdom of crowds.** Ask one slightly-overconfident expert (a deep tree) and you
    get a sharp but unreliable opinion. Ask **hundreds of diverse experts and take
    the majority vote**, and the idiosyncratic errors cancel while the shared signal
    survives. The catch: the crowd must be **diverse** — a thousand experts who all
    think alike are no better than one. That diversity is exactly what bootstrap
    sampling + random feature subsets manufacture.

    **Two dials of randomness:**
    1. **Bootstrap rows** — each tree sees a different ~63% of the data (the rest are
       its *out-of-bag* set, useful for free validation).
    2. **Random features per split** — each split chooses from a random handful of
       features, so no single dominant feature makes every tree look the same.

    **What averaging does to the boundary.** A single tree's decision boundary is a
    jagged staircase that fences off noise. Average many such staircases and the
    boundary becomes **smooth and stable** — the noise-driven wiggles disagree and
    wash out, the real structure agrees and remains.

    ```mermaid
    flowchart TD
        D["Training data"] --> B1["bootstrap 1 + random features"] --> T1["Tree 1"]
        D --> B2["bootstrap 2 + random features"] --> T2["Tree 2"]
        D --> Bd["...M bootstraps..."] --> Tm["Tree M"]
        T1 --> V["Average / majority vote"]
        T2 --> V
        Tm --> V
        V --> P["Low-variance prediction"]
    ```

    Run the cells: build a forest from scratch and watch variance collapse as trees
    are added.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # Reuse the circular-boundary dataset from Notebook 06 (a single tree overfits it).
    def make_circle_data(n=600, seed=0):
        r = np.random.default_rng(seed)
        X = r.uniform(-3, 3, (n, 2))
        y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 4.0).astype(int)
        flip = r.random(n) < 0.08
        y[flip] = 1 - y[flip]
        return X, y

    X, y = make_circle_data()
    Xtr, ytr, Xte, yte = X[:400], y[:400], X[400:], y[400:]
    print("train:", Xtr.shape, "test:", Xte.shape, "| balance:", np.bincount(y))
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 The variance of an average — the heart of the matter
    Suppose each tree gives an estimate with variance $\sigma^2$, and any two trees
    have correlation $\rho$. The variance of the average of $M$ such trees is
    $$\operatorname{Var}\!\Big(\tfrac1M\sum_{m}T_m\Big)=\rho\,\sigma^2+\frac{1-\rho}{M}\,\sigma^2.$$
    Read this carefully — it explains the entire algorithm:
    - As $M\to\infty$, the second term vanishes and variance $\to\rho\sigma^2$. **More
      trees help, but only down to a floor set by the correlation $\rho$.**
    - To push the floor lower you must **reduce $\rho$** — make the trees more
      different. That is precisely what the **random feature subset** per split does.
    - Bias is unchanged by averaging unbiased-ish trees, so the forest keeps the
      single tree's low bias while slashing variance. **Forests reduce variance, not
      bias** — remember this; it's why boosting (which reduces bias) is the
      complement.

    ### 4.2 Bagging (bootstrap aggregating)
    A **bootstrap sample** draws $n$ rows from the training set *with replacement*.
    The probability a given row is *omitted* from one bootstrap is
    $(1-\tfrac1n)^n\to e^{-1}\approx0.368$, so each tree is trained on ~63% of the
    data and ~37% are its **out-of-bag (OOB)** rows. Classification aggregates by
    **majority vote**; regression by **mean**.

    ### 4.3 The "random" in Random Forest
    At **every split**, restrict the candidate features to a random subset of size
    $m_{\text{try}}$ (defaults: $\sqrt{d}$ for classification, $d/3$ for regression).
    Lower $m_{\text{try}}$ → more decorrelation (lower $\rho$, lower variance floor)
    but each tree is individually weaker (higher bias). $m_{\text{try}}$ is therefore
    the **main tuning knob**, trading the two terms in §4.1.

    ### 4.4 Out-of-bag error — free cross-validation
    Predict each training row using **only the trees for which it was OOB** (never
    trained on it), then score. This is an almost-unbiased estimate of test error
    **without a held-out set or extra training** — a genuinely useful freebie.

    ### 4.5 Feature importance (and its traps)
    - **Impurity (Gini) importance:** total impurity decrease attributed to each
      feature, averaged over trees. Fast but **biased toward high-cardinality and
      continuous features** (more split points = more chances to look useful).
    - **Permutation importance:** shuffle one feature's values and measure the drop
      in accuracy. Slower but model-agnostic and more trustworthy; still misleading
      under correlated features. (Notebook 13 develops SHAP as the principled
      alternative.)
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    We build a compact CART tree that supports **random feature subsets per split**,
    then a forest that **bootstraps** rows, fits a tree to each, votes, and computes
    **OOB** error. This is a complete Random Forest in ~50 lines of NumPy.
    """),

    code(r"""
    # 5.1 A compact CART tree with random feature subsets at each split.
    def gini(y):
        if len(y) == 0:
            return 0.0
        _, c = np.unique(y, return_counts=True)
        p = c / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(X, y, feat_idx):
        n = len(y); parent = gini(y); best_gain, best = 0.0, None
        for f in feat_idx:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            for t in (vals[:-1] + vals[1:]) / 2:
                left = X[:, f] <= t
                nl, nr = left.sum(), n - left.sum()
                if nl == 0 or nr == 0:
                    continue
                gain = parent - (nl * gini(y[left]) + nr * gini(y[~left])) / n
                if gain > best_gain:
                    best_gain, best = gain, (f, t)
        return best

    def _build(X, y, depth, max_depth, min_samples, m_try, rng):
        if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:
            return {"leaf": True, "pred": int(round(y.mean()))}
        d = X.shape[1]
        feat_idx = rng.choice(d, size=min(m_try, d), replace=False)   # RANDOM features
        split = _best_split(X, y, feat_idx)
        if split is None:
            return {"leaf": True, "pred": int(round(y.mean()))}
        f, t = split; left = X[:, f] <= t
        return {"leaf": False, "feature": f, "threshold": t,
                "left": _build(X[left], y[left], depth + 1, max_depth, min_samples, m_try, rng),
                "right": _build(X[~left], y[~left], depth + 1, max_depth, min_samples, m_try, rng)}

    def tree_predict(node, X):
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            nd = node
            while not nd["leaf"]:
                nd = nd["left"] if x[nd["feature"]] <= nd["threshold"] else nd["right"]
            out[i] = nd["pred"]
        return out
    """),

    code(r"""
    # 5.2 The forest: bootstrap rows + per-split feature randomness + voting + OOB.
    class RandomForestScratch:
        def __init__(self, n_trees=40, max_depth=8, min_samples=2, m_try=None, seed=0):
            self.n_trees, self.max_depth = n_trees, max_depth
            self.min_samples, self.m_try, self.seed = min_samples, m_try, seed

        def fit(self, X, y):
            n, d = X.shape
            self.m_try = self.m_try or max(1, int(np.sqrt(d)))   # sqrt(d) default
            rng = np.random.default_rng(self.seed)
            self.trees, self.oob_masks = [], []
            for _ in range(self.n_trees):
                idx = rng.integers(0, n, n)                       # bootstrap (with replacement)
                oob = np.ones(n, bool); oob[np.unique(idx)] = False
                self.trees.append(_build(X[idx], y[idx], 0, self.max_depth,
                                         self.min_samples, self.m_try, rng))
                self.oob_masks.append(oob)
            return self

        def predict_proba_stagewise(self, X):
            # cumulative vote fraction after each added tree -> lets us watch variance fall
            votes = np.zeros(len(X)); preds = []
            for m, tree in enumerate(self.trees, 1):
                votes += tree_predict(tree, X)
                preds.append(votes / m)
            return preds                                          # list of P(1) arrays

        def predict(self, X):
            votes = sum(tree_predict(t, X) for t in self.trees)
            return (votes / self.n_trees >= 0.5).astype(int)

        def oob_score(self, X, y):
            n = len(y); vote = np.zeros(n); count = np.zeros(n)
            for tree, oob in zip(self.trees, self.oob_masks):
                p = tree_predict(tree, X[oob])
                vote[oob] += p; count[oob] += 1
            valid = count > 0
            pred = (vote[valid] / count[valid] >= 0.5).astype(int)
            return np.mean(pred == y[valid])

    rf = RandomForestScratch(n_trees=40, max_depth=8).fit(Xtr, ytr)
    print(f"forest test accuracy : {np.mean(rf.predict(Xte) == yte):.3f}")
    print(f"OOB accuracy (free!) : {rf.oob_score(Xtr, ytr):.3f}")

    # single deep tree for comparison (m_try = all features, one tree)
    single = _build(Xtr, ytr, 0, 8, 2, Xtr.shape[1], np.random.default_rng(1))
    print(f"single tree test acc : {np.mean(tree_predict(single, Xte) == yte):.3f}")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    The three figures that *prove* the variance arithmetic: a smoother boundary,
    test error falling and plateauing with more trees, and OOB tracking test error.
    """),

    code(r"""
    # Figure 1 — single jagged tree vs smooth averaged forest boundary.
    def plot_boundary(ax, predict_fn, title):
        xx, yy = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
        Z = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.4)
        ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap="RdBu", edgecolor="k", s=8)
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(2 * np.cos(th), 2 * np.sin(th), "g--", lw=2)
        ax.set_title(title); ax.set_aspect("equal")

    small_forest = RandomForestScratch(n_trees=25, max_depth=8, seed=3).fit(Xtr, ytr)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_boundary(axes[0], lambda G: tree_predict(single, G), "Single deep tree (jagged)")
    plot_boundary(axes[1], small_forest.predict, "Forest of 25 trees (smooth)")
    plt.suptitle("Figure 1 — Averaging decorrelated trees smooths the boundary")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** Same data, same true circle (green). The **single tree** (left)
    fences off noisy points with a jagged staircase — high variance. The **forest**
    (right) is visibly smoother and hugs the true circle better: each tree's
    noise-driven steps point in different directions and cancel under averaging,
    while the shared circular signal reinforces. We didn't reduce bias (still
    axis-aligned steps), we reduced *variance* — exactly §4.1.
    """),

    code(r"""
    # Figure 2 — test error falls and PLATEAUS as trees are added (it does not overfit).
    stage_probs = rf.predict_proba_stagewise(Xte)
    test_err = [np.mean((p >= 0.5).astype(int) != yte) for p in stage_probs]

    fig, ax = plt.subplots()
    ax.plot(range(1, len(test_err) + 1), test_err, color="tab:blue")
    ax.axhline(test_err[-1], color="r", ls="--", alpha=0.6, label="plateau")
    ax.set_xlabel("number of trees"); ax.set_ylabel("test error")
    ax.set_title("Figure 2 — More trees -> lower variance -> error plateaus (never rises)")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 2.** Test error drops quickly and then **flattens** — adding trees can
    only reduce the $\frac{1-\rho}{M}\sigma^2$ term toward the $\rho\sigma^2$ floor;
    it never *increases* error. This is the crucial practical property: **you cannot
    overfit a Random Forest by adding trees** (unlike boosting, where too many rounds
    *does* overfit — Notebook 08). `n_estimators` is a compute/accuracy tradeoff, not
    a regularization risk: use as many as your latency budget allows.
    """),

    code(r"""
    # Figure 3 — OOB accuracy tracks test accuracy, and m_try (decorrelation) matters.
    mtrys = [1, 2]                              # only 2 features here, so 1 = sqrt(2)-ish
    results = {}
    for mt in mtrys:
        f = RandomForestScratch(n_trees=40, max_depth=8, m_try=mt, seed=5).fit(Xtr, ytr)
        results[mt] = (f.oob_score(Xtr, ytr), np.mean(f.predict(Xte) == yte))

    fig, ax = plt.subplots()
    labels = [f"m_try={mt}" for mt in mtrys]
    oob = [results[mt][0] for mt in mtrys]; test = [results[mt][1] for mt in mtrys]
    xpos = np.arange(len(mtrys))
    ax.bar(xpos - 0.2, oob, 0.4, label="OOB accuracy")
    ax.bar(xpos + 0.2, test, 0.4, label="test accuracy")
    ax.set_xticks(xpos); ax.set_xticklabels(labels); ax.set_ylim(0.8, 1.0)
    ax.set_title("Figure 3 — OOB ~ test accuracy; m_try controls the variance floor")
    ax.legend()
    plt.show()
    for mt in mtrys:
        print(f"m_try={mt}: OOB={results[mt][0]:.3f}, test={results[mt][1]:.3f}")
    """),

    md(r"""
    **Figure 3.** Two things at once. First, **OOB accuracy closely matches test
    accuracy** — confirming OOB as a free, honest validation estimate (§4.4); you can
    tune a forest with no separate validation split. Second, **`m_try`** (features
    considered per split) shifts results by changing tree correlation $\rho$: smaller
    `m_try` decorrelates more (lower variance) but weakens individual trees (higher
    bias). With only 2 features the effect is mild; on wide datasets it's the main
    knob. In sklearn the default $\sqrt{d}$ is a strong starting point.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Can't extrapolate** | Flat predictions outside training range | Leaves are constants (inherited from trees) | Use a model with trend (linear) for extrapolation |
    | **Biased importances** | High-cardinality/continuous features look "important" | Impurity importance favors many-split features | Use **permutation importance** or SHAP (Ntbk 13) |
    | **Correlated trees** | Variance barely drops with more trees | `m_try` too high → trees too similar (high $\rho$) | Lower `max_features`; ensure enough features |
    | **Memory / latency** | Large model, slow inference | Hundreds of deep trees stored & traversed | Limit depth/#trees; quantize; or use boosting |
    | **Bias unchanged** | Underfits a hard pattern no tree captures | Averaging fixes variance, **not bias** | Boosting (08); better features |
    | **Imbalance** | Minority class under-predicted | Majority dominates votes/impurity | `class_weight="balanced"`, balanced bootstrap, good metrics (Ntbk 12) |

    The cell makes the **importance bias** concrete: a *pure-noise* high-cardinality
    feature can rank as "important" under impurity importance.
    """),

    code(r"""
    from sklearn.ensemble import RandomForestClassifier

    # Build data: 3 useful features + 1 high-cardinality NOISE feature (random continuous).
    n = 800
    Xg = rng.normal(size=(n, 3))
    yg = (Xg[:, 0] + Xg[:, 1] - Xg[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    noise = rng.normal(size=(n, 1)) * 100          # irrelevant, high-variance continuous
    Xall = np.hstack([Xg, noise])

    rf_imp = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xall, yg)
    print("impurity importances:", rf_imp.feature_importances_.round(3),
          "  (index 3 is PURE NOISE)")
    print("-> the noise feature grabs nonzero importance purely from its many split points.")
    print("Use permutation importance on held-out data to expose it as worthless.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    sklearn's `RandomForestClassifier`/`Regressor` add C-speed trees, `n_jobs=-1`
    parallelism (trees are independent → linear speedup), `oob_score=True`,
    `class_weight`, and both importance flavors. What it buys over our scratch
    forest: orders-of-magnitude speed, parallel training/inference, and tested
    numerics. We confirm it matches our scratch accuracy and reproduce OOB.
    """),

    code(r"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score

    skrf = RandomForestClassifier(n_estimators=200, max_depth=8, oob_score=True,
                                  n_jobs=-1, random_state=0).fit(Xtr, ytr)
    print(f"sklearn forest test acc : {accuracy_score(yte, skrf.predict(Xte)):.3f}")
    print(f"sklearn OOB score       : {skrf.oob_score_:.3f}")
    print(f"scratch forest test acc : {np.mean(rf.predict(Xte) == yte):.3f}")

    # Permutation importance (trustworthy) on the noisy dataset from Section 7:
    skrf2 = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xall, yg)
    perm = permutation_importance(skrf2, Xall, yg, n_repeats=10, random_state=0)
    print("\\npermutation importances:", perm.importances_mean.round(3),
          "  (index 3 noise ~ 0, as it should be)")
    """),

    md(r"""
    **Scratch vs production.** Our forest and sklearn's reach essentially the same
    accuracy and OOB — the algorithm is identical; sklearn parallelizes across cores
    and runs trees in C. Note the payoff of **permutation importance**: the
    high-cardinality noise feature that impurity importance flagged (§7) now correctly
    scores ~0. In production, prefer permutation importance or SHAP (Notebook 13) for
    any decision that depends on "which features matter."
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Customer Churn Prediction

    **Scenario.** A subscription business predicts which customers will **churn next
    month** from usage, billing, tenure, and support-ticket features, to target
    retention offers.

    **Why a Random Forest fits the first production model:**
    - **Strong out-of-the-box accuracy** on heterogeneous tabular features with
      **minimal tuning** and **no scaling** — fast path to a useful baseline.
    - **OOB error** gives an immediate, leakage-resistant performance read without
      carving out a validation set.
    - **Feature importance** offers a first-pass answer to "what drives churn?" for
      the retention team (validated with permutation importance / SHAP).
    - **Robust** to outliers and noisy columns that would derail a linear model.

    **Business objectives:** rank customers by churn risk so limited retention budget
    targets the highest-risk, highest-value accounts.

    **Cost of mistakes**
    - **Miss a churner** (false negative) → lost lifetime value.
    - **Flag a loyal customer** (false positive) → wasted incentive + possible
      annoyance.
    Because budget is limited, the real objective is **ranking quality** (precision
    among the top-k contacted), tuned via the threshold — not raw accuracy
    (Notebooks 09, 12).

    **Constraints:** must score the full base nightly (throughput); importances must
    be defensible to stakeholders; model retrained as behavior shifts.

    **KPIs:** ROC/PR-AUC, precision@k (k = retention capacity), realized churn within
    the contacted vs control group (a proper A/B test, Notebook 02), and importance
    stability over time.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Parallelism & throughput.** Trees are independent — training and inference
      parallelize linearly (`n_jobs=-1`). Great for batch scoring of a large base.
    - **Latency / memory.** A forest is hundreds of trees; per-prediction latency and
      model size scale with `n_estimators × depth`. For tight latency, cap depth/#trees,
      or prefer a boosted model that hits similar accuracy with fewer, shallower trees.
    - **No overfitting from more trees.** Set `n_estimators` as high as your
      compute/latency budget allows; it only reduces variance (Fig 2). Real tuning is
      `max_features`, `max_depth`, `min_samples_leaf`.
    - **OOB for cheap monitoring.** Track OOB score across retrains as a stability
      signal; large drops flag data issues.
    - **Importance drift.** Monitor feature importances over time — a sudden reshuffle
      often means a broken feature pipeline or genuine concept drift (Notebook 45).
    - **Explainability.** No coefficients; use **SHAP** (Notebook 13) for per-prediction
      explanations in regulated/decisioning contexts.
    - **Retraining.** Cheap and parallel; pin seeds so importances and scores are
      reproducible.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Single tree vs Random Forest vs Gradient Boosting:**

    | Dimension | Single Tree | Random Forest | Gradient Boosting (XGBoost) |
    |---|---|---|---|
    | Accuracy (tabular) | Moderate | High | **Highest (usually)** |
    | Reduces… | — | **Variance** | **Bias** |
    | Overfit risk from more trees | n/a | **None** (plateaus) | **Yes** (needs early stopping) |
    | Tuning effort | Low | **Low** | Higher |
    | Training | Fast | **Parallel** | Sequential |
    | Interpretability | High (shallow) | Low (SHAP) | Low (SHAP) |
    | Latency / size | Lowest | Higher (many trees) | Medium |
    | Robust to noise/outliers | Moderate | **High** | Moderate |

    **Bagging vs Boosting — the fundamental contrast (memorize this):**

    | | Bagging (Random Forest) | Boosting (Notebook 08) |
    |---|---|---|
    | Trees trained | Independently, in **parallel** | Sequentially, each fixes the last |
    | Attacks | **Variance** | **Bias** |
    | Base learner | Deep (low-bias, high-var) trees | Shallow (high-bias) "stumps" |
    | More trees | Safe (plateaus) | Can **overfit** |
    | Tuning | Light | Heavier (LR, depth, rounds) |

    **Senior lesson:** Random Forest is the **robust default** — pick it when you want
    strong accuracy with minimal effort and low overfitting risk; reach for boosting
    when you need to squeeze out the last few points and can afford the tuning.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *How does a forest reduce variance?* → Average decorrelated trees; cite
      $\rho\sigma^2+\frac{1-\rho}{M}\sigma^2$ (Section 4.1).
    - *Why bootstrap AND random features?* → Bootstrap alone leaves trees correlated;
      feature subsets lower $\rho$, the variance floor.

    **Deep-dive questions**
    - *What is OOB error and why useful?* → Score each row with the trees that didn't
      train on it; free, near-unbiased validation (Section 4.4).
    - *Why doesn't adding trees overfit?* → Averaging only shrinks the variance term
      toward a floor; it can't increase error (Fig 2). Contrast with boosting.
    - *Pitfalls of feature importance?* → Impurity importance is biased to
      high-cardinality features; use permutation/SHAP (Section 4.5, §7).

    **Whiteboard questions**
    - "Implement bagging with feature subsampling." (Section 5.)
    - "Derive the variance of an average of M correlated estimators." (Section 4.1.)

    **Strong vs weak answers**
    - *"How many trees should you use?"*
      - **Weak:** "More overfits, so be careful."
      - **Strong:** "More trees never increase test error for a forest — it only
        lowers variance toward a floor — so I'd use as many as latency allows and
        tune `max_features`/`max_depth` instead. (That 'more overfits' intuition is
        true for *boosting*, not bagging.)"
    - *"Random Forest or XGBoost?"*
      - **Weak:** "XGBoost, it wins Kaggle."
      - **Strong:** "RF for a robust, low-tuning baseline that won't overfit; boosting
        when I need maximum accuracy and can invest in tuning + early stopping. They
        attack different error terms — variance vs bias."

    **Follow-ups:** "Trees correlated despite bagging — why and fix?" (`max_features`
    too high). "Regression forest aggregation?" (mean). "Why no scaling?" (split
    order-invariance, inherited from trees).

    **Common mistakes:** thinking more trees overfit a forest; trusting impurity
    importance blindly; confusing bagging with boosting; forgetting forests don't
    extrapolate; ignoring that averaging doesn't fix bias.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define a Random Forest in terms of bagging + feature randomness.
    2. **Why was it invented?** What single weakness of decision trees does it cure,
       and how?
    3. **How does it work?** Walk bootstrap → random-feature splits → vote/average.
    4. **Why does it work?** Use the correlated-average variance formula to explain
       why decorrelation matters as much as the number of trees.
    5. **When to use it?** When is a forest the right default model?
    6. **When NOT to use it?** Name two situations (extrapolation, ultra-low latency)
       where you'd pick something else.
    7. **Tradeoffs?** Forest vs single tree vs boosting; bagging vs boosting.
    8. **How would you productionize it?** OOB monitoring, importance drift,
       parallelism, and explainability.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Compute the OOB fraction $(1-1/n)^n$ for $n=10,100,1000$ and explain why it
       converges to $\approx0.368$.
    2. Using the variance formula, explain why a forest of correlated trees
       ($\rho=0.9$) barely improves on a single tree no matter how many you add.

    **Beginner → Intermediate (coding)**
    3. Add **probability outputs** (vote fraction) to the scratch forest and plot a
       calibration curve; is the forest well-calibrated?
    4. Sweep `m_try` over a wider dataset (≥10 features) and plot OOB error vs
       `m_try`; locate the variance/bias sweet spot.

    **Intermediate (analysis)**
    5. Empirically estimate tree-pair **correlation** $\rho$ for bagging-only vs
       full Random Forest and connect the numbers to the variance formula.
    6. Reproduce the **importance-bias** experiment (§7) and show permutation
       importance fixes it; then break permutation importance with two *correlated*
       informative features and explain why.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive $\operatorname{Var}(\frac1M\sum T_m)=\rho\sigma^2+\frac{1-\rho}{M}\sigma^2$
       from the definition of variance of a sum, and interpret both terms.
    8. *Design:* build the churn system of §9 end-to-end — nightly batch scoring,
       precision@k targeting under a fixed retention budget, OOB-based monitoring,
       importance-drift alerts, and a retention A/B test to prove causal lift.
    9. *Diagnose:* a forest's OOB accuracy is excellent but live accuracy is poor.
       List the top three causes (leakage, drift, train/serve skew) and how you'd
       confirm each.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    A Random Forest = **bagging** (bootstrap rows) + **random feature subsets per
    split**, aggregated by vote/mean. The correlated-average variance formula
    $\rho\sigma^2+\frac{1-\rho}{M}\sigma^2$ explains everything: more trees shrink the
    second term to a floor set by tree correlation $\rho$, and the random features
    lower $\rho$. It **reduces variance, not bias**, gives **OOB** validation for
    free, can't be overfit by adding trees, needs little tuning, and parallelizes
    trivially — the robust default for tabular ML.

    **Next:** `08 · Gradient Boosting and XGBoost` — the complementary ensemble.
    Instead of averaging independent low-bias trees to cut *variance*, we add shallow
    trees **sequentially**, each correcting the previous ensemble's errors, to cut
    *bias* — the technique that tends to win tabular competitions outright.
    """),
]

build("phase1_classical_ml/07_random_forest.ipynb", cells)
