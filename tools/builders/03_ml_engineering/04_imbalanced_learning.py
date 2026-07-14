"""Builder for Lesson MLE-04 — Imbalanced Learning.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # MLE-04 · Imbalanced Learning
    ### Section 03 — ML Engineering Foundations · *ML/AI Senior Mastery Curriculum*

    > The highest-value classification problems are almost all **imbalanced**: fraud
    > (<1%), disease (rare), churn, ad clicks, defaults, manufacturing defects. The
    > positive class — the one you actually care about — is a tiny minority, and a
    > model trained naively learns to ignore it (CML-02 and MLE-01). This notebook is
    > the toolkit for the rare-class problem: **resampling** (random + SMOTE),
    > **class weights / cost-sensitive learning**, and **threshold moving** — plus the
    > senior judgment about which actually helps, the leakage trap that invalidates
    > most beginner attempts, and the calibration damage resampling silently does.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - *Why* imbalance breaks naive training and accuracy (recap + deepen CML-02 and
      MLE-01), and the three families of fixes.
    - **Data-level**: random over/under-sampling and **SMOTE** (synthetic
      interpolation) — implemented from scratch.
    - **Algorithm-level**: **class weights / cost-sensitive loss** — derived and coded.
    - **Threshold moving**: the often-best, simplest fix on a well-calibrated model.
    - The two traps every senior must flag: **resampling before the split is leakage**,
      and **resampling distorts probability calibration**.
    - Choosing the right approach (and metric) for a given cost structure.

    **Why it matters in industry**
    - Imbalance is the *norm* in high-stakes ML; mishandling it ships a model that
      "looks accurate" and catches nothing (Lesson MLE-01's accuracy paradox).
    - The fixes interact subtly with metrics (Lesson MLE-01), validation (Lesson MLE-02),
      and calibration — getting the *combination* right is the senior skill.

    **Typical interview questions**
    - "Your fraud model predicts 'never fraud' — what do you do?"
    - "Oversampling vs undersampling vs SMOTE vs class weights — tradeoffs?"
    - "How does SMOTE work, and when does it fail?"
    - "Why must resampling happen *inside* cross-validation?"
    - "Does oversampling change your predicted probabilities?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **The naive failure.** A classifier minimizing average error (CML-02 and CML-05) on
    99%-negative data discovers the laziest possible solution: **predict negative
    always**. It's 99% accurate and 0% useful. The loss and the metric both reward
    ignoring the minority — so we must change one or both.

    **Three historical lines of attack.**
    1. **Re-balance the data (data-level).** The oldest idea: duplicate minority rows
       (random oversampling) or drop majority rows (random undersampling) until classes
       are balanced. **SMOTE** (Chawla et al., 2002) refined oversampling by
       *synthesizing* new minority points via interpolation instead of duplicating —
       reducing the overfitting that exact copies cause.
    2. **Re-weight the loss (algorithm-level).** Cost-sensitive learning makes a
       minority error cost more in the objective (`class_weight`,
       `scale_pos_weight`) — equivalent in spirit to oversampling but without
       touching the data.
    3. **Move the decision threshold (decision-level).** Keep the model and *its
       probabilities*, but choose a threshold below 0.5 to trade precision for recall
       (Lesson MLE-01). Increasingly, practitioners argue this is the **cleanest** fix:
       a well-trained, **calibrated** model already ranks correctly, and the imbalance
       "problem" is really a *threshold* problem.

    **Why the modern view matters.** A recurring senior insight (and interview
    discriminator): resampling and class weights mostly **shift the implicit
    threshold** while *damaging calibration*. So before reaching for SMOTE, ask: "Is my
    model's *ranking* (AUC/PR) actually bad, or is my *threshold* just wrong?" Often
    it's the latter — and then threshold moving on a calibrated model beats elaborate
    resampling.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The lazy-student analogy.** A student graded only on overall accuracy, facing an
    exam that's 99% easy questions and 1% hard ones, learns to ace the easy ones and
    skip the hard ones — 99%! To force attention on the hard 1%, you can: (a) **show
    more hard questions** (oversample/SMOTE), (b) **make hard questions worth more
    points** (class weights), or (c) **lower the bar for what counts as "answered"** on
    hard questions (threshold moving).

    **What each fix does geometrically:**
    - **Random oversampling** duplicates minority points — the boundary shifts toward
      the majority, but exact copies invite overfitting.
    - **SMOTE** creates *new* minority points along lines between existing minority
      neighbors — it fills in the minority region rather than stacking duplicates.
    - **Class weights** tell the loss that each minority example counts like many —
      the boundary moves without changing the data.
    - **Threshold moving** leaves the model untouched and simply says "predict positive
      when $p > t$" for some $t < 0.5$.

    **The two warnings that separate seniors from juniors:**
    1. **Resample only the *training* fold.** Balancing the whole dataset before
       splitting leaks minority information into the test fold (duplicates/synthetics
       of test points appear in train) — fake-good CV (Lesson MLE-02).
    2. **Resampling decalibrates.** Changing the class ratio changes the model's
       implied base rate, so its probabilities no longer match reality — you must
       recalibrate or correct the prior if the probability is consumed.

    ```mermaid
    flowchart TD
        I["Imbalanced data<br/>(rare positives)"] --> D["Data-level:<br/>over/under-sample, SMOTE"]
        I --> A["Algorithm-level:<br/>class weights / cost-sensitive"]
        I --> T["Decision-level:<br/>move threshold (MLE-01)"]
        D --> W1["⚠ resample TRAIN only · decalibrates"]
        A --> M["boundary shifts; calibration ~ok"]
        T --> M2["cleanest if model is calibrated"]
    ```

    Run the cells — first, watch a naive model ignore the rare class.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # 2D imbalanced data: 5% positives, overlapping with the majority.
    n = 2000
    n_pos = int(0.05 * n)
    Xneg = rng.normal([0, 0], 1.2, (n - n_pos, 2))
    Xpos = rng.normal([2.2, 2.2], 1.0, (n_pos, 2))
    X = np.vstack([Xneg, Xpos])
    y = np.r_[np.zeros(n - n_pos), np.ones(n_pos)]
    perm = rng.permutation(n); X, y = X[perm], y[perm]
    print(f"positives: {int(y.sum())}/{n}  ({y.mean():.1%}) -- a rare class")
    """),

    code(r"""
    # The naive baseline: optimize accuracy and the minority class vanishes.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, precision_score, accuracy_score

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    base = LogisticRegression().fit(Xtr, ytr)
    pred = base.predict(Xte)
    print(f"accuracy : {accuracy_score(yte, pred):.3f}  (looks great...)")
    print(f"recall   : {recall_score(yte, pred):.3f}  (...but catches few positives)")
    print(f"precision: {precision_score(yte, pred, zero_division=0):.3f}")
    print("\\nHigh accuracy, poor recall: the model leans heavily toward the majority.")
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Why imbalance hurts
    The training loss is an average over examples (Lesson CML-05). With 99% negatives,
    the gradient is dominated by the majority class, and the easiest way to reduce loss
    is to predict the majority. The Bayes-optimal decision at threshold 0.5 also
    favors the majority when the prior $P(y{=}1)$ is tiny. Both the **objective** and
    the **default threshold** are mis-aligned with "find the rare positives."

    ### 4.2 Cost-sensitive learning / class weights
    Attach a weight $w_c$ to each class and minimize the **weighted** loss
    $$J=-\frac1{\sum_i w_{y_i}}\sum_i w_{y_i}\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big].$$
    Setting $w_1/w_0 = n_0/n_1$ (inverse frequency) makes the two classes contribute
    equally — the gradient $\frac1N\sum w_{y_i}(p_i-y_i)x_i$ now "hears" the minority.
    This is **mathematically close to oversampling** the minority by the same ratio,
    but without duplicating data.

    ### 4.3 SMOTE — synthetic minority oversampling
    For a minority point $x_i$, pick one of its $k$ minority nearest neighbors $x_j$
    and create a synthetic point on the segment between them:
    $$x_{\text{new}} = x_i + \lambda\,(x_j - x_i),\qquad \lambda\sim U(0,1).$$
    This **fills the minority region** with plausible interpolated examples rather than
    stacking exact duplicates, reducing overfitting. *Assumptions:* the minority class
    is locally convex and features are continuous — interpolating categorical or
    high-dimensional sparse features produces nonsense (§7).

    ### 4.4 Threshold moving (recap of Lesson MLE-01)
    Keep the model; choose the threshold $t$ minimizing expected cost
    $C_{FP}\,FP(t)+C_{FN}\,FN(t)$, or hitting a recall/precision target. For a
    **calibrated** model this is often the best fix — it changes the *decision*, not
    the *probabilities*, so calibration is preserved.

    ### 4.5 The calibration consequence of resampling
    If you train on data resampled to a balanced ratio, the model learns
    $P_{\text{resampled}}(y{=}1\mid x)$, **not** the true $P(y{=}1\mid x)$. The
    predicted probabilities are systematically inflated for the minority. To recover
    true probabilities you must **correct the prior** or **recalibrate** (Platt/
    isotonic) on data with the *real* class ratio. Class weights distort calibration
    less than aggressive resampling; threshold moving not at all.

    ### 4.6 The validation rule (recap of Lesson MLE-02)
    **Resampling is part of model fitting**, so it must happen **inside** each CV fold,
    on the training portion only. Balancing the full dataset first leaks synthetic/
    duplicated minority points across the train/test boundary, producing optimistic and
    meaningless scores. Use an imblearn `Pipeline` (or manual in-fold resampling).
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    We implement random over/under-sampling, **SMOTE**, and a **class-weighted logistic
    regression** — the three core mechanisms — in pure NumPy.
    """),

    code(r"""
    # 5.1 Random oversampling, random undersampling, and SMOTE from scratch.
    def random_oversample(X, y, seed=0):
        r = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        extra = r.choice(pos, size=len(neg) - len(pos), replace=True)   # duplicate minority
        idx = np.concatenate([neg, pos, extra])
        return X[idx], y[idx]

    def random_undersample(X, y, seed=0):
        r = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        keep_neg = r.choice(neg, size=len(pos), replace=False)          # drop majority
        idx = np.concatenate([keep_neg, pos])
        return X[idx], y[idx]

    def smote(X, y, k=5, seed=0):
        r = np.random.default_rng(seed)
        Xmin = X[y == 1]
        n_needed = int((y == 0).sum() - (y == 1).sum())
        synth = np.empty((n_needed, X.shape[1]))
        for s in range(n_needed):
            i = r.integers(len(Xmin))
            d = np.linalg.norm(Xmin - Xmin[i], axis=1)
            nn = np.argsort(d)[1:k + 1]                                 # k nearest minority neighbors
            j = nn[r.integers(len(nn))]
            lam = r.random()
            synth[s] = Xmin[i] + lam * (Xmin[j] - Xmin[i])             # interpolate
        Xnew = np.vstack([X, synth])
        ynew = np.r_[y, np.ones(n_needed)]
        return Xnew, ynew

    for name, fn in [("oversample", random_oversample), ("undersample", random_undersample),
                     ("SMOTE", smote)]:
        Xr, yr = fn(Xtr, ytr)
        print(f"{name:12s}: {len(yr)} rows, balance {yr.mean():.2f}")
    """),

    code(r"""
    # 5.2 Class-weighted logistic regression (cost-sensitive) from scratch.
    def sigmoid(z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def fit_weighted_logistic(X, y, lr=0.1, steps=3000, class_weight=None):
        A = np.c_[np.ones(len(X)), X]
        w = np.zeros(A.shape[1])
        if class_weight is None:
            sw = np.ones(len(y))
        else:
            sw = np.where(y == 1, class_weight[1], class_weight[0])     # per-sample weights
        for _ in range(steps):
            p = sigmoid(A @ w)
            grad = A.T @ (sw * (p - y)) / sw.sum()                      # weighted gradient (Section 4.2)
            w -= lr * grad
        return w

    def predict_proba(X, w):
        return sigmoid(np.c_[np.ones(len(X)), X] @ w)

    # inverse-frequency weights make the minority "count" as much as the majority
    cw = {0: 1.0, 1: (ytr == 0).sum() / (ytr == 1).sum()}
    w_plain = fit_weighted_logistic(Xtr, ytr)
    w_weighted = fit_weighted_logistic(Xtr, ytr, class_weight=cw)
    from sklearn.metrics import recall_score, precision_score
    for name, w in [("plain", w_plain), ("class-weighted", w_weighted)]:
        pr = (predict_proba(Xte, w) >= 0.5).astype(int)
        print(f"{name:15s}: recall={recall_score(yte, pr):.3f}, "
              f"precision={precision_score(yte, pr, zero_division=0):.3f}")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Four pictures: SMOTE's synthetic points, the boundary shift from class weights,
    the recall/precision tradeoff across methods, and the **calibration damage** that
    resampling does.
    """),

    code(r"""
    # Figure 1 — SMOTE fills the minority region with interpolated points.
    Xsm, ysm = smote(Xtr, ytr)
    synth = Xsm[len(Xtr):]                                  # the newly created points
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(Xtr[ytr == 0][:, 0], Xtr[ytr == 0][:, 1], s=8, alpha=0.3, label="majority")
    ax.scatter(Xtr[ytr == 1][:, 0], Xtr[ytr == 1][:, 1], s=25, color="tab:red", label="minority (real)")
    ax.scatter(synth[:, 0], synth[:, 1], s=10, color="tab:green", alpha=0.5, label="SMOTE synthetic")
    ax.set_title("Figure 1 — SMOTE interpolates new minority points between neighbors")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 1.** SMOTE doesn't copy minority points — it creates **new** ones on the
    line segments between a minority point and its minority neighbors (green). This
    densifies the minority region so the classifier sees a fuller picture of it,
    avoiding the exact-duplicate overfitting of random oversampling. The flip side is
    visible too: where minority points are sparse or near the majority cloud, synthetic
    points can land in **majority territory**, manufacturing label noise (§7) — which
    is why SMOTE assumes a locally-clean, continuous minority region.
    """),

    code(r"""
    # Figure 2 — class weights move the decision boundary to recover the minority.
    def boundary(ax, predict_fn, title):
        xx, yy = np.meshgrid(np.linspace(-4, 6, 200), np.linspace(-4, 6, 200))
        Z = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.3)
        ax.scatter(Xtr[ytr == 0][:, 0], Xtr[ytr == 0][:, 1], s=6, alpha=0.3, color="tab:blue")
        ax.scatter(Xtr[ytr == 1][:, 0], Xtr[ytr == 1][:, 1], s=18, color="tab:red")
        ax.set_title(title); ax.set_aspect("equal")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    boundary(axes[0], lambda G: (predict_proba(G, w_plain) >= 0.5).astype(int),
             "Plain: boundary ignores the minority")
    boundary(axes[1], lambda G: (predict_proba(G, w_weighted) >= 0.5).astype(int),
             "Class-weighted: boundary protects the minority")
    plt.suptitle("Figure 2 — Class weights shift the boundary toward the rare class")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** The plain model (left) places its boundary to maximize overall
    accuracy, swallowing much of the (red) minority into the majority region. With
    **inverse-frequency class weights** (right), each minority error costs ~19× more,
    so the boundary moves to carve out the minority region — recall rises sharply at
    some cost to precision. Note: this looks a lot like simply *lowering the threshold*
    — which is the next figure's point.
    """),

    code(r"""
    # Figure 3 — compare strategies by precision/recall (evaluate on the ORIGINAL test set).
    from sklearn.metrics import precision_recall_fscore_support

    def evaluate(name, proba, thresh=0.5):
        pr = (proba >= thresh).astype(int)
        p, r, f, _ = precision_recall_fscore_support(yte, pr, average="binary", zero_division=0)
        return name, p, r, f

    results = [
        evaluate("baseline", predict_proba(Xte, w_plain)),
        evaluate("class weights", predict_proba(Xte, w_weighted)),
    ]
    # SMOTE-trained model
    w_smote = fit_weighted_logistic(*smote(Xtr, ytr))
    results.append(evaluate("SMOTE", predict_proba(Xte, w_smote)))
    # threshold moving on the PLAIN calibrated model (no resampling at all)
    results.append(evaluate("threshold=0.15", predict_proba(Xte, w_plain), thresh=0.15))

    labels = [r[0] for r in results]
    prec = [r[1] for r in results]; rec = [r[2] for r in results]; f1 = [r[3] for r in results]
    xp = np.arange(len(labels)); width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xp - width, prec, width, label="precision")
    ax.bar(xp, rec, width, label="recall")
    ax.bar(xp + width, f1, width, label="F1")
    ax.set_xticks(xp); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_title("Figure 3 — Strategies trade precision for recall (test set unchanged)")
    ax.legend()
    plt.show()
    for n, p, r, f in results:
        print(f"{n:16s}: P={p:.2f} R={r:.2f} F1={f:.2f}")
    """),

    md(r"""
    **Figure 3.** All three "imbalance fixes" do essentially the same thing — **buy
    recall at the cost of precision** — by effectively lowering the decision threshold.
    Strikingly, **threshold moving on the plain model** (no resampling, no reweighting)
    achieves a similar recall/precision trade as class weights and SMOTE, with less
    machinery and *without* harming calibration. This is the senior takeaway: before
    SMOTE-ing, check whether your model's **ranking** (PR-AUC) is actually poor or
    whether you just need a better **threshold** (Lesson MLE-01). Often it's the latter.
    """),

    code(r"""
    # Figure 4 — resampling DECALIBRATES: oversampled model overstates P(positive).
    def reliability(y, p, bins=8):
        edges = np.linspace(0, 1, bins + 1); xs, ys = [], []
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1])
            if m.sum() > 10:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        return np.array(xs), np.array(ys)

    fig, ax = plt.subplots()
    for w, name, c in [(w_plain, "plain (calibrated)", "tab:blue"),
                       (w_smote, "SMOTE-trained", "tab:red")]:
        p = predict_proba(Xte, w)
        xs, ys = reliability(yte, p)
        brier = np.mean((p - yte) ** 2)
        ax.plot(xs, ys, "o-", color=c, label=f"{name} (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Figure 4 — Resampling inflates probabilities (decalibration)")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 4.** The plain model (blue) tracks the diagonal — its probabilities are
    roughly honest. The **SMOTE-trained** model (red) bows **above** the diagonal: it
    was trained on a 50/50 world, so it systematically *overstates* the probability of
    the (truly rare) positive class, and its Brier score is worse. If you only need a
    *ranking* or a yes/no decision, this may be fine; but if the **probability is
    consumed** (pricing, expected-loss, risk thresholds), resampling demands a
    **recalibration** step on real-ratio data or a prior correction. Threshold moving
    avoids this entirely.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Resample before split** | CV ≫ live; "perfect" minority recall | Synthetic/dup minority points leak across folds | Resample **inside** the fold (imblearn Pipeline, MLE-02) |
    | **Decalibration** | Probabilities too high for minority | Trained on a balanced (fake) prior | Recalibrate / prior-correct; or use class weights / threshold |
    | **SMOTE on wrong data** | Noisy synthetic points; worse model | Categorical/sparse/high-dim or overlapping classes | SMOTENC/encodings; clean overlap; prefer class weights |
    | **Undersampling discards signal** | Higher variance, lost majority info | Threw away most of the data | Keep data; oversample/weight; or ensemble undersampling |
    | **Oversampling overfit** | Train ≫ test; memorized duplicates | Exact minority copies | SMOTE instead; regularize |
    | **Wrong metric** | "Improved accuracy," worse business outcome | Accuracy on imbalanced data | PR-AUC, recall@precision, cost (MLE-01) |
    | **Over-fixing imbalance** | Tons of false positives | Pushed recall with no cost basis | Set threshold/weights from the **cost matrix** |

    The cell shows the **leakage** failure — SMOTE applied before the split inflates CV.
    """),

    code(r"""
    # SMOTE before splitting leaks; SMOTE inside the fold is honest.
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    def cv_smote(leaky):
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        scores = []
        Xall, yall = (smote(X, y) if leaky else (X, y))      # LEAKY: resample everything first
        # For the honest case we resample only the train part of each fold below.
        for tr, te in skf.split(X, y):
            if leaky:
                # train on resampled-everything (test points' synthetics may be in train) -- LEAK
                Xt, yt = smote(X[tr], y[tr]) if False else (Xall, yall)  # illustrative
            if leaky:
                w = fit_weighted_logistic(Xall, yall, steps=1500)
            else:
                Xt, yt = smote(X[tr], y[tr])
                w = fit_weighted_logistic(Xt, yt, steps=1500)
            pr = (predict_proba(X[te], w) >= 0.5).astype(int)
            scores.append(f1_score(y[te], pr, zero_division=0))
        return np.mean(scores)

    # cleaner: explicit leaky vs honest
    def cv_honest():
        skf = StratifiedKFold(5, shuffle=True, random_state=0); s = []
        for tr, te in skf.split(X, y):
            Xt, yt = smote(X[tr], y[tr])                      # resample TRAIN ONLY
            w = fit_weighted_logistic(Xt, yt, steps=1500)
            s.append(f1_score(y[te], (predict_proba(X[te], w) >= 0.5).astype(int), zero_division=0))
        return np.mean(s)

    def cv_leaky():
        Xall, yall = smote(X, y)                              # resample BEFORE splitting -> leak
        skf = StratifiedKFold(5, shuffle=True, random_state=0); s = []
        for tr, te in skf.split(Xall, yall):
            w = fit_weighted_logistic(Xall[tr], yall[tr], steps=1500)
            s.append(f1_score(yall[te], (predict_proba(Xall[te], w) >= 0.5).astype(int), zero_division=0))
        return np.mean(s)

    print(f"LEAKY  (SMOTE before split) F1: {cv_leaky():.3f}   <- optimistic, fake")
    print(f"HONEST (SMOTE inside fold)  F1: {cv_honest():.3f}   <- trustworthy")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    The `imbalanced-learn` library provides SMOTE and friends plus an
    **imblearn `Pipeline`** that applies resampling **train-fold-only** automatically
    (the leak-safe pattern). sklearn offers `class_weight="balanced"` everywhere, and
    XGBoost has `scale_pos_weight`. The import is wrapped defensively; if absent, we
    fall back to `class_weight`, which needs no extra package.
    """),

    code(r"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import make_scorer, f1_score

    f1 = make_scorer(f1_score, zero_division=0)

    # sklearn class_weight='balanced' = inverse-frequency weights (Section 4.2), no extra lib
    plain_cv = cross_val_score(LogisticRegression(max_iter=1000),
                               X, y, cv=StratifiedKFold(5), scoring=f1).mean()
    bal_cv = cross_val_score(LogisticRegression(max_iter=1000, class_weight="balanced"),
                             X, y, cv=StratifiedKFold(5), scoring=f1).mean()
    print(f"F1  plain: {plain_cv:.3f}   class_weight='balanced': {bal_cv:.3f}")

    # imbalanced-learn (optional) -- leak-safe Pipeline that resamples train folds only
    try:
        from imblearn.over_sampling import SMOTE as ImbSMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipe = ImbPipeline([("smote", ImbSMOTE(random_state=0)),
                            ("clf", LogisticRegression(max_iter=1000))])
        smote_cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring=f1).mean()
        print(f"imblearn SMOTE-in-pipeline F1: {smote_cv:.3f} (resampling is train-fold-only)")
    except Exception as e:
        print(f"[imbalanced-learn not installed: {type(e).__name__}] "
              f"class_weight='balanced' above is the no-dependency alternative.")
    """),

    md(r"""
    **Scratch vs production.** Our hand-written SMOTE and weighted-logistic taught the
    mechanics; `imbalanced-learn`'s **`Pipeline`** is what you ship, because it
    guarantees resampling happens *inside* each CV fold (and not at serving) — the
    single most important correctness property (§7). `class_weight="balanced"` is the
    zero-dependency option that often matches SMOTE and preserves calibration better;
    `scale_pos_weight` is XGBoost's equivalent. Reach for SMOTE only when class weights/
    threshold moving underperform *and* your features are continuous and clean.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Rare-Disease Screening

    **Scenario.** A screening model flags patients for a disease present in ~0.5% of
    the screened population. Missing a case (FN) is far worse than a false alarm (FP),
    which triggers a (costly, stressful, but survivable) follow-up test.

    **Why this is the textbook imbalance problem:**
    - **Accuracy is meaningless** — "all healthy" scores 99.5% (Lesson MLE-01).
    - The objective is **recall at an acceptable false-positive rate**, set by the
      enormous FN:FP cost asymmetry.
    - The output may feed clinical risk discussions, so **calibrated probabilities**
      matter — which constrains how aggressively we resample.

    **Approach (senior playbook):**
    1. Train a strong model with **`class_weight`** (preserves calibration better than
      heavy resampling); evaluate with **PR-AUC** and recall@FP-rate.
    2. **Move the threshold** to the cost-optimal point from the FN/FP matrix
      (Lesson MLE-01) — usually the biggest lever.
    3. Only if ranking is still poor, try **SMOTE inside the CV fold**, then
      **recalibrate** on real-ratio data.

    **Cost of mistakes:** FN = missed disease (catastrophic); FP = unnecessary
    follow-up (costly/anxiety). The ratio sets weights and threshold. **Over-fixing**
    imbalance (chasing recall blindly) floods clinicians with false positives.

    **Constraints:** clinical capacity caps the false-positive volume; outputs must be
    auditable and calibrated; rigorous, leak-free validation (Lesson MLE-02).

    **KPIs:** recall at fixed FP-rate (sensitivity at fixed specificity), PR-AUC,
    calibration error, and the realized catch-rate vs review burden in deployment.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Resample train-fold-only, always.** Use an imblearn `Pipeline`; never resample
      before the split or at serving (you don't resample live data). This is the #1
      correctness rule (§7, Lesson MLE-02).
    - **Mind calibration.** If the probability is consumed downstream, prefer class
      weights or threshold moving (calibration-preserving), or **recalibrate** a
      resampled model on real-ratio data (Lesson MLE-01).
    - **Threshold is the main production lever** and is cheap to re-tune as costs and
      **base rates drift** — monitor the positive rate over time; a shifting prior
      changes the optimal threshold even if the model is unchanged.
    - **Metric discipline.** Track PR-AUC / recall@precision / cost, never bare
      accuracy (Lesson MLE-01). Report with confidence intervals (few positives → noisy
      metrics; bootstrap, Lesson FND-02).
    - **Don't over-correct.** Balancing to 50/50 is rarely optimal; tune the
      sampling/weight ratio and threshold to the **business cost**, not to "balanced."
    - **Combine sparingly.** Stacking SMOTE + class weights + threshold moving triple-
      counts the minority and floods false positives; pick the minimal effective fix.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    | Approach | Mechanism | Pros | Cons | Calibration impact |
    |---|---|---|---|---|
    | **Random oversample** | Duplicate minority | Simple; keeps all data | Overfits duplicates | Distorted |
    | **Random undersample** | Drop majority | Fast; small data | **Discards signal**; variance | Distorted |
    | **SMOTE** | Interpolate minority | Less overfit than dup; fills region | Continuous-only; noisy near overlap | Distorted |
    | **Class weights** | Reweight loss | No data change; simple | Still shifts threshold implicitly | Mildly distorted |
    | **Threshold moving** | Move decision point | **Cleanest**; preserves probabilities | Needs a good/calibrated model | **None** |

    **Decision guide (senior):**
    - **First**: train a strong model + **move the threshold** from the cost matrix.
      Check if PR-AUC (ranking) is actually the problem.
    - **If ranking is weak**: try **class weights** (calibration-friendly).
    - **If still weak and features are continuous/clean**: **SMOTE inside CV**, then
      **recalibrate**.
    - **Undersample** only when the majority is huge and compute-bound (or in an
      ensemble like BalancedBagging).

    **Senior lesson:** most "imbalance" problems are really **threshold + metric**
    problems. Reach for the simplest decision-level fix before complicating the data
    pipeline — and never forget the leakage and calibration consequences.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Fraud model predicts 'never fraud' — fix it?* → It's optimizing accuracy on
      imbalance. Switch to PR metrics, move the threshold from the cost matrix, add
      class weights; resample only if ranking is poor.
    - *Oversample vs undersample vs SMOTE vs class weights?* → Section 11 table — and
      note they mostly shift the threshold.

    **Deep-dive questions**
    - *How does SMOTE work and when does it fail?* → Interpolate between minority
      neighbors; fails on categorical/sparse/overlapping data (Sections 4.3, 7).
    - *Why resample inside CV?* → Resampling before the split leaks minority info
      across folds → fake-good CV (Fig in §7, Lesson MLE-02).
    - *Does oversampling change probabilities?* → Yes — trains on a fake prior, inflates
      minority probability; recalibrate or prior-correct (Fig 4).

    **Whiteboard questions**
    - "Implement SMOTE." (Section 5.1.)
    - "Implement class-weighted logistic loss/gradient." (Section 5.2.)

    **Strong vs weak answers**
    - *"How do you handle 1% positives?"*
      - **Weak:** "SMOTE to balance it."
      - **Strong:** "First I'd check whether *ranking* (PR-AUC) is bad or just the
        *threshold*. I'd train with class weights, move the threshold from the FN/FP
        costs, and use PR-based metrics. SMOTE only if ranking is still weak — applied
        inside CV — and then recalibrate, since resampling distorts probabilities."
    - *"Your SMOTE model has great CV but fails live."*
      - **Weak:** "Needs more data."
      - **Strong:** "Almost certainly SMOTE was applied before the split — synthetic
        minority points leaked across folds. I'd move resampling inside an imblearn
        Pipeline and re-measure; the honest score will be lower."

    **Follow-ups:** "Categorical features — still SMOTE?" (SMOTENC / encode first, or
    class weights). "Probabilities feed pricing — which fix?" (class weights/threshold,
    or recalibrate). "How pick the resample ratio?" (tune to cost, not to 50/50).

    **Common mistakes:** resampling before the split; reporting accuracy; balancing to
    50/50 reflexively; ignoring calibration after resampling; SMOTE on categorical data;
    stacking every fix and flooding false positives.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Why does class imbalance break naive training and accuracy?
    2. **Why was it invented?** What gap did SMOTE fill over random oversampling?
    3. **How does it work?** Explain SMOTE, class weights, and threshold moving.
    4. **Why does it work?** Why do all three essentially trade precision for recall?
    5. **When to use it?** Order the fixes you'd try and justify the order.
    6. **When NOT to use it?** When does resampling hurt (calibration, SMOTE on
       categorical, leakage)?
    7. **Tradeoffs?** Resampling vs class weights vs threshold moving.
    8. **How would you productionize it?** In-fold resampling, recalibration, cost-
       based threshold, base-rate-drift monitoring.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. A dataset is 2% positive. State the accuracy of "always negative" and explain
       why recall and PR-AUC are the metrics to watch.
    2. Explain in two sentences why training on SMOTE-balanced data makes predicted
       probabilities too high.

    **Beginner → Intermediate (coding)**
    3. Show empirically that `class_weight='balanced'` with weight ratio $n_0/n_1$ gives
       similar recall to oversampling the minority by the same ratio.
    4. Implement **Borderline-SMOTE** (only synthesize from minority points near the
       boundary) and compare it to plain SMOTE on the §3 data.

    **Intermediate (analysis)**
    5. Reproduce the leakage demo (§7) and quantify how the optimistic gap grows as the
       minority gets rarer (5% → 1% → 0.5%).
    6. After SMOTE training, apply **isotonic recalibration** on a real-ratio holdout
       and show the reliability diagram (Fig 4) returns to the diagonal while recall is
       preserved.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive why class weighting by inverse frequency is approximately
       equivalent to oversampling by the same ratio (from the weighted-loss gradient).
    8. *Design:* build the rare-disease screening system of §9 — model + class weights,
       cost-based threshold, in-fold resampling, recalibration, PR-based monitoring,
       and base-rate-drift alerts.
    9. *Diagnose:* a teammate's SMOTE pipeline shows F1 0.9 offline but floods
       clinicians with false positives live. Identify the two most likely causes
       (resample-before-split leakage; decalibration + wrong threshold) and the fix.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Imbalanced learning is about forcing a model — and your metrics — to attend to a
    rare but important class. The three families are **data-level** (over/under-sample,
    **SMOTE**), **algorithm-level** (**class weights / cost-sensitive**), and
    **decision-level** (**threshold moving**), and they mostly do the same thing: trade
    precision for recall by shifting the effective threshold. The senior moves: start
    with a calibrated model + threshold from the **cost matrix**, prefer class weights
    over heavy resampling, **resample inside the CV fold** (never before — leakage,
    Lesson MLE-02), **recalibrate** after resampling, and judge with **PR-based metrics**
    (Lesson MLE-01), not accuracy.

    **Related lesson:** `MLE-05 · Explainability (SHAP)` — our tree ensembles (CML-04 and CML-05) and the models
    here are accurate but opaque. We close Section 03 by making any model's predictions
    *explainable* with a principled, game-theoretic method, essential for trust,
    debugging, and regulated decisions.
    """),
]

build("03_ml_engineering/04_imbalanced_learning.ipynb", cells)
