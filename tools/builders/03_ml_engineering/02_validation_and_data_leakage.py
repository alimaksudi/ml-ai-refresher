"""Builder for Lesson MLE-02 — Validation and Data Leakage.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # MLE-02 · Validation and Data Leakage
    ### Core ML Evidence — required before trees, ensembles, and tuning

    **Prerequisites:** FND-03, CML-01, CML-02, and MLE-01. You should understand the
    prediction unit, a holdout split, model fitting, and task-aligned metrics.

    > Lesson FND-03 introduced the first holdout and Lesson MLE-01 gave us task-aligned
    > *metrics*. This notebook makes the **process of
    > measuring** trustworthy — because the most common way to ship a broken model is
    > not a bad algorithm, it's a **leaky evaluation** that reports a number that was
    > never real. "Looked amazing offline, died in production" is, nine times out of
    > ten, **data leakage**. A senior engineer is professionally paranoid about
    > validation: they treat the test set as sacred, fit every transform inside the
    > fold, respect time and groups, and can name the half-dozen ways information
    > sneaks from the future or the answer into the features. Getting this wrong
    > invalidates *everything* downstream — so this is arguably the highest-leverage
    > notebook in Section 03.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The **train / validation / test** split and the distinct job of each — and why
      the test set is a *one-time* final exam.
    - **K-fold, stratified, group, and time-series cross-validation** — and exactly
      when each is mandatory.
    - The **leakage taxonomy**: target leakage, train–test contamination,
      **preprocessing leakage**, temporal leakage, group leakage — with live demos of
      the inflation each causes.
    - **Nested cross-validation** for honest hyperparameter tuning.
    - Why **pipelines** exist: to fit every transform on train-only, automatically.
    - The generalization gap and reading **learning curves**.

    **Why it matters in industry**
    - Leakage is the #1 cause of the "great offline, terrible online" failure — and
      it can cost a launch, a quarter, or a career.
    - Correct validation is the foundation under *every* metric, model comparison,
      and A/B decision; if it's wrong, all of Sections 02–03 is wrong.
    - **Point-in-time correctness** and train/serve skew are core to feature stores
      and MLOps (PROD-03 and PROD-05).

    **Typical interview questions**
    - "What's the difference between the validation set and the test set?"
    - "Give three concrete examples of data leakage."
    - "Why must you fit the scaler inside cross-validation, not before?"
    - "How do you cross-validate time-series data?"
    - "Your model scores 0.99 offline and 0.65 live — what's your first hypothesis?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **The original sin: evaluating on training data.** The first lesson of ML is that
    a model's error on data it was trained on is wildly optimistic — a deep tree
    memorizes its training set perfectly (Lesson CML-03). So the **holdout** test set
    was introduced: measure error on data the model has *never seen*, as a proxy for
    future performance.

    **Why cross-validation.** A single holdout is noisy (you got lucky/unlucky with
    the split) and wasteful with small data. **K-fold cross-validation** (Stone,
    1974) rotates the holdout across $k$ slices so every point is tested once and the
    estimate averages out split luck — at $k\times$ the compute.

    **Why leakage became the central concern.** As pipelines grew complex — feature
    engineering, selection, scaling, target encoding — information started **leaking**
    from the test set (or from the future, or from the label) into training *through
    the preprocessing*, producing offline numbers that were pure fiction. Kaufman et
    al. (2012) catalogued this; Kaggle is littered with leaderboards won and then
    invalidated by leaks. The reproducibility crisis in ML-for-science is largely a
    leakage crisis: a 2023 survey found leakage errors across *hundreds* of published
    papers.

    **The senior reframing.** Validation isn't a checkbox at the end — it's a
    *discipline* applied throughout. The question shifts from "what's my CV score?"
    to "**is my CV score measuring what production will actually see?**" Everything in
    this notebook serves that one question.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Studying for an exam.** Training data is your study material. The **validation
    set** is the practice exam you take repeatedly to tune your study strategy
    (hyperparameters, model choice). The **test set** is the *real* exam — you take it
    **once**, at the end. If you keep peeking at the real exam to adjust your studying,
    your "score" no longer predicts how you'll do on a *new* exam. That's why the test
    set is touched exactly once, and why tuning on the test set is cheating yourself.

    **Leakage = the answers were on the study sheet.** If a practice question
    accidentally includes the answer, you'll ace it and learn nothing. Data leakage is
    any way the model gets information at training time that it **won't have at
    prediction time** — a feature computed from the future, a column derived from the
    label, or statistics (mean, scaler, selected features) computed using the test
    data. The model "learns" the leak, the offline score soars, and production
    collapses because the leak isn't available live.

    **The subtlest, most common leak: preprocessing before splitting.** Fit a
    `StandardScaler` (or feature selector, or imputer) on the *whole* dataset and the
    training folds now contain information about the test fold's distribution. The fix
    is mechanical: **every fit happens on train-only, inside the fold** — which is
    exactly what a `Pipeline` enforces.

    ```mermaid
    flowchart TD
        D["All data"] --> Split["Split FIRST"]
        Split --> Tr["Train folds"]
        Split --> Te["Test fold (untouched)"]
        Tr --> Fit["fit transforms + model<br/>on TRAIN ONLY"]
        Fit --> Apply["apply learned transform"]
        Te --> Apply
        Apply --> Eval["honest score"]
        Bad["fit scaler/selector on ALL data"]:::bad -.->|LEAK| Eval
        classDef bad fill:#fdd,stroke:#c00;
    ```

    Run the cells: we'll make pure-noise features look 90% accurate via a single
    leak.
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
    # THE classic leak: select features using the WHOLE dataset's labels, THEN cross-validate.
    # Data is PURE NOISE with NO relationship to y. An honest pipeline must score ~50%.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    n, p = 200, 5000
    X = rng.normal(size=(n, p))                       # pure noise features
    y = rng.integers(0, 2, n)                         # random labels, unrelated to X

    # --- LEAKY: pick the 20 features most correlated with y using ALL the data ---
    corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(p)])
    top = np.argsort(-corr)[:20]
    leaky_cv = cross_val_score(LogisticRegression(max_iter=500), X[:, top], y,
                               cv=StratifiedKFold(5)).mean()
    print(f"LEAKY CV accuracy (selection saw all labels): {leaky_cv:.3f}  <- looks predictive!")
    print(f"Truth: features are random noise; honest accuracy must be ~0.50.")
    """),

    md(r"""
    **The leak in action.** We selected the 20 features most correlated with the
    labels *using the entire dataset* — including the rows that later become test
    folds. Even though every feature is pure noise, some will correlate with random
    labels *by chance*, and because selection peeked at all the labels, those spurious
    correlations persist into every fold. The "cross-validated" accuracy lands well
    above 50% — **completely fictional**. We fix it in §5 by moving selection inside
    the CV loop, and the score collapses back to chance. This single mistake has
    invalidated countless real analyses.
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 The goal: estimate true risk, not empirical risk
    We minimize **empirical risk** $\hat R=\frac1n\sum L(y_i,\hat f(x_i))$ on training
    data, but we *care* about **true risk** $R=\mathbb E_{(x,y)}[L(y,\hat f(x))]$ on
    unseen data. Training error underestimates $R$ (the **generalization gap**);
    validation/test error on held-out data is an (approximately) unbiased estimate —
    *as long as the held-out data leaks nothing*.

    ### 4.2 The three-way split and their jobs
    - **Train** — fit model parameters.
    - **Validation** — choose *hyperparameters* and compare models. Reused many times,
      so it slowly becomes "seen."
    - **Test** — a final, **single-use** estimate of generalization. Touch it once;
      every peek erodes its honesty.

    ### 4.3 K-fold cross-validation
    Split data into $k$ folds; train on $k-1$, validate on the held-out fold; rotate
    so each fold is validated once; average the $k$ scores. Trades $k\times$ compute
    for a lower-variance estimate that uses all data. **Stratified** K-fold preserves
    class proportions per fold (essential for imbalanced data, Lesson MLE-01).
    **LOOCV** ($k=n$) is nearly unbiased but high-variance and expensive.

    ### 4.4 When plain K-fold is *wrong*
    Random K-fold assumes rows are **IID** (Lesson FND-02). When they're not, shuffling
    leaks:
    - **Time series:** future must never train a model evaluated on the past. Use
      **forward-chaining / `TimeSeriesSplit`** — train on $[0,t]$, test on $[t,t+1]$.
    - **Grouped data** (multiple rows per user/patient/device): all of a group's rows
      must stay on one side, or the model "recognizes" the group. Use **GroupKFold**.

    ### 4.5 Nested cross-validation
    If you tune hyperparameters on the *same* CV you report, the score is optimistically
    biased (you selected the config that got lucky on those folds). **Nested CV** uses
    an **inner** loop to tune and an **outer** loop to estimate generalization — the
    only honest way to report performance of a *tuned* model.

    ### 4.6 The leakage taxonomy
    | Type | What leaks | Example |
    |---|---|---|
    | **Target leakage** | A feature encodes the label | "account_closed_date" predicting churn |
    | **Train–test contamination** | Test rows seen in training | Duplicate rows; selecting features on all data |
    | **Preprocessing leakage** | Stats fit on full data | `StandardScaler`/impute/encode before split |
    | **Temporal leakage** | Future info in past prediction | Using next month's data; shuffling a time series |
    | **Group leakage** | Same entity in train & test | Same patient's images in both sets |

    The unifying definition: **any information available at training time but not at
    prediction time**. If a feature won't exist (with the same meaning) at serving,
    it's a leak.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    K-fold CV in pure NumPy, then the **honest** version of the §3 experiment —
    feature selection *inside* each fold — proving the inflated score was entirely the
    leak.
    """),

    code(r"""
    # 5.1 K-fold cross-validation from scratch.
    def kfold_indices(n, k, seed=0):
        idx = np.random.default_rng(seed).permutation(n)
        folds = np.array_split(idx, k)
        for i in range(k):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(k) if j != i])
            yield train, test

    def cross_validate(fit_predict, X, y, k=5):
        scores = []
        for train, test in kfold_indices(len(y), k):
            yhat = fit_predict(X[train], y[train], X[test])
            scores.append(np.mean(yhat == y[test]))
        return np.array(scores)

    def logreg_fit_predict(Xtr, ytr, Xte):
        m = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        return m.predict(Xte)

    # sanity check on REAL signal
    from sklearn.datasets import make_classification
    Xr, yr = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=0)
    s = cross_validate(logreg_fit_predict, Xr, yr, k=5)
    print(f"scratch 5-fold CV on real data: {s.mean():.3f} +/- {s.std():.3f}")
    """),

    code(r"""
    # 5.2 HONEST version of the leak demo: select features INSIDE each fold.
    def fit_predict_with_selection(Xtr, ytr, Xte, n_select=20):
        # selection sees ONLY the training labels of this fold
        corr = np.array([abs(np.corrcoef(Xtr[:, j], ytr)[0, 1]) for j in range(Xtr.shape[1])])
        top = np.argsort(-corr)[:n_select]
        m = LogisticRegression(max_iter=500).fit(Xtr[:, top], ytr)
        return m.predict(Xte[:, top])

    honest = cross_validate(fit_predict_with_selection, X, y, k=5)   # X,y are the noise data
    print(f"HONEST CV accuracy (selection inside fold): {honest.mean():.3f} +/- {honest.std():.3f}")
    print(f"Compare to the LEAKY {leaky_cv:.3f} from Section 3.")
    print("-> The 'predictive power' was 100% leakage. Honest score is chance (~0.50).")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Four pictures: how the folds rotate, the leakage inflation made stark, why a time
    series must not be shuffled, and the learning curve that shows the generalization
    gap.
    """),

    code(r"""
    # Figure 1 — how K-fold rotates the held-out slice across the data.
    k = 5; n_show = 20
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for fold in range(k):
        for i in range(n_show):
            is_test = (i % k) == fold
            ax.add_patch(plt.Rectangle((i, k - fold - 1), 1, 0.9,
                         color="tab:red" if is_test else "tab:blue",
                         alpha=0.85 if is_test else 0.45))
    ax.set_xlim(0, n_show); ax.set_ylim(0, k)
    ax.set_yticks([k - f - 0.55 for f in range(k)]); ax.set_yticklabels([f"fold {f+1}" for f in range(k)])
    ax.set_xlabel("data points")
    ax.set_title("Figure 1 — 5-fold CV: red = held out (tested once), blue = trained")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="tab:blue", alpha=0.45, label="train"),
                       Patch(color="tab:red", alpha=0.85, label="validation")], loc="upper right")
    plt.show()
    """),

    md(r"""
    **Figure 1.** Each row is one fold's split. The red (validation) slice rotates so
    **every point is held out exactly once**, and every point trains a model in the
    other four folds. Averaging the five validation scores gives a lower-variance
    estimate than any single holdout, using all the data. The discipline that makes
    this honest: whatever you *fit* (scaler, selector, model) must be fit only on the
    blue portion of each row — never across rows.
    """),

    code(r"""
    # Figure 2 — leakage inflation: the same noise data, leaky vs honest evaluation.
    fig, ax = plt.subplots()
    bars = ax.bar(["LEAKY\n(select on all data)", "HONEST\n(select in-fold)"],
                  [leaky_cv, honest.mean()], color=["tab:red", "tab:green"])
    ax.axhline(0.5, color="k", ls="--", label="true accuracy (random labels)")
    ax.set_ylabel("cross-validated accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Figure 2 — Leakage manufactures predictive power from noise")
    for b, v in zip(bars, [leaky_cv, honest.mean()]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 2.** The two bars use **identical, meaningless data**. The only
    difference is *where the feature selection happens*: peeking at all labels (red)
    fabricates ~0.9 accuracy; selecting inside each fold (green) correctly reports
    chance. If you ever see an offline score that seems too good — *especially* with
    many features and few rows — suspect a leak of exactly this kind before believing
    it. This is why the rule "fit nothing outside the fold" is non-negotiable.
    """),

    code(r"""
    # Figure 3 — time-series data must use forward-chaining, not random folds.
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    N = 30
    # WRONG: random K-fold shuffles future into the training set
    rngc = np.random.default_rng(1)
    test_fold = rngc.integers(0, 3, N)
    axes[0].scatter(range(N), [0] * N, c=["tab:red" if t == 0 else "tab:blue" for t in test_fold], s=80)
    axes[0].set_title("WRONG: random split on a time series\n(future leaks into training)")
    axes[0].set_yticks([])

    # RIGHT: expanding-window forward chaining
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)
    for s, (tr, te) in enumerate(tscv.split(np.arange(N))):
        axes[1].scatter(tr, [s] * len(tr), color="tab:blue", s=40)
        axes[1].scatter(te, [s] * len(te), color="tab:red", s=40)
    axes[1].set_title("RIGHT: TimeSeriesSplit (train on past, test on future)")
    axes[1].set_xlabel("time ->"); axes[1].set_ylabel("split")
    plt.suptitle("Figure 3 — Respect the arrow of time")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** **Left:** a random split scatters test points (red) throughout the
    timeline, so the model trains on data from *after* the points it's evaluated on —
    impossible at serving time, and a guaranteed optimistic bias (temporal leakage).
    **Right:** `TimeSeriesSplit` always trains on an initial segment and tests on the
    *next* segment, mimicking how the model will actually be used (predict the future
    from the past). Any time-ordered problem — forecasting, fraud, churn — must
    validate this way, and features must be **point-in-time correct** (Lesson PROD-03).
    """),

    code(r"""
    # Figure 4 — learning curve: the generalization gap shrinks with more data.
    from sklearn.model_selection import learning_curve
    from sklearn.ensemble import RandomForestClassifier
    sizes, train_sc, val_sc = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=0), Xr, yr,
        train_sizes=np.linspace(0.1, 1.0, 8), cv=5)
    fig, ax = plt.subplots()
    ax.plot(sizes, train_sc.mean(1), "o-", label="training score")
    ax.plot(sizes, val_sc.mean(1), "s-", label="validation score")
    ax.fill_between(sizes, val_sc.mean(1) - val_sc.std(1), val_sc.mean(1) + val_sc.std(1), alpha=0.2)
    ax.set_xlabel("training set size"); ax.set_ylabel("accuracy")
    ax.set_title("Figure 4 — Learning curve: gap = overfitting; convergence = enough data")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 4.** The **training score** starts near-perfect; the **validation score**
    starts lower — the vertical distance is the **generalization gap** (overfitting).
    As data grows, the two curves converge: more data is the most reliable cure for
    overfitting. If the curves have converged and both are low, you're **underfitting**
    (need a better model/features, not more data); if there's a persistent large gap,
    you're **overfitting** (regularize or get more data). Reading this plot tells you
    *which* problem you have — and it's only meaningful when the validation split is
    leak-free.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Preprocessing leakage** | Offline ≫ online | Scaler/selector/imputer fit on all data | Fit inside the fold via `Pipeline` |
    | **Target leakage** | One feature "too good"; near-perfect score | Feature derived from / proxies the label | Audit features for post-outcome info; ablate |
    | **Temporal leakage** | Great backtest, bad live | Shuffled time series; future features | `TimeSeriesSplit`; point-in-time features |
    | **Group leakage** | High CV, poor on new entities | Same user/patient in train & test | `GroupKFold` |
    | **Tuning on test** | Test ≈ val, both optimistic | Repeated peeking / no nested CV | Single-use test set; **nested CV** |
    | **Train/serve skew** | Offline ≠ online despite clean CV | Features computed differently at serving | Shared feature pipeline; monitor (PROD-05) |
    | **High-variance CV** | Scores swing across folds | Too few samples / unstable model | More folds/repeats; report CI (FND-02) |
    | **Distribution shift** | CV honest but live degrades | Train and production differ | Drift monitoring; representative splits |

    The cell demonstrates **target leakage** — adding one feature that secretly
    encodes the label makes a model look perfect.
    """),

    code(r"""
    # Target leakage: a feature accidentally derived from the label -> fake perfection.
    Xc, yc = make_classification(n_samples=1000, n_features=8, n_informative=4, random_state=0)
    # a leaked feature: the label plus tiny noise (e.g., a field populated AFTER the outcome)
    leaked = yc + rng.normal(0, 0.01, len(yc))
    X_leak = np.column_stack([Xc, leaked])

    clean_cv = cross_val_score(LogisticRegression(max_iter=500), Xc, yc, cv=5).mean()
    leak_cv = cross_val_score(LogisticRegression(max_iter=500), X_leak, yc, cv=5).mean()
    print(f"CV accuracy WITHOUT leaked feature : {clean_cv:.3f}")
    print(f"CV accuracy WITH leaked feature    : {leak_cv:.3f}   <- suspiciously perfect")
    print("\\nA near-1.0 score and one dominant feature is a classic target-leakage signature.")
    print("In production that field is unknown at predict time, so the model collapses.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    sklearn's `Pipeline` is the single most important anti-leakage tool: it ensures
    every transform is `fit` on training folds only and merely `transform`-ed on
    validation/test. Combined with the right splitter (`StratifiedKFold`,
    `GroupKFold`, `TimeSeriesSplit`) and `cross_val_score`/`GridSearchCV`, it makes
    correct validation the path of least resistance.
    """),

    code(r"""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

    # WRONG vs RIGHT scaling+selection, on the noise data from Section 3.
    from sklearn.feature_selection import SelectKBest
    # WRONG: transform the whole X first, then CV (leak)
    Xsel_all = SelectKBest(f_classif, k=20).fit_transform(X, y)   # selection sees all labels
    wrong = cross_val_score(LogisticRegression(max_iter=500), Xsel_all, y, cv=5).mean()
    # RIGHT: put selection + model in a Pipeline so selection is re-fit each fold
    pipe = make_pipeline(SelectKBest(f_classif, k=20), StandardScaler(),
                         LogisticRegression(max_iter=500))
    right = cross_val_score(pipe, X, y, cv=5).mean()
    print(f"WRONG (select before CV): {wrong:.3f}   RIGHT (Pipeline): {right:.3f}")

    # Nested CV: tune C in an inner loop, estimate generalization in an outer loop.
    inner = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
                         {"logisticregression__C": [0.01, 0.1, 1, 10]},
                         cv=StratifiedKFold(3))
    nested = cross_val_score(inner, Xr, yr, cv=StratifiedKFold(5))   # outer loop
    print(f"\\nnested CV accuracy (honest, tuned): {nested.mean():.3f} +/- {nested.std():.3f}")
    """),

    md(r"""
    **Scratch vs production.** Our hand-rolled CV taught the mechanics; in practice you
    let `Pipeline` + `cross_val_score` enforce the discipline so you *can't* leak a
    preprocessing step by accident — note the `Pipeline` version reports honest chance
    accuracy while the "select first" version inflates. **Nested CV** (inner loop tunes
    `C`, outer loop scores) is the only honest way to report a *tuned* model's
    performance; reporting the best inner score directly would be optimistically
    biased. Make the `Pipeline` your default unit of modeling — it's the cheapest
    insurance in ML.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — The Hospital Readmission Model That Failed

    **Scenario.** A hospital builds a model to predict 30-day readmission risk.
    Offline AUC is a stellar **0.95**; deployed, it's barely better than chance and
    flags the wrong patients. Post-mortem reveals **three** leaks — a composite of the
    most common real-world failures.

    1. **Target leakage.** A feature `discharge_disposition` included the value
       "transferred for readmission" — populated *because* of the outcome. The model
       was reading the answer. (Available offline; meaningless/absent at prediction
       time.)
    2. **Group leakage.** Patients with multiple admissions had rows in *both* train
       and test; the model memorized patient identities rather than learning risk.
       Fix: `GroupKFold` on patient ID.
    3. **Preprocessing leakage.** Lab-value normalization statistics were computed on
       the full dataset, slipping test-set distribution info into training.

    **Business objectives:** identify high-risk patients early to target interventions
    and reduce costly readmissions.

    **Cost of mistakes:** a falsely-confident model **wastes clinician time, misses
    at-risk patients, and erodes trust in ML** — plus the regulatory and patient-safety
    stakes of a medical tool that doesn't work.

    **Constraints:** **point-in-time correctness** (only data available at discharge),
    grouped data (patients), strict auditability.

    **The fix and KPIs:** rebuild with `GroupKFold`, a feature audit removing
    post-outcome fields, and all preprocessing inside a `Pipeline`; honest AUC drops to
    ~0.72 — *lower but real*. KPIs: grouped-CV AUC, **offline-vs-online gap**, feature
    point-in-time validation, and realized reduction in readmissions from the
    intervention (a proper experiment, Lesson FND-02).
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Pipelines everywhere.** Wrap all preprocessing + model in a `Pipeline` so
      fitting is train-only by construction; serialize the *whole* pipeline so serving
      applies the exact same transforms (prevents train/serve skew).
    - **Point-in-time correctness.** For any time-ordered problem, every feature must
      be computed from data available *at prediction time*. This is the core promise of
      a **feature store** (Lesson PROD-03) and the hardest part of real ML systems.
    - **Backtesting.** Validate temporally with rolling/expanding windows that mirror
      the deployment cadence; a random-split "validation" of a forecasting model is
      worthless.
    - **Guard the test set.** Touch it once. Track how many times the validation set has
      been used to tune — repeated reuse silently overfits it ("validation set
      decay"); refresh it periodically.
    - **Offline↔online reconciliation.** Always confirm offline metrics with an online
      A/B test on the real KPI (Lesson FND-02); a large gap is a leakage/skew alarm
      (PROD-05 and PROD-06).
    - **Leakage review** as a checklist before launch: for each feature, "is this
      available, with this meaning, at prediction time?" If not, drop it.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Validation strategies:**

    | Strategy | Bias of estimate | Variance | Cost | Use when |
    |---|---|---|---|---|
    | Single holdout | Higher | High (one split) | Cheapest | Big data, quick checks |
    | K-fold (k=5/10) | Low | Moderate | $k\times$ | **Default** for IID tabular |
    | Stratified K-fold | Low | Moderate | $k\times$ | **Imbalanced** classification |
    | LOOCV (k=n) | Lowest | **High** | $n\times$ (expensive) | Very small datasets |
    | GroupKFold | Low (honest) | Moderate | $k\times$ | **Grouped/entity** data |
    | TimeSeriesSplit | Honest for time | Moderate | $k\times$ | **Temporal** data |
    | Nested CV | Honest for tuned model | Higher | $k_{in}\times k_{out}$ | Reporting a tuned model |

    **Holdout vs cross-validation:**

    | Dimension | Single holdout | K-fold CV |
    |---|---|---|
    | Data efficiency | Wastes held-out data | Uses all data |
    | Estimate stability | Noisy | Averaged, stabler |
    | Compute | 1× | $k$× |
    | Best for | Large datasets / final test | Model selection, small data |

    **Senior lesson:** the validation scheme must **mirror how the model is used in
    production**. The fanciest CV is worthless if it shuffles time or groups; a humble
    holdout that respects them beats it every time.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Validation set vs test set?* → Validation tunes hyperparameters (reused); test
      is the single-use final generalization estimate.
    - *What is data leakage?* → Information available at training but not at
      prediction time, inflating offline metrics.

    **Deep-dive questions**
    - *Three concrete leaks.* → Preprocessing-before-split, target leakage, temporal/
      group leakage (Sections 4.6, 7 — give examples).
    - *Why fit the scaler inside CV?* → Fitting on all data leaks test-fold
      distribution into training (Fig 2). `Pipeline` enforces train-only fitting.
    - *How to CV time series?* → Forward-chaining/`TimeSeriesSplit`; never shuffle;
      point-in-time features (Fig 3).
    - *Why nested CV?* → Tuning and reporting on the same CV is optimistically biased.

    **Whiteboard questions**
    - "Implement K-fold CV." (Section 5.1.)
    - "Design validation for a churn model with repeat customers and time order."
      (GroupKFold + temporal split + point-in-time features.)

    **Strong vs weak answers**
    - *"Model is 0.99 offline, 0.65 live."*
      - **Weak:** "Overfitting; regularize."
      - **Strong:** "First hypothesis is **leakage** or train/serve skew. I'd audit
        features for post-outcome info, check that preprocessing was fit train-only,
        verify temporal/group splits, and reconcile the offline feature pipeline with
        serving — *before* touching the model."
    - *"Your CV looks great."*
      - **Weak:** "Ship it."
      - **Strong:** "Does the CV mirror production? Right splitter for time/groups,
        all transforms in a pipeline, test set untouched, tuned via nested CV? Only
        then do I trust the number."

    **Follow-ups:** "How detect target leakage?" (one feature dominates / near-perfect
    score / ablation). "Cost of LOOCV?" (high variance + $n$× compute). "When is a
    single holdout fine?" (very large data, final test).

    **Common mistakes:** scaling/selecting before splitting; shuffling time series;
    same entity in train & test; tuning on the test set; trusting a too-good offline
    number; no offline↔online check.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define validation, cross-validation, and data leakage.
    2. **Why was it invented?** Why isn't training error enough, and why CV over a
       single holdout?
    3. **How does it work?** Describe K-fold and how a `Pipeline` prevents leakage.
    4. **Why does it work?** Why does held-out (leak-free) error estimate true risk?
    5. **When to use it?** Match scenarios to stratified / group / time-series CV.
    6. **When NOT to use it?** When does plain K-fold give a dishonest estimate?
    7. **Tradeoffs?** Holdout vs K-fold vs LOOCV; nested CV cost.
    8. **How would you productionize it?** Pipelines, point-in-time features,
       backtesting, test-set discipline, offline↔online reconciliation.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. For each of these, name the leakage type: (a) imputing missing values using the
       full dataset's mean; (b) a "days_until_churn" feature; (c) random-splitting
       stock prices; (d) the same customer in train and test.
    2. Explain why the test set should be used only once.

    **Beginner → Intermediate (coding)**
    3. Extend the scratch CV to **StratifiedKFold** (preserve class balance) and show
       it matters on an imbalanced dataset.
    4. Reproduce the §3 leak, then fix it three ways: in-fold selection, a `Pipeline`,
       and removing selection entirely. Confirm all honest versions ≈ chance.

    **Intermediate (analysis)**
    5. Implement **GroupKFold** from scratch and demonstrate the optimistic bias of
       plain K-fold on grouped data (same entity in multiple rows).
    6. Build a **nested CV** by hand and compare its score to the (biased) flat-CV
       best-config score; quantify the optimism.

    **Senior (interview + production design)**
    7. *Whiteboard:* design the validation scheme for a credit model with repeat
       borrowers, time order, and engineered aggregate features; identify every place
       leakage could enter and how you'd prevent it.
    8. *Design:* propose an offline↔online validation system — backtesting cadence,
       point-in-time feature checks, a guarded test set, and the monitoring that would
       catch a leak that slipped to production.
    9. *Audit:* you inherit a model with offline AUC 0.97. Write the step-by-step
       leakage audit you'd run before trusting it.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    A metric is only as trustworthy as the data it's measured on. **Cross-validation**
    estimates true risk by rotating a held-out slice; **leakage** — preprocessing fit
    on all data, target-derived features, shuffled time, shared groups — secretly
    feeds the answer in and produces offline numbers that evaporate in production. The
    defenses are mechanical and non-negotiable: **split first, fit transforms inside a
    `Pipeline`, respect time and groups, tune with nested CV, and touch the test set
    once.** Validation that mirrors production is the foundation everything else stands
    on.

    **Related lesson:** `MLE-03 · Feature Engineering` — with trustworthy evaluation in hand, we turn
    to the lever that most often beats model choice: constructing, transforming, and
    encoding features (leak-free, of course) that make the signal learnable.
    """),
]

build("03_ml_engineering/02_validation_and_data_leakage.ipynb", cells)
