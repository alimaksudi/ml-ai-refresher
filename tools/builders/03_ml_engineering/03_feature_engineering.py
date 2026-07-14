"""Builder for Lesson MLE-03 — Feature Engineering.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # MLE-03 · Feature Engineering
    ### Section 03 — ML Engineering Foundations · *ML/AI Senior Mastery Curriculum*

    > *"Applied machine learning is basically feature engineering."* — Andrew Ng.
    > For tabular problems, **how you represent the data usually matters more than
    > which model you pick.** A linear model with the right features beats a tuned
    > gradient-boosting machine on raw ones; a great feature can lift a weak model
    > above a strong one. This notebook is the craft of turning raw columns into
    > signal a model can use — scaling, encoding, interactions, time, missingness,
    > and selection — and, just as importantly, doing all of it **leak-free**
    > (Lesson MLE-02). The single most dangerous technique here, **target encoding**,
    > is also one of the most powerful, and we treat its leakage trap head-on.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Numeric transforms**: standardization, min-max, **log / power** for skew,
      and **binning** — and which models need them (Lessons FND-04, CML-01, and CML-02) vs which don't
      (trees, Lessons CML-03 through CML-05).
    - **Categorical encoding**: one-hot, ordinal, and **target/mean encoding** done
      **leak-free** via out-of-fold cross-fitting + smoothing (the MLE-02 link).
    - **Interactions & polynomial features** — making nonlinearity learnable by linear
      models.
    - **Datetime & cyclical (sin/cos) encoding** so "23:00" and "00:00" are neighbors.
    - **Missing-value** strategies (and treating missingness as signal).
    - **Feature selection**: filter / wrapper / embedded, and why fewer good features
      can beat many.
    - The meta-principle: **good features often beat fancier models**.

    **Why it matters in industry**
    - Feature work is where most real-world accuracy gains come from — and where most
      **leakage bugs** are born.
    - Encoding/scaling choices interact with model choice; getting them wrong silently
      caps performance.
    - Features must be **reproducible at serving time** (train/serve skew, feature
      stores — MLE-02 and PROD-03).

    **Typical interview questions**
    - "How do you encode a high-cardinality categorical feature?"
    - "Why log-transform a feature? When does scaling matter?"
    - "What is target encoding and how do you keep it from leaking?"
    - "How would you encode the hour of day or day of week?"
    - "More features or better features — and how do you select?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **Models are simple; the world is messy.** The algorithms of Section 02 expect clean
    numeric matrices, but raw data is skewed, categorical, temporal, and full of holes.
    Historically, the gap between "raw data" and "what the model can use" was bridged
    by **hand-crafted features** — and for decades, *that craft was the job*. Winning
    Kaggle solutions, credit scorecards, and search-ranking systems were won on
    features, not exotic models.

    **Why representation matters so much.** A linear model can only fit a line; give it
    $\log(\text{income})$ instead of income, or the *product* of two features, and
    suddenly the relationship it needs *is* linear in the new space. This is the same
    idea as a kernel or a basis expansion (Lesson CML-01's polynomial features): we do
    the model's nonlinearity *for* it by transforming the inputs. Tree models need
    less of this (they split on raw values), which is itself a key engineering
    decision.

    **The deep-learning twist — and why this still matters.** Deep nets *learn*
    representations from raw signals (pixels, tokens), which is why feature
    engineering faded for images and text (Sections 04–05). But for **tabular data** —
    still the majority of business ML — engineered features + gradient boosting
    remain state of the art, and even deep tabular models benefit from good encodings.
    Moreover, the *discipline* transfers: embeddings (NLP-01 and NLP-02) are learned
    features, and prompt design (Lesson NLP-04) is feature engineering for LLMs.

    **The cautionary half.** Feature engineering is also the **#1 source of data
    leakage** (Lesson MLE-02): target encoding, aggregates, and "helpful" derived columns
    routinely smuggle the answer (or the future) into training. So this notebook is
    inseparable from the previous one — every technique here is shown *and* its leak
    risk addressed.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Feature engineering = changing the coordinate system so the signal is obvious.**
    Three recurring moves:

    1. **Reshape distributions.** Income, prices, counts are right-skewed; a linear
       model fixates on the few huge values. $\log(1+x)$ pulls the tail in and often
       *linearizes* the relationship — the model's job becomes easy.
    2. **Make categories numeric — carefully.** A model can't consume "city =
       Boston". One-hot turns it into indicator columns (safe but explodes for high
       cardinality); **target encoding** replaces a category with its average outcome
       (compact and powerful, but leaks unless cross-fit).
    3. **Encode structure the model can't infer.** Hours are *cyclical* (23 is next to
       0), but the raw number 23 looks far from 0; sin/cos coordinates fix that.
       Interactions ($x_1\cdot x_2$) let a linear model express "the effect of A
       depends on B" — which trees discover automatically but linear models cannot.

    **The senior mindset:** match the encoding to **both** the data's structure **and**
    the model. Linear/NN models need scaling and explicit nonlinearity; trees are
    scale-invariant and find interactions themselves. And *every* transform that uses
    the target (encoding, aggregates) must be fit **inside the CV fold** or it leaks.

    ```mermaid
    flowchart LR
        Raw["Raw columns<br/>(skewed, categorical, time, missing)"] --> N["Numeric: scale / log / bin"]
        Raw --> C["Categorical: one-hot / target-encode (leak-free)"]
        Raw --> T["Time: cyclical sin/cos, lags"]
        Raw --> M["Missing: impute + missing-flag"]
        N --> F["Feature matrix"]
        C --> F
        T --> F
        M --> F
        F --> Sel["Selection: filter / wrapper / embedded"]
        Sel --> Model["Model (Section 02)"]
    ```

    Run the cells — start with a transform that turns an unlearnable relationship
    into a trivial one.
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
    # Figure 1 — a log transform linearizes a skewed relationship a linear model can then fit.
    income = rng.lognormal(mean=10, sigma=1.0, size=300)      # heavily right-skewed
    spend = 0.5 * np.log(income) + rng.normal(0, 0.3, 300)    # truly linear in LOG(income)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # raw: relationship looks curved/compressed; a line fits poorly
    wr = np.polyfit(income, spend, 1)
    axes[0].scatter(income, spend, s=12, alpha=0.6)
    axes[0].plot(np.sort(income), np.polyval(wr, np.sort(income)), "r", lw=2)
    r2_raw = 1 - np.var(spend - np.polyval(wr, income)) / np.var(spend)
    axes[0].set_title(f"Raw income (skewed): linear R2 = {r2_raw:.2f}")
    axes[0].set_xlabel("income"); axes[0].set_ylabel("spend")

    # log-transformed: relationship is straight; the same linear model nails it
    logi = np.log(income)
    wl = np.polyfit(logi, spend, 1)
    axes[1].scatter(logi, spend, s=12, alpha=0.6, color="tab:green")
    axes[1].plot(np.sort(logi), np.polyval(wl, np.sort(logi)), "r", lw=2)
    r2_log = 1 - np.var(spend - np.polyval(wl, logi)) / np.var(spend)
    axes[1].set_title(f"log(income): linear R2 = {r2_log:.2f}")
    axes[1].set_xlabel("log(income)"); axes[1].set_ylabel("spend")
    plt.suptitle("Figure 1 — The right transform makes the model's job trivial")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** Same data, same linear model — only the input representation changed.
    On **raw income** (left) the relationship is compressed into a corner by the skew
    and a straight line fits badly. After a **log transform** (right) the relationship
    is genuinely linear and $R^2$ jumps. We didn't use a fancier model; we changed the
    coordinate system to match the structure. This is the essence of feature
    engineering — and note a **tree** wouldn't care (it splits on order, invariant to
    monotonic transforms), which is exactly the kind of model-dependent judgment a
    senior engineer makes.
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Numeric transforms
    - **Standardization** $z=\frac{x-\mu}{\sigma}$ — zero mean, unit variance. Required
      for distance/gradient/regularized models (Lessons FND-04, CML-01, and CML-02); irrelevant to trees.
    - **Min–max** $\frac{x-\min}{\max-\min}\in[0,1]$ — bounded range; sensitive to
      outliers. **Robust scaling** uses median/IQR instead.
    - **Log / power (Box–Cox, Yeo–Johnson)** — tame right skew, stabilize variance,
      linearize multiplicative relationships (Fig 1). Use $\log(1+x)$ for
      non-negatives with zeros.
    - **Binning/discretization** — convert a continuous feature into ordinal buckets;
      can capture nonlinearity for linear models but discards information.

    **Fit statistics on TRAIN only.** $\mu,\sigma,\min,\max$, quantile edges must come
    from training data and be applied to validation/test — fitting on all data is
    preprocessing leakage (Lesson MLE-02).

    ### 4.2 Categorical encoding
    - **One-hot**: one binary column per category. Safe and lossless, but
      **dimensionality explodes** with cardinality and creates sparse, correlated
      columns. Drop-one to avoid the dummy-variable trap in linear models.
    - **Ordinal/label**: map categories to integers. Valid only when there's a true
      order (low/med/high); otherwise it invents a false ordering that misleads linear
      models (trees tolerate it better).
    - **Target (mean) encoding**: replace category $c$ with the average target among
      rows of that category,
      $$\text{enc}(c)=\frac{n_c\,\bar y_c+\alpha\,\bar y}{n_c+\alpha}\quad(\text{smoothed toward the global mean }\bar y).$$
      Compact, handles high cardinality, often very predictive — **but it uses the
      target, so it leaks unless computed out-of-fold** (§4.3). Smoothing $\alpha$
      shrinks rare categories toward the prior to avoid overfitting them.

    ### 4.3 Leak-free target encoding (the crucial bit)
    Encoding a row using its *own* target is leakage. The fix: **out-of-fold
    cross-fitting** — to encode fold $k$, compute category means from the *other*
    folds only (and at serving, from the full training set). This is the MLE-02
    "fit inside the fold" rule applied to a target-derived feature. Without it, CV is
    wildly optimistic (§6, Fig 3).

    ### 4.4 Interactions & polynomial features
    A linear model is additive: it can't express "effect of $x_1$ depends on $x_2$".
    Adding the product $x_1 x_2$ (and powers $x_1^2,\dots$) expands the basis so the
    model can fit curves and interactions — the same trick as Lesson CML-01's polynomial
    regression. Trees and boosting find interactions automatically, so this is mainly
    for linear/NN models. Beware the combinatorial blow-up (degree-$d$ over $p$
    features → $O(p^d)$ terms) and overfitting.

    ### 4.5 Datetime & cyclical encoding
    Extract components (year, month, dayofweek, hour, is_holiday) and **lags/rolling
    aggregates** (point-in-time correct! Lesson MLE-02). Cyclical features (hour, month,
    angle) must encode wrap-around: map $t$ with period $T$ to
    $$\big(\sin\tfrac{2\pi t}{T},\ \cos\tfrac{2\pi t}{T}\big),$$
    so $t=0$ and $t=T-1$ are adjacent on the circle (Fig 2).

    ### 4.6 Missing values
    Missingness is information. Options: drop (wasteful), **mean/median/mode impute**
    (simple, fit on train), model-based (KNN/iterative) impute, or — often best —
    **impute + add a binary "was-missing" flag** so the model can learn that
    missingness itself is predictive. Tree libraries (XGBoost/LightGBM) handle NaNs
    natively by learning a default direction.

    ### 4.7 Feature selection
    - **Filter** (model-free): rank by correlation, $\chi^2$, **mutual information**;
      fast, ignores interactions.
    - **Wrapper**: search subsets by model performance (RFE, forward/backward); accurate
      but expensive and leak-prone if done outside CV.
    - **Embedded**: selection inside training — **L1/Lasso** zeros coefficients
      (Lesson CML-01), tree **importances** (Lesson CML-04). Usually the best cost/quality
      trade.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    A feature-engineering toolkit in NumPy: numeric transforms, one-hot, **smoothed
    out-of-fold target encoding** (the leak-free version), cyclical encoding, and
    interactions. The target-encoder is the one to study closely.
    """),

    code(r"""
    # 5.1 Numeric transforms (fit on train, apply to test) and one-hot from scratch.
    def fit_standardizer(x):
        return x.mean(), x.std() + 1e-12          # stats from TRAIN only

    def standardize(x, mu, sd):
        return (x - mu) / sd

    def log1p(x):
        return np.log1p(x)                         # log(1+x), safe for zeros

    def one_hot(cats, categories=None):
        categories = categories if categories is not None else np.unique(cats)
        mapping = {c: i for i, c in enumerate(categories)}
        out = np.zeros((len(cats), len(categories)))
        for i, c in enumerate(cats):
            if c in mapping:
                out[i, mapping[c]] = 1.0
        return out, categories

    x = rng.lognormal(3, 1, 6)
    mu, sd = fit_standardizer(x)
    print("standardized:", standardize(x, mu, sd).round(2))
    oh, cats = one_hot(np.array(["a", "b", "a", "c"]))
    print("one-hot of [a,b,a,c]:\\n", oh, "\\ncategories:", cats)
    """),

    code(r"""
    # 5.2 Smoothed target encoding with OUT-OF-FOLD cross-fitting (leak-free, MLE-02).
    def smoothed_means(cat_tr, y_tr, alpha=10.0):
        prior = y_tr.mean()
        means = {}
        for c in np.unique(cat_tr):
            m = cat_tr == c
            n, ybar = m.sum(), y_tr[m].mean()
            means[c] = (n * ybar + alpha * prior) / (n + alpha)   # shrink rare cats to prior
        return means, prior

    def apply_means(cat, means, prior):
        return np.array([means.get(c, prior) for c in cat])

    def oof_target_encode(cat, y, n_folds=5, alpha=10.0, seed=0):
        # encode each row using category means computed from OTHER folds only
        idx = np.random.default_rng(seed).permutation(len(y))
        folds = np.array_split(idx, n_folds)
        enc = np.empty(len(y))
        for i in range(n_folds):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            means, prior = smoothed_means(cat[tr], y[tr], alpha)
            enc[te] = apply_means(cat[te], means, prior)
        return enc

    # demo categories
    cat = np.array(["x", "y", "x", "z", "y", "x"])
    yy = np.array([1, 0, 1, 0, 1, 1.0])
    print("OOF target-encoded:", oof_target_encode(cat, yy, n_folds=3, alpha=1.0).round(2))
    """),

    code(r"""
    # 5.3 Cyclical encoding and interaction/polynomial features.
    def cyclical(values, period):
        ang = 2 * np.pi * values / period
        return np.sin(ang), np.cos(ang)

    def add_interactions(X):
        # append all pairwise products x_i * x_j (i<j) to capture interactions
        n, d = X.shape
        extra = [X[:, i] * X[:, j] for i in range(d) for j in range(i + 1, d)]
        return np.column_stack([X] + extra) if extra else X

    hours = np.array([0, 6, 12, 18, 23])
    s, c = cyclical(hours, 24)
    print("hour 23 sin/cos:", round(s[-1], 3), round(c[-1], 3),
          "| hour 0:", round(s[0], 3), round(c[0], 3), "-> adjacent on the circle")
    print("interactions of a 3-col matrix -> shape:", add_interactions(rng.normal(size=(4, 3))).shape)
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Three pictures: why cyclical encoding is necessary, the **target-encoding leakage
    trap** (the most important figure in this notebook), and the principle that good
    features beat a fancier model.
    """),

    code(r"""
    # Figure 2 — cyclical encoding places hour 23 next to hour 0 (raw integer can't).
    hrs = np.arange(24)
    s, c = cyclical(hrs, 24)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(hrs, np.zeros_like(hrs), c=hrs, cmap="twilight", s=80)
    axes[0].set_title("Raw hour: 23 and 0 look FAR apart")
    axes[0].set_xlabel("hour"); axes[0].set_yticks([])
    axes[1].scatter(s, c, c=hrs, cmap="twilight", s=80)
    for h in [0, 6, 12, 18, 23]:
        axes[1].annotate(str(h), (s[h], c[h]))
    axes[1].set_title("sin/cos encoding: 23 and 0 are NEIGHBORS")
    axes[1].set_xlabel("sin(2pi h/24)"); axes[1].set_ylabel("cos(2pi h/24)")
    axes[1].set_aspect("equal")
    plt.suptitle("Figure 2 — Cyclical features respect wrap-around")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** As a raw integer, hour 23 is maximally distant from hour 0 — yet
    they're one hour apart. A model fed the raw number learns a false discontinuity at
    midnight. Mapping the hour onto a **circle** via $(\sin,\cos)$ places 23 and 0
    adjacent, so "1 hour before midnight" and "1 hour after" are correctly close. The
    same applies to month, day-of-week, and any angular feature. (Trees can partly
    cope by splitting, but linear/NN models *need* this.)
    """),

    code(r"""
    # Figure 3 — TARGET-ENCODING LEAKAGE: naive (full-data) encoding inflates CV.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    # high-cardinality categorical that DOES carry signal (each category has a true rate)
    n, n_cat = 3000, 300
    true_rate = rng.uniform(0.1, 0.9, n_cat)
    cat_idx = rng.integers(0, n_cat, n)
    y = (rng.random(n) < true_rate[cat_idx]).astype(int)
    cat = cat_idx.astype(str)

    # NAIVE (leaky): encode using the WHOLE dataset's target, then cross-validate
    means_all, prior_all = smoothed_means(cat, y, alpha=1.0)
    naive_enc = apply_means(cat, means_all, prior_all).reshape(-1, 1)
    leaky = cross_val_score(LogisticRegression(), naive_enc, y, cv=StratifiedKFold(5)).mean()

    # HONEST: out-of-fold encoding
    oof_enc = oof_target_encode(cat, y, n_folds=5, alpha=1.0).reshape(-1, 1)
    honest = cross_val_score(LogisticRegression(), oof_enc, y, cv=StratifiedKFold(5)).mean()

    fig, ax = plt.subplots()
    bars = ax.bar(["NAIVE target encode\n(leaky)", "OUT-OF-FOLD\n(honest)"],
                  [leaky, honest], color=["tab:red", "tab:green"])
    ax.set_ylabel("CV accuracy"); ax.set_ylim(0, 1)
    for b, v in zip(bars, [leaky, honest]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_title("Figure 3 — Naive target encoding leaks; OOF encoding is honest")
    plt.show()
    print(f"leaky CV {leaky:.3f} vs honest CV {honest:.3f} -- the gap is pure leakage.")
    """),

    md(r"""
    **Figure 3.** Target encoding is powerful for high-cardinality categoricals — but
    the **naive** version (compute each category's mean target over the *whole*
    dataset, then evaluate) lets each row peek at its own label, inflating CV accuracy.
    The **out-of-fold** version encodes each fold using only the *other* folds' targets,
    matching what's available at serving time, and reports the honest (lower) score.
    With low smoothing $\alpha$ and rare categories the leak is enormous. This is the
    canonical feature-engineering leak from Lesson MLE-02 — *always* cross-fit any
    target-derived feature.
    """),

    code(r"""
    # Figure 4 — good features beat a fancier model: linear+cyclical vs RF on raw hour.
    from sklearn.ensemble import RandomForestClassifier
    # signal depends on a cyclical hour (peak demand near midnight)
    N = 1500
    hour = rng.integers(0, 24, N)
    prob = 0.5 + 0.4 * np.cos(2 * np.pi * hour / 24)          # peaks at hour 0/24
    label = (rng.random(N) < prob).astype(int)

    raw = hour.reshape(-1, 1).astype(float)
    s, c = cyclical(hour, 24); feat = np.column_stack([s, c])

    lr_raw = cross_val_score(LogisticRegression(), raw, label, cv=5).mean()
    lr_cyc = cross_val_score(LogisticRegression(), feat, label, cv=5).mean()
    rf_raw = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0),
                             raw, label, cv=5).mean()

    fig, ax = plt.subplots()
    bars = ax.bar(["Linear\n(raw hour)", "Random Forest\n(raw hour)", "Linear\n(sin/cos hour)"],
                  [lr_raw, rf_raw, lr_cyc],
                  color=["tab:gray", "tab:orange", "tab:green"])
    ax.set_ylabel("CV accuracy"); ax.set_ylim(0, 1)
    for b, v in zip(bars, [lr_raw, rf_raw, lr_cyc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_title("Figure 4 — A good feature beats a fancier model")
    plt.show()
    """),

    md(r"""
    **Figure 4.** The signal is cyclical in the hour. A **linear model on the raw hour**
    (gray) is near-useless — it can only fit a monotonic trend. A **Random Forest on the
    raw hour** (orange) does better (it can split out buckets) but still works hard.
    A **linear model with sin/cos features** (green) — the *simplest* model with the
    *right* representation — matches or beats the forest. The lesson senior engineers
    internalize: **spend your effort on features before reaching for a heavier model**;
    representation is leverage.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Target-encoding leakage** | CV ≫ live; rare categories "too predictive" | Encoding uses own row's label | **Out-of-fold** cross-fit + smoothing (Fig 3) |
    | **Preprocessing leakage** | Optimistic CV | Scaler/encoder fit on all data | Fit inside `Pipeline`/fold (MLE-02) |
    | **One-hot explosion** | Huge sparse matrix; slow; overfit | High-cardinality one-hot | Target encoding, hashing, embeddings |
    | **False ordinal order** | Linear model misled | Label-encoding unordered categories | One-hot / target encode instead |
    | **Skew ignored** | Linear model dominated by outliers | Heavy-tailed feature untransformed | log / power / robust scaling |
    | **Cyclical as integer** | Discontinuity at wrap (23→0) | Linear encoding of periodic feature | sin/cos encoding (Fig 2) |
    | **Imputation distorts** | Biased estimates; lost signal | Mean-fill without a flag | Impute + **missing indicator** |
    | **Train/serve skew** | Offline ≠ online | Features computed differently live | Shared pipeline; feature store (PROD-03) |
    | **Too many features** | Overfit; slow; unstable | Kitchen-sink feature dump | Selection (L1, importance, MI); fewer-better |

    The cell shows the **one-hot vs target-encoding** tradeoff on high cardinality.
    """),

    code(r"""
    # High cardinality: one-hot explodes dimensionality; target encoding stays compact.
    n_cat_demo = 500
    cat_demo = rng.integers(0, n_cat_demo, 2000).astype(str)
    oh, _ = one_hot(cat_demo)
    print(f"one-hot of {n_cat_demo} categories  -> matrix shape {oh.shape} (one column PER category)")
    print(f"target encoding                  -> matrix shape (2000, 1)  (single compact column)")
    print("\\nOne-hot is safe but blows up width (sparsity, memory, overfitting);")
    print("target encoding is compact and powerful but MUST be cross-fit (Fig 3).")
    print("For very high cardinality, also consider feature hashing or learned embeddings (NLP-01-21).")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    scikit-learn's `ColumnTransformer` + `Pipeline` is the production pattern: declare
    per-column transforms (scale numerics, one-hot categoricals, impute), and the
    pipeline guarantees everything is **fit on train only** and reproduced identically
    at serving. Libraries also provide `StandardScaler`, `PowerTransformer`,
    `KBinsDiscretizer`, `OneHotEncoder`, `SimpleImputer`, `PolynomialFeatures`, and
    (in `category_encoders`/`TargetEncoder`) cross-fitted target encoding.
    """),

    code(r"""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import pandas as pd

    # a small mixed-type dataset with a missing value
    df = pd.DataFrame({
        "income": rng.lognormal(10, 1, 500),
        "age": rng.integers(18, 80, 500).astype(float),
        "city": rng.choice(["NYC", "LA", "SF", "CHI"], 500),
    })
    df.loc[rng.choice(500, 40, replace=False), "age"] = np.nan      # inject missingness
    target = (np.log(df["income"]) + (df["city"] == "SF") * 1.5
              + rng.normal(0, 1, 500) > 11).astype(int)

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([("num", numeric, ["income", "age"]),
                             ("cat", categorical, ["city"])])
    pipe = Pipeline([("features", pre), ("model", LogisticRegression(max_iter=1000))])

    score = cross_val_score(pipe, df, target, cv=5).mean()
    print(f"leak-free ColumnTransformer pipeline CV accuracy: {score:.3f}")
    print("Every transform (impute, scale, one-hot) is fit on TRAIN folds only -- no leakage.")
    """),

    md(r"""
    **Scratch vs production.** Our NumPy transforms taught the mechanics; in production
    you compose them in a `ColumnTransformer`/`Pipeline` so (1) numerics get
    imputed+scaled, categoricals get one-hot, all **fit on train folds only**; (2) the
    *entire* fitted pipeline serializes and serves, eliminating train/serve skew; and
    (3) `handle_unknown="ignore"` survives unseen categories at inference. For
    cross-fitted target encoding use `sklearn.preprocessing.TargetEncoder` or the
    `category_encoders` library — they implement the out-of-fold logic from §5.2 so you
    don't hand-roll (and accidentally leak) it.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Features for Ride-Hailing Demand & Pricing

    **Scenario.** A ride-hailing company predicts demand (and sets surge pricing) per
    city zone per 15-minute window. The raw logs are timestamps, zone IDs, weather, and
    event calendars — almost none of it model-ready.

    **The feature work is the project:**
    - **Cyclical time:** hour-of-day and day-of-week as sin/cos (Fig 2) — demand is
      profoundly periodic (rush hours, weekends).
    - **High-cardinality zones:** thousands of zone IDs → **out-of-fold target
      encoding** of historical demand per zone (leak-free, Fig 3), not one-hot.
    - **Lags & rolling aggregates:** demand in the previous windows, last week
      same-hour — **point-in-time correct** (MLE-02 and PROD-03) or you leak the future.
    - **Interactions:** weather × hour (rain at rush hour ≠ rain at 3am), event × zone.
    - **Missing weather:** impute + a "weather_missing" flag.

    **Business objectives:** accurate short-horizon demand forecasts that drive pricing
    and driver positioning.

    **Cost of mistakes:** under-forecast → undersupply, long waits, lost rides;
    over-forecast → idle drivers, unnecessary surge, rider churn. A **leaky** feature
    (e.g., using future demand or naive target encoding) makes the offline model look
    great and the live system misprice in real time — directly costly.

    **Constraints:** features must be computable in real time at serving with the same
    code as training (feature store), and respect the arrow of time.

    **KPIs:** forecast error (MAE/RMSE per zone-window, Lesson MLE-01), realized
    wait-times and driver utilization, and — critically — the **offline-vs-online gap**
    that would expose a leak.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **One pipeline, train and serve.** Serialize the *fitted* `Pipeline`/feature
      transforms and run the identical code at inference to prevent **train/serve
      skew** — the most common production feature bug.
    - **Point-in-time correctness.** Lags and aggregates must use only data available at
      prediction time; this is the core job of a **feature store** (Lesson PROD-03) and a
      frequent leakage source (Lesson MLE-02).
    - **Cross-fit anything that touches the target.** Target/leave-one-out encodings,
      target-based aggregates — always out-of-fold; never fit on the row's own label.
    - **Unknown categories at serving.** New categories appear live; handle gracefully
      (`handle_unknown`, fall back to the prior in target encoding).
    - **Feature drift & freshness.** Monitor feature distributions (Lesson PROD-05); stale
      aggregates or a shifted encoding degrade the model silently.
    - **Cost/latency.** Some features (rolling windows, joins) are expensive to compute
      online; precompute and cache. Selection (§4.7) also reduces serving cost.
    - **Reproducibility & lineage.** Version feature definitions; a changed transform
      invalidates the trained model.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Categorical encoding:**

    | Encoding | Dimensionality | Handles high cardinality | Leak risk | Best for |
    |---|---|---|---|---|
    | One-hot | High (1/category) | Poorly (explodes) | None | Low-cardinality, linear/NN |
    | Ordinal/label | Low (1 col) | Yes | None | **Truly ordered** categories; trees |
    | Target/mean | Low (1 col) | **Well** | **High (must cross-fit)** | High-cardinality, predictive cats |
    | Hashing | Fixed | Yes | None | Very high cardinality, streaming |
    | Learned embedding | Low-dim dense | **Well** | Low | Deep models, huge cardinality (NLP-01) |

    **Scaling — does the model need it?**

    | Model | Needs scaling? | Needs explicit nonlinearity? |
    |---|---|---|
    | Linear / Logistic / SVM / NN | **Yes** | **Yes** (interactions, polynomials) |
    | Tree / RF / Gradient Boosting | No (order-invariant) | No (finds splits/interactions) |
    | kNN / k-means (distance) | **Yes** | Depends |

    **Feature selection methods:**

    | Method | Cost | Captures interactions | Leak-safe-by-default |
    |---|---|---|---|
    | Filter (corr / MI) | Cheap | No | If done in-fold |
    | Wrapper (RFE) | Expensive | Yes | Only inside CV |
    | Embedded (L1, importances) | Cheap–moderate | Yes | Yes (part of training) |

    **Senior lesson:** the encoding/scaling decision is **model-dependent** and the
    target-encoding decision is **leakage-dependent**. Stating *why* a transform fits
    *this* model and *how* you keep it leak-free is the senior signal.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Encode a high-cardinality categorical?* → One-hot explodes; prefer **target
      encoding (cross-fit + smoothed)**, hashing, or embeddings.
    - *When does scaling matter?* → Distance/gradient/regularized models (linear, SVM,
      kNN, NN); **never** for trees.

    **Deep-dive questions**
    - *What is target encoding and how prevent leakage?* → Replace category with
      smoothed mean target; compute **out-of-fold** (Fig 3, §4.3).
    - *Why log-transform?* → Tame skew, stabilize variance, linearize multiplicative
      effects (Fig 1) — for linear/NN, not trees.
    - *Encode hour-of-day?* → sin/cos so 23 and 0 are adjacent (Fig 2).

    **Whiteboard questions**
    - "Implement out-of-fold target encoding." (Section 5.2.)
    - "Design a leak-free feature pipeline for mixed-type data." (`ColumnTransformer`.)

    **Strong vs weak answers**
    - *"Your features gave 0.99 CV but failed live."*
      - **Weak:** "Overfitting."
      - **Strong:** "Likely a leaky feature — naive target encoding or a future-derived
        aggregate. I'd cross-fit target encodings, verify point-in-time correctness,
        and reconcile the offline pipeline with serving."
    - *"More features or a better model?"*
      - **Weak:** "Bigger model."
      - **Strong:** "Usually better features — representation is leverage (the cyclical
        example beats a forest on raw hour). I'd engineer and select features first,
        then consider model complexity, and always measure with leak-free CV."

    **Follow-ups:** "City has 50k values — encode it?" (target/hashing/embedding).
    "Missing 30% of a column?" (impute + flag, or model-based; consider dropping if
    non-informative). "Polynomial features for a tree?" (unnecessary — it finds
    interactions).

    **Common mistakes:** naive target encoding (leak); scaling before split; label-
    encoding unordered categories; ignoring skew for linear models; integer-encoding
    cyclical features; dumping thousands of features without selection; computing
    aggregates that peek at the future.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define feature engineering and why representation matters.
    2. **Why was it invented?** Why transform inputs instead of always using a bigger
       model?
    3. **How does it work?** Explain target encoding and how to make it leak-free.
    4. **Why does it work?** Why does the right transform (log, sin/cos, interaction)
       let a *simpler* model succeed?
    5. **When to use it?** Which transforms matter for linear/NN models vs trees?
    6. **When NOT to use it?** When is a transform unnecessary or harmful (trees +
       scaling, naive target encoding)?
    7. **Tradeoffs?** One-hot vs target encoding; filter vs wrapper vs embedded
       selection.
    8. **How would you productionize it?** Pipeline/serving parity, point-in-time
       features, cross-fitting, and drift monitoring.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. For each model — logistic regression, random forest, kNN — state whether it
       needs feature scaling and why.
    2. Explain why one-hot encoding a 10,000-category column is a bad idea and name two
       alternatives.

    **Beginner → Intermediate (coding)**
    3. Extend the scratch target encoder with **leave-one-out** encoding and compare its
       leakage behavior to out-of-fold k-fold encoding.
    4. Engineer datetime features (cyclical hour + dayofweek + is_weekend) for a
       synthetic demand series and show CV error drops vs raw timestamp.

    **Intermediate (analysis)**
    5. Reproduce Figure 3 and sweep the smoothing $\alpha$; show how it trades rare-
       category overfitting against signal, and how it interacts with the leak.
    6. Compare **filter (mutual information)**, **wrapper (RFE)**, and **embedded
       (L1)** selection on a dataset with many noise features; report selected-feature
       overlap and CV accuracy.

    **Senior (interview + production design)**
    7. *Whiteboard:* design the full, leak-free feature pipeline for the ride-hailing
       demand model of §9 — cyclical time, OOF zone encoding, point-in-time lags,
       weather imputation+flag — and identify every place leakage could enter.
    8. *Design:* propose a feature-store-backed system guaranteeing train/serve parity
       and point-in-time correctness for target-based aggregates, including how you'd
       monitor feature drift.
    9. *Audit:* a teammate's model uses a `customer_lifetime_value` feature and scores
       suspiciously well. Explain why this is likely target leakage and how you'd
       confirm and fix it.
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Feature engineering is changing the data's representation so the signal becomes
    learnable: **scale/transform** numerics for the models that need it,
    **encode** categoricals (one-hot for low cardinality, cross-fit **target encoding**
    for high), make **time cyclical**, add **interactions** for linear models, handle
    **missingness** as signal, and **select** down to the features that matter. The
    recurring senior themes: representation often beats model complexity, the right
    transform is **model-dependent**, and every target-derived feature must be
    **leak-free** (Lesson MLE-02) and **point-in-time correct** at serving.

    **Related lesson:** `MLE-04 · Imbalanced Learning` — we return to the rare-positive problem that
    haunted CML-02 and MLE-01 and tackle it head-on: resampling, class weights,
    threshold-moving, and the metrics that keep you honest when one class is 1% of the
    data.
    """),
]

build("03_ml_engineering/03_feature_engineering.ipynb", cells)
