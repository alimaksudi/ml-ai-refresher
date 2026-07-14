"""Builder for Notebook 09 — Evaluation Metrics.

Run:  python3 tools/builders/phase2_09_evaluation_metrics.py
Emits: notebooks/phase2_ml_engineering/09_evaluation_metrics.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # 09 · Evaluation Metrics
    ### Core ML Evidence — taught immediately after logistic regression

    **Prerequisites:** Notebooks 04 and 05. You should understand continuous
    predictions, class probabilities, thresholds, and the difference between a
    model score and a decision.

    > We now have one regression model and one classifier. Before learning flexible
    > trees or tuning any model, we need the most consequential question in applied
    > ML: **how do you know your
    > model is any good?** Pick the wrong metric and you will confidently ship a
    > model that destroys value — a fraud detector that catches no fraud but reports
    > "99% accuracy," a churn model optimized for the wrong threshold. A senior
    > engineer treats metric selection as a *business decision*, derives the metrics
    > from a confusion matrix on demand, and knows exactly when **accuracy lies**,
    > why **ROC can flatter** an imbalanced model, and how to turn probabilities into
    > **cost-optimal decisions**. This notebook makes you fluent in all of it.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The **confusion matrix** as the source of all classification metrics, built
      from scratch.
    - **Precision, recall, F1, accuracy** — their formulas, what each protects
      against, and **why accuracy is misleading on imbalanced data**.
    - **ROC/AUC** vs **Precision–Recall/AP** curves from scratch — and the crucial
      rule of thumb: *PR for rare positives, ROC for balanced*.
    - **Threshold selection from a cost matrix** — the bridge from a probability to a
      decision (continuing Notebook 05).
    - **Calibration** (reliability diagrams, **Brier score**) — when the *probability
      itself* matters, not just the ranking.
    - **Regression metrics** (MAE/MSE/RMSE/R²/MAPE) and when each is right.
    - **Multi-class averaging** (micro / macro / weighted) and what each hides.

    **Why it matters in industry**
    - The metric you optimize **is** the behavior you get — a misaligned metric is a
      misaligned product.
    - Imbalanced problems (fraud, disease, churn, ads) are everywhere, and naive
      accuracy is actively dangerous there (Notebook 12).
    - Offline metric → online business outcome is a senior engineer's core
      translation skill.

    **Typical interview questions**
    - "Define precision and recall. When do you care about each?"
    - "Why is accuracy a bad metric for a 1%-positive fraud problem?"
    - "ROC-AUC vs PR-AUC — when do they disagree and which do you trust?"
    - "How do you choose a classification threshold?"
    - "What is calibration and how is it different from discrimination?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **From a single number to a matrix.** Early classifiers were judged by
    **accuracy** — the fraction correct. It's intuitive and was fine when classes
    were balanced. But accuracy collapses *all* error types into one number, hiding
    the distinction that actually matters: a cancer screen that misses tumors
    (false negatives) and one that over-refers healthy patients (false positives) can
    have *identical* accuracy yet wildly different real-world consequences.

    **The confusion matrix and signal-detection theory (WWII radar).** Operators had
    to decide "blip = enemy plane or noise?" — trading missed planes against false
    alarms. This produced the **Receiver Operating Characteristic (ROC)** curve and
    the language of true/false positives/negatives. Medicine adopted the same
    framework (**sensitivity** = recall, **specificity**), because the cost of the two
    error types is so different there.

    **Precision/recall from information retrieval.** As search engines emerged, "how
    many returned documents are relevant (precision) and how many relevant documents
    did we return (recall)?" became the natural questions — and **F1** was introduced
    to summarize the tradeoff in one number.

    **Why metrics multiplied.** Each was invented because the previous one *lied* in
    some regime: accuracy lies under imbalance, ROC-AUC flatters models on rare
    positives, point predictions hide miscalibration, R² hides outlier sensitivity.
    The senior takeaway is not "memorize formulas" but **know which lie each metric
    tells, so you can choose the one that tells the truth for your problem**.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Four outcomes, two error types.** Every binary prediction is one of four cells:

    |  | Predicted Positive | Predicted Negative |
    |---|---|---|
    | **Actually Positive** | True Positive (TP) ✅ | False Negative (FN) ❌ *miss* |
    | **Actually Negative** | False Positive (FP) ❌ *false alarm* | True Negative (TN) ✅ |

    The whole game is that **FN and FP usually cost very different amounts.** A missed
    fraud (FN) loses money; a falsely-declined good customer (FP) loses goodwill.
    Medicine: a missed tumor (FN) can be fatal; a false alarm (FP) means an
    unnecessary biopsy. **You cannot pick a metric without knowing this asymmetry.**

    **Precision answers "when I say positive, am I right?"** — it protects against
    **false alarms**. **Recall answers "of all real positives, how many did I
    catch?"** — it protects against **misses**. They trade off: lowering the threshold
    catches more positives (↑recall) but raises false alarms (↓precision).

    **A model gives a *score*; a threshold turns it into a *decision*.** Most models
    output a probability; *you* choose the cut-point. ROC and PR curves show
    performance across *all* thresholds at once; the cost matrix picks the *one*
    threshold to deploy.

    ```mermaid
    flowchart LR
        S["Model scores<br/>P(positive)"] --> T{"threshold t"}
        T -->|"score >= t"| Pos["predict positive"]
        T -->|"score < t"| Neg["predict negative"]
        Pos --> C["Confusion matrix<br/>TP FP FN TN"]
        Neg --> C
        C --> M["precision · recall · F1<br/>ROC/PR · cost"]
        M -->|"sweep all t"| Curves["ROC & PR curves"]
        M -->|"pick best t"| Decision["deployed threshold"]
    ```

    Run the cells: first, the demo that should make you distrust accuracy forever.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # An IMBALANCED binary problem (3% positives) with realistic model scores.
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    Xc, yc = make_classification(n_samples=5000, n_features=10, n_informative=4,
                                 weights=[0.97, 0.03], class_sep=0.9, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.4, random_state=0, stratify=yc)
    model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    scores = model.predict_proba(Xte)[:, 1]          # P(positive) for each test point
    print(f"test positives: {yte.sum()} / {len(yte)}  ({yte.mean():.1%}) -> heavily imbalanced")
    """),

    code(r"""
    # The ACCURACY PARADOX: a model that predicts 'never positive' scores ~97%.
    dummy_pred = np.zeros_like(yte)                  # always predict the majority class
    dummy_acc = np.mean(dummy_pred == yte)
    model_pred = (scores >= 0.5).astype(int)
    model_acc = np.mean(model_pred == yte)
    print(f"'always negative' accuracy : {dummy_acc:.3f}   <- USELESS model")
    print(f"logistic @0.5    accuracy  : {model_acc:.3f}")
    print(f"...but 'always negative' catches {0}/{yte.sum()} positives (recall = 0).")
    print("Accuracy rewards ignoring the rare class. This is why we need precision/recall.")
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Metrics from the confusion matrix
    $$\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\qquad
    \text{Precision}=\frac{TP}{TP+FP},\qquad
    \text{Recall (TPR/Sensitivity)}=\frac{TP}{TP+FN}.$$
    $$\text{Specificity}=\frac{TN}{TN+FP},\qquad
    \text{FPR}=1-\text{Specificity}=\frac{FP}{FP+TN},\qquad
    F_1=\frac{2\,PR}{P+R}.$$
    $F_1$ is the **harmonic** mean of precision and recall (harsh on imbalance between
    them: you need *both* high). $F_\beta$ weights recall $\beta$ times as much as
    precision when misses matter more.

    ### 4.2 Why accuracy lies under imbalance
    If positives are a fraction $\pi$ of the data, the trivial "always negative"
    classifier scores accuracy $1-\pi$. With $\pi=0.03$ that's **97%** — yet it has
    zero recall and is worthless. Accuracy is dominated by the majority class.
    Precision/recall and PR-AUC focus on the rare positive class and don't reward
    ignoring it. (Notebook 12 is devoted to learning under imbalance.)

    ### 4.3 ROC curve and AUC
    Sweep the threshold from high to low and plot **TPR vs FPR**. The **Area Under the
    Curve (AUC)** has a beautiful interpretation: it is the probability that a random
    positive is scored higher than a random negative,
    $$\text{AUC}=P\big(\text{score}(x^+)>\text{score}(x^-)\big).$$
    AUC = 0.5 is random, 1.0 is perfect. It measures **ranking/discrimination**,
    independent of threshold and of calibration. *Caveat:* because FPR uses the large
    negative count in its denominator, ROC-AUC can look **deceptively good on
    imbalanced data** — a few thousand false positives barely move FPR.

    ### 4.4 Precision–Recall curve and Average Precision
    Plot **precision vs recall** across thresholds. **Average Precision (AP)** is the
    area under it, $\text{AP}=\sum_n (R_n-R_{n-1})P_n$. The PR curve's baseline is the
    positive rate $\pi$ (not 0.5), so it **exposes** poor performance on rare
    positives that ROC hides. **Rule: use PR-AUC when positives are rare and you care
    about them; ROC-AUC when classes are balanced or you care about both errors
    symmetrically.**

    ### 4.5 Threshold selection from a cost matrix
    Assign costs to each error: $C_{FP}$ and $C_{FN}$ (benefits of TP/TN often 0).
    Expected cost at threshold $t$ is $\;\text{Cost}(t)=C_{FP}\,FP(t)+C_{FN}\,FN(t)$.
    Choose $t^*=\arg\min_t \text{Cost}(t)$. **The optimal threshold is rarely 0.5** —
    it's wherever the marginal cost of a false alarm equals the marginal benefit of a
    catch. This operationalizes Notebook 05's "decouple model from decision."

    ### 4.6 Calibration vs discrimination
    A model can **rank** perfectly (AUC 1.0) yet be **miscalibrated** — its "0.9" may
    only be right 70% of the time. **Calibration** = do predicted probabilities match
    empirical frequencies? Measured by a **reliability diagram** (predicted vs
    observed, per bin) and the **Brier score** $\frac1n\sum(p_i-y_i)^2$ (lower
    better; the MSE of probabilities). Calibration matters whenever the *probability*
    is consumed — pricing, expected-value decisions, thresholding (Notebooks 05, 38).

    ### 4.7 Regression metrics
    - **MAE** $\frac1n\sum|y-\hat y|$ — robust, in target units, optimizes the median.
    - **MSE/RMSE** $\sqrt{\frac1n\sum(y-\hat y)^2}$ — penalizes large errors heavily
      (outlier-sensitive), optimizes the mean (Gaussian-NLL link, Notebook 04).
    - **R²** — fraction of variance explained (Notebook 04); can go negative.
    - **MAPE** $\frac1n\sum|\frac{y-\hat y}{y}|$ — scale-free %, but explodes near
      $y=0$ and is asymmetric.

    ### 4.8 Multi-class averaging
    Compute per-class precision/recall, then average: **macro** (unweighted mean —
    treats every class equally, surfaces rare-class failures), **weighted** (by class
    support), **micro** (pool all TP/FP/FN — dominated by frequent classes, equals
    accuracy for single-label). *Which average you report changes the story.*
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    Every metric in this notebook, in pure NumPy, verified against sklearn in §8.
    Understanding the loops is what lets you *debug* a metric and reason about edge
    cases (zero positives, ties) instead of trusting a black box.
    """),

    code(r"""
    # 5.1 Confusion matrix and the metrics derived from it.
    def confusion(y_true, y_pred):
        TP = int(np.sum((y_pred == 1) & (y_true == 1)))
        TN = int(np.sum((y_pred == 0) & (y_true == 0)))
        FP = int(np.sum((y_pred == 1) & (y_true == 0)))
        FN = int(np.sum((y_pred == 0) & (y_true == 1)))
        return TP, FP, FN, TN

    def metrics(y_true, y_pred):
        TP, FP, FN, TN = confusion(y_true, y_pred)
        acc = (TP + TN) / (TP + TN + FP + FN)
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

    print("logistic @ threshold 0.5:")
    for k, v in metrics(yte, (scores >= 0.5).astype(int)).items():
        print(f"  {k:9s}: {v:.3f}")
    print("\\nNote the gap: high accuracy, but recall tells the real (worse) story.")
    """),

    code(r"""
    # 5.2 ROC curve + AUC, and PR curve + AP, all from scratch (threshold sweep).
    def roc_curve_scratch(y, s):
        order = np.argsort(-s); y = y[order]
        P, N = y.sum(), len(y) - y.sum()
        tpr = np.r_[0, np.cumsum(y) / P]
        fpr = np.r_[0, np.cumsum(1 - y) / N]
        return fpr, tpr

    def pr_curve_scratch(y, s):
        order = np.argsort(-s); y = y[order]
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        precision = tp / (tp + fp)
        recall = tp / y.sum()
        return recall, precision

    def auc(x, y):
        return float(np.trapezoid(y, x))            # area via trapezoid rule

    def average_precision(recall, precision):
        return float(np.sum(np.diff(np.r_[0, recall]) * precision))

    fpr, tpr = roc_curve_scratch(yte, scores)
    rec, prec = pr_curve_scratch(yte, scores)
    print(f"ROC-AUC (scratch): {auc(fpr, tpr):.4f}   <- looks great...")
    print(f"PR-AUC / AP (scratch): {average_precision(rec, prec):.4f}   <- ...the honest number")
    print(f"(PR baseline = positive rate = {yte.mean():.3f}; that's the 'random' floor)")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    The figures that make metric choice intuitive: the precision/recall threshold
    tradeoff, ROC-vs-PR (and why they disagree under imbalance), cost-based threshold
    selection, and a calibration diagram.
    """),

    code(r"""
    # Figure 1 — precision, recall, F1 as functions of the decision threshold.
    ts = np.linspace(0.01, 0.99, 99)
    P, R, F = [], [], []
    for t in ts:
        m = metrics(yte, (scores >= t).astype(int))
        P.append(m["precision"]); R.append(m["recall"]); F.append(m["f1"])

    fig, ax = plt.subplots()
    ax.plot(ts, P, label="precision")
    ax.plot(ts, R, label="recall")
    ax.plot(ts, F, label="F1")
    best_t = ts[int(np.argmax(F))]
    ax.axvline(best_t, color="k", ls="--", label=f"max-F1 @ t={best_t:.2f}")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("metric")
    ax.set_title("Figure 1 — Threshold trades precision against recall")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 1.** As the threshold rises, the model is more conservative about saying
    "positive": **precision climbs** (fewer false alarms) while **recall falls**
    (more misses). F1 balances them and peaks somewhere in between — but note that the
    F1-optimal threshold is **not** business-optimal unless false alarms and misses
    cost the same. The default 0.5 is almost never the right operating point on
    imbalanced data; you choose $t$ from the curve that matches your costs (Fig 3).
    """),

    code(r"""
    # Figure 2 — ROC vs PR: ROC looks rosy, PR tells the truth on imbalanced data.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, color="tab:blue", lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="random (0.5)")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve (can flatter imbalance)"); axes[0].legend()

    axes[1].plot(rec, prec, color="tab:red", lw=2,
                 label=f"AP = {average_precision(rec, prec):.3f}")
    axes[1].axhline(yte.mean(), color="k", ls="--", alpha=0.5,
                    label=f"random = {yte.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve (honest on imbalance)"); axes[1].legend()
    plt.suptitle("Figure 2 — Same model, two stories")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** **Same scores, same model — very different impressions.** The ROC
    curve hugs the top-left and reports a high AUC, because FPR's denominator (the
    huge negative count) makes false positives look cheap. The PR curve's baseline is
    the 3% positive rate, and the curve reveals that precision drops sharply as we
    chase recall — the honest picture for a rare-positive problem. **When positives
    are rare and matter (fraud, disease), report PR-AUC/AP, not ROC-AUC.** ROC is the
    right choice for balanced classes or when both error types matter symmetrically.
    """),

    code(r"""
    # Figure 3 — pick the threshold that MINIMIZES expected cost (not max accuracy/F1).
    C_FN, C_FP = 50.0, 1.0                            # a miss costs 50x a false alarm
    costs = []
    for t in ts:
        pred = (scores >= t).astype(int)
        TP, FP, FN, TN = confusion(yte, pred)
        costs.append(C_FP * FP + C_FN * FN)
    costs = np.array(costs)
    t_star = ts[int(np.argmin(costs))]

    fig, ax = plt.subplots()
    ax.plot(ts, costs, color="tab:purple")
    ax.axvline(t_star, color="r", ls="--", label=f"cost-optimal t = {t_star:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="naive t = 0.5")
    ax.set_xlabel("threshold"); ax.set_ylabel(f"expected cost (C_FN={C_FN}, C_FP={C_FP})")
    ax.set_title("Figure 3 — The business-optimal threshold is NOT 0.5")
    ax.legend()
    plt.show()
    print(f"Because a miss costs {C_FN/C_FP:.0f}x a false alarm, the optimal threshold "
          f"drops to {t_star:.2f} (catch more positives).")
    """),

    md(r"""
    **Figure 3.** With misses 50× costlier than false alarms, the expected-cost curve
    is minimized at a threshold **well below 0.5** — we deliberately accept more false
    alarms to avoid expensive misses. The naive 0.5 (gray) sits at a much higher cost.
    This is the single most important production habit in classification: **derive the
    threshold from the cost matrix**, never default to 0.5. (Different segments can
    even use different thresholds.)
    """),

    code(r"""
    # Figure 4 — calibration: does '0.8' really mean 80%? Reliability diagram + Brier.
    def reliability(y, p, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        xs, ys = [], []
        for i in range(bins):
            mask = (p >= edges[i]) & (p < edges[i + 1])
            if mask.sum() > 5:
                xs.append(p[mask].mean()); ys.append(y[mask].mean())
        return np.array(xs), np.array(ys)

    # an overconfident model: push probabilities toward 0/1
    overconf = np.clip((scores - 0.5) * 3 + 0.5, 1e-3, 1 - 1e-3)
    fig, ax = plt.subplots()
    for p, name, c in [(scores, "logistic (well-calibrated)", "tab:blue"),
                       (overconf, "overconfident", "tab:red")]:
        xs, ys = reliability(yte, p)
        brier = np.mean((p - yte) ** 2)
        ax.plot(xs, ys, "o-", color=c, label=f"{name} (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Figure 4 — Reliability diagram: on the diagonal = calibrated")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 4.** A perfectly calibrated model lies on the diagonal: of the cases it
    calls "0.8", 80% are truly positive. Logistic regression (blue) tracks it
    closely (it optimizes log loss, which encourages calibration). The
    **overconfident** model (red) bows away — when it says 0.9 the truth is lower —
    and its **Brier score** (probability MSE) is worse. Crucially, *both could have
    the same AUC*: **calibration (are the probabilities right?) is independent of
    discrimination (is the ranking right?)**. Fix miscalibration with Platt/isotonic
    scaling when the probability is consumed downstream.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Accuracy on imbalance** | "99% accurate," useless model | Majority class dominates | Precision/recall, **PR-AUC**, balanced acc (Ntbk 12) |
    | **ROC-AUC over-optimism** | Great AUC, terrible precision live | FPR insensitive to rare-class FP | Report **PR-AUC/AP** for rare positives |
    | **Default 0.5 threshold** | Wrong precision/recall mix | Ignoring cost asymmetry | Choose $t$ from the **cost matrix** (Fig 3) |
    | **Miscalibration** | Probabilities don't match reality | Regularization/imbalance/tree models | Reliability diagram; Platt/isotonic |
    | **Single metric tunnel vision** | Optimizing one number harms others | Goodhart's law | Track a metric *suite* + guardrails |
    | **MAPE near zero / RMSE outliers** | Exploding % error or outlier-driven RMSE | Wrong regression metric | Match metric to error distribution & cost |
    | **Macro vs micro confusion** | Multi-class score hides rare-class failure | Wrong averaging | Report macro *and* micro; per-class breakdown |
    | **Test-set leakage** | Inflated metrics offline (Ntbk 10) | Info from test in train | Strict splits, pipelines |

    The cell shows how **the same predictions** yield very different multi-class
    scores depending on the averaging — a real source of miscommunication.
    """),

    code(r"""
    # Multi-class averaging: micro vs macro tell different stories under imbalance.
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.datasets import make_classification as mk
    from sklearn.linear_model import LogisticRegression as LR

    Xm, ym = mk(n_samples=3000, n_features=10, n_informative=6, n_classes=3,
                weights=[0.8, 0.15, 0.05], random_state=0)
    mdl = LR(max_iter=1000).fit(Xm[:2000], ym[:2000])
    pred = mdl.predict(Xm[2000:]); true = ym[2000:]
    for avg in ["micro", "macro", "weighted"]:
        f = f1_score(true, pred, average=avg, zero_division=0)
        print(f"F1 ({avg:8s}) = {f:.3f}")
    print("\\nmicro ~ accuracy (dominated by the 80% majority class);")
    print("macro is much lower because the rare 5% class is predicted poorly.")
    print("Reporting only 'micro' would hide that failure.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    `sklearn.metrics` provides all of these (`confusion_matrix`, `classification_report`,
    `roc_auc_score`, `average_precision_score`, `roc_curve`, `precision_recall_curve`,
    `brier_score_loss`, regression metrics) with correct handling of ties, edge cases,
    and multi-class. We verify our scratch numbers match.
    """),

    code(r"""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_score, recall_score, f1_score,
                                 brier_score_loss, mean_absolute_error,
                                 mean_squared_error, r2_score)

    print("=== classification (verify scratch == sklearn) ===")
    print(f"ROC-AUC  scratch {auc(fpr, tpr):.4f} | sklearn {roc_auc_score(yte, scores):.4f}")
    print(f"AP       scratch {average_precision(rec, prec):.4f} | "
          f"sklearn {average_precision_score(yte, scores):.4f}")
    pred = (scores >= 0.5).astype(int)
    print(f"precision sklearn {precision_score(yte, pred):.4f} | "
          f"recall {recall_score(yte, pred):.4f} | f1 {f1_score(yte, pred):.4f}")
    print(f"Brier    sklearn {brier_score_loss(yte, scores):.4f}")

    print("\\n=== regression metrics: MAE vs RMSE and outlier sensitivity ===")
    yt = rng.normal(10, 3, 500); yp = yt + rng.normal(0, 1, 500)
    yp_out = yp.copy(); yp_out[:5] += 30                  # a few big errors
    for label, p in [("clean", yp), ("with outliers", yp_out)]:
        mae = mean_absolute_error(yt, p)
        rmse = mean_squared_error(yt, p) ** 0.5
        print(f"{label:13s}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2_score(yt, p):.3f}")
    print("-> RMSE jumps far more than MAE under outliers (squares the big errors).")
    """),

    md(r"""
    **Scratch vs production.** Our hand-rolled AUC/AP/precision/recall match sklearn
    to the displayed precision — proving (again) there's no magic, just bookkeeping
    over the confusion matrix and a threshold sweep. The library adds correct
    tie-handling, multi-class/multi-label support, sample weighting, and speed. The
    regression block shows the key judgment call: **RMSE balloons under outliers while
    MAE barely moves** — pick MAE (robust, optimizes the median) or RMSE (penalizes
    large misses, optimizes the mean) based on whether big errors are catastrophic or
    just noise.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Choosing the Metric for Cancer Screening

    **Scenario.** An ML model flags mammograms for radiologist review. It outputs a
    probability of malignancy; the hospital must choose how to evaluate and threshold
    it.

    **Why metric choice is the whole game here:**
    - Malignancy is **rare** (~0.5% of screens) → **accuracy is meaningless** ("all
      benign" scores 99.5%). Use **recall (sensitivity)** and **PR-AUC**.
    - A **false negative** (missed cancer) can be fatal; a **false positive** means an
      unnecessary follow-up (stressful, costly, but recoverable). The cost asymmetry
      is enormous → optimize **recall at a tolerable false-positive rate**, and set the
      threshold from that cost matrix, not 0.5.
    - The output feeds clinical decisions, so **calibration matters**: "30% malignancy
      risk" must mean 30%.

    **Business objectives:** maximize cancers caught (recall) subject to a cap on
    false positives the radiology team can absorb; provide trustworthy probabilities.

    **Cost of mistakes:** FN = missed cancer (catastrophic); FP = unnecessary biopsy
    (costly/anxiety-inducing). The ratio drives the operating threshold.

    **Constraints:** regulatory scrutiny, radiologist capacity (caps FP volume),
    requirement for calibrated, explainable outputs.

    **KPIs:** recall at fixed FP-rate (or sensitivity at fixed specificity), PR-AUC,
    calibration error (reliability/Brier), and the realized catch-rate vs review
    burden in deployment. *Reporting plain accuracy here would be malpractice.*
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Pick the metric from the business cost, then freeze it.** The optimized metric
      becomes the product's behavior (Goodhart's law) — choose deliberately and track
      a **suite + guardrails**, not one number.
    - **Threshold is a deployment parameter.** Re-tune it as base rates and costs shift
      — the model can stay fixed while the threshold moves. Support per-segment
      thresholds.
    - **Monitor metrics over time.** A drop in precision/recall at a fixed threshold
      signals drift or a pipeline break (Notebook 45); track the **score distribution**
      too.
    - **Calibration drift.** Probabilities can decalibrate as data shifts; periodically
      re-fit a calibrator and watch the Brier score / reliability.
    - **Offline ≠ online.** Great offline metrics with bad live results usually mean
      **leakage** (Notebook 10) or train/serve skew — validate with an online A/B test
      (Notebook 02) measuring the real KPI.
    - **Class imbalance** is the norm in high-value problems; default to PR-based
      metrics and never report bare accuracy there (Notebook 12).
    - **Confidence intervals** on metrics: bootstrap them (Notebook 02) — a 0.91 vs
      0.92 AUC on 200 positives may not be a real difference.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Classification metric selection:**

    | Metric | Measures | Best when | Blind spot |
    |---|---|---|---|
    | Accuracy | Overall correctness | **Balanced** classes, equal costs | Lies under imbalance |
    | Precision | Correctness of positive calls | False alarms costly | Ignores misses |
    | Recall | Coverage of real positives | Misses costly (fraud, cancer) | Ignores false alarms |
    | F1 | P/R balance | Need both, unknown costs | Hides cost asymmetry |
    | ROC-AUC | Ranking, threshold-free | Balanced; both errors matter | Over-optimistic on rare positives |
    | PR-AUC / AP | Ranking on positives | **Rare positives** | Less intuitive baseline |
    | Brier / log loss | Calibration + sharpness | Probabilities consumed | Not a decision metric alone |

    **Regression metric selection:**

    | Metric | Penalizes | Use when | Watch out |
    |---|---|---|---|
    | MAE | Errors linearly | Robust, outliers are noise | Less sensitive to big misses |
    | RMSE/MSE | Errors quadratically | Large errors are costly | Outlier-dominated |
    | R² | Variance unexplained | Communicate fit | Negative on bad test data |
    | MAPE | Relative % error | Scale-free comparison | Explodes near $y=0$; asymmetric |

    **Multi-class averaging:** *micro* (≈accuracy, majority-dominated) vs *macro*
    (rare classes count equally) vs *weighted* (by support). Always report the
    per-class breakdown for high-stakes minority classes.

    **Senior lesson:** there is no universal "best metric" — only the metric that
    encodes *your* cost structure. Stating that explicitly is the senior signal.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Define precision and recall; when each?* → Precision = correctness of positive
      calls (guard false alarms); recall = coverage of positives (guard misses).
    - *Why does accuracy fail on imbalanced data?* → "Always majority" scores $1-\pi$;
      rewards ignoring the rare class (the demo).

    **Deep-dive questions**
    - *ROC-AUC vs PR-AUC — when disagree?* → Under heavy imbalance ROC looks great
      while PR is poor; trust PR for rare positives (Fig 2; explain the FPR
      denominator).
    - *What is AUC, probabilistically?* → $P(\text{score}^+>\text{score}^-)$.
    - *Calibration vs discrimination?* → Right probabilities vs right ranking;
      independent (Fig 4).

    **Whiteboard questions**
    - "Compute precision/recall/F1 from a confusion matrix." (Section 5.1.)
    - "Sketch how to build a ROC curve from scores." (Threshold sweep; Section 5.2.)

    **Strong vs weak answers**
    - *"My fraud model is 99% accurate."*
      - **Weak:** "Excellent."
      - **Strong:** "On ~1% fraud, 99% is what 'always legit' scores. I need recall,
        precision, and PR-AUC at a cost-chosen threshold — accuracy is meaningless
        here."
    - *"How do you set the threshold?"*
      - **Weak:** "0.5."
      - **Strong:** "From the cost matrix: minimize $C_{FP}\cdot FP+C_{FN}\cdot FN$,
        or hit a precision/recall target the business specifies. 0.5 is almost never
        right on imbalanced data."

    **Follow-ups:** "Probabilities feed pricing — which metric?" (calibration/Brier).
    "Report one number to a VP?" (the business KPI, with a CI). "Multi-class
    imbalanced — macro or micro?" (macro to expose rare-class failure).

    **Common mistakes:** reporting accuracy on imbalanced data; trusting ROC-AUC for
    rare positives; defaulting to 0.5; conflating calibration with AUC; using RMSE
    when outliers are just noise; MAPE with near-zero targets; ignoring metric CIs.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Draw the confusion matrix and define the four cells.
    2. **Why was it invented?** Why did we move beyond accuracy to precision/recall
       and ROC?
    3. **How does it work?** Derive precision, recall, F1; describe building ROC & PR
       curves.
    4. **Why does it work?** Why is AUC threshold-free, and why does PR beat ROC on
       rare positives?
    5. **When to use it?** Map three business scenarios to the right metric.
    6. **When NOT to use it?** When is accuracy/ROC/MAPE the wrong call?
    7. **Tradeoffs?** Precision vs recall; ROC vs PR; MAE vs RMSE; macro vs micro.
    8. **How would you productionize it?** Threshold from costs, calibration, metric
       monitoring, and offline↔online validation.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Worked example:** with `TP=8, FP=2, FN=4, TN=86`, precision is `8/10=0.8`,
    recall is `8/12≈0.667`, and accuracy is `94/100=0.94`. High accuracy hides four
    missed positives.

    **Guided practice (25 min)**
    1. A 1000-sample test set has 20 positives. A model predicts all negative.
       Compute accuracy, precision, recall, F1. Which reveal the problem?
    2. Explain in two sentences why ROC-AUC can be high while PR-AUC is low.

    **Independent practice (45 min)**
    3. Implement **$F_\beta$** and show how $\beta>1$ shifts the optimal threshold
       toward higher recall.
    4. Choose a threshold from an explicit false-positive/false-negative cost table
       using validation predictions.

    **Challenge extension (60 min)**
    5. Bootstrap a **95% CI** for one chosen metric (Notebook 02) and decide whether
       two models with AUC 0.91 vs 0.92 are *really* different.

    <details><summary><strong>Solution and scoring rubric</strong></summary>

    For Question 1: accuracy is 0.98, recall is 0, F1 is 0, and precision is undefined
    (commonly reported as 0); recall/F1 reveal total failure. Award 2 points for the
    confusion-matrix calculation, 2 for metric interpretation, 3 for correct
    validation-only threshold selection, and 3 for uncertainty reasoning. Common
    mistakes: choosing on test data, treating undefined precision as success, and
    claiming overlapping point estimates prove equivalence. **Readiness: 8/10.**
    </details>
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Every classification metric falls out of the **confusion matrix**; the art is
    choosing the one whose lie doesn't matter for *your* problem. **Accuracy lies
    under imbalance**; **ROC-AUC flatters rare-positive problems** while **PR-AUC**
    tells the truth; the **threshold** is a business decision set from a **cost
    matrix**, not 0.5; and **calibration** (Brier/reliability) is a separate axis from
    discrimination (AUC). For regression, MAE vs RMSE encodes how much you fear large
    errors. *Optimize the metric that is your business objective — nothing else.*

    **Next:** `10 · Validation and Data Leakage` — even the perfect metric lies if you
    measure it on contaminated data. We make evaluation *trustworthy*: train/val/test
    discipline, cross-validation, and the leakage traps that produce "too good to be
    true" offline numbers and live disasters.
    """),
]

build("phase2_ml_engineering/09_evaluation_metrics.ipynb", cells)
