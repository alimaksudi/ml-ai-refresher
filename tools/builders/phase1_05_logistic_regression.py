"""Builder for Notebook 05 — Logistic Regression.

Run:  python3 tools/builders/phase1_05_logistic_regression.py
Emits: notebooks/phase1_classical_ml/05_logistic_regression.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # 05 · Logistic Regression
    ### Phase 1 — Classical Machine Learning · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** Notebooks 02, 03, 03A, and 04. You should understand
    probability, a leakage-safe split, a linear score, loss, and gradient descent.

    > Keep linear regression's machinery — a linear score $\mathbf w^\top\mathbf x$ —
    > but change the *question* from "what number?" to "what's the **probability** of
    > the positive class?" The recipe from Notebook 02 tells us exactly how: swap the
    > **Gaussian** likelihood for a **Bernoulli** one. Out falls the **sigmoid**, the
    > **cross-entropy** loss, and a **linear decision boundary**. Logistic regression
    > is the workhorse classifier of industry — credit scoring, fraud, churn, ad
    > click-through — *and* it is literally the final layer of most neural-network
    > classifiers (Notebook 14). Master it and you understand classification.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - Why we model **log-odds** linearly and squash with the **sigmoid** to get a
      calibrated probability in $[0,1]$.
    - The full derivation: linear score → odds → log-odds → sigmoid → Bernoulli
      likelihood → **cross-entropy** → gradient → gradient descent (no closed form).
    - Why **cross-entropy beats squared error** for classification (it doesn't
      saturate — strong gradients even when confidently wrong).
    - That the decision boundary is **linear**, and how to read coefficients as
      **odds ratios**.
    - **Regularization**, **class imbalance**, **threshold selection**, and
      **calibration** — the things that actually determine whether it works in prod.
    - **Softmax** as the multiclass generalization.

    **Why it matters in industry**
    - The default interpretable classifier and the baseline every fraud/credit/churn
      project starts with — often the model that ships in regulated settings.
    - Outputs **probabilities**, which let you make **cost-sensitive decisions**
      (different thresholds for different error costs) rather than just labels.
    - It is the read-out layer of deep classifiers — the concept transfers directly.

    **Typical interview questions**
    - "Derive logistic regression from the Bernoulli likelihood."
    - "Why cross-entropy and not MSE for classification?"
    - "What does a coefficient mean? (odds ratio)"
    - "Your model has 99% accuracy on a 1%-fraud dataset — is it good?"
    - "What is perfect separation and why does it break the fit?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **The problem with using linear regression for classification.** If you code the
    classes as 0/1 and fit a line, three things go wrong: (1) predictions escape
    $[0,1]$ (you get "probabilities" of 1.4 or $-0.3$), (2) squared error is the
    wrong loss for a yes/no outcome, and (3) the fit is dragged around by points far
    from the boundary. We need a model whose output is *always* a valid probability.

    **The logistic function (Verhulst, 1830s; Berkson, 1944).** The sigmoid
    $\sigma(z)=1/(1+e^{-z})$ originally modeled population growth, then Berkson
    proposed the "logit" model for bioassay (dose → probability of response),
    explicitly as an alternative to the older "probit". Modeling the **log-odds** as
    linear was the key move: odds live in $(0,\infty)$, log-odds in
    $(-\infty,\infty)$ — the natural range for a linear score.

    **Generalized Linear Models (Nelder & Wedderburn, 1972).** Logistic regression
    is the GLM with a Bernoulli response and a logit link. This unifies it with
    linear regression (Gaussian + identity link) and Poisson regression (counts) —
    same skeleton, different likelihood. That's the lens we adopt: Notebook 04 and 05
    are the *same algorithm* with a different noise model, exactly as Notebook 02
    foreshadowed.

    **Why it endures.** Like linear regression: interpretable (coefficients are
    log-odds), fast, well-calibrated, regulation-friendly, and the foundation for
    everything from GLMs to the softmax head of an LLM classifier.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **From score to probability.** Compute a linear score $z=\mathbf w^\top\mathbf
    x+b$ (high = looks positive, low = looks negative). Then *squash* it through the
    **sigmoid** into a probability: very positive $z\to1$, very negative $z\to0$,
    $z=0\to0.5$. The 0.5 contour — where $z=0$ — is a **straight line** (hyperplane):
    the **decision boundary**.

    **Log-odds is the natural scale.** The model says the **log-odds** of the
    positive class is linear in the features: $\log\frac{p}{1-p}=\mathbf
    w^\top\mathbf x$. Each coefficient is "how much the log-odds change per unit of
    this feature" — and $e^{w_j}$ is the **odds ratio**, a quantity domain experts
    and regulators actually understand ("each extra late payment multiplies the odds
    of default by 1.4×").

    **Probabilities, not just labels.** The output is a *probability*, so you choose
    the **threshold** to match business costs: flag fraud at $p>0.3$ if missing fraud
    is far costlier than a false alarm. Decoupling the *model* from the *decision* is
    a senior habit.

    ```mermaid
    flowchart LR
        X["Features x"] --> Z["Linear score<br/>z = w·x + b"]
        Z --> S["Sigmoid<br/>p = 1/(1+e^-z)"]
        S --> P["Probability p in (0,1)"]
        P -->|"threshold t (set by cost)"| D["Decision: positive / negative"]
        S -.->|"train: maximize Bernoulli likelihood<br/>= minimize cross-entropy"| W["Weights w"]
    ```

    Run the cells: first see *why linear regression fails*, then meet the sigmoid.
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
    # Figure 1 — why a straight line is the wrong model for a 0/1 target.
    x = np.concatenate([rng.normal(-2, 1, 30), rng.normal(2, 1, 30)])
    y = (x > 0).astype(float)

    w_lin = np.linalg.lstsq(np.c_[np.ones_like(x), x], y, rcond=None)[0]
    grid = np.linspace(-6, 6, 200)
    lin_pred = w_lin[0] + w_lin[1] * grid

    def sigmoid(z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="tab:blue", alpha=0.6, label="data (0/1)")
    ax.plot(grid, lin_pred, "r--", label="linear regression fit")
    ax.plot(grid, sigmoid(2.5 * grid), "g-", lw=2, label="logistic (sigmoid) fit")
    ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="k", lw=0.5)
    ax.set_ylim(-0.4, 1.4); ax.legend()
    ax.set_title("Figure 1 — Linear reg leaves [0,1]; the sigmoid stays a valid probability")
    plt.show()
    """),

    md(r"""
    **Figure 1.** The red linear fit shoots **below 0 and above 1** — nonsensical as
    a probability — and its slope is yanked around by points far from the boundary.
    The green **sigmoid** stays inside $[0,1]$, saturates gracefully at the extremes,
    and transitions smoothly through 0.5 at the boundary. This bounded S-shape is
    precisely what we need, and it is *not* an arbitrary choice — it's what the
    Bernoulli likelihood demands (next section).
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Build the model from the requirement "output a probability"
    We want $p=P(y=1\mid\mathbf x)\in(0,1)$ but a linear score lives on all of
    $\mathbb R$. Bridge them through the **odds** and **log-odds**:
    $$\text{odds}=\frac{p}{1-p}\in(0,\infty),\qquad \log\text{-odds}=\log\frac{p}{1-p}\in(-\infty,\infty).$$
    Model the *log-odds* as linear: $\log\frac{p}{1-p}=\mathbf w^\top\mathbf x$.
    Solving for $p$ inverts to the **sigmoid**:
    $$\boxed{\ p=\sigma(\mathbf w^\top\mathbf x)=\frac{1}{1+e^{-\mathbf w^\top\mathbf x}}\ }$$
    The sigmoid *is* the inverse of the logit — it wasn't pulled from a hat.

    ### 4.2 The loss from the Bernoulli likelihood (Notebook 02)
    Each label is Bernoulli: $P(y\mid\mathbf x)=p^{y}(1-p)^{1-y}$. The log-likelihood
    over the dataset is
    $$\ell(\mathbf w)=\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big].$$
    Its negative is the **binary cross-entropy** loss:
    $$J(\mathbf w)=-\frac1n\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big].$$
    So just as MSE was the Gaussian NLL (Notebook 04), **cross-entropy is the
    Bernoulli NLL.** Choosing the loss = choosing the noise model, again.

    ### 4.3 The gradient — and a beautiful coincidence
    Using $\sigma'=\sigma(1-\sigma)$, the gradient simplifies to
    $$\nabla_{\mathbf w}J=\frac1n X^\top(\mathbf p-\mathbf y).$$
    This is the **exact same form** as linear regression's gradient
    $\frac1n X^\top(X\mathbf w-\mathbf y)$ — only the prediction changed from
    $X\mathbf w$ to $\sigma(X\mathbf w)$. Both are GLMs; the gradient is always
    $\frac1n X^\top(\hat{\mathbf y}-\mathbf y)$. There is **no closed form** (the
    sigmoid is nonlinear), so we fit by **gradient descent** (Notebook 03).

    ### 4.4 Convexity — and why cross-entropy, not MSE
    The cross-entropy loss is **convex** in $\mathbf w$ (its Hessian
    $\frac1n X^\top \text{diag}(p_i(1-p_i)) X$ is positive semidefinite), so gradient
    descent reaches the **global** optimum. If you instead used **MSE on the
    sigmoid**, the loss is **non-convex** *and* its gradient **vanishes when the model
    is confidently wrong** ($p(1-p)\to0$), so learning stalls. Cross-entropy's
    gradient $(p-y)$ stays strong exactly when you're most wrong — we visualize this
    in §6, Fig 3. This is *the* reason classifiers use cross-entropy.

    ### 4.5 Reading coefficients; regularization; multiclass
    - **Odds ratio:** $e^{w_j}$ is the multiplicative change in odds per unit
      increase in feature $j$ (holding others fixed) — the interpretable payoff.
    - **Regularization:** add $\lambda\lVert\mathbf w\rVert_2^2$ (L2, the default in
      sklearn) or L1 — same Gaussian/Laplace-prior logic as Notebook 04, and
      *essential* to prevent weights exploding under near-separable data (§7).
    - **Multiclass:** replace the sigmoid with the **softmax**
      $p_k=e^{z_k}/\sum_j e^{z_j}$ and use categorical cross-entropy — the multiclass
      generalization, and exactly the output layer of neural classifiers.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    A complete logistic-regression classifier in NumPy: a numerically stable
    sigmoid, cross-entropy loss, the $\frac1n X^\top(\mathbf p-\mathbf y)$ gradient,
    and gradient-descent fitting with optional L2. We verify it against sklearn in §8.
    """),

    code(r"""
    # 5.1 Stable sigmoid, BCE loss, and gradient-descent fit (with optional L2).
    def sigmoid(z):
        # branch to avoid overflow in exp for large |z|
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def bce(y, p, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)                  # avoid log(0)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit_logistic(X, y, lr=0.1, steps=5000, l2=0.0):
        A = np.c_[np.ones(len(X)), X]
        w = np.zeros(A.shape[1]); n = len(y)
        history = []
        for _ in range(steps):
            p = sigmoid(A @ w)
            grad = (1 / n) * A.T @ (p - y)            # the GLM gradient from Section 4.3
            grad[1:] += (l2 / n) * w[1:]              # L2 (don't penalize intercept)
            w -= lr * grad
            history.append(bce(y, p))
        return w, history

    def predict_proba(X, w):
        return sigmoid(np.c_[np.ones(len(X)), X] @ w)

    # two Gaussian blobs in 2D
    n = 400
    X = np.vstack([rng.normal([-1.5, -1.5], 1.0, (n // 2, 2)),
                   rng.normal([1.5, 1.5], 1.0, (n // 2, 2))])
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]

    w, hist = fit_logistic(X, y, lr=0.5, steps=3000)
    acc = np.mean((predict_proba(X, w) > 0.5) == y)
    print("learned weights [bias, w1, w2]:", w.round(3))
    print(f"training accuracy: {acc:.3f}")
    print(f"final cross-entropy: {hist[-1]:.4f}")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Three pictures: the learned **decision boundary** with probability contours, the
    **convergence** of cross-entropy under gradient descent, and the decisive
    **why-not-MSE** gradient comparison.
    """),

    code(r"""
    # Figure 2 — decision boundary and probability contours of the fitted model.
    xx, yy = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-5, 5, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = predict_proba(grid, w).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cf = axes[0].contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.8)
    axes[0].contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=2)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=15)
    axes[0].set_title("Probability field; black line = 0.5 boundary (linear)")
    fig.colorbar(cf, ax=axes[0], label="P(y=1)")

    axes[1].plot(hist)
    axes[1].set_xlabel("gradient-descent step"); axes[1].set_ylabel("cross-entropy")
    axes[1].set_title("Convex loss converges smoothly")
    plt.suptitle("Figure 2 — Learned boundary and training convergence")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** **Left:** the probability surface runs smoothly from blue (class 0)
    to red (class 1); the black 0.5 contour — the decision boundary — is a **straight
    line**, because the log-odds are linear in $\mathbf x$. Points near the line are
    uncertain ($p\approx0.5$); points far away are confident. **Right:** because
    cross-entropy is convex (§4.4), gradient descent descends smoothly to the global
    optimum — no local minima to worry about.
    """),

    code(r"""
    # Figure 3 — the decisive reason classifiers use cross-entropy, not MSE.
    # Compare the gradient w.r.t. the logit z for a single positive example (y=1).
    p = np.linspace(0.001, 0.999, 400)
    grad_bce = np.abs(p - 1)                          # d/dz of BCE = (p - y)
    grad_mse = np.abs(2 * (p - 1) * p * (1 - p))      # d/dz of MSE-on-sigmoid

    fig, ax = plt.subplots()
    ax.plot(p, grad_bce, label="cross-entropy |dL/dz|", color="tab:green", lw=2)
    ax.plot(p, grad_mse, label="MSE-on-sigmoid |dL/dz|", color="tab:red", lw=2)
    ax.axvspan(0.001, 0.15, color="orange", alpha=0.15)
    ax.annotate("confidently WRONG\n(p~0 but y=1)", (0.02, 0.6), fontsize=9)
    ax.set_xlabel("predicted probability p (true label y=1)")
    ax.set_ylabel("gradient magnitude w.r.t. logit z")
    ax.set_title("Figure 3 — MSE's gradient vanishes when confidently wrong; CE's doesn't")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 3.** Consider a positive example ($y=1$) the model gets *confidently
    wrong* ($p\approx0$, orange zone). **Cross-entropy's** gradient is $|p-1|\approx
    1$ — a strong push to correct the error. **MSE-on-sigmoid's** gradient is
    $|2(p-1)\,p(1-p)|\to0$ because the $p(1-p)$ sigmoid-derivative term **vanishes**
    at the extremes. So with MSE, the most badly misclassified points produce
    almost *no* learning signal — training stalls. Cross-entropy is the loss whose
    gradient is largest exactly where you most need to learn. (Plus MSE+sigmoid is
    non-convex.) This is why every classifier, up to and including LLMs, trains on
    cross-entropy.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Perfect separation** | Weights → ∞; loss → 0; absurd certainty | Linearly separable data; MLE is unbounded | **L2 regularization** (always); bounded solvers |
    | **Nonlinear boundary** | Underfits; high error despite tuning | True boundary isn't linear | Feature engineering (interactions, polynomials); kernels; trees/NNs |
    | **Class imbalance** | 99% "accuracy" by always predicting majority | Accuracy is the wrong metric | `class_weight`, resampling, **threshold tuning**, PR-AUC (Ntbks 09, 12) |
    | **Miscalibration** | Predicted 0.9 ≠ 90% empirical | Regularization/imbalance shift probabilities | Platt / isotonic calibration; reliability plots |
    | **Multicollinearity** | Unstable, uninterpretable coefficients | Correlated features (Notebook 01) | L2; drop/merge features |
    | **Wrong threshold** | Too many false positives/negatives | Using default 0.5 regardless of costs | Choose threshold from the cost matrix / PR curve |

    The cell demonstrates **perfect separation**: on separable data, unregularized
    logistic regression's weights grow without bound (the loss can always be reduced
    by making the sigmoid steeper), while L2 keeps them finite.
    """),

    code(r"""
    # Perfect separation: weights blow up without regularization; L2 tames them.
    Xsep = np.vstack([rng.normal([-3, 0], 0.4, (50, 2)),
                      rng.normal([3, 0], 0.4, (50, 2))])   # cleanly separable
    ysep = np.r_[np.zeros(50), np.ones(50)]

    w_none, _ = fit_logistic(Xsep, ysep, lr=0.5, steps=8000, l2=0.0)
    w_reg, _ = fit_logistic(Xsep, ysep, lr=0.5, steps=8000, l2=2.0)
    print(f"||w|| without L2 : {np.linalg.norm(w_none):8.2f}   <- growing toward infinity")
    print(f"||w|| with L2    : {np.linalg.norm(w_reg):8.2f}   <- bounded, stable")
    print("Both classify perfectly, but the unregularized model is absurdly overconfident.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    sklearn's `LogisticRegression` uses robust solvers (lbfgs, liblinear, saga),
    supports L1/L2/elastic-net penalties, multinomial softmax, and `class_weight`
    for imbalance. Crucially it is **regularized by default** (`C=1.0`, where
    `C=1/λ`) — which is *why* you rarely hit the perfect-separation blow-up in
    practice, but also why you must scale features for the penalty to be fair.
    """),

    code(r"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss, accuracy_score

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0)).fit(X, y)
    p_sklearn = clf.predict_proba(X)[:, 1]
    p_scratch = predict_proba(X, w)

    print(f"sklearn  accuracy {accuracy_score(y, p_sklearn > 0.5):.3f} | "
          f"log-loss {log_loss(y, p_sklearn):.4f}")
    print(f"scratch  accuracy {accuracy_score(y, p_scratch > 0.5):.3f} | "
          f"log-loss {log_loss(y, p_scratch):.4f}")
    print(f"\\nprobability correlation (scratch vs sklearn): "
          f"{np.corrcoef(p_sklearn, p_scratch)[0,1]:.4f}")

    # Coefficients as ODDS RATIOS — the interpretable payoff:
    coefs = clf.named_steps["logisticregression"].coef_[0]
    print("\\nodds ratios exp(w):", np.exp(coefs).round(3),
          "  (>1 raises odds of class 1, <1 lowers them)")
    """),

    md(r"""
    **Scratch vs production.** Our hand-written classifier matches sklearn's
    probabilities closely — the difference is sklearn's default L2 penalty and a
    second-order solver (lbfgs) that converges faster than vanilla GD. The library
    also gives `predict_proba`, `class_weight`, calibrated multinomial softmax, and
    sparse support. And it hands you the coefficients as **odds ratios**, the
    quantity you'll put in front of a domain expert or regulator.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Credit Default Scoring

    **Scenario.** A lender predicts the **probability a loan applicant defaults**,
    using income, debt ratio, payment history, etc. The probability drives the
    approve/decline decision and the interest rate.

    **Why logistic regression?**
    - **Regulation (e.g. fair-lending / ECOA, "right to explanation").** Decisions
      must be explainable; **odds ratios** give a defensible, signed reason per
      feature ("each missed payment multiplies default odds by 1.5×"). This is why
      scorecards — essentially logistic regression — dominate consumer credit.
    - **Calibration matters as much as ranking:** the predicted probability feeds
      expected-loss and pricing calculations, so 0.2 must *mean* 20%.

    **Business objectives:** approve good borrowers, decline likely defaulters, price
    risk correctly, stay compliant.

    **Cost of mistakes (asymmetric!)**
    - **False negative** (approve a defaulter): lose principal — expensive.
    - **False positive** (decline a good borrower): lose the interest margin +
      goodwill — cheaper, but real.
    The cost asymmetry sets the **decision threshold**, *not* the default 0.5. This
    is the core senior insight: the model outputs a probability; the *business*
    chooses the cut-point.

    **Constraints:** protected attributes and proxies handled per law; monotonicity
    often required (more debt ⇒ never *lower* predicted risk); full model
    documentation.

    **KPIs:** AUC / KS (ranking), calibration error (reliability), approval rate,
    realized default rate by score band, fairness across protected groups, and
    expected profit at the chosen threshold.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Threshold ≠ 0.5.** Choose the operating point from the cost matrix or a
      precision/recall target (Notebook 09). One model can serve many thresholds for
      different segments.
    - **Calibration.** Regularization and imbalance distort probabilities; validate
      with **reliability diagrams** and apply **Platt/isotonic** calibration when the
      probability itself is consumed (pricing, expected loss).
    - **Latency / cost.** Inference is a dot product + one sigmoid — microseconds,
      the cheapest model to serve; trivially scalable.
    - **Explainability.** Coefficients → odds ratios out of the box; for nonlinear
      feature engineering, pair with SHAP (Notebook 13). A major reason it ships in
      regulated domains.
    - **Monitoring & drift.** Track score-distribution shift, calibration over time,
      and class-balance changes. Watch coefficient stability across retrains
      (instability ⇒ multicollinearity or pipeline change).
    - **Imbalance.** Use `class_weight`/resampling and PR-based metrics; never report
      plain accuracy on a skewed problem (Notebook 12).
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Logistic Regression vs alternatives:**

    | Dimension | Logistic Regression | XGBoost | SVM (RBF) | Neural Net |
    |---|---|---|---|---|
    | Accuracy on complex data | Lower (linear) | **High** | High | **High** |
    | Probability quality | **Well-calibrated** | Decent (needs calib.) | Poor (not native) | Decent |
    | Interpretability | **High (odds ratios)** | Low (SHAP) | Low | Low |
    | Nonlinearity/interactions | Manual only | **Automatic** | Via kernel | **Automatic** |
    | Latency / cost | **Lowest** | Medium | Medium–High | Higher |
    | Data needed | Low | Medium | Medium | High |
    | Regulatory suitability | **High** | Lower | Low | Lower |

    **L1 vs L2 penalty:**

    | Dimension | L2 (Ridge, default) | L1 (Lasso) |
    |---|---|---|
    | Effect on weights | Shrinks smoothly | Drives some to **zero** |
    | Feature selection | No | **Yes** |
    | Correlated features | Splits weight | Picks one |
    | Use when | Many useful features | Want a sparse, simple model |

    **Senior lesson:** logistic regression is the baseline whose *calibrated
    probabilities + interpretability* are often worth more than a few points of AUC
    from a black box — especially where decisions must be explained.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Derive logistic regression from the Bernoulli likelihood.* (Sections 4.1–4.3.)
    - *What does a coefficient mean?* → Change in **log-odds** per unit feature;
      $e^{w_j}$ is the odds ratio.

    **Deep-dive questions**
    - *Why cross-entropy and not MSE?* → Convexity + non-vanishing gradient when
      confidently wrong (Section 4.4, Fig 3). Be able to *show* the $p(1-p)$ term.
    - *Why is there no closed form, unlike linear regression?* → The sigmoid makes
      the score nonlinear; solve by GD/Newton (IRLS).
    - *What is perfect separation?* → MLE pushes weights to infinity; fix with L2.

    **Whiteboard questions**
    - "Implement logistic regression with gradient descent." (Section 5 — and state
      the gradient is $\frac1n X^\top(\mathbf p-\mathbf y)$.)
    - "Derive the gradient using $\sigma'=\sigma(1-\sigma)$."

    **Strong vs weak answers**
    - *"99% accuracy on 1% fraud — good model?"*
      - **Weak:** "Yes, 99% is great."
      - **Strong:** "Accuracy is meaningless here — predicting 'never fraud' scores
        99%. I'd look at precision/recall, PR-AUC, and the confusion matrix at a
        cost-chosen threshold, and probably use class weights." (Ntbks 09, 12.)
    - *"Your probabilities are off."*
      - **Weak:** "Retrain."
      - **Strong:** "Check calibration with a reliability diagram; regularization or
        imbalance may have distorted them — apply Platt/isotonic calibration and
        validate on held-out data."

    **Follow-ups:** "How do you pick the threshold?" (cost matrix / PR target).
    "Multiclass?" (softmax + categorical cross-entropy). "Relationship to neural
    nets?" (it *is* the one-layer, sigmoid/softmax output).

    **Common mistakes:** reporting accuracy on imbalanced data; using 0.5 blindly;
    confusing calibration with discrimination/ranking; forgetting L2 enables fitting
    near-separable data; interpreting raw coefficients without scaling.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define logistic regression and what it outputs.
    2. **Why was it invented?** Why not just use linear regression for 0/1 targets?
    3. **How does it work?** Walk log-odds → sigmoid → cross-entropy → GD.
    4. **Why does it work?** Why is cross-entropy the right loss (convex, non-
       saturating), and why is the boundary linear?
    5. **When to use it?** Name three production settings where it's the right call.
    6. **When NOT to use it?** Give two situations where it underperforms and why.
    7. **Tradeoffs?** vs XGBoost; L1 vs L2; ranking vs calibration.
    8. **How would you productionize it?** Threshold selection, calibration,
       monitoring, and explainability for a regulated decision.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Worked example:** a logit of zero gives sigmoid probability `0.5`; threshold
    `0.5` predicts positive. A logit of `ln(3)` gives odds 3:1 and probability 0.75.

    **Guided practice (25 min):** calculate sigmoid probabilities for logits
    `[-2,0,2]`, then predict at threshold 0.5. Hint: use `1/(1+exp(-z))`.
    Self-check: probabilities are approximately `[0.119,0.5,0.881]`.

    **Independent practice (45 min):** fit a mean-class-frequency baseline and a
    logistic regression on one stratified split. Report the confusion matrix and list
    false positives and false negatives separately without yet choosing an advanced
    metric.

    **Challenge extension (60 min):** sweep three thresholds and calculate the total
    cost when a false negative costs 10 and a false positive costs 1. Choose from
    validation data, not test data. Calibration and fairness audits remain later work.

    <details><summary><strong>Solution and scoring rubric</strong></summary>

    Award 2 points for correct sigmoid values, 3 for a stratified split and baseline,
    3 for correct error counting, and 2 for a threshold decision tied to cost. Common
    mistakes: treating logits as probabilities, fitting on the test set, assuming
    threshold 0.5 is mandatory, and reporting accuracy without error types.
    **Readiness threshold: 8/10.**
    </details>
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Logistic regression is linear regression's classification twin: the **same
    linear score**, a **sigmoid** to make it a probability, a **Bernoulli
    likelihood** giving the **cross-entropy** loss, and **gradient descent** to fit
    it. Cross-entropy (not MSE) because it's convex and keeps a strong gradient when
    confidently wrong. It outputs calibrated probabilities and interpretable odds
    ratios — which is why it's the default classifier in fraud, credit, and churn,
    and the read-out layer of deep classifiers.

    **Next in the canonical route:** `09 · Evaluation Metrics` turns these scores,
    thresholds, and error types into task-aligned evidence before flexible models.
    """),
]

build("phase1_classical_ml/05_logistic_regression.ipynb", cells)
