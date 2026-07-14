"""Builder for Notebook 04 — Linear Regression.

Run:  python3 tools/builders/phase1_04_linear_regression.py
Emits: notebooks/phase1_classical_ml/04_linear_regression.ipynb
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # 04 · Linear Regression
    ### Phase 1 — Classical Machine Learning · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** Notebooks 01–02 and 03A. You should be able to reason about
    vectors, samples, a prediction target, a naive baseline, and a train/test split.

    > This is the first *complete* learning algorithm, and it is where the first
    > Phase-0 pillars snap together: the **geometry** of Notebook 01 (least squares
    > is a projection onto a column space), the **probability** of Notebook 02 (MSE
    > is the negative log-likelihood of Gaussian noise; Ridge/Lasso are Gaussian/
    > Laplace priors), and the **data contract** of Notebook 03A. This notebook
    > gives gradient descent its first concrete objective: squared loss. Notebook
    > 03 follows and generalizes that optimization method. Master linear regression deeply and
    > you have a template for every supervised model that follows — logistic
    > regression, neural nets, even the read-out layer of an LLM are the same idea
    > with different link functions and losses.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - The linear model $\hat y=\mathbf w^\top\mathbf x+b$ as a weighted sum, and
      why "linear" means *linear in the parameters* (so polynomials still count).
    - The squared-error **loss function** as the explicit quantity learning will
      minimize, first with a tiny numeric example.
    - Two ways to fit it — the **closed-form normal equations** and a preview of
      **gradient descent** — and when each is the right tool. Notebook 03 derives
      and diagnoses iterative optimization in depth after this concrete example.
    - The **probabilistic justification**: minimizing MSE = maximizing Gaussian
      likelihood (Notebook 02), which is why squared error and not something else.
    - The **Gauss–Markov** assumptions and what each one buys you (and what breaks
      when it's violated).
    - **Ridge (L2)** and **Lasso (L1)** regularization as MAP estimation, and the
      **bias–variance tradeoff** they navigate.
    - How to *diagnose* a regression with residual plots, $R^2$, and learning curves.

    **Why it matters in industry**
    - It is the **interpretable baseline** every serious project starts with — and
      in regulated domains (credit, insurance, healthcare) it's often the model that
      actually ships because you can explain every coefficient.
    - Fast to train, trivial to serve, easy to monitor — the cost/latency floor.
    - The mental model transfers directly to GLMs, logistic regression, and the
      linear layers inside deep nets.

    **Typical interview questions**
    - "Derive the OLS solution. Why squared error and not absolute error?"
    - "State the assumptions of linear regression. Which matter most in practice?"
    - "Ridge vs Lasso — what's the difference and when do you use each?"
    - "What is the bias–variance tradeoff and how does regularization move along it?"
    - "Your $R^2$ is high but predictions are bad on new data. What's going on?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **Least squares (Legendre 1805, Gauss 1809).** As in Notebook 02, the method
    was born from astronomy/geodesy: fit a model to noisy measurements by minimizing
    the sum of squared errors. Gauss showed that *if* the noise is Gaussian, least
    squares **is** maximum likelihood — the first principled reason to square the
    residuals rather than, say, take absolute values.

    **"Regression" (Galton, 1880s).** Francis Galton studied how tall parents have
    tall children, but *less extreme* than themselves — heights "regress toward the
    mean." The name stuck even though we now use it for any continuous-target
    prediction. The phenomenon (regression to the mean) is itself a classic
    interview trap (Section 7).

    **Why it still matters after 200 years.** Newer models (trees, boosting, deep
    nets) beat linear regression on raw accuracy for complex data. So why teach it
    first and why does it still ship?
    - **Interpretability:** each coefficient is a quantified, signed, auditable
      effect — essential where decisions must be explained or are legally regulated.
    - **Data efficiency & stability:** with few rows or many features, a regularized
      linear model often *generalizes better* than a high-variance flexible model.
    - **Speed:** closed-form or a few GD steps; microsecond inference.
    - **Foundation:** it is the simplest member of the family (GLMs → logistic →
      neural nets) and the cleanest place to learn loss design, regularization, and
      the bias–variance tradeoff that govern *all* of them.

    The throughline of Phase 1: start with the most transparent model, understand
    *why* it works and where it fails, then earn the right to add complexity.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The line of best fit.** Given a cloud of points, draw the straight line (or
    hyperplane, in higher dimensions) that comes *closest* to all of them. "Closest"
    means the **vertical** gaps between each point and the line — the **residuals** —
    are as small as possible *on average*. We square those gaps (so positives and
    negatives don't cancel, and big misses are penalized more) and minimize the sum.

    **Two complementary pictures of the same fit:**
    - *Statistics view:* find the slope/intercept that minimize total squared
      residual — the wiggling-the-ruler picture.
    - *Linear-algebra view (Notebook 01):* the predictions $\hat{\mathbf y}=X\mathbf
      w$ can only land in the column space of $X$; least squares is the **orthogonal
      projection** of the target $\mathbf y$ onto that subspace. The residual is
      what's left over, perpendicular to everything $X$ can express.

    **Why squared error specifically?** Because (Notebook 02) it's the negative
    log-likelihood when we assume the noise is Gaussian. Squared error isn't a
    random choice — it's the *right* loss under a specific, checkable assumption
    about the world.

    ```mermaid
    flowchart LR
        X["Features X"] --> M["Linear model<br/>y_hat = Xw + b"]
        M --> R["Residuals<br/>y - y_hat"]
        R --> L["Loss = mean(residual^2)<br/>(= Gaussian NLL)"]
        L -->|"minimize: normal equations<br/>OR gradient descent"| W["Best weights w"]
        W -.->|"+ L2 / L1 penalty"| Reg["Ridge / Lasso<br/>(bias-variance control)"]
    ```

    Run the cells to see the line, the residuals, and what happens when the
    assumptions break.
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
    # Figure 1 — the line of best fit and the residuals it minimizes.
    n = 40
    x = np.sort(rng.uniform(0, 10, n))
    y = 2.0 + 1.3 * x + rng.normal(0, 1.5, n)          # true: intercept 2, slope 1.3

    # fit a simple OLS line (we derive this properly in Section 4/5)
    A = np.c_[np.ones(n), x]
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ w

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="tab:blue", label="data")
    ax.plot(x, yhat, color="tab:red", lw=2, label=f"fit: y = {w[0]:.2f} + {w[1]:.2f}x")
    for xi, yi, yh in zip(x, y, yhat):
        ax.plot([xi, xi], [yi, yh], color="gray", lw=0.8, alpha=0.7)  # residual segments
    ax.set_title("Figure 1 — Best-fit line minimizes the squared vertical residuals")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 1.** The gray segments are residuals — the vertical misses. Ordinary
    least squares (OLS) chooses the one line that minimizes the *sum of their
    squares*. Note it's **vertical** distance (error in predicting $y$), not
    perpendicular distance — we're predicting $y$ from $x$, not modeling a symmetric
    relationship (that would be PCA / total least squares). Squaring means a few
    large misses dominate the fit — the source of OLS's outlier sensitivity (§7).
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 The model and the loss
    With features $\mathbf x\in\mathbb R^d$ (a bias term folded in as a constant 1),
    weights $\mathbf w$, and stacked data $X\in\mathbb R^{n\times(d+1)}$:
    $$\hat{\mathbf y}=X\mathbf w,\qquad
    J(\mathbf w)=\frac1n\lVert X\mathbf w-\mathbf y\rVert_2^2\ \ (\text{MSE}).$$

    ### 4.2 Why squared error — the probabilistic view (Notebook 02)
    Assume $y_i=\mathbf w^\top\mathbf x_i+\varepsilon_i$ with $\varepsilon_i\sim
    \mathcal N(0,\sigma^2)$ IID. The log-likelihood is
    $\ell(\mathbf w)=-\frac{1}{2\sigma^2}\sum_i (y_i-\mathbf w^\top\mathbf x_i)^2+\text{const}$.
    **Maximizing $\ell$ is exactly minimizing $\sum_i (y_i-\hat y_i)^2$.** So OLS is
    MLE under Gaussian noise — that is the justification for the square.

    ### 4.3 The closed-form solution (Notebook 01)
    Setting $\nabla_{\mathbf w}J=0$ gives the **normal equations**
    $X^\top X\,\mathbf w=X^\top\mathbf y$, hence
    $$\boxed{\ \hat{\mathbf w}_{\text{OLS}}=(X^\top X)^{-1}X^\top\mathbf y\ }$$
    — the orthogonal projection of $\mathbf y$ onto $X$'s column space. **As warned
    in Notebook 01, we don't actually invert $X^\top X$ in code** (it squares the
    condition number); we use QR/SVD via `lstsq`, or gradient descent for large $n$.

    ### 4.4 Gauss–Markov assumptions (what makes OLS trustworthy)
    Under (1) **linearity** in parameters, (2) **exogeneity** $\mathbb E[\varepsilon
    \mid X]=0$, (3) **homoscedasticity** (constant error variance), (4)
    **uncorrelated errors** (independence), and (5) **no perfect multicollinearity**,
    OLS is **BLUE** — the Best (lowest-variance) Linear Unbiased Estimator. Add
    **normality** of errors and you also get exact $t$/$F$ tests and confidence
    intervals. Each violated assumption maps to a specific failure in §7.

    ### 4.5 Regularization = MAP with a prior (Notebook 02)
    Put a prior on the weights and do MAP instead of MLE:
    - **Ridge (L2):** add $\lambda\lVert\mathbf w\rVert_2^2$. Closed form
      $\hat{\mathbf w}=(X^\top X+\lambda I)^{-1}X^\top\mathbf y$. The $+\lambda I$
      lifts the smallest singular values off zero — **fixing both overfitting and
      ill-conditioning/multicollinearity** at once. This is a **Gaussian prior** on
      $\mathbf w$. It shrinks coefficients smoothly but never to exactly zero.
    - **Lasso (L1):** add $\lambda\lVert\mathbf w\rVert_1$. A **Laplace prior**; its
      diamond geometry (Notebook 01, Fig 4) drives some coefficients **exactly to
      zero**, doing automatic feature selection. No closed form (non-smooth) —
      solved by coordinate descent or subgradient/proximal methods.
    - **Elastic Net:** a mix of both.

    ### 4.6 The bias–variance tradeoff
    Expected test error decomposes as
    $$\mathbb E[(y-\hat f(x))^2]=\underbrace{(\text{bias})^2}_{\text{too simple}}+\underbrace{\text{variance}}_{\text{too sensitive}}+\underbrace{\sigma^2}_{\text{irreducible}}.$$
    A too-simple model (e.g. a line for a curve) has high **bias** (underfits); a
    too-flexible model (high-degree polynomial) has high **variance** (overfits,
    chasing noise). Regularization deliberately *adds a little bias to remove a lot
    of variance*. We will see this U-shaped curve directly in §6.

    ### 4.7 Goodness of fit: $R^2$
    $R^2=1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$ — the fraction of target
    variance the model explains (1 = perfect, 0 = no better than predicting the
    mean, and it can go **negative** on test data for a bad model). High training
    $R^2$ with poor test performance is the signature of overfitting (§7).
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    A tiny linear-regression toolkit in pure NumPy: OLS (the stable way and the
    naive way, to feel the difference), gradient descent, Ridge in closed form, and
    an $R^2$ scorer. We verify each against the others.
    """),

    code(r"""
    # 5.1 A minimal OLS / Ridge / GD toolkit.
    def add_bias(X):
        return np.c_[np.ones(len(X)), X]

    def fit_ols(X, y):                       # stable: QR/SVD via lstsq (Notebook 01)
        return np.linalg.lstsq(add_bias(X), y, rcond=None)[0]

    def fit_ridge(X, y, lam):                # closed form: (A^T A + lam I)^-1 A^T y
        A = add_bias(X)
        I = np.eye(A.shape[1]); I[0, 0] = 0.0   # don't penalize the intercept
        return np.linalg.solve(A.T @ A + lam * I, A.T @ y)

    def fit_gd(X, y, lr=0.05, steps=5000):   # gradient descent (Notebook 03)
        A = add_bias(X); w = np.zeros(A.shape[1]); n = len(y)
        for _ in range(steps):
            w -= lr * (2 / n) * A.T @ (A @ w - y)
        return w

    def predict(X, w):
        return add_bias(X) @ w

    def r2(y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

    # synthetic multi-feature data
    n, d = 300, 3
    Xd = rng.normal(size=(n, d))
    true_w = np.array([5.0, 1.5, -2.0, 0.8])   # [intercept, w1, w2, w3]
    yd = add_bias(Xd) @ true_w + rng.normal(0, 0.5, n)

    w_ols = fit_ols(Xd, yd)
    w_gd = fit_gd(Xd, yd)
    print("OLS (lstsq) :", w_ols.round(3))
    print("GD          :", w_gd.round(3))
    print("true        :", true_w)
    print("GD ~ OLS:", np.allclose(w_ols, w_gd, atol=1e-2), "| train R^2:",
          round(r2(yd, predict(Xd, w_ols)), 4))
    """),

    code(r"""
    # 5.2 Why we don't invert X^T X: the naive normal-equation formula loses accuracy
    # on ill-conditioned data, while lstsq stays solid (the Notebook-01 lesson, applied).
    Xill = Xd.copy()
    Xill[:, 2] = Xill[:, 1] + 1e-6 * rng.normal(size=n)   # column 2 ~ column 1 (collinear)
    A = add_bias(Xill)
    w_naive = np.linalg.inv(A.T @ A) @ A.T @ yd           # DON'T do this in production
    w_stable = np.linalg.lstsq(A, yd, rcond=None)[0]
    w_ridge = fit_ridge(Xill, yd, lam=1.0)
    print(f"cond(X^T X)          : {np.linalg.cond(A.T @ A):.2e}")
    print("naive inv coefs      :", w_naive.round(2), " <- wild / unstable")
    print("ridge (lam=1) coefs  :", w_ridge.round(2), " <- tamed by L2 penalty")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    The three figures every regression practitioner reads instinctively: residual
    diagnostics, the bias–variance U-curve, and the regularization path.
    """),

    code(r"""
    # Figure 2 — residual diagnostics: healthy vs heteroscedastic (variance grows with x).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    xh = np.sort(rng.uniform(0, 10, 200))
    # left: well-behaved (constant variance)
    yg = 1 + 0.8 * xh + rng.normal(0, 1.0, 200)
    wg = np.linalg.lstsq(np.c_[np.ones(200), xh], yg, rcond=None)[0]
    resg = yg - np.c_[np.ones(200), xh] @ wg
    axes[0].scatter(np.c_[np.ones(200), xh] @ wg, resg, s=12, color="tab:green")
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_title("Healthy: residuals are a structureless band")

    # right: heteroscedastic (variance increases with x) -> a fan shape
    yb = 1 + 0.8 * xh + rng.normal(0, 0.3 * xh + 0.1, 200)
    wb = np.linalg.lstsq(np.c_[np.ones(200), xh], yb, rcond=None)[0]
    resb = yb - np.c_[np.ones(200), xh] @ wb
    axes[1].scatter(np.c_[np.ones(200), xh] @ wb, resb, s=12, color="tab:red")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_title("Heteroscedastic: tell-tale 'fan' (assumption violated)")
    for ax in axes:
        ax.set_xlabel("fitted value"); ax.set_ylabel("residual")
    plt.suptitle("Figure 2 — Residual plots diagnose assumption violations")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** The residuals-vs-fitted plot is the single most informative
    regression diagnostic. **Left:** a featureless horizontal band — assumptions
    hold. **Right:** a **fan** that widens with the fitted value —
    *heteroscedasticity* (error variance isn't constant). OLS is still unbiased but
    no longer minimum-variance, and your confidence intervals are wrong. Fixes:
    transform the target (e.g. $\log y$), use weighted least squares, or use
    robust/heteroscedasticity-consistent standard errors. *Any* visible structure
    (a curve, a trend) means the model is missing something.
    """),

    code(r"""
    # Figure 3 — the bias-variance tradeoff via polynomial degree.
    def poly_design(x, degree):
        xs = (x - x.mean()) / x.std()                 # scale for conditioning (Notebook 03)
        return np.vander(xs, degree + 1, increasing=True)

    xt = np.sort(rng.uniform(-3, 3, 60))
    ytrue = np.sin(xt) + 0.3 * xt
    yobs = ytrue + rng.normal(0, 0.3, len(xt))
    # train/test split
    idx = rng.permutation(len(xt)); tr, te = idx[:40], idx[40:]

    degrees = range(1, 16)
    train_err, test_err = [], []
    for dg in degrees:
        Ad = poly_design(xt, dg)
        w = np.linalg.lstsq(Ad[tr], yobs[tr], rcond=None)[0]
        train_err.append(np.mean((Ad[tr] @ w - yobs[tr]) ** 2))
        test_err.append(np.mean((Ad[te] @ w - yobs[te]) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(list(degrees), train_err, "o-", label="train error")
    axes[0].plot(list(degrees), test_err, "s-", label="test error")
    axes[0].set_yscale("log"); axes[0].set_xlabel("polynomial degree (complexity)")
    axes[0].set_ylabel("MSE (log)"); axes[0].legend()
    axes[0].set_title("Train error always drops; test error is U-shaped")

    xx = np.linspace(-3, 3, 200)
    for dg, color in [(1, "tab:blue"), (4, "tab:green"), (15, "tab:red")]:
        w = np.linalg.lstsq(poly_design(xt, dg), yobs, rcond=None)[0]
        xs = (xx - xt.mean()) / xt.std()
        axes[1].plot(xx, np.vander(xs, dg + 1, increasing=True) @ w,
                     color=color, label=f"degree {dg}")
    axes[1].scatter(xt, yobs, s=12, color="gray", alpha=0.6)
    axes[1].set_ylim(-3, 3); axes[1].legend()
    axes[1].set_title("Underfit (1) · good (4) · overfit (15)")
    plt.suptitle("Figure 3 — Bias-variance: complexity vs generalization")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** **Left:** training error falls monotonically with model complexity
    — a more flexible model *always* fits the training data better, which is exactly
    why training error is a liar (Notebook 10). Test error is **U-shaped**: it
    improves while we reduce bias, then worsens as variance takes over. The bottom of
    the U is the sweet spot. **Right:** degree-1 underfits (high bias, misses the
    curve), degree-15 overfits (high variance, wiggles through noise and explodes at
    the edges), degree-4 is just right. Regularization lets us pick a flexible basis
    *and* pull it back toward simplicity — the next figure.
    """),

    code(r"""
    # Figure 4 — the Ridge regularization path: coefficients shrink as lambda grows.
    # Use a collinear design so the effect is dramatic.
    p = 8
    Xc = rng.normal(size=(150, p))
    Xc[:, 1] = Xc[:, 0] + 0.01 * rng.normal(size=150)   # near-duplicate features
    Xc[:, 3] = Xc[:, 2] + 0.01 * rng.normal(size=150)
    beta = rng.normal(size=p)
    yc = Xc @ beta + rng.normal(0, 0.5, 150)

    lams = np.logspace(-2, 4, 50)
    paths = []
    for lam in lams:
        A = np.c_[np.ones(150), Xc]
        I = np.eye(p + 1); I[0, 0] = 0
        w = np.linalg.solve(A.T @ A + lam * I, A.T @ yc)
        paths.append(w[1:])                              # drop intercept
    paths = np.array(paths)

    fig, ax = plt.subplots()
    for j in range(p):
        ax.plot(lams, paths[:, j], label=f"w{j}")
    ax.set_xscale("log"); ax.set_xlabel("lambda (regularization strength)")
    ax.set_ylabel("coefficient value")
    ax.set_title("Figure 4 — Ridge path: stronger penalty shrinks all coefficients toward 0")
    ax.legend(ncol=2, fontsize=8)
    plt.show()
    """),

    md(r"""
    **Figure 4.** At $\lambda\to0$ (left) we recover OLS — and on this collinear data
    the coefficients are large and unstable (the near-duplicate features fight each
    other with huge offsetting weights). As $\lambda$ grows, Ridge **shrinks** every
    coefficient smoothly toward zero, splitting the credit between correlated
    features and stabilizing the solution. Lasso's path would instead drive some
    coefficients to *exactly* zero (selection). Choosing $\lambda$ is a
    cross-validation problem (Notebook 10): pick the value that minimizes *test*
    error — the bottom of an implicit U-curve.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause (assumption violated) | Mitigation |
    |---|---|---|---|
    | **Nonlinearity** | Curved structure in residual plot; underfit | Linearity assumption | Add polynomial/interaction features; use a nonlinear model (trees, §Ntbk 06) |
    | **Heteroscedasticity** | Fan-shaped residuals (Fig 2) | Constant-variance assumption | Transform target ($\log$); weighted LS; robust SEs |
    | **Outliers** | One point swings the whole line | Squared loss over-weights extremes | Huber/quantile loss; robust regression; investigate the point |
    | **Multicollinearity** | Huge, sign-flipping, unstable coefficients | Near-dependent features (Notebook 01) | Drop/merge features; **Ridge**; PCA |
    | **Overfitting** | High train $R^2$, poor test $R^2$ | Too many features / too flexible | Regularize; fewer features; more data; CV |
    | **Extrapolation** | Confident, wildly wrong outside training range | Model assumed valid everywhere | Flag out-of-range inputs; don't extrapolate |
    | **Regression to the mean** | "Worst performers improved after intervention!" | Mistaking noise reversion for an effect | Use a control group; randomized experiment |

    The cell shows OLS's **outlier fragility** — one bad point, and squared error
    drags the entire fit.
    """),

    code(r"""
    # A single outlier wrecks an OLS fit (squared loss over-penalizes the big residual).
    xo = np.sort(rng.uniform(0, 10, 50))
    yo = 2 + 1.0 * xo + rng.normal(0, 0.7, 50)
    yo_out = yo.copy(); yo_out[25] += 40                  # inject one extreme outlier

    A = np.c_[np.ones(50), xo]
    w_clean = np.linalg.lstsq(A, yo, rcond=None)[0]
    w_dirty = np.linalg.lstsq(A, yo_out, rcond=None)[0]

    fig, ax = plt.subplots()
    ax.scatter(xo, yo_out, s=18, color="tab:blue")
    ax.plot(xo, A @ w_clean, "g-", lw=2, label=f"no outlier: slope {w_clean[1]:.2f}")
    ax.plot(xo, A @ w_dirty, "r--", lw=2, label=f"with outlier: slope {w_dirty[1]:.2f}")
    ax.set_title("Figure 5 — One outlier tilts the OLS line (squared-loss fragility)")
    ax.legend()
    plt.show()
    print("The single outlier changed the slope materially — MSE has no robustness.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    scikit-learn provides `LinearRegression` (OLS), `Ridge`, `Lasso`, and
    `ElasticNet`, plus the all-important `Pipeline` + `StandardScaler` (regularized
    models *require* scaled features, since the penalty treats all coefficients on
    one scale — Notebook 03's conditioning lesson again). What the library adds:
    robust solvers, cross-validated variants (`RidgeCV`, `LassoCV`) that pick
    $\lambda$ for you, sparse support, and battle-tested numerics.
    """),

    code(r"""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)

    ols = LinearRegression().fit(Xtr, ytr)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1)).fit(Xtr, ytr)

    for name, model in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
        print(f"{name:6s}  test R^2 = {model.score(Xte, yte):.4f}")

    # Lasso's sparsity: how many coefficients did it zero out?
    lasso_coef = lasso.named_steps["lasso"].coef_
    print(f"\\nLasso zeroed {np.sum(np.abs(lasso_coef) < 1e-8)}/{len(lasso_coef)} coefficients "
          f"(automatic feature selection).")

    # Verify sklearn OLS == our scratch OLS on the same data:
    w_scratch = fit_ols(Xtr, ytr)
    print("scratch vs sklearn OLS intercept match:",
          np.allclose(w_scratch[0], ols.intercept_, atol=1e-6))
    """),

    md(r"""
    **Scratch vs production.** sklearn's `LinearRegression` returns the same fit as
    our `fit_ols` — no magic, just better solvers and ergonomics. The real
    value-adds: `Pipeline` guarantees the scaler is fit on *train only* (preventing
    leakage, Notebook 10), `RidgeCV`/`LassoCV` choose the penalty by cross-
    validation, and everything is vectorized and stable. Note Lasso zeroing
    coefficients on the collinear features — exactly the L1 sparsity we predicted
    geometrically in Notebook 01.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Insurance Premium Pricing

    **Scenario.** An insurer must set **annual premiums** from customer features
    (age, vehicle, region, claim history). A linear/GLM model predicts expected
    claim cost; the premium is that plus a loaded margin.

    **Why linear (not a black box) here?**
    - **Regulation:** in most jurisdictions insurers must *justify* pricing and prove
      it isn't unfairly discriminatory. Every coefficient is an auditable,
      signed effect — "+$X per year of age" — which regulators can inspect.
    - **Stability & monitoring:** coefficients are easy to track over time and across
      segments for fairness and drift.

    **Business objectives:** price risk accurately, stay compliant, remain
    competitive (overpricing loses customers; underpricing loses money).

    **Cost of mistakes**
    - **Underprediction** → premiums too low → underwriting losses.
    - **Overprediction** → premiums too high → customers churn to competitors.
    - **Unexplainable model** → regulatory rejection, fines, reputational damage.
    The *asymmetry* of these costs may call for quantile/asymmetric loss, not plain
    MSE.

    **Constraints:** protected attributes (and proxies) must be handled per law;
    coefficients must be signed sensibly (monotonicity often required); the model
    must be documented.

    **KPIs:** loss ratio (claims/premiums), predictive error on held-out claims,
    coefficient stability across retrains, fairness metrics across protected groups,
    and competitiveness (quote-to-bind rate).
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Latency / throughput / cost.** Inference is a single dot product — microsecond
      latency, trivial to scale, the cheapest possible model to serve. Training is
      closed-form or a few GD steps.
    - **Interpretability.** Coefficients *are* the explanation — but only comparable
      across features if inputs are **standardized**; otherwise a big coefficient may
      just reflect a small-scale feature. Report standardized coefficients for
      importance.
    - **Monitoring & drift.** Watch the **residual distribution** over time: a
      growing bias or widening spread signals concept drift (Notebook 45). Track
      coefficient stability across retrains — sudden swings often mean
      multicollinearity or a data-pipeline change.
    - **Retraining.** Cheap to retrain; schedule on a cadence or trigger on residual
      drift. Persist the scaler with the model so serving matches training.
    - **Explainability & regulation.** Linear models are the gold standard for
      auditability — a major reason they persist in finance, insurance, and
      healthcare even when a boosted tree scores higher.
    - **Reliability.** Guard against **extrapolation**: flag or clip inputs outside
      the training range, where linear extrapolation is confidently wrong.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **OLS vs Ridge vs Lasso vs Elastic Net:**

    | Dimension | OLS | Ridge (L2) | Lasso (L1) | Elastic Net |
    |---|---|---|---|---|
    | Handles multicollinearity | Poorly | **Well** | Picks one of a group | Well |
    | Feature selection | No | No (shrinks) | **Yes (zeros)** | Yes |
    | Closed form | Yes | Yes | No | No |
    | Best when | Few, clean features | Many correlated features | Sparse true model | Correlated + sparse |
    | Prior (Notebook 02) | None (MLE) | Gaussian | Laplace | Mix |

    **Linear regression vs flexible models (preview of Phase 1):**

    | Dimension | Linear Regression | XGBoost / Neural Net |
    |---|---|---|
    | Accuracy on complex data | Lower | **Higher** |
    | Interpretability | **High** (signed coefficients) | Low (needs SHAP, Ntbk 13) |
    | Latency / cost | **Lowest** | Higher |
    | Data needed | Low | High |
    | Captures nonlinearity/interactions | Only if hand-engineered | **Automatically** |
    | Regulatory suitability | **High** | Lower (explainability burden) |
    | Maintenance | Low | Higher |

    **The senior lesson:** start linear. It's the baseline that tells you whether the
    extra accuracy of a complex model is worth its interpretability, latency, and
    maintenance cost. Often it isn't.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Derive the OLS estimator.* (Section 4.3 — normal equations from $\nabla J=0$.)
    - *Why squared error?* → Gaussian-noise MLE (Section 4.2). Bonus: contrast with
      MAE (Laplace noise, robust to outliers, predicts the median not the mean).

    **Deep-dive questions**
    - *State the Gauss–Markov assumptions and what each buys you.* (Section 4.4.)
    - *Ridge vs Lasso — mechanism and use case.* → L2 shrinks (Gaussian prior), L1
      selects (Laplace prior, diamond corners); cite Notebook 01 Fig 4.
    - *Explain bias–variance and how $\lambda$ moves along it.* (Sections 4.6, 6.)

    **Whiteboard questions**
    - "Implement OLS and Ridge from scratch." (Section 5 — and explain why you use
      `lstsq`/`solve`, not `inv`.)
    - "Given a residual-vs-fitted plot, diagnose the model." (Section 6, Fig 2.)

    **Strong vs weak answers**
    - *"Your $R^2$ is 0.95 on train but the model is useless live."*
      - **Weak:** "Need more data."
      - **Strong:** "Train $R^2$ measures fit, not generalization — classic
        overfitting. I'd check the train/test gap, add regularization or remove
        features, cross-validate $\lambda$, and confirm there's no leakage inflating
        the offline number."
    - *"Two features are correlated — does it matter?"*
      - **Weak:** "No, regression handles it."
      - **Strong:** "It inflates coefficient variance and makes them unstable/
        uninterpretable (multicollinearity). Predictions may be fine, but individual
        coefficients aren't trustworthy. I'd use Ridge, drop/merge the features, or
        report them jointly."

    **Follow-ups:** "How do you choose $\lambda$?" (CV). "MAE vs MSE — when each?"
    (outliers → MAE/Huber; predict mean → MSE, median → MAE). "Is a high coefficient
    an important feature?" (only if standardized).

    **Common mistakes:** inverting $X^\top X$ in code; forgetting to scale before
    regularizing; interpreting raw coefficients across different feature scales;
    confusing correlation/causation in coefficients; trusting train $R^2$;
    extrapolating beyond the training range.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Define the linear model and the OLS objective in one breath.
    2. **Why was it invented?** Why least squares, and why does it still ship in
       2020s production?
    3. **How does it work?** Derive the normal equations *or* describe the GD fit.
    4. **Why does it work?** Why is squared error the "right" loss (probabilistically)?
    5. **When to use it?** Name three situations where you'd pick it over XGBoost.
    6. **When NOT to use it?** Name three assumption violations and their symptoms.
    7. **Tradeoffs?** OLS vs Ridge vs Lasso; linear vs flexible models.
    8. **How would you productionize it?** Pipeline, scaling, monitoring, retraining,
       and (for a regulated domain) explainability.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Worked example:** for points `(0,1)` and `(2,5)`, the line has slope 2 and
    intercept 1. Predictions are `(1,5)`, residuals are zero, and MSE is zero.

    **Guided practice (25 min):** calculate predictions, residuals, and MSE for
    `x=[0,1,2]`, `y=[1,3,6]`, using `ŷ=1+2x`. Hint: make a three-column table.
    Self-check: predictions are `[1,3,5]`; MSE is `1/3`.

    **Independent practice (45 min):** split a small regression dataset once, fit a
    mean-prediction baseline and sklearn `LinearRegression` on training data, then
    report test MAE and RMSE. Explain whether the model earns its complexity.

    **Challenge extension (60 min):** add one extreme outlier to training data only,
    refit, and compare coefficients and test errors. Do not use k-fold CV, Lasso
    coordinate descent, calibration, or fairness tooling yet; those belong to later
    modules.

    <details><summary><strong>Solution and scoring rubric</strong></summary>

    The guided result is `residuals=[0,0,1]`, squared errors `[0,0,1]`, MSE `1/3`.
    For the independent task, full credit requires split-before-fit, a baseline,
    metrics on untouched test rows, and a plain-language conclusion. Award 3 points
    for the calculation, 4 for leak-free code, and 3 for interpretation. Common
    mistakes: evaluating on training data, fitting preprocessing before splitting,
    and confusing residual with squared error. **Readiness threshold: 8/10.**
    </details>
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Linear regression is Phase 0 made concrete: a linear model (01) fit by minimizing
    a Gaussian-NLL loss (02), with Ridge/
    Lasso as priors that trade a little bias for much less variance. It is the
    interpretable, fast, regulation-friendly baseline against which every fancier
    model must justify itself.

    **Next in the canonical route:** `03 · Optimization and Gradient Descent` uses
    this concrete squared-loss surface to derive iterative learning. Logistic
    regression follows after optimization.
    """),
]

build("phase1_classical_ml/04_linear_regression.ipynb", cells)
