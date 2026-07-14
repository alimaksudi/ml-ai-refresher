"""Builder for Lesson FND-02 — Probability and Statistics.


Every cell body is a RAW string (r\"\"\"...\"\"\") so LaTeX backslashes and code
escapes survive verbatim.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # FND-02 · Probability and Statistics
    ### Section 01 — Mathematical Foundations · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** PRE-01 through PRE-04 and FND-01. You should be able to
    read fractions, summations, logarithms, subscripts, and basic derivatives.
    **Estimated time:** 4–6 hours including exercises.

    > Lesson FND-01 gave us the *geometry* of data (vectors, matrices, SVD). This
    > notebook gives us the *reasoning under uncertainty* that turns geometry into
    > **learning**. The single most important idea you will take away: **almost
    > every loss function in ML is a negative log-likelihood.** MSE, cross-entropy,
    > and L2/L1 regularization are not arbitrary — they fall out of probability the
    > moment you state your assumptions. A senior engineer can derive them on
    > demand and, more importantly, knows *which assumption* each one encodes.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - Probability as both **long-run frequency** and **degree of belief**, and why
      ML quietly uses both.
    - Random variables, the distributions that matter in ML (Bernoulli, Binomial,
      Poisson, Gaussian, Exponential), and **expectation / variance / covariance**.
    - **Bayes' theorem** as belief-updating, with the base-rate trap that fools
      most people (and most candidates).
    - **Maximum Likelihood Estimation (MLE)** — derived from scratch — and the
      punchline that **MSE = Gaussian NLL** and **cross-entropy = Bernoulli NLL**.
    - **MAP** estimation and why **L2 regularization is a Gaussian prior** and
      **L1 is a Laplace prior**.
    - The **Law of Large Numbers** and **Central Limit Theorem** — why the Gaussian
      is everywhere and why your metrics' error bars shrink like $1/\sqrt n$.
    - Estimator **bias & variance**, the **bootstrap**, confidence intervals, and
      **hypothesis testing** (the math behind A/B tests).

    **Why it matters in industry**
    - Loss design, calibration, and uncertainty estimates all live here.
    - A/B testing and experimentation are applied statistics; a wrong test ships a
      revenue-losing change with "statistical significance."
    - Data/concept **drift** is literally a change in a probability distribution.

    **Typical interview questions**
    - "Derive MLE for a Gaussian. Why does that justify squared-error loss?"
    - "Explain Bayes' theorem and the base-rate fallacy with a numeric example."
    - "What is the Central Limit Theorem and why should an ML engineer care?"
    - "What does a p-value actually mean — and what does it *not* mean?"
    - "Why is the L2 penalty equivalent to a Gaussian prior on the weights?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **Gambling (1650s).** Probability began with Pascal and Fermat settling a
    dispute about how to split the stakes of an interrupted game of chance. The
    early view was purely **frequentist**: probability = the long-run fraction of
    times an event happens.

    **Bayes & Laplace (1760s–1810s).** Thomas Bayes (published posthumously) and
    Laplace formalized the opposite direction: not "given a known coin, what data
    will I see?" but "**given the data I saw, what should I believe about the
    coin?**" This *inverse* question — updating beliefs with evidence — is the core
    loop of learning from data.

    **Gauss & Legendre (c. 1805).** The Gaussian ("normal") distribution arose from
    modeling measurement error in astronomy. Gauss showed that *if* errors are
    normal, the least-squares estimate is the maximum-likelihood estimate — the
    first bridge from probability to the optimization we do today.

    **Fisher (1920s).** R.A. Fisher made **maximum likelihood** the central
    estimation principle and built the machinery of hypothesis testing,
    significance, and experimental design that A/B testing still runs on.

    **The frequentist–Bayesian split.** Two camps emerged. Frequentists treat
    parameters as fixed unknowns and data as random (p-values, confidence
    intervals). Bayesians treat parameters as random and update a prior into a
    posterior. Modern ML is pragmatically **both**: MLE/cross-entropy is
    frequentist; regularization, priors, and Bayesian deep learning are Bayesian.
    Knowing which lens you're using — and its assumptions — is senior-level fluency.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Probability is a number in [0,1] that measures uncertainty.** Two readings,
    both useful:
    - *Frequentist:* "If I repeat this many times, the event happens this fraction
      of the time." (coin flips, click-through rate)
    - *Bayesian:* "This is my degree of belief, which I update as evidence arrives."
      (Is this transaction fraud? Start with a base rate, update with features.)

    **A distribution is a shape that says where outcomes land.** Discrete outcomes
    get a *probability mass function* (bars that sum to 1); continuous outcomes get
    a *probability density function* (a curve whose area is 1). Expectation is the
    balance point of the shape; variance is its spread.

    **Bayes' theorem is belief-updating.** Posterior ∝ Likelihood × Prior. Start
    with a prior belief, see data, multiply, renormalize. Do it again tomorrow with
    today's posterior as the new prior. *That loop is what "learning" means.*

    ```mermaid
    flowchart LR
        P["Prior belief<br/>P(theta)"] --> M["x Likelihood<br/>P(data | theta)"]
        D["Observed data"] --> M
        M --> N["Normalize"]
        N --> Post["Posterior belief<br/>P(theta | data)"]
        Post -.->|"becomes tomorrow's prior"| P
    ```

    **Central Limit Theorem — the reason the Gaussian is everywhere.** Average
    enough independent things and the average looks Gaussian, *whatever* the
    original distribution. This is why measurement noise, aggregate metrics, and
    sampling error are so often bell-shaped — and why your A/B metric's uncertainty
    shrinks like $1/\sqrt n$.

    Run the cells and watch these claims become pictures.
    """),

    code(r"""
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    print("NumPy", np.__version__)
    """),

    code(r"""
    # Figure 1 — a gallery of the distributions ML actually uses (pmf/pdf from formulas).
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Binomial(n=20, p=0.3): number of successes in 20 independent trials
    n, p = 20, 0.3
    ks = np.arange(0, n + 1)
    binom = np.array([math.comb(n, k) * p**k * (1 - p)**(n - k) for k in ks])
    axes[0, 0].bar(ks, binom, color="tab:blue")
    axes[0, 0].set_title("Binomial(20, 0.3) — # successes (PMF)")

    # Poisson(lambda=4): count of rare events in a fixed window
    lam = 4.0
    ks2 = np.arange(0, 15)
    pois = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in ks2])
    axes[0, 1].bar(ks2, pois, color="tab:orange")
    axes[0, 1].set_title("Poisson(4) — event counts (PMF)")

    # Gaussian(0,1): the bell curve
    x = np.linspace(-4, 4, 400)
    gauss = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    axes[1, 0].plot(x, gauss, color="tab:green")
    axes[1, 0].fill_between(x, gauss, alpha=0.2, color="tab:green")
    axes[1, 0].set_title("Gaussian(0, 1) — continuous (PDF)")

    # Exponential(rate=1): waiting times; skewed, heavy right tail
    xe = np.linspace(0, 6, 400)
    expo = np.exp(-xe)
    axes[1, 1].plot(xe, expo, color="tab:red")
    axes[1, 1].fill_between(xe, expo, alpha=0.2, color="tab:red")
    axes[1, 1].set_title("Exponential(1) — waiting times (PDF)")

    plt.suptitle("Figure 1 — Distributions that recur across ML")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** Each shape encodes a different *generative story*. **Binomial**:
    how many of $n$ independent yes/no trials succeed (conversions out of visitors).
    **Poisson**: counts of rare events per window (fraud attempts/hour, server
    errors/min). **Gaussian**: sums/averages of many small effects (the CLT below).
    **Exponential**: time between events; note its skew and heavy right tail — a
    reminder that *not everything is Gaussian*, and assuming so is a classic failure
    (Section 7). When you choose a likelihood/loss, you are choosing one of these
    stories for your data.
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    **Notation bridge.** $P(A)$ means the probability of event $A$;
    $P(A\mid B)$ means probability of $A$ after restricting attention to cases
    where $B$ is true; $\sum$ adds discrete possibilities; $\int$ accumulates a
    continuous range; $\mathbb E$ means probability-weighted average; and a hat
    such as $\hat\mu$ marks an estimate calculated from observed data. Notebook
    PRE-04 introduces each symbol with small examples.

    ### 4.1 Axioms, random variables, expectation
    Kolmogorov's axioms: probabilities are non-negative, the whole space has
    probability 1, and probabilities of disjoint events add. A **random variable**
    $X$ maps outcomes to numbers. Its **expectation** (mean) and **variance**:
    $$\mathbb E[X]=\sum_x x\,p(x)\ \ (\text{or}\ \int x\,f(x)\,dx),\qquad
    \operatorname{Var}(X)=\mathbb E[(X-\mathbb E[X])^2]=\mathbb E[X^2]-\mathbb E[X]^2.$$
    **Read and symbols:** $X$ is a random variable; lowercase $x$ is one possible
    value; $p(x)$ is its probability in the discrete case; $f(x)$ is density in
    the continuous case; $\mathbb E[X]$ is the long-run weighted average;
    $\operatorname{Var}(X)$ is average squared distance from that mean. Density is
    not itself probability—the integral over a range gives probability.
    **Covariance** measures co-movement: $\operatorname{Cov}(X,Y)=\mathbb
    E[(X-\mathbb E X)(Y-\mathbb E Y)]$; the covariance matrix is the statistical
    cousin of $X^\top X$ from Lesson FND-01 (and PCA diagonalizes it).

    ### 4.2 Joint, marginal, conditional; independence
    $P(X,Y)$ is joint; summing out a variable gives a **marginal**; $P(X\mid
    Y)=P(X,Y)/P(Y)$ is **conditional**. $X,Y$ are **independent** iff
    $P(X,Y)=P(X)P(Y)$. The IID assumption ("independent and identically
    distributed") underlies nearly every train/test split — and violating it
    (time series, grouped data) silently breaks evaluation (Lesson MLE-02).

    **Symbols and example:** the comma in $P(X,Y)$ means both variable outcomes;
    the vertical bar means “given.” If 6 of 8 premium users renew, then
    $P(\text{renew}\mid\text{premium})=6/8$. Independence means learning one event
    does not change the probability assigned to the other.

    ### 4.3 Bayes' theorem
    $$P(\theta\mid D)=\frac{P(D\mid\theta)\,P(\theta)}{P(D)}\ \propto\ \underbrace{P(D\mid\theta)}_{\text{likelihood}}\;\underbrace{P(\theta)}_{\text{prior}}.$$
    **Read aloud:** “probability of theta given data equals likelihood times prior
    divided by evidence.” **Symbols:** $\theta$ (theta) is an unknown model
    parameter or hypothesis; $D$ is observed data; $P(\theta)$ is the prior belief;
    $P(D\mid\theta)$ is compatibility of data with the hypothesis; $P(D)$
    normalizes all hypotheses; $\propto$ means “proportional to.”
    The denominator $P(D)=\int P(D\mid\theta)P(\theta)\,d\theta$ just renormalizes.
    **Base-rate fallacy:** even a 99%-accurate test for a 1-in-10,000 disease yields
    mostly false positives, because the prior $P(\text{disease})$ is tiny. We'll
    compute this in Section 5 — it is a guaranteed interview question.

    ### 4.4 Maximum Likelihood Estimation — derived
    Given IID data $x_1,\dots,x_n$ and a model $p(x\mid\theta)$, the **likelihood**
    is $L(\theta)=\prod_i p(x_i\mid\theta)$. We maximize the **log**-likelihood
    (sums are nicer than products, and it's monotonic):
    $$\ell(\theta)=\sum_i \log p(x_i\mid\theta).$$

    **Symbols:** IID means each $x_i$ is generated independently by the same
    distribution; $n$ is sample size; $\prod$ means multiply repeated terms;
    $L$ is likelihood; lowercase $\ell$ is log-likelihood; $\log$ reverses an
    exponential and turns products into sums.

    **Gaussian case.** With $p(x\mid\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$,
    $$\ell(\mu,\sigma^2)=-\frac{n}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_i (x_i-\mu)^2.$$
    **Symbols:** $\mu$ (mu) is the Gaussian center; $\sigma^2$ (sigma squared) is
    variance; $\pi\approx3.14159$ is the circle constant; $\exp$ means the natural
    exponential; $(x_i-\mu)^2$ is squared distance from the center. The first term
    penalizes excessive spread and the second penalizes poor fit.
    Set $\partial\ell/\partial\mu=0$: $\;\hat\mu=\frac1n\sum_i x_i$ (the sample mean).
    Set $\partial\ell/\partial\sigma^2=0$: $\;\hat\sigma^2=\frac1n\sum_i (x_i-\hat\mu)^2$.

    > **The payoff.** Maximizing the Gaussian log-likelihood over $\mu$ is *exactly*
    > minimizing $\sum_i (x_i-\mu)^2$ — squared error. So **MSE is not arbitrary: it
    > is the negative log-likelihood under the assumption of Gaussian noise.**
    > Likewise, **cross-entropy is the NLL under a Bernoulli/Categorical model**
    > (Lesson CML-02). Choosing a loss = choosing a noise model.

    ### 4.5 MAP and the prior–regularizer bridge
    Maximizing the posterior instead of the likelihood gives the **MAP** estimate:
    $$\hat\theta_{\text{MAP}}=\arg\max_\theta\big[\log P(D\mid\theta)+\log P(\theta)\big].$$
    **Read and symbols:** “theta-hat MAP is the value of theta that maximizes log
    likelihood plus log prior.” A hat marks an estimate; $\arg\max$ returns the
    input value producing the largest expression; MAP means maximum a posteriori.
    A **Gaussian prior** $\theta\sim\mathcal N(0,\tau^2)$ contributes $-\frac{1}{2\tau^2}\lVert\theta\rVert_2^2$
    — i.e. **L2 (Ridge) regularization**. A **Laplace prior** gives $\lVert\theta\rVert_1$
    — **L1 (Lasso)**. Regularization is just MAP with a belief that weights are small.

    ### 4.6 LLN and the Central Limit Theorem
    **Law of Large Numbers:** the sample mean converges to the true mean as
    $n\to\infty$. **Central Limit Theorem:** for IID $X_i$ with mean $\mu$ and
    finite variance $\sigma^2$,
    $$\frac{\bar X_n-\mu}{\sigma/\sqrt n}\ \xrightarrow{\ d\ }\ \mathcal N(0,1).$$
    **Read and symbols:** $\bar X_n$ is the average of $n$ observations; $\mu$ and
    $\sigma$ are population mean and standard deviation; $\sigma/\sqrt n$ is the
    standard error; the arrow marked $d$ means the distribution approaches;
    $\mathcal N(0,1)$ is a normal distribution with mean zero and variance one.
    The standard error of a mean shrinks like $\sigma/\sqrt n$ — *quadruple your
    sample to halve your error bar.* This is the engine behind confidence intervals
    and A/B-test sample-size math.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    We implement, in plain NumPy/stdlib: (1) MLE for a Gaussian — and verify it
    equals the sample mean/variance; (2) the base-rate Bayes calculation;
    (3) Bayesian belief-updating for a coin; (4) the bootstrap; (5) a two-proportion
    hypothesis test (the math inside an A/B test). No scipy yet.
    """),

    code(r"""
    # 5.1 MLE for a Gaussian == sample mean and (1/n) variance. Verify against the formula.
    def gaussian_mle(data):
        mu = data.mean()
        var = ((data - mu) ** 2).mean()      # note: 1/n (MLE), not 1/(n-1)
        return mu, var

    truth_mu, truth_sigma = 5.0, 2.0
    data = rng.normal(truth_mu, truth_sigma, size=10_000)
    mu_hat, var_hat = gaussian_mle(data)
    print(f"true (mu, sigma^2)      = ({truth_mu}, {truth_sigma**2})")
    print(f"MLE  (mu_hat, var_hat)  = ({mu_hat:.3f}, {var_hat:.3f})")

    # The MLE variance is BIASED low by a factor (n-1)/n; the unbiased estimator divides by n-1.
    n = len(data)
    print(f"\\nMLE var (1/n)      = {var_hat:.4f}")
    print(f"unbiased (1/(n-1)) = {var_hat * n / (n - 1):.4f}")
    print(f"numpy ddof=0 / ddof=1: {data.var(ddof=0):.4f} / {data.var(ddof=1):.4f}")
    """),

    code(r"""
    # 5.2 The base-rate fallacy, in numbers (a guaranteed interview question).
    # A disease affects 1 in 10,000. A test is 99% sensitive and 99% specific.
    # You test positive. What's the probability you actually have the disease?
    prior = 1 / 10_000
    sensitivity = 0.99            # P(test+ | disease)
    specificity = 0.99            # P(test- | healthy)  => false-positive rate = 0.01

    p_pos_given_disease = sensitivity
    p_pos_given_healthy = 1 - specificity
    p_pos = p_pos_given_disease * prior + p_pos_given_healthy * (1 - prior)
    posterior = p_pos_given_disease * prior / p_pos

    print(f"P(disease | positive test) = {posterior:.4f}  ({posterior:.1%})")
    print("Despite a '99% accurate' test, a positive result is ~99% likely a FALSE alarm,")
    print("because the disease is so rare. The prior dominates. This is base-rate neglect.")
    """),

    code(r"""
    # 5.3 Bayesian updating for a coin's bias, computed on a grid (no conjugacy needed).
    # Posterior(p) is proportional to likelihood p^heads * (1-p)^tails, with a uniform prior.
    p_grid = np.linspace(0, 1, 500)

    def posterior(heads, tails, prior=None):
        if prior is None:
            prior = np.ones_like(p_grid)           # uniform = Beta(1,1)
        like = p_grid ** heads * (1 - p_grid) ** tails
        post = prior * like
        return post / np.trapezoid(post, p_grid)   # normalize so area = 1

    true_p = 0.7
    flips = (rng.random(200) < true_p).astype(int)
    stages = [0, 1, 5, 20, 200]
    posteriors = {}
    for s in stages:
        h = int(flips[:s].sum()); t = s - h
        posteriors[s] = posterior(h, t)
    # MAP estimate after all data:
    map_p = p_grid[np.argmax(posteriors[200])]
    print(f"true p = {true_p}; MAP estimate after 200 flips = {map_p:.3f}")
    """),

    code(r"""
    # 5.4 The bootstrap: estimate the sampling distribution (and a CI) of ANY statistic,
    # with no distributional assumptions, by resampling the data with replacement.
    sample = rng.gamma(shape=2.0, scale=2.0, size=200)   # skewed population

    def bootstrap(data, stat=np.median, B=5000):
        n = len(data)
        idx = rng.integers(0, n, size=(B, n))            # B resamples of size n
        return stat(data[idx], axis=1)

    boot_medians = bootstrap(sample, np.median, B=5000)
    ci = np.percentile(boot_medians, [2.5, 97.5])
    print(f"sample median            = {np.median(sample):.3f}")
    print(f"95% bootstrap CI (median)= [{ci[0]:.3f}, {ci[1]:.3f}]")
    """),

    code(r"""
    # 5.5 Two-proportion z-test from scratch — the engine inside an A/B test.
    def two_proportion_ztest(x1, n1, x2, n2):
        p1, p2 = x1 / n1, x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p2 - p1) / se
        # two-sided p-value via the standard-normal CDF, using math.erf (no scipy)
        cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
        p_value = 2 * (1 - cdf)
        return z, p_value

    # Control: 1000 visitors, 100 conversions (10%). Treatment: 1000, 130 (13%).
    z, pval = two_proportion_ztest(100, 1000, 130, 1000)
    print(f"control 10.0%  vs  treatment 13.0%")
    print(f"z = {z:.3f}, two-sided p-value = {pval:.4f}")
    print("p < 0.05 -> reject 'no difference'; the lift is unlikely to be pure noise.")
    """),

    # ============================================ 6. Visualization
    md(r"""
    ## 6 · Visualization

    Three pictures cement the theory: Bayesian updating sharpening with data, the
    CLT manufacturing a bell curve out of a skewed population, and the bootstrap
    distribution of a statistic.
    """),

    code(r"""
    # Figure 2 — Bayesian updating: the posterior concentrates as evidence accrues.
    fig, ax = plt.subplots()
    for s in stages:
        ax.plot(p_grid, posteriors[s], label=f"after {s} flips")
    ax.axvline(true_p, color="k", ls="--", lw=1, label=f"true p={true_p}")
    ax.set_xlabel("p (coin bias)"); ax.set_ylabel("posterior density")
    ax.set_title("Figure 2 — Posterior over coin bias sharpens with data")
    ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 2.** With zero flips the posterior is flat — we believe nothing in
    particular. Each batch of evidence multiplies the likelihood in and the belief
    *concentrates* around the truth (0.7), getting narrower (more certain) as $n$
    grows. This is learning made literal: prior → evidence → sharper posterior.
    """),

    code(r"""
    # Figure 3 — the Central Limit Theorem in action on a skewed (exponential) population.
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    pop_mean, pop_sd = 1.0, 1.0                      # Exponential(1): mean=sd=1, very skewed
    for ax, nsamp in zip(axes, [1, 2, 10, 50]):
        means = rng.exponential(1.0, size=(20_000, nsamp)).mean(axis=1)
        ax.hist(means, bins=60, density=True, alpha=0.6, color="tab:blue")
        xs = np.linspace(means.min(), means.max(), 200)
        se = pop_sd / np.sqrt(nsamp)                 # CLT-predicted standard error
        ax.plot(xs, np.exp(-(xs - pop_mean)**2 / (2 * se**2)) / (se * np.sqrt(2 * np.pi)),
                "r", lw=2)
        ax.set_title(f"mean of n={nsamp}")
    plt.suptitle("Figure 3 — CLT: averages become Gaussian (red) even from a skewed source")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** The source is a heavily right-skewed exponential. With $n=1$ the
    histogram *is* that skew. But the distribution of the **sample mean** marches
    toward the Gaussian (red curve) as $n$ grows, and tightens with standard error
    $\sigma/\sqrt n$. This is *why* aggregate metrics and estimation errors are so
    often bell-shaped, and why confidence intervals use the normal distribution.
    """),

    code(r"""
    # Figure 4 — the bootstrap distribution and its 95% percentile interval.
    fig, ax = plt.subplots()
    ax.hist(boot_medians, bins=50, density=True, alpha=0.7, color="tab:purple")
    ax.axvline(ci[0], color="r", ls="--"); ax.axvline(ci[1], color="r", ls="--",
                                                       label="95% CI")
    ax.axvline(np.median(sample), color="k", lw=2, label="observed median")
    ax.set_title("Figure 4 — Bootstrap sampling distribution of the median")
    ax.set_xlabel("median of a resample"); ax.legend()
    plt.show()
    """),

    md(r"""
    **Figure 4.** We never derived a formula for the standard error of a *median* —
    we didn't have to. Resampling the data with replacement thousands of times and
    recomputing the statistic *empirically reconstructs its sampling distribution*.
    The middle 95% gives a confidence interval. The bootstrap is the senior
    engineer's universal tool for "what's the uncertainty on this metric?" when no
    clean formula exists (AUC, recall@k, revenue-per-user, …).
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Base-rate neglect** | "99% accurate" test, yet most positives are false | Ignoring a small prior $P(\text{event})$ | Always reason with Bayes; report precision at the true base rate |
    | **p-hacking / multiple comparisons** | Lots of "significant" findings that don't replicate | Testing many hypotheses; stopping when $p<0.05$ | Pre-register; correct (Bonferroni / FDR); hold-out confirmation |
    | **Peeking at A/B tests** | "Significant!" early, reverts later | Repeatedly testing inflates false-positive rate | Fix sample size in advance, or use sequential/Bayesian tests |
    | **Assuming Gaussian** | Models blow up on outliers; bad tail risk | Heavy-tailed data (income, latency, losses) | Check the data; use robust stats, log-transforms, heavier-tailed likelihoods |
    | **Correlation ⇒ causation** | A feature "works" then fails after a change | Confounders; non-causal association | Experiments / causal inference, not just correlation |
    | **Simpson's paradox** | Trend reverses after aggregation | A lurking confounder across groups | Segment; condition on the confounder |
    | **Non-IID data** | Great offline, bad online | Temporal/grouped leakage breaks independence | Time-based & grouped splits (Lesson MLE-02) |

    The next cell demonstrates **p-hacking** quantitatively — pure noise will hand
    you "significant" results if you test enough hypotheses.
    """),

    code(r"""
    # Demonstrate p-hacking: test 100 PURE-NOISE features against a random label.
    # Under the null, ~5% will be "significant" at p<0.05 — by construction, not signal.
    n = 500
    label = rng.normal(size=n)
    false_hits = 0
    n_features = 100
    for _ in range(n_features):
        feature = rng.normal(size=n)                  # independent of label, by design
        # correlation t-statistic -> two-sided p-value via normal approx
        r = np.corrcoef(feature, label)[0, 1]
        z = r * math.sqrt(n - 1)                       # rough large-n approximation
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        if p < 0.05:
            false_hits += 1
    print(f"'significant' noise features at p<0.05: {false_hits}/{n_features}")
    print("Expected ~5 by pure chance. Test enough hypotheses and noise looks like signal.")
    print("Fix: multiple-comparison correction + out-of-sample confirmation.")
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    `scipy.stats` provides every distribution (with `pdf/cdf/ppf/rvs`), and proper
    statistical tests; `numpy.random.Generator` is the modern, fast, reproducible
    sampler. In real pipelines you also reach for `statsmodels` (regression
    inference, multiple-testing correction) and dedicated experimentation platforms
    for A/B tests. Below: scipy reproduces our scratch results and adds exact tests.
    """),

    code(r"""
    from scipy import stats

    # Distributions are objects with pdf/cdf/ppf/rvs — no more hand-rolled formulas.
    print("P(X<=12) for Binomial(20,0.3):", stats.binom.cdf(12, 20, 0.3).round(4))
    print("97.5th percentile of N(0,1):  ", stats.norm.ppf(0.975).round(4))

    # Exact two-sided test that a coin with 130/1000 differs from p=0.10:
    res = stats.binomtest(130, 1000, 0.10)
    print("binom test p-value:", round(res.pvalue, 4))

    # Welch's t-test for two continuous samples (unequal variances):
    a = rng.normal(0.0, 1.0, 300)
    b = rng.normal(0.3, 1.5, 320)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"Welch t-test: t={t:.3f}, p={p:.4f}")

    # Our scratch z-test vs scipy's normal CDF — same machinery, validated:
    z = (0.13 - 0.10) / math.sqrt(0.115 * 0.885 * (2 / 1000))
    print("scratch p-value via scipy norm.sf:", round(2 * stats.norm.sf(abs(z)), 4))
    """),

    md(r"""
    **Scratch vs production.** Our hand-rolled erf-based p-values match scipy — the
    point of Section 5 was to prove there's no magic. What the libraries add:
    *numerically stable* tails, *exact* (non-asymptotic) tests for small samples,
    correct handling of edge cases, and a vast catalogue of distributions and tests
    you should not reimplement in anger. Know the math; call the library.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — A/B Testing a Checkout Change

    **Scenario.** An e-commerce team proposes a new one-click checkout. Current
    conversion is **~10%**. They want to ship only if the new flow *truly* lifts
    conversion, because a worse flow directly loses revenue.

    **Business objectives:** maximize conversion without degrading the funnel; make
    a *reliable* ship/no-ship decision.

    **Cost of mistakes**
    - **False positive** (ship a flow that isn't actually better): lost revenue +
      eng cost of a change that must be rolled back.
    - **False negative** (miss a real improvement): permanent opportunity cost.
    These costs set $\alpha$ (false-positive tolerance) and power $1-\beta$.

    **Design (the statistics).**
    1. Pick the **minimum detectable effect** (e.g. +1pt absolute, 10%→11%) — the
       smallest lift worth shipping.
    2. Compute **sample size** so the test has, say, 80% power at $\alpha=0.05$.
       Sample size scales like $1/\text{MDE}^2$ (CLT again) — small effects need
       *huge* traffic. The cell below shows the magnitude.
    3. **Randomize** users to control/treatment; run until the pre-computed $n$.
    4. Analyze with the **two-proportion test** from Section 5 — *no peeking*.

    **Constraints:** finite traffic (time budget), novelty effects, and the need
    not to harm the live funnel during the test.

    **KPIs:** primary = conversion rate (with CI); guardrails = revenue/user, page
    latency, refund rate. A win on the primary that tanks a guardrail is not a win.
    """),

    code(r"""
    # Rough sample-size intuition for an A/B test (normal approximation).
    # n per arm ~ (z_alpha/2 + z_beta)^2 * 2 * p(1-p) / MDE^2
    p0 = 0.10
    z_alpha = stats.norm.ppf(0.975)     # two-sided alpha=0.05
    z_beta = stats.norm.ppf(0.80)       # 80% power
    for mde in [0.02, 0.01, 0.005]:
        n = (z_alpha + z_beta) ** 2 * 2 * p0 * (1 - p0) / mde ** 2
        print(f"MDE = {mde*100:>4.1f}pt absolute  ->  ~{math.ceil(n):,} users PER ARM")
    print("\\nHalving the effect you want to detect ~4x's the traffic needed (1/MDE^2).")
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Experimentation at scale.** Pre-compute sample sizes; *don't* peek. Use
      **sequential** or **Bayesian** testing if you need to stop early without
      inflating false positives. Track guardrail metrics, not just the target.
    - **Drift = distribution change.** Monitor feature/label distributions over
      time (population stability index, KL divergence, KS test). A shift in $P(x)$
      is data drift; a shift in $P(y\mid x)$ is concept drift (Lesson PROD-05).
    - **Uncertainty in serving.** Report **calibrated** probabilities and intervals,
      not just point predictions — downstream decisions (pricing, routing,
      thresholds) depend on the *distribution*, not the mean.
    - **Heavy tails.** Latency, revenue, and losses are rarely Gaussian. Monitor
      **percentiles (p95/p99)**, not means; one whale or one outage moves the mean
      but the business runs on the tail.
    - **Reproducibility.** Seed every sampler (`np.random.default_rng(seed)`), log
      seeds, and version the data snapshot a statistic was computed on.
    - **Multiple comparisons in monitoring.** Hundreds of dashboards each alert at
      5% → constant false alarms. Apply corrections / sensible thresholds.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Frequentist vs Bayesian:**

    | Dimension | Frequentist (p-values, CIs) | Bayesian (posteriors) |
    |---|---|---|
    | Interpretation | Long-run error rates | Direct probability of hypotheses |
    | Priors | None (pro & con) | Required (encode knowledge; can bias) |
    | Small data | Weaker, asymptotic | Priors regularize, often better |
    | Compute | Cheap, closed-form tests | Often needs MCMC/VI |
    | Stopping early | Inflates false positives | Naturally supports sequential decisions |
    | Industry use | Default A/B testing | Personalization, bandits, small-sample |

    **MLE vs MAP (regularized):**

    | Dimension | MLE | MAP (= MLE + prior/penalty) |
    |---|---|---|
    | Overfitting | Prone (esp. small $n$) | Controlled by the prior |
    | Bias/variance | Low bias, higher variance | Trades a little bias for much less variance |
    | ML analogue | Unregularized fit | Ridge (Gaussian prior) / Lasso (Laplace) |

    **Parametric vs nonparametric uncertainty (formula vs bootstrap):**

    | Dimension | Closed-form (CLT) CI | Bootstrap |
    |---|---|---|
    | Assumptions | Distributional / large $n$ | Almost none |
    | Works for any statistic | No (need a formula) | **Yes** (median, AUC, recall@k) |
    | Cost | Instant | Many resamples (compute) |
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *Difference between probability and likelihood?* → Probability: data varies,
      parameters fixed. Likelihood: data fixed (observed), viewed as a function of
      parameters. We maximize the latter.
    - *Expectation vs variance vs covariance in one line each?* (Section 4.1)

    **Deep-dive questions**
    - *Derive MLE for a Gaussian; connect to MSE.* (Section 4.4 — be able to write
      it on a board.)
    - *Why is L2 regularization a Gaussian prior?* (Section 4.5)
    - *State the CLT precisely and give its standard error.* (Section 4.6)

    **Whiteboard questions**
    - "Disease prevalence 0.5%, test 95% sensitive/specific. P(disease | positive)?"
      → Plug into Bayes (Section 5.2). The answer being *low* is the whole point.
    - "Design an A/B test for a 1pt conversion lift: what sample size, what test,
      what could go wrong?" (Sections 5.5, 9, 10.)

    **Strong vs weak answers**
    - *"What's a p-value?"*
      - **Weak:** "The probability the null hypothesis is true." (Wrong.)
      - **Strong:** "The probability of seeing data *at least this extreme* **if the
        null were true**. It is *not* P(H0 | data), and a small p-value doesn't
        measure effect size or importance."
    - *"Is your metric improvement real?"*
      - **Weak:** "The number went up."
      - **Strong:** "I'd put a confidence interval on the delta (bootstrap if no
        formula), check it excludes zero at a pre-set $\alpha$, confirm power and
        sample size, and watch guardrail metrics."

    **Follow-ups:** "Now you ran 50 variants — what changes?" (multiple comparisons).
    "Your data is heavy-tailed — does the t-test still apply?" (robustness, larger
    $n$, transforms).

    **Common mistakes:** equating p-value with P(H0); forgetting the base rate;
    assuming Gaussianity; peeking at experiments; reporting means for heavy-tailed
    metrics; ignoring the IID assumption when splitting data.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **What is it?** Distinguish probability, likelihood, prior, and posterior.
    2. **Why was it invented?** Why did the Bayesian "inverse" question matter for
       learning from data?
    3. **How does it work?** Walk through Bayes' theorem and one MLE derivation.
    4. **Why does it work?** Why does maximizing Gaussian likelihood give squared
       error, and Bernoulli likelihood give cross-entropy?
    5. **When to use it?** When would you choose Bayesian over frequentist?
    6. **When NOT to use it?** When does the Gaussian/CLT assumption break, and what
       do you do instead?
    7. **Tradeoffs?** MLE vs MAP; closed-form CI vs bootstrap.
    8. **How would you productionize it?** Design a trustworthy A/B test and a drift
       monitor; say what you'd track and how you'd avoid false positives.
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Estimated time:** 90–150 minutes. Use the hints for the first attempt, then
    compare with the expected results and rubric.

    **Beginner (conceptual)**
    1. A test is 99% sensitive and 99% specific for a disease with prevalence 2%.
       Compute P(disease | positive) by hand, then verify with code.
    2. Explain in two sentences why MSE corresponds to assuming Gaussian noise.

    **Beginner → Intermediate (coding)**
    3. Extend `gaussian_mle` to fit a Gaussian by **numerically** maximizing the
       log-likelihood (grid or gradient ascent) and confirm it matches the closed
       form.
    4. Implement MLE for a **Bernoulli** ($\hat p=\frac1n\sum x_i$) and show that
       maximizing its log-likelihood equals minimizing **cross-entropy**.

    **Intermediate (analysis)**
    5. Reproduce Figure 3 (CLT) starting from a **different** skewed distribution
       (e.g. log-normal). At what $n$ does the mean look Gaussian? Does a
       heavy-tailed source (e.g. Cauchy) ever converge — and why not?
    6. Use the bootstrap to put a 95% CI on the **AUC** of a toy classifier. Why is
       there no simple closed-form CI for AUC?

    **Senior (interview + production design)**
    7. *Whiteboard:* derive MAP for linear regression with a Gaussian prior and show
       it equals Ridge; identify $\lambda$ in terms of the prior/noise variances.
    8. *Design:* an A/B test shows $p=0.04$ after 3 days, but the team peeked daily.
       Explain why the result is untrustworthy and design a correct protocol
       (fixed-$n$ or sequential), including guardrail metrics and a rollback rule.
    9. *Drift:* propose a monitoring system that distinguishes **data drift**
       ($P(x)$ changes) from **concept drift** ($P(y\mid x)$ changes), naming the
       statistics you'd compute and the alert thresholds you'd set.

    <details>
    <summary><strong>Hints, expected results, and scoring rubric</strong></summary>

    1. Out of 10,000 people, expect 200 diseased and 198 true positives. Of 9,800
       non-diseased people, expect 98 false positives. The answer is
       `198/(198+98) ≈ 66.9%`, not 99%.
    2. With fixed variance, Gaussian negative log-likelihood differs from squared
       error only by constants and a positive scale.
    3. The numerical optimum should agree with sample mean and population variance
       within the grid or optimizer tolerance.
    4. The Bernoulli MLE is the observed positive fraction. Clip probabilities away
       from exactly zero and one before taking logs.
    5. A log-normal source approaches a normal sampling distribution for the mean
       as $n$ grows. A Cauchy source violates the ordinary CLT's finite-moment
       conditions.
    6. Resample paired `(label, score)` observations. Report a percentile interval
       and explain that AUC depends on rankings of many positive-negative pairs.
    7. Award 2 points for the Gaussian likelihood, 2 for the Gaussian prior, and 1
       for identifying $\lambda$ using noise and prior variances.
    8. Full credit identifies repeated peeking as inflated Type-I error and specifies
       a fixed-horizon or valid sequential design, guardrails, and rollback.
    9. Full credit separates unlabeled feature-distribution monitoring from delayed
       labeled performance or residual monitoring and uses effect sizes rather than
       p-values alone.

    A score of 12/15 across Questions 7–9 indicates senior-level readiness.
    </details>
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    Probability is the language of uncertainty; statistics is how we reason from
    data to beliefs and decisions. The throughline for ML: **losses are negative
    log-likelihoods** (Gaussian → MSE, Bernoulli → cross-entropy), **regularizers
    are priors** (Gaussian → L2, Laplace → L1), and the **CLT/bootstrap** give us
    the error bars that make metrics trustworthy.

    **Related lesson:** `FND-04 · Optimization and Gradient Descent` — having defined *what* to
    minimize (a likelihood-derived loss), we now study *how* to minimize it, the
    algorithm at the heart of every model in this curriculum.
    """),
]

build("01_ml_foundations/02_probability_and_statistics.ipynb", cells)
