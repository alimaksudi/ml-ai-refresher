"""Build FND-02: Probability and Statistics."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # FND-02 · Probability and Statistics

    **Prerequisites:** PRE-06 and FND-01  
    **Estimated study time:** 8–10 hours, including practice  
    **Next lesson:** FND-03 · Data Workflow, EDA, and Cleaning

    PRE-06 taught the language of events, conditional probability, Bayes' rule,
    random variables, expected value, variance, and standard deviation. This lesson
    uses that language to answer a harder question:

    > When a sample changes, which differences are meaningful and which could be
    > ordinary sampling variation?

    We will calculate everything with small numbers before using a library.

    ### Scope boundary

    This lesson does **not** derive model-specific losses or regularization:

    - Gaussian likelihood and squared-error regression belong in CML-01;
    - Bernoulli likelihood and cross-entropy belong in CML-02;
    - MAP estimation, Ridge, and Lasso belong after regression is understood;
    - production drift monitoring belongs in PROD-05.

    Here, the goal is a dependable foundation for reasoning from samples.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - distinguish a population, sample, parameter, statistic, estimator, and estimate;
    - recognize Bernoulli, binomial, and normal distributions from their data stories;
    - calculate mean, variance, covariance, and correlation manually;
    - explain why covariance is not yet causation;
    - distinguish a data distribution from a sampling distribution;
    - explain the Law of Large Numbers and Central Limit Theorem without slogans;
    - calculate standard error and a confidence interval;
    - state a null and alternative hypothesis;
    - interpret a p-value correctly;
    - separate statistical significance from practical importance;
    - explain false positives, false negatives, and statistical power;
    - build and check a small A/B analysis;
    - use a bootstrap when a simple uncertainty formula is unavailable.

    ### What mastery looks like

    You are ready to continue when you can explain the full chain:

    ```mermaid
    flowchart LR
        A[Population] --> B[Sample]
        B --> C[Statistic]
        C --> D[Sampling uncertainty]
        D --> E[Interval or test]
        E --> F[Scoped decision]
    ```

    An interval or test does not repair biased sampling, leakage, or a badly defined
    metric. Statistical machinery starts **after** the data question is valid.
    """),

    md(r"""
    ## 2 · The practical problem: did checkout really improve?

    An online shop tests a new checkout page:

    | Group | Visitors | Purchases | Observed conversion |
    | --- | ---: | ---: | ---: |
    | Old checkout | 1,000 | 100 | 10% |
    | New checkout | 1,000 | 130 | 13% |

    The observed difference is 3 percentage points. It is tempting to announce that
    the new page is better. But these 2,000 visitors are only samples from future
    visitors.

    The real question is:

    > Is the 3-point difference large relative to the amount these sample rates would
    > naturally move from one random sample to another?

    We need four separate ideas:

    1. **Effect:** what difference did we observe?
    2. **Uncertainty:** how much would the estimate change across samples?
    3. **Evidence:** how compatible is this result with a no-difference model?
    4. **Decision:** is the effect valuable enough and safe enough to act on?

    Statistics supports the decision. It does not make the business decision for us.
    """),

    code(r"""
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    random_generator = np.random.default_rng(42)

    old_visitors, old_purchases = 1_000, 100
    new_visitors, new_purchases = 1_000, 130

    old_rate = old_purchases / old_visitors
    new_rate = new_purchases / new_visitors
    observed_difference = new_rate - old_rate

    print(f"old conversion: {old_rate:.1%}")
    print(f"new conversion: {new_rate:.1%}")
    print(f"observed difference: {observed_difference:.1%} points")

    assert np.isclose(old_rate, 0.10)
    assert np.isclose(new_rate, 0.13)
    assert np.isclose(observed_difference, 0.03)
    """),

    md(r"""
    ## 3 · Population, sample, parameter, and statistic

    These words describe different objects. Mixing them creates vague conclusions.

    | Term | Simple meaning | Checkout example |
    | --- | --- | --- |
    | Population | Every case we want to understand | All eligible future visitors |
    | Sample | Cases actually observed | Visitors included in the experiment |
    | Parameter | Fixed but unknown population quantity | True future conversion rate |
    | Statistic | Number calculated from a sample | Observed sample conversion rate |
    | Estimator | Rule used to calculate an estimate | purchases divided by visitors |
    | Estimate | The estimator's result for one sample | 0.13 for the new page |

    A parameter is not random after the population is fixed, but we do not know it.
    A statistic changes when the sample changes.

    For binary outcomes $x_i\in\{0,1\}$, the sample conversion rate is:

    $$
    \hat p=\frac{1}{n}\sum_{i=1}^{n}x_i
    $$

    **Symbols:** $x_i$ is visitor $i$'s outcome, where 1 means purchase and 0 means
    no purchase; $n$ is the number of visitors; $\sum$ means add all outcomes;
    $\hat p$ reads “p-hat” and is the sample estimate of population rate $p$.

    If five outcomes are $[1,0,1,0,0]$, then:

    $$
    \hat p=\frac{1+0+1+0+0}{5}=\frac{2}{5}=0.40
    $$

    The estimate is 40%. It does not prove that the population parameter is exactly
    40%.
    """),

    code(r"""
    tiny_purchase_sample = np.array([1, 0, 1, 0, 0])
    tiny_rate = tiny_purchase_sample.mean()

    print("outcomes:", tiny_purchase_sample)
    print("purchases:", tiny_purchase_sample.sum())
    print("sample size:", tiny_purchase_sample.size)
    print("sample conversion:", tiny_rate)

    assert tiny_purchase_sample.sum() == 2
    assert np.isclose(tiny_rate, 0.40)
    """),

    md(r"""
    ### Sampling quality comes before formulas

    A larger biased sample can be worse than a smaller representative sample.

    Examples of bad sampling include:

    - showing the new checkout only to loyal customers;
    - excluding visitors whose page failed to load;
    - comparing weekdays for one version with weekends for the other;
    - counting the same returning visitor many times in one group;
    - stopping the experiment when the result first looks attractive.

    Random assignment helps make experiment groups comparable. It does not guarantee
    perfect balance in one finite sample, so we still inspect the groups and data flow.
    """),

    md(r"""
    ## 4 · Distributions describe uncertain outcomes

    A **random variable** maps an uncertain outcome to a number. A **distribution**
    describes which values that variable can take and how probable they are.

    ### 4.1 Bernoulli: one yes-or-no trial

    A Bernoulli variable $X$ is 1 with probability $p$ and 0 with probability $1-p$:

    $$
    P(X=x)=p^x(1-p)^{1-x},\qquad x\in\{0,1\}
    $$

    **Symbols:** $X$ is the uncertain binary variable; $x$ is an observed value;
    $p$ is the probability of 1; $1-p$ is the probability of 0.

    If $p=0.10$, then $P(X=1)=0.10$ and $P(X=0)=0.90$.

    ### 4.2 Binomial: the number of successes in repeated trials

    If $K$ counts successes in $n$ independent Bernoulli trials with the same $p$:

    $$
    P(K=k)=\binom{n}{k}p^k(1-p)^{n-k}
    $$

    **Symbols:** $K$ is the success count; $k$ is one possible count; $n$ is the
    number of trials; $\binom{n}{k}$ counts which trials could be successes.

    For $n=3$, $p=0.5$, and exactly $k=2$ successes:

    $$
    P(K=2)=\binom{3}{2}(0.5)^2(0.5)^1=3(0.125)=0.375
    $$

    ### 4.3 Normal: a continuous bell-shaped model

    A normal distribution is described by mean $\mu$ and variance $\sigma^2$:

    $$
    X\sim\mathcal N(\mu,\sigma^2)
    $$

    The symbol $\sim$ means “is distributed as.” The normal model is symmetric and
    useful for many measurement errors and sampling distributions. Real data is not
    automatically normal; counts, waiting times, income, and latency may be skewed.
    """),

    code(r"""
    possible_purchases = np.arange(0, 11)
    number_of_trials = 10
    purchase_probability = 0.30

    binomial_probabilities = np.array(
        [
            math.comb(number_of_trials, purchase_count)
            * purchase_probability**purchase_count
            * (1 - purchase_probability) ** (number_of_trials - purchase_count)
            for purchase_count in possible_purchases
        ]
    )

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(possible_purchases, binomial_probabilities, color="tab:blue")
    axis.set_xlabel("purchases among 10 visitors")
    axis.set_ylabel("probability")
    axis.set_title("Binomial distribution: n=10, p=0.30")
    figure.tight_layout()
    plt.show()

    print("sum of probabilities:", binomial_probabilities.sum())
    print("most likely purchase count:", possible_purchases[np.argmax(binomial_probabilities)])

    assert np.isclose(binomial_probabilities.sum(), 1.0)
    """),

    md(r"""
    ## 5 · Mean, variance, covariance, and correlation

    PRE-06 introduced mean and variance. We now connect them to relationships between
    two variables.

    ### 5.1 Mean and sample variance

    For observations $x_1,\ldots,x_n$, the sample mean is:

    $$
    \bar x=\frac{1}{n}\sum_{i=1}^{n}x_i
    $$

    The sample variance is:

    $$
    s_x^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar x)^2
    $$

    **Symbols:** $\bar x$ is the sample mean; $s_x^2$ is sample variance; $x_i$ is
    observation $i$; $n-1$ is the sample-variance denominator. Variance has squared
    units. Standard deviation $s_x=\sqrt{s_x^2}$ returns to the original unit.

    For $x=[1,2,3]$, $\bar x=2$. The squared deviations are $[1,0,1]$, so:

    $$
    s_x^2=\frac{1+0+1}{3-1}=1
    $$

    ### 5.2 Covariance

    Covariance checks whether two variables tend to move together:

    $$
    s_{xy}=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)
    $$

    **Symbols:** $s_{xy}$ is sample covariance; $\bar x$ and $\bar y$ are sample
    means. Positive covariance means the variables often move in the same direction;
    negative means they often move in opposite directions; near zero means no clear
    linear co-movement.

    ### 5.3 Correlation removes units

    $$
    r_{xy}=\frac{s_{xy}}{s_xs_y}
    $$

    **Symbols:** $r_{xy}$ is sample correlation; $s_x$ and $s_y$ are sample standard
    deviations. Correlation usually lies from -1 to 1.

    Correlation measures linear association, not causation. A hidden variable can
    influence both measured variables, and a curved relationship can have low linear
    correlation.
    """),

    code(r"""
    study_hours = np.array([1.0, 2.0, 3.0])
    exam_scores = np.array([2.0, 4.0, 6.0])

    hours_mean = study_hours.mean()
    scores_mean = exam_scores.mean()
    hours_deviation = study_hours - hours_mean
    scores_deviation = exam_scores - scores_mean

    hours_variance = np.sum(hours_deviation**2) / (study_hours.size - 1)
    scores_variance = np.sum(scores_deviation**2) / (exam_scores.size - 1)
    covariance = np.sum(hours_deviation * scores_deviation) / (study_hours.size - 1)
    correlation = covariance / np.sqrt(hours_variance * scores_variance)

    print("hours deviations:", hours_deviation)
    print("score deviations:", scores_deviation)
    print("sample variance of hours:", hours_variance)
    print("sample covariance:", covariance)
    print("sample correlation:", correlation)

    assert np.isclose(hours_variance, 1.0)
    assert np.isclose(covariance, 2.0)
    assert np.isclose(correlation, 1.0)
    """),

    md(r"""
    ### Covariance matrix and FND-01

    With several numerical features, covariance values form a matrix. Diagonal
    entries are variances; off-diagonal entries are covariances.

    $$
    S=
    \begin{bmatrix}
    s_x^2 & s_{xy}\\
    s_{xy} & s_y^2
    \end{bmatrix}
    $$

    **Symbols:** $S$ is the covariance matrix; $s_x^2$ and $s_y^2$ are feature
    variances; $s_{xy}$ is their covariance. The matrix is symmetric because
    $s_{xy}=s_{yx}$.

    Later, PCA will use this matrix to find directions of high variation. We do not
    need eigenvectors yet.
    """),

    code(r"""
    paired_measurements = np.column_stack([study_hours, exam_scores])
    covariance_matrix = np.cov(paired_measurements, rowvar=False, ddof=1)

    print("data shape:", paired_measurements.shape)
    print("covariance matrix:\n", covariance_matrix)

    assert covariance_matrix.shape == (2, 2)
    assert np.allclose(covariance_matrix, [[1, 2], [2, 4]])
    assert np.allclose(covariance_matrix, covariance_matrix.T)
    """),

    md(r"""
    ## 6 · An estimator has its own sampling distribution

    Imagine drawing many samples of the same size from one population. Calculate the
    sample mean each time. Those means form a new distribution called the **sampling
    distribution of the mean**.

    Do not confuse these:

    | Distribution | What varies? | Example |
    | --- | --- | --- |
    | Data distribution | Individual observations | One visitor's spending |
    | Sampling distribution | A statistic across repeated samples | Mean spending from samples of 50 visitors |

    An estimator can be judged by:

    - **bias:** whether its repeated-sample average misses the true parameter;
    - **variance:** how widely it changes across samples;
    - **standard error:** the standard deviation of its sampling distribution.

    Analogy: arrows tightly grouped away from the bullseye have low variance but high
    bias. Arrows spread around the bullseye have low bias but high variance. We want
    estimates that are centered correctly and reasonably stable.
    """),

    code(r"""
    population = random_generator.exponential(scale=20.0, size=200_000)
    population_mean = population.mean()

    sample_size = 40
    repeated_sample_means = np.array(
        [
            random_generator.choice(population, size=sample_size, replace=True).mean()
            for _ in range(5_000)
        ]
    )

    estimator_bias = repeated_sample_means.mean() - population_mean
    estimated_standard_error = repeated_sample_means.std(ddof=1)

    print("population mean:", round(population_mean, 3))
    print("average sample mean:", round(repeated_sample_means.mean(), 3))
    print("estimated bias:", round(estimator_bias, 3))
    print("estimated standard error:", round(estimated_standard_error, 3))

    assert abs(estimator_bias) < 0.3
    assert estimated_standard_error > 0
    """),

    md(r"""
    ## 7 · Law of Large Numbers: one estimate becomes more stable

    The Law of Large Numbers says that, under suitable conditions, the sample mean
    approaches the population mean as sample size grows:

    $$
    \bar X_n\longrightarrow\mu
    \quad\text{as}\quad n\longrightarrow\infty
    $$

    **Symbols:** $\bar X_n$ is the mean of $n$ observations; $\mu$ is the population
    mean; the arrow means the sample mean approaches the population mean; $\infty$
    means sample size grows without bound.

    It does **not** say every larger sample is closer than the previous sample. The
    path can wiggle. It says large-sample deviation becomes less likely.

    It also does not repair bias. A huge sample of only loyal customers still may not
    represent all customers.
    """),

    code(r"""
    coin_flips = random_generator.binomial(n=1, p=0.60, size=5_000)
    running_sample_sizes = np.arange(1, coin_flips.size + 1)
    running_rates = np.cumsum(coin_flips) / running_sample_sizes

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(running_sample_sizes, running_rates, label="running sample rate")
    axis.axhline(0.60, color="tab:red", linestyle="--", label="population rate 0.60")
    axis.set_xlabel("number of observations")
    axis.set_ylabel("sample rate")
    axis.set_title("Law of Large Numbers: the estimate settles with more data")
    axis.legend()
    figure.tight_layout()
    plt.show()

    print("rate after 10 flips:", running_rates[9])
    print("rate after 5,000 flips:", running_rates[-1])

    assert abs(running_rates[-1] - 0.60) < 0.03
    """),

    md(r"""
    ## 8 · Central Limit Theorem: repeated means become bell-shaped

    The Central Limit Theorem concerns a **sampling distribution**, not necessarily
    the raw data. For independent observations with finite variance, sufficiently
    large sample means are approximately normal:

    $$
    \frac{\bar X_n-\mu}{\sigma/\sqrt n}
    \overset{d}{\longrightarrow}
    \mathcal N(0,1)
    $$

    **Symbols:** $\bar X_n$ is a sample mean; $\mu$ and $\sigma$ are population mean
    and standard deviation; $\sigma/\sqrt n$ is the standard deviation of the sample
    mean; $\overset{d}{\longrightarrow}$ means the distribution approaches;
    $\mathcal N(0,1)$ is a standard normal distribution.

    The raw population may remain skewed. The repeated sample means become more
    bell-shaped and narrower.

    The approximation can be poor with tiny samples, strong dependence, or extremely
    heavy tails. Always inspect the data-generating situation.
    """),

    code(r"""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))

    for axis, current_sample_size in zip(axes, [1, 5, 40]):
        simulated_means = random_generator.exponential(
            scale=1.0,
            size=(10_000, current_sample_size),
        ).mean(axis=1)
        axis.hist(simulated_means, bins=50, density=True, color="tab:blue", alpha=0.75)
        axis.set_title(f"means of n={current_sample_size}")
        axis.set_xlabel("sample mean")
        axis.set_ylabel("density")

    figure.suptitle("The source is skewed; repeated means become more bell-shaped")
    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 9 · Standard error measures estimate-to-estimate movement

    Standard deviation describes variation among individual observations. Standard
    error describes variation among repeated estimates.

    For a sample mean, estimate the standard error with:

    $$
    \operatorname{SE}(\bar x)=\frac{s}{\sqrt n}
    $$

    **Symbols:** $\operatorname{SE}$ means standard error; $\bar x$ is the sample
    mean; $s$ is sample standard deviation; $n$ is sample size.

    Example: if $s=12$ minutes and $n=36$ deliveries:

    $$
    \operatorname{SE}(\bar x)=\frac{12}{\sqrt{36}}=\frac{12}{6}=2\text{ minutes}
    $$

    To halve standard error, we need about four times as many independent observations:

    $$
    \frac{s}{\sqrt{4n}}=\frac{1}{2}\frac{s}{\sqrt n}
    $$

    More data has diminishing returns. Dependence also reduces effective information;
    1,000 repeated events from ten users are not equivalent to 1,000 independent users.
    """),

    code(r"""
    delivery_standard_deviation = 12.0

    for delivery_count in [36, 144, 576]:
        standard_error = delivery_standard_deviation / np.sqrt(delivery_count)
        print(f"n={delivery_count:>3}: standard error={standard_error:.2f} minutes")

    assert np.isclose(delivery_standard_deviation / np.sqrt(36), 2.0)
    assert np.isclose(delivery_standard_deviation / np.sqrt(144), 1.0)
    """),

    md(r"""
    ## 10 · Confidence intervals show a plausible estimation range

    A confidence interval combines an estimate with a margin of error:

    $$
    \text{estimate}\ \pm\ \text{critical value}\times\text{standard error}
    $$

    For a mean with unknown population standard deviation, a common interval is:

    $$
    \bar x\pm t^*\frac{s}{\sqrt n}
    $$

    **Symbols:** $\bar x$ is the sample mean; $t^*$ is a critical value from a
    t-distribution; $s$ is sample standard deviation; $n$ is sample size.

    Suppose $\bar x=50$, $s=12$, $n=36$, and the chosen critical value is about 2:

    $$
    50\pm2\left(\frac{12}{6}\right)=50\pm4=[46,54]
    $$

    Correct interpretation of a 95% frequentist confidence procedure:

    > If we repeated the sampling method many times, about 95% of intervals built by
    > this procedure would contain the true parameter, when its assumptions hold.

    After one interval is calculated, the parameter is fixed. Avoid saying there is
    a 95% frequentist probability that this particular fixed parameter is inside.

    A narrow interval from biased data is precisely wrong, not trustworthy.
    """),

    code(r"""
    from scipy import stats

    delivery_times = np.array([44, 48, 51, 47, 55, 52, 49, 46, 58, 50], dtype=float)
    delivery_count = delivery_times.size
    mean_delivery_time = delivery_times.mean()
    sample_standard_deviation = delivery_times.std(ddof=1)
    standard_error = sample_standard_deviation / np.sqrt(delivery_count)
    critical_value = stats.t.ppf(0.975, df=delivery_count - 1)
    margin_of_error = critical_value * standard_error
    confidence_interval = (
        mean_delivery_time - margin_of_error,
        mean_delivery_time + margin_of_error,
    )

    print("sample mean:", round(mean_delivery_time, 3))
    print("sample standard deviation:", round(sample_standard_deviation, 3))
    print("standard error:", round(standard_error, 3))
    print("t critical value:", round(critical_value, 3))
    print("95% confidence interval:", tuple(round(value, 3) for value in confidence_interval))

    library_interval = stats.t.interval(
        confidence=0.95,
        df=delivery_count - 1,
        loc=mean_delivery_time,
        scale=standard_error,
    )

    assert np.allclose(confidence_interval, library_interval)
    assert confidence_interval[0] < mean_delivery_time < confidence_interval[1]
    """),

    md(r"""
    ## 11 · Hypothesis tests ask whether data conflicts with a model

    A hypothesis test begins with two claims:

    - **null hypothesis $H_0$:** the reference claim, often no difference;
    - **alternative hypothesis $H_1$:** the competing claim.

    For the checkout experiment:

    $$
    H_0:p_{new}-p_{old}=0
    $$

    $$
    H_1:p_{new}-p_{old}\ne0
    $$

    **Symbols:** $p_{new}$ and $p_{old}$ are population conversion rates; $H_0$ and
    $H_1$ name the two hypotheses; $\ne$ means not equal.

    A test statistic measures how far the observed effect is from the null value in
    standard-error units:

    $$
    z=\frac{\text{observed difference}-\text{null difference}}
            {\text{standard error under }H_0}
    $$

    The **p-value** is the probability, assuming $H_0$ and the test assumptions are
    true, of a result at least as extreme as the observed result.

    A p-value is **not**:

    - the probability that $H_0$ is true;
    - the probability the result happened “by chance”;
    - the size or business value of the effect;
    - proof that assumptions or data collection were valid.

    We choose a significance level $\alpha$ before seeing the result. If $p<\alpha$,
    we reject $H_0$. Otherwise we fail to reject it. “Fail to reject” does not prove
    equality; the study may simply be imprecise.
    """),

    code(r"""
    def two_proportion_z_test(successes_a, total_a, successes_b, total_b):
        '''Return rates, difference, pooled standard error, z score, and two-sided p-value.'''
        for successes, total in [(successes_a, total_a), (successes_b, total_b)]:
            if not 0 <= successes <= total:
                raise ValueError("successes must lie between zero and total")
            if total <= 0:
                raise ValueError("total must be positive")

        rate_a = successes_a / total_a
        rate_b = successes_b / total_b
        difference = rate_b - rate_a
        pooled_rate = (successes_a + successes_b) / (total_a + total_b)
        null_standard_error = math.sqrt(
            pooled_rate * (1 - pooled_rate) * (1 / total_a + 1 / total_b)
        )
        z_score = difference / null_standard_error
        p_value = 2 * stats.norm.sf(abs(z_score))
        return rate_a, rate_b, difference, null_standard_error, z_score, p_value


    test_result = two_proportion_z_test(100, 1_000, 130, 1_000)
    rate_a, rate_b, difference, null_se, z_score, p_value = test_result

    print(f"old rate: {rate_a:.3f}")
    print(f"new rate: {rate_b:.3f}")
    print(f"difference: {difference:.3f}")
    print(f"null standard error: {null_se:.4f}")
    print(f"z score: {z_score:.3f}")
    print(f"two-sided p-value: {p_value:.4f}")

    assert np.isclose(difference, 0.03)
    assert 0 < p_value < 0.05
    """),

    md(r"""
    ### Expected result and its limit

    The p-value is about 0.036. At a pre-declared $\alpha=0.05$, this test rejects
    the no-difference hypothesis.

    That narrow statement does not prove the new page caused the improvement unless
    assignment and data collection were valid. It also does not tell us whether a
    3-point lift is worth shipping. For that, inspect the effect and its interval.
    """),

    md(r"""
    ## 12 · Effect size, errors, power, and practical importance

    ### Effect size

    The absolute conversion change is:

    $$
    \Delta=\hat p_{new}-\hat p_{old}=0.13-0.10=0.03
    $$

    **Symbols:** $\Delta$ (delta) means change; the hats mark sample estimates.
    This is a 3-percentage-point absolute lift. Relative lift is $0.03/0.10=30\%$.
    Always state which one you mean.

    ### Two possible testing errors

    | Reality and decision | Name | Checkout meaning |
    | --- | --- | --- |
    | No real effect, but reject $H_0$ | Type I error | Ship a page that is not better |
    | Real effect, but fail to reject $H_0$ | Type II error | Miss a useful improvement |

    The pre-declared significance level $\alpha$ controls the long-run Type I error
    rate under the test assumptions. **Power** is the probability of detecting a
    specified real effect:

    $$
    \text{power}=1-\beta
    $$

    **Symbols:** $\beta$ is the Type II error probability for the specified effect.

    Power generally increases with:

    - a larger true effect;
    - a larger sample;
    - lower measurement noise;
    - a less strict significance threshold.

    Decide the minimum effect worth detecting before collecting data. A massive sample
    can make a tiny, useless difference statistically significant.
    """),

    code(r"""
    unpooled_difference_standard_error = math.sqrt(
        old_rate * (1 - old_rate) / old_visitors
        + new_rate * (1 - new_rate) / new_visitors
    )
    difference_margin = stats.norm.ppf(0.975) * unpooled_difference_standard_error
    difference_interval = (
        observed_difference - difference_margin,
        observed_difference + difference_margin,
    )

    relative_lift = observed_difference / old_rate

    print(f"absolute lift: {observed_difference:.1%} points")
    print(f"relative lift: {relative_lift:.1%}")
    print("approximate 95% interval for absolute lift:",
          tuple(f"{value:.1%}" for value in difference_interval))

    assert np.isclose(relative_lift, 0.30)
    assert difference_interval[0] > 0
    """),

    md(r"""
    ## 13 · Bootstrap: let resampling approximate uncertainty

    Some statistics have no simple standard-error formula. The **bootstrap** treats
    the observed sample as a stand-in population:

    1. draw a new sample of the same size **with replacement**;
    2. calculate the statistic;
    3. repeat many times;
    4. inspect the distribution of bootstrap statistics.

    “With replacement” means an observed row may appear more than once in a resample.
    Without replacement, every full-size resample would contain the same rows and
    produce no useful variation.

    For percentile interval endpoints:

    $$
    [q_{0.025},q_{0.975}]
    $$

    **Symbols:** $q_{0.025}$ and $q_{0.975}$ are the 2.5th and 97.5th percentiles of
    the bootstrap statistics.

    Bootstrap limitations matter:

    - it cannot repair an unrepresentative original sample;
    - ordinary row bootstrap is wrong for dependent time-series or grouped data;
    - very small samples may not represent the population shape;
    - a basic percentile interval is not ideal for every statistic.
    """),

    code(r"""
    customer_spending = np.array([12, 14, 15, 18, 21, 23, 25, 28, 40, 90], dtype=float)

    def bootstrap_statistic(data, statistic_function, repetitions=5_000, seed=123):
        data = np.asarray(data, dtype=float)
        if data.ndim != 1 or data.size < 2:
            raise ValueError("data must be a one-dimensional sample with at least two values")
        generator = np.random.default_rng(seed)
        resample_indices = generator.integers(0, data.size, size=(repetitions, data.size))
        resamples = data[resample_indices]
        return np.apply_along_axis(statistic_function, 1, resamples)


    bootstrap_medians = bootstrap_statistic(customer_spending, np.median)
    median_interval = np.percentile(bootstrap_medians, [2.5, 97.5])

    print("sample median:", np.median(customer_spending))
    print("bootstrap median interval:", median_interval)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(bootstrap_medians, bins=30, color="tab:purple", alpha=0.75)
    axis.axvline(median_interval[0], color="tab:red", linestyle="--")
    axis.axvline(median_interval[1], color="tab:red", linestyle="--")
    axis.set_xlabel("bootstrap sample median")
    axis.set_ylabel("count")
    axis.set_title("Bootstrap distribution of the median")
    figure.tight_layout()
    plt.show()

    assert median_interval[0] <= np.median(customer_spending) <= median_interval[1]
    """),

    md(r"""
    ## 14 · Mini-project, practice, and mastery checkpoint

    ### Mini-project: checked checkout experiment report

    **Goal:** turn experiment counts into a transparent statistical report without
    turning the p-value into the decision.

    **Input columns:**

    | Column | Meaning |
    | --- | --- |
    | `variant` | Stable group name |
    | `visitors` | Assigned visitors |
    | `purchases` | Visitors who purchased |
    | `revenue` | Total group revenue |

    **Required workflow:**

    1. validate counts and unique variant names;
    2. calculate conversion rates;
    3. calculate absolute and relative effect;
    4. calculate a two-sided test and an interval for the conversion difference;
    5. calculate revenue per visitor as a guardrail;
    6. return evidence separately from the ship/no-ship decision.
    """),

    code(r"""
    experiment_data = pd.DataFrame(
        {
            "variant": ["old", "new"],
            "visitors": [1_000, 1_000],
            "purchases": [100, 130],
            "revenue": [25_000.0, 29_000.0],
        }
    )

    required_columns = {"variant", "visitors", "purchases", "revenue"}
    if set(experiment_data.columns) != required_columns:
        raise ValueError("experiment columns do not match the required contract")
    if experiment_data["variant"].duplicated().any():
        raise ValueError("variant names must be unique")
    if (experiment_data["visitors"] <= 0).any():
        raise ValueError("visitor counts must be positive")
    if (experiment_data["purchases"] < 0).any():
        raise ValueError("purchase counts cannot be negative")
    if (experiment_data["purchases"] > experiment_data["visitors"]).any():
        raise ValueError("purchases cannot exceed visitors")

    report = experiment_data.copy()
    report["conversion_rate"] = report["purchases"] / report["visitors"]
    report["revenue_per_visitor"] = report["revenue"] / report["visitors"]

    old_row = report.loc[report["variant"] == "old"].iloc[0]
    new_row = report.loc[report["variant"] == "new"].iloc[0]
    project_test = two_proportion_z_test(
        int(old_row["purchases"]),
        int(old_row["visitors"]),
        int(new_row["purchases"]),
        int(new_row["visitors"]),
    )
    project_difference = project_test[2]
    project_p_value = project_test[5]
    project_relative_lift = project_difference / old_row["conversion_rate"]

    project_standard_error = math.sqrt(
        old_row["conversion_rate"] * (1 - old_row["conversion_rate"]) / old_row["visitors"]
        + new_row["conversion_rate"] * (1 - new_row["conversion_rate"]) / new_row["visitors"]
    )
    project_margin = stats.norm.ppf(0.975) * project_standard_error
    project_interval = (
        project_difference - project_margin,
        project_difference + project_margin,
    )

    print(report.to_string(index=False))
    print(f"\nabsolute conversion lift: {project_difference:.1%} points")
    print(f"relative conversion lift: {project_relative_lift:.1%}")
    print(f"two-sided p-value: {project_p_value:.4f}")
    print("approximate 95% lift interval:", tuple(f"{value:.1%}" for value in project_interval))
    print("evidence statement: data conflicts with equal rates at alpha=0.05")
    print("decision statement: shipping still requires effect, guardrail, and validity review")

    assert np.isclose(project_difference, 0.03)
    assert np.isclose(project_relative_lift, 0.30)
    assert project_p_value < 0.05
    assert new_row["revenue_per_visitor"] > old_row["revenue_per_visitor"]
    """),

    md(r"""
    ### Worked example

    A sample has values $[2,4,6]$.

    - Mean: $(2+4+6)/3=4$.
    - Deviations: $[-2,0,2]$.
    - Squared deviations: $[4,0,4]$.
    - Sample variance: $(4+0+4)/(3-1)=4$.
    - Sample standard deviation: $\sqrt4=2$.
    - Standard error of the mean: $2/\sqrt3\approx1.155$.

    Each number answers a different question. Standard deviation describes the three
    observations; standard error describes how their mean would vary across samples.

    ### Guided practice

    1. Label each item: population, sample, parameter, or statistic.
    2. Calculate the conversion rate for 18 purchases among 120 visitors.
    3. Calculate mean, sample variance, and standard deviation for $[1,3,5]$.
    4. For $x=[1,2,3]$ and $y=[6,4,2]$, calculate covariance and correlation.
    5. Calculate standard error when $s=15$ and $n=25$.
    6. Explain the difference between a data distribution and sampling distribution.

    ### Independent practice

    7. Simulate repeated Bernoulli samples for $p=0.2$ with sizes 10, 100, and 1,000.
       Compare the spread of their sample rates.
    8. Recreate the CLT plot with a log-normal source.
    9. Build a 95% t-interval for a small numerical sample and verify with SciPy.
    10. State $H_0$ and $H_1$ for a two-sided difference in average delivery time.
    11. Explain a p-value without using the phrase “probability the null is true.”
    12. Bootstrap the mean and median of a skewed sample. Compare their intervals.

    ### Challenge

    Rebuild the checkout mini-project without copying its code. Add:

    - a minimum practically important lift;
    - a guardrail rule for revenue per visitor;
    - validation for missing values and exactly two variants;
    - an explicit assumptions list;
    - six meaningful assertions;
    - a conclusion that separates evidence from action.

    ### Self-check

    Before reporting any result, answer:

    - What population does this sample represent?
    - How were observations selected or assigned?
    - What is the effect in understandable units?
    - What uncertainty measure was used, and why?
    - Which assumptions could fail?
    - Was the threshold chosen before viewing results?
    - Is the result practically important?
    """),

    md(r"""
    ### Solution and scoring rubric

    1. A population is the full target group; a sample is observed cases; a parameter
       describes the population; a statistic is calculated from the sample.
    2. $18/120=0.15=15\%$.
    3. Mean 3; deviations $[-2,0,2]$; sample variance 4; standard deviation 2.
    4. The variables move in opposite directions; covariance is negative and
       correlation is -1.
    5. $15/\sqrt{25}=3$.
    6. Individual values form the data distribution; repeated estimates form a
       sampling distribution.
    7. All rate distributions center near 0.2; their spread shrinks with sample size.
    8. Raw log-normal values remain skewed while sufficiently large sample means become
       more bell-shaped.
    9. The manual interval should match `stats.t.interval` within rounding.
    10. $H_0:\mu_A-\mu_B=0$ and $H_1:\mu_A-\mu_B\ne0$.
    11. A p-value is a tail probability calculated under the null model and assumptions.
    12. In skewed data, mean and median can have noticeably different bootstrap behavior.

    Challenge scoring:

    | Skill | Points |
    | --- | ---: |
    | Input contract and valid unit of analysis | 3 |
    | Correct effect, interval, and test | 4 |
    | Correct p-value interpretation | 3 |
    | Practical threshold and guardrail | 3 |
    | Assumptions and limitations | 3 |
    | Assertions and readable report | 4 |
    | **Total** | **20** |

    ### Common mistakes

    - Treating a sample statistic as the exact population parameter.
    - Using $n$ instead of $n-1$ without stating whether population or sample variance
      is intended.
    - Saying covariance or correlation proves causation.
    - Confusing standard deviation with standard error.
    - Saying the CLT makes raw data normal.
    - Interpreting a 95% confidence interval as a 95% frequentist probability about
      one already-fixed parameter.
    - Saying a p-value is the probability that the null hypothesis is true.
    - Treating “not significant” as proof of no effect.
    - Ignoring effect size because $p<0.05$.
    - Repeatedly checking a test and stopping at the first small p-value.
    - Bootstrapping rows that are dependent by user, group, or time.
    - Using more data to hide a biased sampling process.

    ### Readiness threshold

    Score at least **16/20** on the challenge and answer all Quick Check questions
    without notes. If standard deviation, standard error, confidence interval, and
    p-value blur together, repeat Sections 6–12 before continuing.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    1. Why can two random samples produce different statistics?
    2. What is the difference between a parameter and a statistic?
    3. What does covariance add after learning variance?
    4. What is the difference between standard deviation and standard error?
    5. What does the CLT say about sample means?
    6. What does a 95% confidence procedure guarantee in repeated use?
    7. What exactly is a p-value conditioned on?
    8. Why must practical importance be checked separately?
    9. What are Type I and Type II errors?
    10. Why does an ordinary bootstrap require representative, appropriately
        independent observations?

    ### Teach it back

    Explain this story to someone who has never studied statistics:

    > We observe a sample, calculate an effect, estimate how much that effect would
    > move across samples, and make a limited claim whose strength depends on both
    > the data-collection design and the statistical assumptions.

    If you can explain each link with the checkout example and no jargon, the
    foundation is solid.

    ### Memory aid

    **A sample gives an estimate; uncertainty tells us how carefully to trust it.**

    ### Next dependency

    FND-03 uses these ideas during data exploration. A mean, correlation, or unusual
    group difference is evidence to investigate—not automatically a population truth
    or causal result.
    """),
]


build("01_ml_foundations/02_probability_and_statistics.ipynb", cells)
