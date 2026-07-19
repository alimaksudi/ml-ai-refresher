"""Builder for PRE-06 — Probability, Random Variables, and Statistics."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # PRE-06 · Probability, Random Variables, and Statistics

    *Reason about uncertainty without pretending one outcome is guaranteed*

    | Lesson detail | Value |
    | --- | --- |
    | Prerequisites | PRE-01 through PRE-04 |
    | Estimated study time | 6–8 hours across two sessions |
    | Main outcome | Calculate and interpret events, conditional probability, expectation, and spread |
    | Next mathematical lessons | FND-01, then FND-02 |

    Probability describes uncertainty before an outcome is known. Statistics uses
    observed data to describe or learn about the process that produced it.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - distinguish experiment, outcome, sample space, and event;
    - calculate theoretical and empirical probability;
    - use complements, unions, and intersections;
    - build a joint count table and obtain marginal probabilities;
    - calculate conditional probability with the correct denominator;
    - distinguish independent events from mutually exclusive events;
    - update a probability with Bayes' rule using counts;
    - explain a random variable and a probability distribution;
    - distinguish a discrete PMF from continuous density intuition;
    - calculate expected value, population variance, and standard deviation;
    - distinguish population quantities from sample estimates;
    - verify results with seeded simulation without confusing simulation with proof.
    """),

    md(r"""
    ## 2 · The problem we are trying to solve

    A monitoring system raises an alert. The alert catches 90% of real incidents,
    but real incidents are rare. How likely is a real incident after an alert?

    Many people answer 90%. That reverses the condition:

    - $P(\text{alert}\mid\text{incident})$ asks how alerts behave for known incidents.
    - $P(\text{incident}\mid\text{alert})$ asks what an observed alert means.

    The second answer also depends on the incident base rate and false-alert rate.
    We need a language for outcomes, events, denominators, and updated beliefs.
    """),

    md(r"""
    ## 3 · Three mental pictures

    ### The bag

    A bag contains known proportions but the next draw is hidden. This builds
    long-run frequency intuition.

    ### The filter

    Conditional probability first filters to cases satisfying the condition, then
    calculates a fraction inside that smaller group.

    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #4c78a8; border-radius: 10px; padding: 14px 18px; background: #eef5ff; color: #172b4d; text-align: center;"><strong>All cases</strong><br>choose the condition</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 14px 18px; background: #fff4e8; color: #4a2b0b; text-align: center;"><strong>Filtered group</strong><br>new denominator</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 14px 18px; background: #eef8ec; color: #173d17; text-align: center;"><strong>Target inside group</strong><br>conditional fraction</div>
    </div>

    ### The shape

    A probability distribution spreads total probability across possible values.
    Expected value is its balance point; variance and standard deviation describe
    spread. None of these guarantees the next outcome.
    """),

    md(r"""
    ## 4 · Mathematical foundations

    ### 4.1 Experiment, outcome, sample space, and event

    For one fair die roll:

    - **experiment:** roll the die;
    - **outcome:** the observed face, such as 4;
    - **sample space:** every possible outcome;
    - **event:** a selected set of outcomes.

    $$
    S=\{1,2,3,4,5,6\}
    $$

    $$
    A=\{2,4,6\}
    $$

    **Symbols:** $S$ names the sample space; braces form a set; $A$ is the event
    “even outcome.” An event occurs when the observed outcome belongs to its set.
    """),

    md(r"""
    ### 4.2 Theoretical and empirical probability

    For equally likely outcomes:

    $$
    P(A)=\frac{\text{outcomes in }A}{\text{outcomes in }S}
    $$

    **Symbols:** $P(A)$ means probability of event $A$. For even die outcomes,
    $P(A)=3/6=0.5$.

    Empirical probability comes from observations:

    $$
    \widehat{P}(A)=\frac{\text{observed occurrences of }A}{\text{observed trials}}
    $$

    The hat marks an estimate. Ten rolls might produce six evens, giving 0.6. That
    does not change the fair-die theoretical probability of 0.5. With more fair,
    independent trials, empirical frequency often gets closer to the underlying
    probability, though no finite sample must match exactly.
    """),

    md(r"""
    ### 4.3 Probability rules, complements, unions, and intersections

    Probabilities satisfy $0\le P(A)\le1$, and the whole sample space has probability
    1. The complement means “not $A$”:

    $$
    P(A^c)=1-P(A)
    $$

    **Symbols:** superscript $c$ means complement. If failure probability is 0.03,
    non-failure probability is 0.97.

    - $A\cup B$, read “A union B,” means A or B or both.
    - $A\cap B$, read “A intersection B,” means both A and B.

    $$
    P(A\cup B)=P(A)+P(B)-P(A\cap B)
    $$

    **Symbols:** subtracting the intersection avoids counting shared outcomes twice.
    If 40% use mobile, 30% purchase, and 15% do both, mobile-or-purchase probability
    is $0.40+0.30-0.15=0.55$.
    """),

    md(r"""
    ### 4.4 Joint and marginal probability from a count table

    Suppose 100 users produce:

    | | Purchase | No purchase | Total |
    | --- | ---: | ---: | ---: |
    | Mobile | 12 | 28 | 40 |
    | Desktop | 18 | 42 | 60 |
    | Total | 30 | 70 | 100 |

    A **joint** probability uses an interior cell:

    $$
    P(\text{mobile}\cap\text{purchase})=\frac{12}{100}=0.12
    $$

    A **marginal** probability uses an edge total:

    $$
    P(\text{purchase})=\frac{30}{100}=0.30
    $$

    **Symbols:** intersection means both properties. “Marginal” comes from totals
    traditionally written in table margins.
    """),

    md(r"""
    ### 4.5 Conditional probability changes the denominator

    $$
    P(A\mid B)=\frac{P(A\cap B)}{P(B)}
    $$

    **Symbols:** the vertical bar means “given B.” Filter to $B$ first, so $P(B)$ is
    the denominator and must be greater than zero.

    Given mobile, purchase probability is:

    $$
    P(\text{purchase}\mid\text{mobile})=\frac{12}{40}=0.30
    $$

    Given purchase, mobile probability is:

    $$
    P(\text{mobile}\mid\text{purchase})=\frac{12}{30}=0.40
    $$

    Same shared count, different denominator. Conditional direction matters.
    """),

    md(r"""
    ### 4.6 Independent is not the same as mutually exclusive

    Events $A$ and $B$ are **independent** when learning one does not change the
    probability of the other:

    $$
    P(A\cap B)=P(A)P(B)
    $$

    **Symbols:** multiplication joins probabilities only under independence or an
    equivalent conditional calculation.

    **Mutually exclusive** events cannot happen together, so their intersection is
    empty. On one die roll, “even” and “odd” are mutually exclusive. They are not
    independent: learning that the result is even makes odd impossible.

    Repeated coin flips may be modeled as independent. Two measurements from the
    same person usually are not independent merely because they are separate rows.
    """),

    md(r"""
    ### 4.7 Bayes' rule reverses a condition

    $$
    P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}
    $$

    **Symbols:** $P(A)$ is the prior or base rate; $P(B\mid A)$ describes evidence
    under $A$; $P(A\mid B)$ is the updated probability after observing $B$.

    Counts make the rule easier. Among 1,000 periods:

    - 10 contain an incident;
    - the alert catches 9 of those incidents;
    - among 990 normal periods, a 5% false-alert rate creates about 50 alerts.

    There are about 59 alerts total, only 9 tied to incidents:

    $$
    P(\text{incident}\mid\text{alert})\approx\frac{9}{59}\approx0.153
    $$

    High detection does not guarantee a high post-alert probability when the base
    rate is small. FND-02 will formalize priors, likelihoods, and posteriors.
    """),

    md(r"""
    ### 4.8 Random variables and distributions

    A random variable maps outcomes to numbers. Capital $X$ names the uncertain
    quantity; lowercase $x$ names one possible value.

    For the number of failures in one request, $X\in\{0,1\}$ might have:

    | $x$ | $P(X=x)$ |
    | ---: | ---: |
    | 0 | 0.98 |
    | 1 | 0.02 |

    A discrete **probability mass function (PMF)** assigns probabilities that sum to
    one. A continuous **probability density function (PDF)** is different: height is
    density, and probability is area over an interval. PRE-04's integral provides
    the accumulation idea. A single exact point usually has zero probability under
    a continuous model even when nearby intervals have positive probability.
    """),

    md(r"""
    ### 4.9 Expected value is a probability-weighted average

    For a discrete random variable:

    $$
    \mathbb{E}[X]=\sum_x xP(X=x)
    $$

    **Symbols:** $\mathbb{E}$ means expected value, $x$ is one possible value, and
    the sum combines every value weighted by its probability.

    If an incident costs $50{,}000$ with probability 0.02 and otherwise costs zero:

    $$
    \mathbb{E}[X]=50{,}000(0.02)+0(0.98)=1{,}000
    $$

    This is a long-run average cost per period, not a promised $1,000 bill each
    period. Most periods cost zero; rare periods cost much more.
    """),

    md(r"""
    ### 4.10 Mean, variance, and standard deviation

    For a population of $n$ observed values:

    $$
    \mu=\frac{1}{n}\sum_{i=1}^{n}x_i
    $$

    $$
    \sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2
    $$

    $$
    \sigma=\sqrt{\sigma^2}
    $$

    **Symbols:** $\mu$ is population mean, $\sigma^2$ is population variance,
    $\sigma$ is standard deviation, and $x_i$ is observation $i$.

    For $(2,4,6)$, mean is 4; deviations are $(-2,0,2)$; squared deviations are
    $(4,0,4)$; variance is $8/3\approx2.67$; standard deviation is about 1.63.

    Variance has squared units. Standard deviation returns to the original unit. A
    sample statistic often divides squared deviations by $n-1$ when estimating a
    larger population's variance; FND-02 explains why and adds uncertainty intervals.
    """),

    md(r"""
    ## 5 · Worked example: one table, four questions

    **Worked example:** use the 100-user table from Section 4.

    - Joint mobile-and-purchase: $12/100=0.12$.
    - Marginal purchase: $30/100=0.30$.
    - Purchase given mobile: $12/40=0.30$.
    - Mobile given purchase: $12/30=0.40$.

    The numerator 12 repeats, but the question chooses the denominator. Before any
    conditional calculation, say the filtered group aloud.
    """),

    code(r"""
    user_counts = {
        "mobile_purchase": 12,
        "mobile_total": 40,
        "purchase_total": 30,
        "all_users": 100,
    }

    joint = user_counts["mobile_purchase"] / user_counts["all_users"]
    purchase_given_mobile = user_counts["mobile_purchase"] / user_counts["mobile_total"]
    mobile_given_purchase = user_counts["mobile_purchase"] / user_counts["purchase_total"]

    print("joint:", joint)
    print("purchase given mobile:", purchase_given_mobile)
    print("mobile given purchase:", mobile_given_purchase)

    assert joint == 0.12
    assert purchase_given_mobile == 0.30
    assert mobile_given_purchase == 0.40
    """),

    md(r"""
    ## 6 · Simulation connects probability to observed frequency

    Simulation repeats a declared random process. It is useful for intuition and
    complex systems, but it does not repair incorrect assumptions or prove a formula.
    """),

    code(r"""
    import numpy as np

    rng = np.random.default_rng(42)
    trial_sizes = [10, 100, 1_000, 10_000]

    for trial_count in trial_sizes:
        draws = rng.choice(["blue", "red"], size=trial_count, p=[0.7, 0.3])
        observed_blue_rate = np.mean(draws == "blue")
        print(f"trials={trial_count:>5} observed blue rate={observed_blue_rate:.3f}")
    """),

    md(r"""
    Small samples may wander far from 0.7. Larger samples tend to stabilize near the
    declared probability, but exact convergence is not guaranteed at a chosen finite
    size. The explicit seed makes the demonstration reproducible.
    """),

    md(r"""
    ## 7 · Visualize a PMF and repeated estimates

    The left plot shows a discrete probability distribution whose bar heights sum to
    one. The right plot shows empirical estimates across increasing sample sizes.
    """),

    code(r"""
    import matplotlib.pyplot as plt

    values = np.array([0, 1])
    probabilities = np.array([0.98, 0.02])

    rng_plot = np.random.default_rng(7)
    sizes = np.array([10, 30, 100, 300, 1_000, 3_000])
    estimates = [rng_plot.binomial(size, 0.2) / size for size in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(values, probabilities, color=["tab:blue", "tab:red"])
    axes[0].set_xticks(values, ["no failure", "failure"])
    axes[0].set_ylabel("probability")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Discrete failure PMF")

    axes[1].plot(sizes, estimates, marker="o", label="empirical estimate")
    axes[1].axhline(0.2, color="tab:red", linestyle="--", label="declared p=0.2")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of trials (log scale)")
    axes[1].set_ylabel("observed success fraction")
    axes[1].set_title("Observed frequency stabilizes")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    assert np.isclose(probabilities.sum(), 1.0)
    """),

    md(r"""
    ## 8 · NumPy statistics with manual anchors

    NumPy calculates summaries quickly, but we still verify one small dataset by
    hand and state whether it represents a full population or a sample.
    """),

    code(r"""
    observations = np.array([2.0, 4.0, 6.0])
    population_mean = observations.mean()
    deviations = observations - population_mean
    squared_deviations = deviations ** 2
    population_variance = squared_deviations.mean()
    population_std = np.sqrt(population_variance)
    sample_variance = observations.var(ddof=1)

    print("mean:", population_mean)
    print("deviations:", deviations)
    print("squared deviations:", squared_deviations)
    print("population variance:", population_variance)
    print("population standard deviation:", population_std)
    print("sample variance estimate:", sample_variance)

    assert population_mean == 4
    assert np.isclose(population_variance, 8 / 3)
    assert sample_variance == 4
    """),

    md(r"""
    `ddof=1` changes the divisor from $n$ to $n-1$. Do not choose it by habit: first
    state whether the values are the entire population of interest or a sample used
    to estimate a larger process.
    """),

    md(r"""
    ## 9 · Real-world case: alerts and the base-rate trap

    For 10,000 periods with a 1% incident rate:

    - 100 periods contain incidents;
    - 90% detection produces 90 true alerts;
    - 9,900 normal periods with 5% false alerts produce 495 false alerts;
    - total alerts are 585.

    $$
    P(\text{incident}\mid\text{alert})=\frac{90}{585}\approx0.154
    $$

    **Symbols:** the condition “alert” creates the denominator 585. Only 90 of those
    alerts correspond to incidents.

    This does not make the detector useless. It means alert handling should account
    for base rate, false-alert cost, and additional evidence.
    """),

    md(r"""
    ## 10 · Common mistakes and recovery habits

    - Reversing $P(A\mid B)$ and $P(B\mid A)$: say the filtered denominator first.
    - Using all observations as a conditional denominator: filter before dividing.
    - Adding overlapping probabilities without subtracting their intersection.
    - Calling mutually exclusive events independent.
    - Assuming rows are independent because they are stored separately.
    - Treating empirical frequency as exact probability from a small sample.
    - Treating expected value as the next guaranteed outcome.
    - Comparing variance values while ignoring squared units.
    - Switching between population and sample variance without naming the goal.
    - Treating density height as probability instead of accumulating area.
    """),

    md(r"""
    ## 11 · Compare the related concepts

    | Concept | Main question | Denominator or weighting | Common confusion |
    | --- | --- | --- | --- |
    | Joint probability | How often do both occur? | All cases | Confused with conditional |
    | Marginal probability | How often does one property occur overall? | All cases | Ignores other variable |
    | Conditional probability | How often inside a filtered group? | Condition group | Direction reversed |
    | Independence | Does one event change the other's probability? | Product rule | Confused with exclusivity |
    | Expected value | What is the probability-weighted average? | Probabilities | Treated as guaranteed result |
    | Variance | How spread are squared deviations? | Population or sample divisor | Units are squared |
    | Standard deviation | What is spread in original units? | Square root of variance | Treated as a universal range |
    """),

    md(r"""
    ## 12 · Readiness check

    Without notes:

    1. Define experiment, outcome, sample space, and event for one coin flip.
    2. Calculate a complement.
    3. Calculate a union with an overlap.
    4. Obtain joint, marginal, and both conditional directions from a count table.
    5. Explain why mutually exclusive non-impossible events are not independent.
    6. Solve one Bayes/base-rate question using counts.
    7. Build a two-value PMF and verify it sums to one.
    8. Calculate expected value, variance, and standard deviation manually.
    9. Explain population versus sample variance.

    **Readiness threshold:** 8/9, including Questions 4, 5, 6, and 8.
    """),

    md(r"""
    ## 13 · Mini-project: support-ticket uncertainty report

    **Dataset columns:** `ticket_id`, `channel` (`chat` or `email`), `escalated`
    (`True` or `False`), and `resolution_minutes`.

    **Workflow:** create at least 20 small records; build a joint channel/escalation
    count table; calculate marginal and both conditional directions; check whether
    channel and escalation look independent; calculate mean, population variance,
    and standard deviation of resolution time; simulate a declared escalation rate
    with a seed; compare declared and observed rates; plot one PMF or frequency chart.

    **Expected output:** labelled counts, probabilities with denominators, statistical
    summaries with units, a reproducible simulation, and one limitation statement.

    **Evaluation:** correct event definitions, denominators, independence reasoning,
    manual check, reproducibility, labelled visualization, and cautious conclusion.
    """),

    md(r"""
    ## 14 · Practice, self-check, and solutions

    ### Worked example

    In 50 requests, 20 are mobile and 8 of the mobile requests fail. Then
    $P(\text{failure}\mid\text{mobile})=8/20=0.40$.

    ### Guided practice

    1. For a fair die, define event $A$ as values above 4 and calculate $P(A)$ and
       $P(A^c)$.
    2. If $P(A)=0.5$, $P(B)=0.4$, and $P(A\cap B)=0.2$, calculate $P(A\cup B)$.
    3. From the 100-user table, calculate purchase given desktop.
    4. Explain whether “even” and “above 3” are mutually exclusive on a die.
    5. A game pays $10 with probability 0.25 and zero otherwise. Find expectation.

    ### Independent practice

    6. Construct a two-by-two count table and calculate joint, marginal, and both
       conditional directions.
    7. Give one independent-event example and one dependent-event example, including
       why each label is justified.
    8. Calculate mean, population variance, and standard deviation for $(1,3,5)$.
    9. Explain why 90% sensitivity does not mean 90% probability after a positive.

    ### Challenge

    Complete the support-ticket mini-project from Section 13 with four assertions
    and a written warning about causation or sampling limitations.

    ### Self-check

    Name every denominator before dividing. Confirm PMF probabilities sum to one.
    Keep original units for standard deviation and avoid promising one future outcome.
    """),

    md(r"""
    ### Solution and scoring rubric

    1. $A=\{5,6\}$, so $P(A)=2/6=1/3$ and $P(A^c)=2/3$.
    2. $0.5+0.4-0.2=0.7$.
    3. Desktop purchases are 18 among 60 desktop users: $18/60=0.30$.
    4. They overlap at 4 and 6, so they are not mutually exclusive.
    5. $10(0.25)+0(0.75)=2.50$.
    6. Answers depend on the table; all four calculations must show denominators.
    7. Independent examples must preserve probability after conditioning; dependent
       examples must change it.
    8. Mean 3; squared deviations $(4,0,4)$; variance $8/3$; standard deviation
       $\sqrt{8/3}\approx1.63$.
    9. Post-positive probability also depends on prevalence and false-positive rate.

    Award one point per exercise, with Questions 6–9 worth two points for reasoning.
    Award five challenge points for table, probabilities, statistics, simulation/
    visualization, and limitation. Maximum: 18.

    **Common mistakes:** wrong conditional denominator, double-counted union,
    independence/exclusivity confusion, expected-value guarantee, and variance-unit
    confusion.

    **Readiness threshold:** 14/18, including correct joint/marginal/conditional
    calculations, independence explanation, expected value, and variance.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Continue when you can move from a story to events, a count table, conditional
    probabilities, a base-rate update, a small distribution, expected value, and
    spread—without changing denominators silently.

    ### Teach it back

    Explain why $P(\text{alert}\mid\text{incident})$ differs from
    $P(\text{incident}\mid\text{alert})$. Then explain how expected value and
    standard deviation describe different properties of a random outcome.

    ### Memory aid

    **Probability starts by naming possible outcomes; conditional probability
    changes the denominator; statistics summarizes what was observed.**
    """),
]


build("00_prerequisites/05_probability_random_variables_and_statistics.ipynb", cells)
