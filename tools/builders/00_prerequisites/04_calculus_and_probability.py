"""Builder for Lesson PRE-04 — Calculus and Probability Intuition."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # PRE-04 · Calculus and Probability Intuition
    ### Prerequisite Phase — Change and Uncertainty Without Hidden Steps

    Calculus describes change. Probability describes uncertainty. This notebook
    builds the intuition and notation needed by optimization and statistics.

    **Estimated time:** 3–4 hours.
    **Prerequisites:** PRE-01 and PRE-02.
    """),

    md(r"""
    ## 1 · Learning Objectives

    You will learn to:

    - interpret average and instantaneous rate of change;
    - read derivative, partial-derivative, and gradient notation;
    - understand an integral as accumulated small pieces;
    - explain exponentials and logarithms as inverse operations;
    - calculate probability from equally likely outcomes and observed frequencies;
    - distinguish joint and conditional probability;
    - compute a small expected value, mean, and variance;
    - identify what a probability statement does and does not claim.
    """),

    md(r"""
    ## 2 · Historical Motivation

    Many engineering questions ask one of two things:

    - How quickly is a quantity changing?
    - How uncertain is the next outcome?

    A model learns by changing parameters to reduce error, so optimization needs
    rates of change. A model predicts from incomplete evidence, so evaluation needs
    probability. These ideas are tools for reasoning, not collections of symbols.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    Imagine a hill. The height is the loss and horizontal position is a model
    parameter. Slope tells us whether moving right raises or lowers the loss.

    Imagine a bag containing seven blue tokens and three red tokens. Before drawing
    one token, the exact outcome is unknown, but the proportions quantify what we
    should expect over many similar draws.
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Average rate of change

    $$\text{average rate}=\frac{f(b)-f(a)}{b-a}.$$

    **Read aloud:** “the average rate equals the change in the output divided by
    the change in the input.”

    **Symbols:** $f$ is a function; $a$ is the starting input; $b$ is the ending
    input; $f(a)$ and $f(b)$ are their outputs; the fraction bar means division.

    **Small example:** if distance rises from 20 km to 80 km while time rises from
    1 hour to 3 hours, the average speed is `(80 − 20) ÷ (3 − 1) = 30 km/hour`.

    ### 4.2 Derivative as local slope

    $$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.$$

    **Read aloud:** “f prime of x is the limit, as h approaches zero, of the change
    in f divided by h.”

    **Symbols:** $f'(x)$ is the derivative or local slope at $x$; $h$ is a small
    input step; $\lim_{h\to0}$ asks what value the ratio approaches as the step is
    made smaller, without dividing by zero.

    **Small example:** for $f(x)=x^2$ at $x=3$, using $h=0.01$ gives
    `((3.01)² − 3²) ÷ 0.01 = 6.01`, close to the exact slope `6`.

    **Meaning:** a positive derivative means the function rises locally; a negative
    derivative means it falls; zero means locally flat. Zero slope does not always
    mean a minimum.

    ### 4.3 Partial derivatives and gradients

    $$\nabla f(x,y)=\begin{bmatrix}\frac{\partial f}{\partial x}\\[2pt]
    \frac{\partial f}{\partial y}\end{bmatrix}.$$

    **Read aloud:** “the gradient of f is a vector containing the partial derivative
    with respect to x and the partial derivative with respect to y.”

    **Symbols:** $\partial$ means change one input while temporarily holding the
    others fixed; $\nabla$ names the gradient; the bracketed result is a vector.

    **Small example:** if $f(x,y)=x^2+y^2$, then the slopes at $(3,4)$ are `6` in
    the x-direction and `8` in the y-direction, so the gradient is `(6, 8)`.

    **Use:** the gradient points toward the fastest local increase. Optimization
    moves in the opposite direction to reduce a loss.

    ### 4.4 Integral as accumulation

    $$\int_a^b f(x)\,dx.$$

    **Read aloud:** “the integral of f of x from a to b, with respect to x.”

    **Symbols:** $\int$ means accumulate many thin pieces; $a$ and $b$ are the
    start and end; $dx$ identifies the input being divided into small widths.

    **Small example:** a constant speed of 5 metres per second for 3 seconds gives
    area `5 × 3 = 15 metres` under the speed-time graph.

    **Meaning:** derivatives describe local change; integrals accumulate local
    amounts. This curriculum rarely asks you to integrate by hand, but probability
    uses integrals to add density over ranges.

    ### 4.5 Exponentials and logarithms

    $$2^3=8\quad\Longleftrightarrow\quad \log_2(8)=3.$$

    **Read aloud:** “two to the power three equals eight if and only if log base two
    of eight equals three.”

    **Symbols:** $\Longleftrightarrow$ means both statements express the same fact;
    the logarithm asks which exponent produces the value.

    **Small example:** $10^2=100$, so $\log_{10}(100)=2$. The natural logarithm
    $\ln$ uses the special base $e\approx2.718$ and appears in likelihoods.

    **Use and limit:** logarithms turn products into sums, which improves numerical
    stability. A real logarithm requires a positive input.

    ### 4.6 Probability from counts

    $$P(A)=\frac{\text{number of outcomes in }A}{\text{number of possible outcomes}}.$$

    **Read aloud:** “the probability of event A equals favourable outcomes divided
    by possible outcomes,” when the outcomes are equally likely.

    **Symbols:** $P$ means probability; $A$ names an event, which is a set of
    outcomes; probability ranges from `0` (impossible) to `1` (certain).

    **Small example:** a fair six-sided die has three even outcomes, so
    $P(\text{even})=3/6=0.5=50\%$.

    ### 4.7 Joint and conditional probability

    $$P(A\mid B)=\frac{P(A\cap B)}{P(B)}.$$

    **Read aloud:** “the probability of A given B equals the probability of A and B
    divided by the probability of B.”

    **Symbols:** the vertical bar means “given that”; $\cap$ means both events occur;
    $P(B)$ must be greater than zero.

    **Small example:** among 20 customers, 8 are premium and 6 of those premium
    customers renewed. Given that a customer is premium, renewal probability is
    `6 ÷ 8 = 0.75`. The denominator is 8, not all 20, because the condition changes
    the group under discussion.

    ### 4.8 Expected value

    $$\mathbb E[X]=\sum_x x\,P(X=x).$$

    **Read aloud:** “the expected value of X is the sum over each possible value x,
    multiplied by the probability that X equals x.”

    **Symbols:** $X$ is a random variable; $x$ is one possible numerical value;
    $\mathbb E$ means long-run probability-weighted average; $\sum$ means add.

    **Small example:** a game pays `$10` with probability `0.2` and `$0` otherwise.
    Expected payout is `10 × 0.2 + 0 × 0.8 = $2`. This does not promise that any
    single play pays `$2`.

    ### 4.9 Mean and variance

    $$\mu=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad
    \sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2.$$

    **Read aloud:** “mu is the average of n values; sigma squared is the average
    squared distance from that mean.”

    **Symbols:** $n$ is the number of values; $x_i$ is value number $i$; $\mu$ is
    the population mean; $\sigma^2$ is variance; squaring prevents positive and
    negative deviations from cancelling.

    **Small example:** for `(2, 4, 6)`, mean is `4`; deviations are `−2, 0, 2`;
    squared deviations are `4, 0, 4`; variance is `8/3 ≈ 2.67`.

    **Limit:** sample variance often divides by $n-1$ when estimating a population
    variance from a sample. Lesson FND-02 explains that distinction.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    We approximate the derivative of $f(x)=x^2$ by using progressively smaller
    steps. This is called a finite difference.
    """),

    code(r"""
    def square(x):
        return x * x

    def approximate_derivative(function, x, step):
        change_in_output = function(x + step) - function(x)
        return change_in_output / step

    for step in [1.0, 0.1, 0.01, 0.001]:
        estimate = approximate_derivative(square, x=3.0, step=step)
        print(f"step={step:<5} derivative estimate={estimate:.3f}")

    assert abs(approximate_derivative(square, 3.0, 0.001) - 6.0) < 0.01
    """),

    md(r"""
    ## 6 · Visualization

    A secant line uses two separated points and shows average slope. As the second
    point moves closer, the line approaches the tangent and its slope approaches
    the derivative.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    x_grid = np.linspace(0, 5, 300)
    y_grid = x_grid ** 2
    x0 = 3.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x_grid, y_grid, label="f(x) = x²")
    for step, color in [(1.0, "tab:orange"), (0.2, "tab:green")]:
        slope = approximate_derivative(square, x0, step)
        line = square(x0) + slope * (x_grid - x0)
        axes[0].plot(x_grid, line, color=color, label=f"step={step}, slope≈{slope:.1f}")
    axes[0].set_ylim(0, 25)
    axes[0].set_title("Average slope approaches local slope")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    outcomes = ["blue", "red"]
    probabilities = [0.7, 0.3]
    axes[1].bar(outcomes, probabilities, color=["tab:blue", "tab:red"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("probability")
    axes[1].set_title("Probabilities must be between 0 and 1")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Treating a derivative as the function value. It is a rate of change.
    - Believing every zero derivative is a minimum; it may be a maximum or flat
      saddle point.
    - Substituting exactly `h = 0` in the derivative ratio and dividing by zero.
    - Reading $P(A\mid B)$ as $P(B\mid A)$. The condition changes the denominator.
    - Interpreting expected value as a guaranteed individual outcome.
    - Treating a probability estimate from a small sample as exact truth.
    - Taking a logarithm of zero or a negative number in ordinary real-number math.
    """),

    md(r"""
    ## 8 · NumPy Implementation

    NumPy can compute descriptive statistics and simulate repeated random trials.
    Simulation illustrates probability but does not replace a derivation.
    """),

    code(r"""
    rng = np.random.default_rng(42)
    draws = rng.choice(["blue", "red"], size=10_000, p=[0.7, 0.3])
    observed_blue_rate = np.mean(draws == "blue")

    values = np.array([2.0, 4.0, 6.0])
    print("observed blue rate:", round(observed_blue_rate, 3))
    print("mean:", np.mean(values))
    print("population variance:", np.var(values))

    assert abs(observed_blue_rate - 0.7) < 0.03
    assert np.mean(values) == 4.0
    """),

    md(r"""
    ## 9 · Realistic Case Study — Expected Incident Cost

    A service incident has a `2%` monthly probability and would cost `$50,000`.
    Ignoring other outcomes, expected monthly incident cost is
    `0.02 × $50,000 = $1,000`.

    This does not mean the company pays `$1,000` each month. Most months cost zero;
    a rare month costs much more. Expected value is useful for long-run budgeting,
    while risk planning must also consider worst cases and variability.
    """),

    md(r"""
    ## 10 · Learning Considerations

    Calculus fluency develops through pictures and small calculations before formal
    proofs. Probability statements must always name the population or condition.
    When stuck, ask: “what changes, what stays fixed, and what group forms the
    denominator?”
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    Exact symbolic derivatives provide insight, while numerical derivatives work
    for many functions but introduce step-size error. Analytical probabilities are
    exact under their assumptions, while simulation handles complex processes but
    has sampling noise. Later notebooks use both and compare them.
    """),

    md(r"""
    ## 12 · Readiness Check

    You are ready for Lesson PRE-05 when you can:

    1. describe a derivative as a local rate of change;
    2. explain each symbol in a two-variable gradient;
    3. explain why an integral represents accumulation;
    4. calculate a conditional probability from counts;
    5. distinguish expected value from a guaranteed result;
    6. calculate mean and population variance for three small values.
    """),

    md(r"""
    ## 13 · Teach-Back

    Explain the hill analogy for a derivative and gradient. Then explain probability
    using a bag of coloured tokens. Finish by describing why repeated observations
    may approach a probability even though individual outcomes remain uncertain.
    """),

    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Estimated time:** 45–60 minutes.

    ### Worked example

    If revenue rises from `$200` to `$260` while customers rise from `10` to `13`,
    average revenue change per added customer is `(260 − 200)/(13 − 10) = $20`.

    ### Guided practice

    1. Approximate the derivative of $f(x)=x^2$ at $x=2$ using $h=0.1$.
    2. A bag has 5 green and 15 yellow tokens. Find $P(\text{green})$.
    3. Among 50 users, 20 use mobile and 12 mobile users purchase. Find
       $P(\text{purchase}\mid\text{mobile})$.
    4. A game pays `$5` with probability `0.4` and loses `$1` otherwise. Find its
       expected value.

    ### Independent practice

    5. For values `(1, 3, 5)`, compute mean and population variance.
    6. Explain why a disease test's $P(\text{positive}\mid\text{disease})$ is not
       the same as $P(\text{disease}\mid\text{positive})$.
    7. State what a negative derivative tells you locally.

    <details>
    <summary><strong>Solutions and scoring</strong></summary>

    1. `((2.1)² − 2²)/0.1 = (4.41 − 4)/0.1 = 4.1`; the exact slope is `4`.
    2. `5/20 = 0.25 = 25%`.
    3. The condition restricts the denominator to 20 mobile users: `12/20 = 60%`.
    4. `5 × 0.4 + (−1) × 0.6 = 1.4`, so expected value is `$1.40` per play.
    5. Mean `3`; deviations `−2, 0, 2`; variance `(4+0+4)/3 = 8/3 ≈ 2.67`.
    6. The first asks how often the test detects known disease; the second asks how
       likely disease is after a positive and also depends on disease prevalence.
    7. A small move to the right decreases the function, while a small move left
       increases it, assuming the local slope remains representative.

    Award one point for each answer and one additional point for showing the
    denominator in Questions 2 and 3. A score of 7/9 is sufficient to continue.
    </details>
    """),
]


build("00_prerequisites/04_calculus_and_probability.ipynb", cells)
