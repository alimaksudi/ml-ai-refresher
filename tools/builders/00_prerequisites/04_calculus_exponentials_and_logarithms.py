"""Builder for PRE-04 — Calculus, Exponentials, and Logarithms."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # PRE-04 · Calculus, Exponentials, and Logarithms

    *Understand change before asking an optimizer to follow it*

    | Lesson detail | Value |
    | --- | --- |
    | Prerequisites | PRE-01, PRE-02, and PRE-03 |
    | Estimated study time | 5–7 hours across two sessions |
    | Main outcome | Interpret and calculate local change, accumulation, exponentials, and logarithms |
    | Next lesson | PRE-06 · Probability, Random Variables, and Statistics |

    Calculus is not a bag of mysterious symbols. It answers two practical questions:
    **How quickly is something changing now?** and **How much change has accumulated?**
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - calculate and interpret average rate of change with units;
    - explain a secant line, tangent line, and limit in plain language;
    - approximate a derivative numerically and calculate basic derivatives by hand;
    - use constant, power, sum, and chain rules without skipping their meaning;
    - interpret a partial derivative while holding other inputs fixed;
    - explain an integral as accumulated small contributions;
    - connect exponentials to repeated multiplicative change;
    - use logarithms as inverse operations and explain their domain;
    - verify small calculations with Python and NumPy;
    - identify when local slope or extrapolation can mislead.

    A formal gradient is an ordered collection of partial derivatives. Its vector
    geometry is intentionally deferred until FND-01 teaches vectors.
    """),

    md(r"""
    ## 2 · The problem we are trying to solve

    A model has a tuning parameter $w$. Its error is:

    $$
    L(w)=(w-3)^2+1
    $$

    **Symbols:** $w$ is the adjustable parameter, $L$ is loss or error, and the
    square makes loss rise as $w$ moves away from 3.

    Testing every possible decimal value is impossible. We need to know:

    - Is loss rising or falling near the current value?
    - How sensitive is loss to a small parameter change?
    - Which direction should an optimizer move?
    - How can many small rates accumulate into a total?

    Derivatives answer the local-change questions. Integrals answer accumulation.
    Exponentials and logarithms describe multiplicative growth and stable likelihood
    calculations used later in machine learning.
    """),

    md(r"""
    ## 3 · Build the picture before the formula

    Imagine hiking on a curved hill.

    - **Height** is the function value.
    - **Average slope** compares two separated locations.
    - **Local slope** describes the ground directly under your feet.

    <div style="display: flex; justify-content: center; gap: 16px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #4c78a8; border-radius: 10px; padding: 14px 18px; background: #eef5ff; color: #172b4d; text-align: center;"><strong>Two separated points</strong><br>average change<br>secant line</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 14px 18px; background: #fff4e8; color: #4a2b0b; text-align: center;"><strong>Move points closer</strong><br>smaller input step<br>limit process</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 14px 18px; background: #eef8ec; color: #173d17; text-align: center;"><strong>One location</strong><br>local change<br>tangent slope</div>
    </div>

    The hill analogy has a limit: a model can have thousands of adjustable inputs,
    not just one visible trail. We start with one input because the reasoning is the
    same and easier to inspect.
    """),

    md(r"""
    ## 4 · Mathematical foundations

    ### 4.1 Average rate of change

    Between inputs $a$ and $b$:

    $$
    \text{average rate}=\frac{f(b)-f(a)}{b-a}
    $$

    **Symbols:** $f$ is the rule, $a$ is the starting input, $b$ is the ending
    input, and the fraction is output change divided by input change.

    For $f(x)=x^2$ from $x=1$ to $x=3$:

    $$
    \frac{f(3)-f(1)}{3-1}=\frac{9-1}{2}=4
    $$

    Read it as “the output rises by four units per input unit on average across this
    interval.” Units are output units divided by input units.
    """),

    md(r"""
    ### 4.2 A limit is a controlled approach

    To estimate the slope of $f(x)=x^2$ at $x=3$, compare $3$ with $3+h$.

    | Step $h$ | Difference quotient | Estimate |
    | ---: | ---: | ---: |
    | $1$ | $((4)^2-3^2)/1$ | $7$ |
    | $0.1$ | $((3.1)^2-3^2)/0.1$ | $6.1$ |
    | $0.01$ | $((3.01)^2-3^2)/0.01$ | $6.01$ |
    | $0.001$ | $((3.001)^2-3^2)/0.001$ | $6.001$ |

    The estimates approach 6. We never substitute $h=0$ into the fraction because
    that would divide by zero. A **limit** asks what value the expression approaches
    as the step becomes arbitrarily small.
    """),

    md(r"""
    ### 4.3 The derivative is local rate of change

    $$
    f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
    $$

    **Symbols:** $f'(x)$, read “f prime of x,” is local slope; $h$ is an input step;
    $\lim_{h\to0}$ means observe the approached value as $h$ gets closer to zero.

    A positive derivative means the function rises locally as $x$ increases. A
    negative derivative means it falls. Zero means locally flat, but a flat point
    can be a minimum, maximum, or neither.

    The derivative is not the function value. At $x=3$, $f(x)=x^2$ has value 9 and
    derivative 6. Height and slope answer different questions.
    """),

    md(r"""
    ### 4.4 Basic derivative rules

    These rules summarize patterns visible from the limit definition.

    **Constant rule:**

    $$
    \frac{d}{dx}c=0
    $$

    **Power rule:**

    $$
    \frac{d}{dx}x^n=nx^{n-1}
    $$

    **Sum rule:** differentiate each added term.

    **Symbols:** $d/dx$ means “differentiate with respect to $x$,” $c$ is a fixed
    number, and $n$ is the exponent.

    For $f(x)=3x^2+2x+5$:

    $$
    f'(x)=6x+2
    $$

    At $x=4$, local slope is $6(4)+2=26$. The constant 5 disappears because it does
    not change when $x$ changes.
    """),

    md(r"""
    ### 4.5 The chain rule handles a rule inside a rule

    Suppose:

    $$
    q(x)=(2x+1)^2
    $$

    The inner rule is $u=2x+1$. The outer rule is $q=u^2$. Change passes through
    both layers:

    $$
    \frac{dq}{dx}=\frac{dq}{du}\frac{du}{dx}
    $$

    $$
    q'(x)=2(2x+1)\times2=4(2x+1)
    $$

    **Symbols:** $dq/du$ measures outer change per inner change; $du/dx$ measures
    inner change per input change. Multiplying connects the rates and cancels the
    conceptual “per $u$” unit.

    At $x=1$, $q'(1)=4(3)=12$. Forgetting the inner factor 2 is the common mistake.
    """),

    md(r"""
    ### 4.6 Partial derivatives: change one input at a time

    A function may have several inputs:

    $$
    f(x,y)=x^2+3y
    $$

    Hold $y$ fixed and change only $x$:

    $$
    \frac{\partial f}{\partial x}=2x
    $$

    Hold $x$ fixed and change only $y$:

    $$
    \frac{\partial f}{\partial y}=3
    $$

    **Symbols:** $\partial$ marks a partial derivative; the denominator names the
    input allowed to change. At $(x,y)=(2,5)$, sensitivities are 4 and 3.

    After FND-01 introduces vectors, FND-04 will collect all partial derivatives
    into the gradient and use it for optimization.
    """),

    md(r"""
    ### 4.7 Integrals accumulate small contributions

    If a rate changes over time, total amount is approximated by many thin
    rectangles:

    $$
    \text{total}\approx\sum_i \text{rate}_i\times\Delta t
    $$

    The integral is the limit of that accumulation:

    $$
    \int_a^b f(x)\,dx
    $$

    **Symbols:** $\int$ means accumulate, $a$ and $b$ are boundaries, and $dx$
    indicates small widths along the $x$ input.

    A constant speed of 5 metres per second for 3 seconds accumulates
    $5\times3=15$ metres. If speed varies, smaller time slices usually give a better
    approximation. PRE-06 uses area under density curves for probability.
    """),

    md(r"""
    ### 4.8 Exponentials model multiplicative change

    Repeated additive change follows a line. Repeated percentage change follows an
    exponential.

    $$
    y=a\,b^x
    $$

    **Symbols:** $a$ is the starting scale, $b$ is the growth factor per input unit,
    and $x$ counts how many growth steps occur.

    Starting with 100 and growing 10% per period:

    $$
    y=100(1.1)^x
    $$

    After two periods, $y=100(1.1)^2=121$. A base between 0 and 1 describes decay.
    The special base $e\approx2.718$ appears when growth is continuous and in many
    probability models.
    """),

    md(r"""
    ### 4.9 Logarithms reverse exponentials

    $$
    2^3=8\quad\Longleftrightarrow\quad\log_2(8)=3
    $$

    **Symbols:** $\log_b(y)$ asks which exponent on base $b$ produces $y$;
    $\Longleftrightarrow$ means both statements express the same fact.

    Important rules:

    $$
    \log(ab)=\log(a)+\log(b)
    $$

    $$
    \log(a^k)=k\log(a)
    $$

    A real logarithm requires a positive input. `log(0)` and the real log of a
    negative number are undefined. Turning products into sums is useful because
    multiplying many tiny probabilities can underflow toward zero; adding their
    logs is numerically safer.
    """),

    md(r"""
    ## 5 · Worked example: approximate, derive, and check

    **Worked example:** for $f(x)=x^3$ at $x=2$:

    1. Numerical estimate with $h=0.001$:

       $$
       \frac{(2.001)^3-2^3}{0.001}\approx12.006
       $$

    2. Power rule:

       $$
       f'(x)=3x^2
       $$

    3. Exact local slope at 2:

       $$
       f'(2)=3(2^2)=12
       $$

    **Symbols:** $h$ is the numerical step; the close agreement is evidence that
    the implementation and symbolic rule are consistent.
    """),

    code(r"""
    def cube(value):
        return value ** 3


    def forward_difference(function, input_value, step):
        if step == 0:
            raise ValueError("step must be non-zero")
        output_change = function(input_value + step) - function(input_value)
        return output_change / step


    for step in [1.0, 0.1, 0.01, 0.001]:
        estimate = forward_difference(cube, 2.0, step)
        print(f"step={step:<5} estimate={estimate:.6f}")

    exact_slope = 3 * 2.0 ** 2
    assert abs(forward_difference(cube, 2.0, 0.001) - exact_slope) < 0.01
    """),

    md(r"""
    A very large step measures a broad interval instead of a local slope. An
    extremely tiny step can suffer floating-point cancellation. Numerical
    differentiation requires a sensible scale and comparison against known cases.
    """),

    md(r"""
    ## 6 · Visualize secants, tangents, and accumulation

    The next plot uses PRE-03 plotting skills. The left panel shows secant slopes
    approaching a tangent. The right panel shows rectangles accumulating area.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    x_grid = np.linspace(0, 4, 300)
    y_grid = x_grid ** 2
    x0 = 2.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x_grid, y_grid, label="f(x)=x²")
    for step, color in [(1.0, "tab:orange"), (0.2, "tab:green")]:
        slope = forward_difference(lambda x: x ** 2, x0, step)
        secant = x0 ** 2 + slope * (x_grid - x0)
        axes[0].plot(x_grid, secant, color=color, label=f"h={step}, slope≈{slope:.1f}")
    axes[0].set_ylim(0, 16)
    axes[0].set_xlabel("input x")
    axes[0].set_ylabel("output f(x)")
    axes[0].set_title("Secant approaches tangent")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    rectangle_left = np.arange(0, 3, 0.5)
    rectangle_height = 2 * rectangle_left
    axes[1].bar(rectangle_left, rectangle_height, width=0.5, align="edge",
                alpha=0.5, edgecolor="black", label="left rectangles")
    axes[1].plot(x_grid[x_grid <= 3], 2 * x_grid[x_grid <= 3], color="tab:red", label="rate=2t")
    axes[1].set_xlabel("time t")
    axes[1].set_ylabel("rate")
    axes[1].set_title("Rectangles approximate accumulation")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 7 · When these tools help—and when they do not

    Use derivatives for local sensitivity, optimization direction, and changing
    rates. Use numerical differences when a symbolic derivative is inconvenient and
    function evaluations are available. Use integrals for accumulated rate, area,
    and continuous probability. Use logs for multiplicative scales and likelihoods.

    Do not assume a local derivative describes distant behavior. Do not declare a
    zero derivative to be a minimum without examining nearby behavior. Do not use a
    numerical derivative without checking step sensitivity. Do not take logs of
    non-positive inputs. Use direct finite sums when values are discrete and few.
    """),

    md(r"""
    ## 8 · NumPy verification

    NumPy can verify exponential/log inverse behavior and approximate an integral
    with a sum. The manual meaning comes first; the library scales it.
    """),

    code(r"""
    inputs = np.array([0.0, 1.0, 2.0])
    exponential_values = np.exp(inputs)
    recovered_inputs = np.log(exponential_values)

    time_left_edges = np.arange(0.0, 3.0, 0.01)
    widths = 0.01
    rates = 2 * time_left_edges
    accumulated_amount = np.sum(rates * widths)

    print("exp values:", exponential_values)
    print("log(exp(x)):", recovered_inputs)
    print("approximate integral of 2t from 0 to 3:", accumulated_amount)

    assert np.allclose(recovered_inputs, inputs)
    assert abs(accumulated_amount - 9.0) < 0.05
    """),

    md(r"""
    The exact integral is 9. The left-rectangle approximation is slightly low
    because the increasing rate is sampled at each interval's left edge. Smaller
    widths reduce this approximation error but require more calculations.
    """),

    md(r"""
    ## 9 · Real-world case: sensitivity of model loss

    Return to $L(w)=(w-3)^2+1$.

    $$
    L'(w)=2(w-3)
    $$

    **Symbols:** $L'(w)$ is loss sensitivity to a small increase in parameter $w$.

    - At $w=1$, slope is $-4$: moving right locally lowers loss.
    - At $w=5$, slope is $4$: moving left locally lowers loss.
    - At $w=3$, slope is $0$ and nearby loss is higher, so this is a minimum.

    FND-04 will turn this reasoning into repeated optimization updates after FND-01
    supplies the vector geometry needed for many parameters.
    """),

    md(r"""
    ## 10 · Common mistakes and recovery habits

    - Confusing function value with derivative: report both height and slope.
    - Substituting $h=0$: approach zero; do not divide by zero.
    - Using one large numerical step: compare several decreasing steps.
    - Applying the power rule to a nested function without the chain rule.
    - Forgetting what is held fixed in a partial derivative.
    - Treating every zero derivative as a minimum: inspect nearby values.
    - Dropping units from a rate: write output unit per input unit.
    - Treating a local line as a global forecast: mark the valid input range.
    - Using `log(0)`: validate positivity or use a justified numerical safeguard.
    """),

    md(r"""
    ## 11 · Compare the related concepts

    | Concept | Question answered | Strength | Limitation |
    | --- | --- | --- | --- |
    | Average rate | How did output change across an interval? | Simple and stable | Hides local variation |
    | Derivative | How is output changing here? | Local sensitivity | Does not describe distant behavior |
    | Partial derivative | How does one input affect output while others stay fixed? | Handles multiple inputs one at a time | Interactions still require care |
    | Integral | How much has a changing rate accumulated? | Connects local pieces to a total | Often needs numerical approximation |
    | Exponential | What happens under repeated proportional change? | Models growth and decay | Extrapolates rapidly |
    | Logarithm | Which exponent produced this value? | Compresses scale and turns products into sums | Requires positive real input |
    """),

    md(r"""
    ## 12 · Readiness check

    Without notes:

    1. Calculate average rate for $f(x)=x^2$ from 1 to 3.
    2. Explain why a limit does not substitute $h=0$ into the difference quotient.
    3. Differentiate $3x^2+2x+5$.
    4. Differentiate $(3x+1)^2$ and explain the inner factor.
    5. Explain what $\partial f/\partial x$ holds fixed.
    6. Approximate an accumulated total with rectangles.
    7. Convert $2^5=32$ into logarithm form.
    8. Explain why logs help with products of tiny positive values.

    **Readiness threshold:** 7/8, including Questions 3, 4, 6, and 8.
    """),

    md(r"""
    ## 13 · Mini-project: service scaling and local cost

    **Goal:** study $C(r)=0.02r^2+5r+100$, where $r$ is requests per second and
    $C$ is estimated hourly infrastructure cost.

    **Workflow:** evaluate costs for several request rates; calculate average change
    across two ranges; approximate local change at one rate with four step sizes;
    derive $C'(r)$; compare numerical and exact slopes; plot cost and a local tangent;
    explain why distant extrapolation may fail.

    **Expected output:** a labelled table, convergence evidence, one plot, and a
    plain-language interpretation with dollars-per-request-rate units.

    **Evaluation:** correct arithmetic, step-size comparison, derivative reasoning,
    units, labelled plot, and a stated model limitation.
    """),

    md(r"""
    ## 14 · Practice, self-check, and solutions

    ### Worked example

    For $f(x)=x^2+4x$, $f'(x)=2x+4$, so $f'(3)=10$.

    ### Guided practice

    1. Find average rate of $f(x)=x^2$ from 2 to 4.
    2. Differentiate $5x^3-2x+7$.
    3. Differentiate $(2x-1)^3$ using the chain rule.
    4. For $g(x,y)=x^2+xy$, find $\partial g/\partial x$ while holding $y$ fixed.
    5. Approximate area under constant rate 4 from time 0 to 6.

    ### Independent practice

    6. Explain why $f'(x)=0$ does not prove a minimum.
    7. Starting at 200 with 5% growth, calculate the value after three periods.
    8. Rewrite $10^3=1000$ as a logarithm.
    9. Use NumPy to verify `log(exp(x))` for three values and explain floating-point
       comparison.

    ### Challenge

    Complete the service-scaling mini-project from Section 13 with three assertions
    and a written recommendation about the useful input range.

    ### Self-check

    For every slope, state output units per input unit. For every symbolic derivative,
    verify one point numerically. For every log, confirm the input is positive.
    """),

    md(r"""
    ### Solution and scoring rubric

    1. $(16-4)/(4-2)=6$.
    2. $15x^2-2$.
    3. $3(2x-1)^2\times2=6(2x-1)^2$.
    4. $2x+y$ because $y$ is held constant.
    5. $4\times6=24$ rate-units times time-units.
    6. The point may be a maximum or another flat point; inspect nearby values.
    7. $200(1.05)^3=231.525$.
    8. $\log_{10}(1000)=3$.
    9. Use `np.allclose` because stored decimals are finite approximations.

    Award one point per exercise, with Questions 3, 4, and 9 worth two points for
    explanation. Award five challenge points for calculation, exact derivative,
    convergence evidence, plot, and limitation. Maximum: 17.

    **Common mistakes:** missing the inner chain factor, dropping units, confusing
    value with slope, declaring every flat point a minimum, and taking a non-positive
    logarithm.

    **Readiness threshold:** 13/17, including correct power rule, chain rule, partial
    derivative, accumulation, and log interpretation.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Continue to PRE-06 when you can move from average change to local change, verify
    a derivative numerically, use power and chain rules, explain one partial
    derivative, approximate accumulation, and reverse an exponential with a log.

    ### Teach it back

    Explain the journey from two-point slope to a derivative, then explain how an
    integral reverses the viewpoint by accumulating small contributions. Finish
    with one reason logarithms matter in machine learning.

    ### Memory aid

    **A derivative measures local change, an integral accumulates change, and a
    logarithm reveals the exponent behind multiplicative change.**
    """),
]


build("00_prerequisites/04_calculus_exponentials_and_logarithms.ipynb", cells)
