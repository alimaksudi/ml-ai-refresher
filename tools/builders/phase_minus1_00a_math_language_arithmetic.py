"""Builder for Notebook 00A — Mathematical Language and Arithmetic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # 00A · Mathematical Language and Arithmetic
    ### Prerequisite Phase — Start Here With No Mathematical Background

    This notebook assumes only that you can read ordinary numbers. It teaches how
    mathematical writing communicates instructions. You do not need to memorize
    the symbols on the first pass. Read them, say them aloud, calculate a small
    example, and use the self-checks.

    **Estimated time:** 2–3 hours, including exercises.
    **Prerequisites:** none.
    """),

    md(r"""
    ## 1 · Learning Objectives

    By the end, you will be able to:

    - read positive and negative numbers on a number line;
    - calculate with `+`, `−`, `×`, and `÷` in the correct order;
    - explain fractions, percentages, powers, roots, and absolute value;
    - treat a letter as a number whose value is unknown or allowed to change;
    - read subscripts, superscripts, parentheses, and summation notation;
    - translate a short formula into plain-English instructions;
    - check a calculation with Python without treating Python as proof.
    """),

    md(r"""
    ## 2 · Historical Motivation

    Mathematical notation is a compression system. The sentence “add the first
    through the fifth measurements and divide by five” is precise but long. A
    formula can express the same instruction compactly. The difficulty is that
    compact writing hides steps from a new reader.

    In this curriculum, a formula is never magic. It is a short program written
    for a human: symbols are inputs, operators are instructions, and the result
    has a meaning.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    Think of a number line as a road:

    - zero is the starting point;
    - positive numbers are positions to the right;
    - negative numbers are positions to the left;
    - adding moves right;
    - subtracting moves left;
    - distance from zero is always non-negative.

    Parentheses are instruction groups. In `2 × (3 + 4)`, finish the grouped
    instruction `3 + 4` before multiplying by 2.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axhline(0, color="black", linewidth=1)
    for value in range(-5, 6):
        ax.plot([value, value], [-0.08, 0.08], color="black")
        ax.text(value, -0.22, str(value), ha="center")
    ax.annotate("add 4", xy=(3, 0.04), xytext=(-1, 0.04),
                arrowprops={"arrowstyle": "->", "color": "tab:blue"},
                color="tab:blue", ha="center", va="bottom")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-0.35, 0.35)
    ax.axis("off")
    ax.set_title("Start at −1 and add 4: move four places right to 3")
    plt.show()
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 The four arithmetic operations

    $$8 + 3 = 11,\qquad 8 - 3 = 5,\qquad 8\times3 = 24,\qquad 8\div2 = 4.$$

    **Read aloud:** “eight plus three equals eleven,” and similarly for the other
    operations.

    **Symbols:** `+` means add, `−` means subtract, `×` means multiply, `÷` means
    divide, and `=` means the expressions on both sides have the same value.

    **Meaning:** addition combines amounts; subtraction finds a remaining amount
    or difference; multiplication repeats equal groups; division splits into equal
    groups. Division by zero is undefined because no number multiplied by zero can
    recover a non-zero amount.

    ### 4.2 Order of operations

    $$2 + 3\times4 = 2 + 12 = 14,\qquad (2+3)\times4=5\times4=20.$$

    **Read aloud:** “two plus three times four”; multiplication happens before
    addition unless parentheses change the grouping.

    **Symbols:** parentheses `( )` group instructions. Multiplication and division
    happen before addition and subtraction. Operations at the same level are read
    left to right.

    **Small example:** `20 ÷ 5 × 2` is `(20 ÷ 5) × 2 = 8`, not `20 ÷ 10`.

    ### 4.3 Fractions, ratios, and percentages

    $$\frac{3}{4}=0.75=75\%.$$

    **Read aloud:** “three divided by four equals zero point seven five equals
    seventy-five percent.”

    **Symbols:** the top number `3` is the numerator, the bottom number `4` is the
    denominator, the fraction bar means division, and `%` means “out of one
    hundred.”

    **Small example:** if 3 of 4 tests pass, the pass rate is `3 ÷ 4 = 0.75`, or
    `75%`. A percentage becomes a decimal by dividing by 100.

    ### 4.4 Powers and roots

    $$3^2=3\times3=9,\qquad \sqrt{9}=3.$$

    **Read aloud:** “three squared equals nine; the square root of nine equals
    three.”

    **Symbols:** in $3^2$, `3` is the base and superscript `2` is the exponent. It
    means use two factors of 3. The radical sign $\sqrt{\phantom{x}}$ asks which
    non-negative number produces the value when multiplied by itself.

    **Use:** squared differences make large errors count more; square roots undo a
    square and return to the original unit.

    ### 4.5 Variables and equations

    $$x+3=8\quad\Longrightarrow\quad x=5.$$

    **Read aloud:** “x plus three equals eight, therefore x equals five.”

    **Symbols:** $x$ is a variable—a named number. The arrow means “this implies.”
    Solving means finding a value that makes the equation true.

    **Small example:** subtract 3 from both sides: `x + 3 − 3 = 8 − 3`, leaving
    `x = 5`. Doing the same operation to both sides preserves equality.

    ### 4.6 Subscripts and summation

    $$x_1+x_2+x_3=\sum_{i=1}^{3}x_i.$$

    **Read aloud:** “x one plus x two plus x three equals the sum of x sub i for i
    from one through three.”

    **Symbols:** $x_i$ means item number $i$; the subscript identifies an item and
    is not multiplication. $\sum$ means repeatedly add. The lower text $i=1$ says
    where counting starts, and upper `3` says where it ends.

    **Small example:** if $(x_1,x_2,x_3)=(4,7,2)$, the sum is `4 + 7 + 2 = 13`.

    ### 4.7 Absolute value

    $$|{-5}|=5,\qquad |3|=3.$$

    **Read aloud:** “the absolute value of negative five is five.”

    **Symbols:** vertical bars around one number mean its distance from zero.
    Distance cannot be negative.

    **Use and limit:** absolute value measures error size without cancellation.
    Do not confuse $|x|$ with probability bars or set-size bars; later notebooks
    will label those meanings when introduced.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    Work this example without code first.

    A service receives latencies `100`, `120`, `80`, and `140` milliseconds.

    1. Sum: `100 + 120 + 80 + 140 = 440`.
    2. Count: there are `4` measurements.
    3. Average: `440 ÷ 4 = 110` milliseconds.
    4. Difference between the largest and smallest: `140 − 80 = 60` milliseconds.

    The average is a representative center. The range of 60 ms tells us that the
    measurements are not all close to that center.
    """),

    code(r"""
    latencies = [100, 120, 80, 140]

    total = 0
    for value in latencies:
        total = total + value

    count = len(latencies)
    average = total / count
    value_range = max(latencies) - min(latencies)

    print("total:", total)
    print("count:", count)
    print("average:", average, "ms")
    print("range:", value_range, "ms")

    assert total == 440
    assert average == 110
    """),

    md(r"""
    ## 6 · Visualization

    A chart is another representation of the same numbers. Height represents
    magnitude. Always read the axis label and unit before interpreting the shape.
    """),

    code(r"""
    labels = ["request 1", "request 2", "request 3", "request 4"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, latencies, color="tab:blue")
    ax.axhline(average, color="tab:red", linestyle="--", label="average = 110 ms")
    ax.set_ylabel("latency (milliseconds)")
    ax.set_title("The same four values shown as bars")
    ax.legend()
    plt.show()
    """),

    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Reading $x_2$ as $x\times2$. A subscript is normally a label or position.
    - Treating `20%` as `20`. In calculations it is usually `0.20`.
    - Ignoring parentheses.
    - Dividing by zero.
    - Thinking $-3^2$ and $(-3)^2$ are identical. The first is `−(3²) = −9`; the
      second is `9`.
    - Copying a calculator result without checking units or scale.
    """),

    md(r"""
    ## 8 · Python and NumPy Implementation

    Python uses `**` for a power, `/` for ordinary division, and `abs()` for
    absolute value. NumPy applies the same operation to many numbers.
    """),

    code(r"""
    values = np.array([4.0, 7.0, 2.0])

    print("sum:", np.sum(values))
    print("mean:", np.mean(values))
    print("squares:", values ** 2)
    print("square roots:", np.sqrt(values))
    print("absolute values:", np.abs(np.array([-5, 3])))

    assert np.sum(values) == 13
    """),

    md(r"""
    ## 9 · Realistic Case Study — Error Rate

    A pipeline processed 2,000 records and 30 failed.

    - Error fraction: `30 ÷ 2,000 = 0.015`.
    - Error percentage: `0.015 × 100 = 1.5%`.
    - Successful records: `2,000 − 30 = 1,970`.

    Reporting “30 errors” lacks context. Reporting “1.5% of 2,000 records” gives
    both the relative rate and the sample size.
    """),

    md(r"""
    ## 10 · Learning Considerations

    Use paper for the first calculation and Python for the check. If the answers
    disagree, inspect one operation at a time. The goal is not mental arithmetic
    speed; it is knowing what operation a symbol requests and whether the result
    is plausible.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    | Representation | Strength | Limitation |
    |---|---|---|
    | Plain sentence | Friendly to read | Long and sometimes ambiguous |
    | Formula | Compact and precise | Dense until symbols are learned |
    | Code | Executable and testable | Can hide the underlying reasoning |
    | Chart | Shows patterns quickly | Can mislead through poor axes or scale |

    Strong technical communication moves between all four representations.
    """),

    md(r"""
    ## 12 · Readiness Check

    You are ready for Notebook 00B when you can answer these without guessing:

    1. What is the difference between a subscript and an exponent?
    2. What does a fraction bar instruct you to do?
    3. Why is `25%` represented as `0.25` in a calculation?
    4. Read $\sum_{i=1}^{3}x_i$ aloud.
    5. Explain why $|{-8}|=8$.
    """),

    md(r"""
    ## 13 · Teach-Back

    Explain these to another beginner in one sentence each:

    - variable;
    - equation;
    - exponent;
    - subscript;
    - summation;
    - percentage;
    - absolute value.

    If you need to use a technical word in the explanation, define that word too.
    """),

    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Estimated time:** 30–45 minutes.

    ### Guided practice

    1. Calculate `5 + 2 × 6`. Hint: multiplication happens first.
    2. Convert `3/5` to a decimal and percentage.
    3. If $(x_1,x_2,x_3)=(2,5,8)$, calculate $\sum_{i=1}^{3}x_i$.
    4. Solve $x-4=9$. Hint: add 4 to both sides.

    ### Independent practice

    5. Calculate $(-4)^2$, $-4^2$, and $\sqrt{16}$.
    6. A model makes 18 mistakes on 600 examples. Compute its error percentage.
    7. Latencies are `50, 70, 60, 100`. Compute their total, average, and range.

    <details>
    <summary><strong>Solutions and scoring</strong></summary>

    1. `17`. Award 1 point for multiplying before adding.
    2. `0.6 = 60%`. Award 1 point for division and 1 for the percentage.
    3. `2 + 5 + 8 = 15`. Award 1 point for expanding the summation.
    4. `x = 13`. Check: `13 − 4 = 9`.
    5. `16`, `−16`, and `4`. Parentheses determine whether the negative sign is
       included in the square.
    6. `18 ÷ 600 = 0.03 = 3%`.
    7. Total `280`, average `70`, range `50`.

    **Score:** 9–10 points means ready; 7–8 means review the missed subsection;
    below 7 means repeat the worked examples before Notebook 00B.
    </details>
    """),
]


build(
    "phase_minus1_onboarding/00a_math_language_and_arithmetic.ipynb",
    cells,
)
