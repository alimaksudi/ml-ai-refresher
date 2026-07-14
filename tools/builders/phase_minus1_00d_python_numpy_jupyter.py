"""Builder for Notebook 00D — Python, NumPy, and Jupyter Foundations."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # 00D · Python, NumPy, and Jupyter Foundations
    ### Prerequisite Phase — Reading and Testing the Code in This Curriculum

    This notebook teaches the small Python and NumPy subset used throughout the
    curriculum. It emphasizes reading code, checking shapes, and verifying results.

    **Estimated time:** 2–3 hours.
    **Prerequisites:** Notebooks 00A–00B; no prior Python required. Notebook 00C
    may be taken before or after this module, but the canonical beginner route puts
    Python here so later mathematics can be explored computationally.
    """),

    md(r"""
    ## 1 · Learning Objectives

    You will learn to:

    - run notebook cells in order and recognize cell state;
    - create variables, lists, functions, loops, and conditions;
    - read common Python errors without treating them as failure;
    - create NumPy arrays and inspect their shape and data type;
    - index, slice, reshape, and combine arrays;
    - distinguish elementwise multiplication from matrix multiplication;
    - create reproducible random values;
    - use assertions as executable self-checks.
    """),

    md(r"""
    ## 2 · Historical Motivation

    Python is widely used in data science because readable code can connect data,
    numerical libraries, visualization, and production systems. NumPy moves repeated
    numeric operations from slow Python loops into optimized array operations.

    A notebook mixes explanation, executable code, and output. That makes it useful
    for learning, but execution order matters: a later cell may depend on a variable
    created earlier.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    A variable is a label attached to a value. A list stores an ordered collection.
    A function gives a reusable instruction a name. A NumPy array adds a rectangular
    shape and numeric operations to a collection.

    ```text
    source code -> Python evaluates instructions -> variables change -> output appears
    ```

    Restarting the kernel clears variables. Running all cells from the top is the
    strongest check that the notebook does not depend on hidden state.
    """),

    md(r"""
    ## 4 · Mathematical and Programming Foundations

    ### 4.1 Assignment is not mathematical equality

    ```python
    count = 3
    count = count + 1
    ```

    The first line attaches `count` to `3`. The second line reads the old value,
    adds one, and stores `4` under the same name. In ordinary algebra, $x=x+1$
    has no solution; in Python, `=` means assignment.

    **Symbols:** in the mathematical statement, $x$ is a number and `=` means equal
    value. In Python code, the name on the left receives the computed value on the
    right.

    ### 4.2 Types

    - `int`: whole number such as `3`;
    - `float`: decimal approximation such as `3.5`;
    - `str`: text such as `"model"`;
    - `bool`: `True` or `False`;
    - `list`: ordered Python collection;
    - `ndarray`: NumPy's numeric array.

    ### 4.3 Function inputs and outputs

    ```python
    def square(value):
        return value * value
    ```

    `def` creates a function, `value` is a parameter, indentation marks the function
    body, and `return` sends the result to the caller.

    ### 4.4 Array shapes

    $$X\in\mathbb R^{n\times d}.$$

    **Read aloud:** “X is a real-valued matrix with n rows and d columns.”

    **Symbols:** $X$ usually stores data; $\in$ means “belongs to”; $\mathbb R$
    means real numbers; $n$ is the number of examples; $d$ is the number of
    features; $n\times d$ is the shape.

    **Small example:** 100 customers with 4 measured features produce shape
    `100 × 4`, represented by `X.shape == (100, 4)`.

    ### 4.5 Elementwise and matrix multiplication

    $$A\odot B\quad\text{versus}\quad AB.$$

    **Read aloud:** “A elementwise-multiplied by B, versus A matrix-multiplied by B.”

    **Symbols:** $\odot$ pairs matching positions; adjacent matrix names represent
    row-by-column multiplication. In NumPy, `A * B` is elementwise and `A @ B` is
    matrix multiplication.

    **Small example:** `[2, 3] * [4, 5]` elementwise gives `[8, 15]`; the dot/matrix
    result `[2, 3] @ [4, 5]` gives `8 + 15 = 23`.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    First compute an average with basic Python. The loop exposes every operation.
    """),

    code(r"""
    values = [10, 20, 30, 40]
    total = 0

    for value in values:
        total = total + value

    average = total / len(values)
    print("total:", total)
    print("average:", average)

    assert total == 100
    assert average == 25
    """),

    md(r"""
    Conditions select which code runs. The `%` operator returns a division remainder,
    so a number is even when dividing by two leaves remainder zero.
    """),

    code(r"""
    def describe_number(value):
        if value < 0:
            sign = "negative"
        elif value == 0:
            sign = "zero"
        else:
            sign = "positive"

        parity = "even" if value % 2 == 0 else "odd"
        return sign, parity

    print("−3:", describe_number(-3))
    print("4:", describe_number(4))
    assert describe_number(4) == ("positive", "even")
    """),

    md(r"""
    ## 6 · Visualization

    Plotting code has three conceptual steps: prepare values, choose visual marks,
    and label the axes so the marks have meaning.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 6)
    y = x ** 2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("input x")
    ax.set_ylabel("output x²")
    ax.set_title("A function evaluated at six inputs")
    ax.grid(alpha=0.3)
    plt.show()
    """),

    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Running cells out of order and relying on a stale variable.
    - Confusing `=` assignment with `==` comparison.
    - Forgetting that Python indexes from zero.
    - Using `*` when matrix multiplication requires `@`.
    - Ignoring array shape and accidental broadcasting.
    - Reusing a variable name for a different meaning.
    - Catching every exception and hiding a real bug.
    - Setting no random seed and then being unable to reproduce a result.
    """),

    md(r"""
    ## 8 · NumPy Implementation

    The following cell demonstrates construction, indexing, slicing, reshaping,
    aggregation, and matrix multiplication.
    """),

    code(r"""
    vector = np.array([10, 20, 30, 40])
    matrix = np.array([[1, 2], [3, 4], [5, 6]])

    print("first value:", vector[0])
    print("middle slice:", vector[1:3])
    print("matrix shape:", matrix.shape)
    print("column means:", matrix.mean(axis=0))
    print("first column:", matrix[:, 0])

    weights = np.array([0.5, 2.0])
    scores = matrix @ weights
    print("row scores:", scores)

    assert matrix.shape == (3, 2)
    assert np.allclose(scores, [4.5, 9.5, 14.5])
    """),

    md(r"""
    Randomness should be explicit and reproducible. Creating two generators with the
    same seed produces the same sequence, which makes debugging and tests repeatable.
    """),

    code(r"""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    sample_a = rng_a.normal(size=5)
    sample_b = rng_b.normal(size=5)

    print(sample_a)
    assert np.allclose(sample_a, sample_b)
    """),

    md(r"""
    ## 9 · Realistic Case Study — Feature Matrix

    Suppose each row represents a customer and columns represent age, monthly spend,
    and number of support contacts. A `(500, 3)` array means 500 examples and three
    features. A weight vector with shape `(3,)` can produce one score per customer:
    `(500, 3) @ (3,) -> (500,)`.

    Writing this shape calculation before running code makes the intended result
    explicit.
    """),

    md(r"""
    ## 10 · Learning Considerations

    Read code one line at a time and state what changes. Before executing an array
    operation, predict the output shape and one value. Use `assert` to turn that
    prediction into a check. Errors are evidence: read the final line first, identify
    the error type, and inspect the referenced line.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    Python loops are explicit and easy to inspect, while NumPy operations are shorter
    and faster for large arrays. Notebooks encourage exploration, while scripts and
    tests provide reproducibility. This curriculum uses both: notebooks teach the
    idea; builder scripts preserve the source.
    """),

    md(r"""
    ## 12 · Readiness Check

    You are ready for Phase 0 when you can:

    1. restart a kernel and run all cells;
    2. explain assignment, comparison, function, loop, and condition;
    3. predict simple array shapes;
    4. distinguish `*` and `@`;
    5. index a row and column;
    6. use a seed and an assertion;
    7. read a traceback from the bottom upward.
    """),

    md(r"""
    ## 13 · Teach-Back

    Explain how a Python list differs from a NumPy array. Then explain why shape is
    part of an array's meaning and why a notebook should be tested by restarting and
    running top-to-bottom.
    """),

    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Estimated time:** 45–60 minutes.

    ### Guided practice

    1. Write a function `double(value)` that returns twice its input.
    2. Use a loop to sum `[3, 5, 7]` without calling `sum`.
    3. Create a NumPy array with shape `(2, 3)` and print its second column.
    4. Predict the result shape of `(4, 3) @ (3, 2)` before running it.

    ### Independent practice

    5. Write `standardize(values)` that subtracts the mean and divides by the
       standard deviation using NumPy.
    6. Create a seeded generator, draw 100 values, and assert that the shape is
       `(100,)`.
    7. Explain why this fails: `np.ones((2, 3)) @ np.ones((4, 2))`.

    <details>
    <summary><strong>Solutions and scoring</strong></summary>

    ```python
    def double(value):
        return 2 * value

    total = 0
    for value in [3, 5, 7]:
        total += value
    assert total == 15

    array = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(array[:, 1], [2, 5])

    # (4, 3) @ (3, 2) -> (4, 2)
    def standardize(values):
        values = np.asarray(values, dtype=float)
        return (values - values.mean()) / values.std()

    rng = np.random.default_rng(7)
    sample = rng.normal(size=100)
    assert sample.shape == (100,)
    ```

    Question 7 fails because the inner dimensions `3` and `4` do not match. Award
    one point per task, plus one point for explaining shapes in Questions 4 and 7.
    A score of 7/9 means ready for Phase 0.
    </details>
    """),
]


build("phase_minus1_onboarding/00d_python_numpy_jupyter.ipynb", cells)
