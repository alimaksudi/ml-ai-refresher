"""Builder for Notebook 00B — Algebra, Functions, and Graphs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # 00B · Algebra, Functions, and Graphs
    ### Prerequisite Phase — Turning Relationships Into Pictures

    Algebra describes how values relate. A function is a repeatable input-to-output
    rule. A graph shows that rule as a picture.

    **Estimated time:** 2–3 hours.
    **Prerequisite:** Notebook 00A or equivalent readiness score.
    """),

    md(r"""
    ## 1 · Learning Objectives

    You will learn to:

    - solve simple equations while preserving equality;
    - distinguish constants, variables, coefficients, and terms;
    - evaluate a function for a concrete input;
    - read coordinates and plot points;
    - interpret slope and intercept in a line;
    - understand vectors as ordered lists and matrices as rectangular tables;
    - check shapes before adding or multiplying arrays.
    """),

    md(r"""
    ## 2 · Historical Motivation

    Arithmetic answers one calculation. Algebra describes a reusable relationship.
    Instead of calculating the cost of one order, we can write a rule for the cost
    of any order. Machine learning uses the same idea: a model is a function whose
    parameters are learned from examples.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    Think of a function as a machine:

    1. place an input in the machine;
    2. follow the rule;
    3. receive one output.

    A graph runs the machine for many inputs. Each point records one
    `(input, output)` pair.

    ```text
    input x  ──>  [ multiply by 2, then add 1 ]  ──>  output y
    ```
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Terms and coefficients

    $$y=2x+1.$$

    **Read aloud:** “y equals two times x plus one.”

    **Symbols:** $x$ is the input variable; $y$ is the output variable; `2` is a
    coefficient multiplying $x$; `1` is a constant. The pieces `2x` and `1` are
    terms. Writing `2x` means `2 × x`.

    **Small example:** if $x=3$, then $y=2\times3+1=7$.

    **Meaning:** every one-unit increase in $x$ increases $y$ by two units, and
    when $x=0$, $y=1$.

    ### 4.2 Solving an equation

    $$3x+2=14\quad\Longrightarrow\quad x=4.$$

    **Read aloud:** “three x plus two equals fourteen, therefore x equals four.”

    **Symbols:** $x$ is the unknown value and the arrow marks the conclusion.

    **Worked steps:** subtract 2 from both sides to obtain $3x=12$; divide both
    sides by 3 to obtain $x=4$; substitute back: $3\times4+2=14$.

    ### 4.3 Functions

    $$f(x)=x^2+1.$$

    **Read aloud:** “f of x equals x squared plus one.”

    **Symbols:** $f$ names the function; parentheses in $f(x)$ contain the input;
    $x^2$ means $x\times x$.

    **Small example:** $f(3)=3^2+1=10$ and $f(-3)=(-3)^2+1=10$.

    **Use and limit:** a function returns one output for each allowed input. Two
    inputs may share an output, but one input cannot produce two different outputs
    under the same deterministic function.

    ### 4.4 Coordinates

    $$(x,y)=(3,7).$$

    **Read aloud:** “the point with x-coordinate three and y-coordinate seven.”

    **Symbols:** an ordered pair stores horizontal position first and vertical
    position second. Order matters: $(3,7)$ and $(7,3)$ are different points.

    ### 4.5 Slope and intercept

    $$y=mx+b,\qquad m=\frac{y_2-y_1}{x_2-x_1}.$$

    **Read aloud:** “y equals slope times x plus intercept; slope equals change in
    y divided by change in x.”

    **Symbols:** $m$ is slope, $b$ is the value of $y$ when $x=0$, $(x_1,y_1)$ and
    $(x_2,y_2)$ are two points, and subscripts label the first and second point.

    **Small example:** from $(1,3)$ to $(3,7)$, slope is
    $(7-3)/(3-1)=4/2=2$. The line rises two units for each one-unit move right.

    **Limit:** slope is undefined for a vertical line because $x_2-x_1=0$.

    ### 4.6 Vectors and matrices

    $$\mathbf{x}=\begin{bmatrix}2\\5\\8\end{bmatrix},\qquad
    A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}.$$

    **Read aloud:** “x is a vector containing two, five, and eight; A is a matrix
    with two rows and three columns.”

    **Symbols:** bold $\mathbf{x}$ denotes a vector; square brackets group values;
    $A$ is a matrix. The vector has shape `3`; the matrix has shape `2 × 3`.

    **Meaning:** one dataset row can be a vector of features; many rows stacked
    together form a matrix.

    ### 4.7 Dot product as multiply-then-add

    $$\mathbf{a}\cdot\mathbf{b}=a_1b_1+a_2b_2.$$

    **Read aloud:** “a dot b equals a one times b one plus a two times b two.”

    **Symbols:** $\cdot$ is the dot product; subscripts pair values in the same
    position.

    **Small example:** if $\mathbf a=(2,3)$ and $\mathbf b=(4,5)$, the result is
    `2 × 4 + 3 × 5 = 23`.

    **Use and limit:** the dot product summarizes how two equal-length vectors
    align. It is undefined when their lengths differ.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    Consider the delivery-cost function `cost = 2 × distance + 5`.

    - `5` is a fixed starting fee.
    - `2` is the cost added per kilometre.
    - At 0 km, cost is 5.
    - At 3 km, cost is `2 × 3 + 5 = 11`.
    """),

    code(r"""
    def delivery_cost(distance_km):
        return 2 * distance_km + 5

    distances = [0, 1, 2, 3, 4, 5]
    costs = [delivery_cost(value) for value in distances]

    for distance, cost in zip(distances, costs):
        print(f"{distance} km -> ${cost}")

    assert delivery_cost(3) == 11
    """),

    md(r"""
    ## 6 · Visualization

    The horizontal axis contains inputs. The vertical axis contains outputs. The
    slope is visible as the line's rise as we move right.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 100)
    y = 2 * x + 5

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, label="cost = 2 × distance + 5")
    ax.scatter([0, 3], [5, 11], color="tab:red", zorder=3)
    ax.annotate("intercept: (0, 5)", (0, 5), xytext=(1, 8),
                arrowprops={"arrowstyle": "->"})
    ax.annotate("3 km costs 11", (3, 11), xytext=(5, 10),
                arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("distance (km)")
    ax.set_ylabel("cost ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()
    """),

    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Moving a term across `=` and changing its sign without understanding that the
      same operation must be applied to both sides.
    - Forgetting that `2x` means multiplication.
    - Swapping x- and y-coordinates.
    - Dividing by zero while calculating slope.
    - Adding arrays with incompatible shapes.
    - Believing a graph proves that one variable causes another. A graph only shows
      the relationship present in the displayed data or function.
    """),

    md(r"""
    ## 8 · NumPy Implementation

    NumPy stores vectors as one-dimensional arrays and matrices as two-dimensional
    arrays. The `.shape` attribute is a safety check.
    """),

    code(r"""
    vector_a = np.array([2, 3])
    vector_b = np.array([4, 5])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])

    manual_dot = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
    numpy_dot = vector_a @ vector_b

    print("vector shape:", vector_a.shape)
    print("matrix shape:", matrix.shape)
    print("manual dot product:", manual_dot)
    print("NumPy dot product:", numpy_dot)

    assert manual_dot == numpy_dot == 23
    """),

    md(r"""
    ## 9 · Realistic Case Study — A Linear Prediction Rule

    A simple model estimates delivery time as:

    `time_minutes = 4 × distance_km + 8`.

    The coefficient `4` says each kilometre adds four minutes. The intercept `8`
    represents preparation time even for a zero-distance delivery. At 5 km, the
    prediction is `4 × 5 + 8 = 28` minutes.

    This rule is useful only if the relationship is approximately linear in the
    operating range. Traffic or weather may break that assumption.
    """),

    md(r"""
    ## 10 · Learning Considerations

    Always attach meaning and units to variables. `x = 5` is less informative than
    `distance_km = 5`. Before matrix operations, write the shape next to each
    object. This habit prevents many errors in later ML notebooks.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    A linear function is easy to explain and calculate, but it cannot represent a
    curved relationship without additional terms. A graph is intuitive but not
    exact enough for repeated calculation. A formula is exact but requires symbol
    fluency. Code scales the calculation but can execute a mistaken rule perfectly.
    """),

    md(r"""
    ## 12 · Readiness Check

    You are ready for Notebook 00C when you can:

    1. evaluate $f(x)=3x-2$ at $x=4$;
    2. explain slope and intercept in plain language;
    3. identify the shape of a vector and matrix;
    4. compute a two-value dot product;
    5. explain why division by zero is invalid.
    """),

    md(r"""
    ## 13 · Teach-Back

    Without notes, explain how the sentence “start at 10 and add 3 for each item”
    becomes a function, a table of values, and a graph. Then explain which part is
    the slope and which part is the intercept.
    """),

    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Estimated time:** 35–50 minutes.

    ### Worked example

    For $f(x)=2x+3$, calculate $f(4)$: replace $x$ with `4`, giving
    `2 × 4 + 3 = 11`.

    ### Guided practice

    1. Solve $2x+5=17$. First subtract 5 from both sides.
    2. For $g(x)=x^2-1$, calculate $g(0)$, $g(2)$, and $g(-2)$.
    3. Find the slope through $(2,5)$ and $(6,13)$.
    4. Compute $(1,3)\cdot(2,4)$.

    ### Independent practice

    5. Write a function for a taxi fare with a `$4` starting fee and `$1.50` per km.
    6. A matrix has 5 customer rows and 3 feature columns. State its shape.
    7. Explain why vectors of lengths 2 and 3 cannot use the position-by-position
       dot-product rule.

    <details>
    <summary><strong>Solutions and scoring</strong></summary>

    1. $2x=12$, so $x=6$. Check: `2 × 6 + 5 = 17`.
    2. `−1`, `3`, and `3`.
    3. $(13-5)/(6-2)=8/4=2$.
    4. `1 × 2 + 3 × 4 = 14`.
    5. `fare(distance) = 1.5 × distance + 4`.
    6. `5 × 3`.
    7. One value would have no matching partner, so multiply-then-add is undefined.

    Award one point per correct answer and one additional point for showing the
    check in Questions 1 and 3. A score of 7/9 is sufficient to continue.
    </details>
    """),
]


build("phase_minus1_onboarding/00b_algebra_functions_graphs.ipynb", cells)
