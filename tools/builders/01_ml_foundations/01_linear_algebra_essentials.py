"""Builder for Lesson FND-01 — Linear Algebra Essentials."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # FND-01 · Linear Algebra Essentials

    *Learn how numbers become vectors, matrices, transformations, and predictions*

    | Lesson detail | Value |
    | --- | --- |
    | Prerequisites | PRE-02 and PRE-03; no formal linear algebra required |
    | Estimated study time | 7–9 hours across two or more sessions |
    | Main outcome | Calculate and explain core vector and matrix operations by hand and in NumPy |
    | Next lesson | FND-02 · Probability and Statistics |

    PRE-03 taught NumPy arrays and shapes as programming objects. This lesson gives
    those arrays mathematical meaning.

    We will move in one direction only:

    **one number → ordered numbers → rectangular numbers → weighted combinations →
    transformations → systems → projections.**

    Eigenvectors, power iteration, SVD, PCA, Ridge, Lasso, and matrix gradients are
    important, but they are not part of this required first pass. Their later lessons
    can now build on a stable foundation instead of competing for attention here.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end of this lesson, you will be able to:

    - distinguish a scalar, vector, and matrix;
    - explain what each vector position and matrix axis represents;
    - calculate vector addition, subtraction, and scalar multiplication;
    - calculate a dot product by multiplying matching positions and adding;
    - distinguish elementwise multiplication from a dot product;
    - calculate Euclidean length and distance;
    - calculate cosine similarity and recognise its zero-vector failure;
    - read matrix shapes before attempting multiplication;
    - distinguish row vectors, column vectors, and one-dimensional NumPy arrays;
    - calculate a transpose, matrix-vector product, and matrix-matrix product;
    - explain matrix multiplication as the composition of transformations;
    - use identity, zero, scaling, reflection, shear, and rotation matrices;
    - represent and solve a small system of linear equations;
    - explain linear combinations, span, dependence, basis, and rank with small numbers;
    - calculate a vector projection and verify its perpendicular residual;
    - connect projection to the least-squares geometry used later in CML-01;
    - complete a checked batch-scoring project with explicit shapes and units.

    Every required operation will appear in this order:

    **intuition → small hand calculation → formula → NumPy → interpretation → failure
    case.**
    """),

    md(r"""
    ## 2 · The practical problem: estimate several delivery routes

    A delivery team estimates the variable part of travel time from three route
    features:

    | Feature position | Meaning | Unit |
    | ---: | --- | --- |
    | 1 | Distance | kilometres |
    | 2 | Stops | stop count |
    | 3 | Rain | `1` for rain, `0` otherwise |

    One route is:

    $$
    \mathbf x = [4, 2, 1]
    $$

    Its positions mean 4 kilometres, 2 stops, and rain present. The ordered meaning
    is part of the data contract. `[2, 4, 1]` contains the same numbers but describes
    a different route.

    The planning rule assigns these contributions:

    $$
    \mathbf w = [2, 3, 5]
    $$

    The units are:

    - 2 minutes per kilometre;
    - 3 minutes per stop;
    - 5 minutes when the rain indicator changes from 0 to 1.

    Multiplying matching positions and adding gives one time estimate. Later, placing
    several route vectors into a matrix will calculate a whole batch at once.

    ```mermaid
    flowchart LR
        S["Scalar<br/>one value"] --> V["Vector<br/>one ordered record"]
        V --> M["Matrix<br/>a batch of records"]
        M --> P["Matrix × weights<br/>one score per row"]
        P --> C["Checked output<br/>shape and units"]
    ```
    """),

    md(r"""
    ## 3 · Scalars, vectors, and matrices

    ### 3.1 Scalar: one number

    A **scalar** is one number. Examples include:

    - `4` kilometres;
    - `2` stops;
    - `5` minutes.

    A scalar has magnitude but no list of components.

    ### 3.2 Vector: an ordered list

    A **vector** is an ordered list of scalars that belong together.

    $$
    \mathbf x = [x_1,x_2,\ldots,x_d]
    $$

    Symbols:

    - bold $\mathbf x$ names the whole vector;
    - $x_j$ is component $j$;
    - $d$ is the number of components, called the vector's dimension;
    - the dots mean the pattern continues.

    The route vector `[4,2,1]` has dimension 3. Its NumPy shape is `(3,)`.

    ### 3.3 Matrix: a rectangular table

    A **matrix** places related vectors into rows or columns. We will use rows as
    observations and columns as features:

    $$
    X =
    \begin{bmatrix}
    4 & 2 & 1 \\
    7 & 1 & 0 \\
    3 & 3 & 1
    \end{bmatrix}
    $$

    This matrix has shape `(3,3)`: three routes and three features.

    $x_{ij}$ means the value in row $i$, column $j$. The first subscript chooses
    an observation; the second chooses a feature.
    """),

    code(r"""
    import matplotlib.pyplot as plt
    import numpy as np

    distance_km = 4.0
    route_vector = np.array([4.0, 2.0, 1.0])
    route_matrix = np.array(
        [
            [4.0, 2.0, 1.0],
            [7.0, 1.0, 0.0],
            [3.0, 3.0, 1.0],
        ]
    )

    print("scalar:", distance_km, "shape: no axes")
    print("vector:", route_vector, "shape:", route_vector.shape)
    print("matrix:\n", route_matrix)
    print("matrix shape:", route_matrix.shape)
    print("row 1, column 2:", route_matrix[0, 1])

    assert np.isscalar(distance_km)
    assert route_vector.shape == (3,)
    assert route_matrix.shape == (3, 3)
    assert route_matrix[0, 1] == 2
    """),

    md(r"""
    Python indexes from zero, while mathematical subscripts usually begin at one.
    Mathematical $x_{1,2}$ corresponds to NumPy `X[0, 1]`.

    ### Orientation matters

    A row vector of three values has mathematical shape $1\times3$. A column vector
    has shape $3\times1$. A NumPy array with shape `(3,)` has one axis and is neither
    explicitly a row nor a column.
    """),

    code(r"""
    one_dimensional = np.array([4.0, 2.0, 1.0])
    row_vector = one_dimensional.reshape(1, 3)
    column_vector = one_dimensional.reshape(3, 1)

    print("1D shape:", one_dimensional.shape)
    print("row shape:", row_vector.shape)
    print("column shape:", column_vector.shape)
    print("1D transpose shape:", one_dimensional.T.shape)

    assert one_dimensional.T.shape == (3,)  # Transpose does not orient a 1D array.
    assert row_vector.T.shape == (3, 1)
    """),

    md(r"""
    Beginner trap: `one_dimensional.T` does not create a column vector. Use
    `.reshape(3,1)` or `[:, None]` when orientation must be explicit.
    """),

    md(r"""
    ## 4 · Vector arithmetic preserves position meaning

    Two vectors can be added or subtracted component by component when their shapes
    and position meanings match.

    Symbols: $\mathbf a$ and $\mathbf b$ name whole vectors; the plus and minus
    signs act on matching components; a number written before a vector is a scalar
    that multiplies every component.

    For:

    $$
    \mathbf a=[2,1]
    \qquad
    \mathbf b=[1,3]
    $$

    addition gives:

    $$
    \mathbf a+\mathbf b=[2+1,1+3]=[3,4]
    $$

    subtraction gives:

    $$
    \mathbf a-\mathbf b=[2-1,1-3]=[1,-2]
    $$

    Multiplying by scalar 2 stretches every component:

    $$
    2\mathbf a=[2(2),2(1)]=[4,2]
    $$

    Do not add vectors merely because their shapes match. `[height,weight]` plus
    `[temperature,price]` is numerically possible but conceptually meaningless.
    """),

    code(r"""
    vector_a = np.array([2.0, 1.0])
    vector_b = np.array([1.0, 3.0])

    vector_sum = vector_a + vector_b
    vector_difference = vector_a - vector_b
    scaled_vector = 2 * vector_a

    print("a + b:", vector_sum)
    print("a - b:", vector_difference)
    print("2a:", scaled_vector)

    assert np.allclose(vector_sum, [3, 4])
    assert np.allclose(vector_difference, [1, -2])
    assert np.allclose(scaled_vector, [4, 2])
    """),

    code(r"""
    figure, axis = plt.subplots(figsize=(6, 5))
    origin = np.zeros(2)

    for vector, colour, label in [
        (vector_a, "tab:blue", "a"),
        (vector_b, "tab:orange", "b"),
        (vector_sum, "tab:green", "a + b"),
    ]:
        axis.quiver(
            *origin,
            *vector,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=colour,
            width=0.012,
            label=label,
        )

    axis.plot(
        [vector_a[0], vector_sum[0]],
        [vector_a[1], vector_sum[1]],
        linestyle="--",
        color="gray",
    )
    axis.set_xlim(-1, 5)
    axis.set_ylim(-3, 5)
    axis.set_aspect("equal")
    axis.axhline(0, color="black", linewidth=0.6)
    axis.axvline(0, color="black", linewidth=0.6)
    axis.set_title("Vector addition: place b at the tip of a")
    axis.legend()
    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    The arrow picture makes direction visible. The component calculation determines
    the exact result. Use both views; the diagram is not a replacement for arithmetic.
    """),

    md(r"""
    ## 5 · Dot products turn two vectors into one weighted total

    The **dot product** multiplies matching components and adds the products.

    For vectors with $d$ components:

    $$
    \mathbf x\cdot\mathbf w
    = \sum_{j=1}^{d}x_jw_j
    $$

    Read it as “x dot w equals the sum of each x component multiplied by its matching
    w component.”

    For the delivery route:

    $$
    \mathbf x=[4,2,1]
    \qquad
    \mathbf w=[2,3,5]
    $$

    $$
    \mathbf x\cdot\mathbf w
    = 4(2)+2(3)+1(5)
    = 8+6+5
    = 19\text{ minutes}
    $$

    Each product has minutes as its output unit, so adding them is meaningful.
    """),

    code(r"""
    contribution_weights = np.array([2.0, 3.0, 5.0])
    component_products = route_vector * contribution_weights
    route_dot_product = route_vector @ contribution_weights

    print("elementwise products:", component_products)
    print("dot product:", route_dot_product, "minutes")
    print("sum of elementwise products:", component_products.sum())

    assert np.allclose(component_products, [8, 6, 5])
    assert route_dot_product == 19
    assert route_dot_product == component_products.sum()
    """),

    md(r"""
    `route_vector * weights` returns another vector because `*` is elementwise.
    `route_vector @ weights` returns one scalar because `@` performs the dot product
    for one-dimensional arrays.

    A dot product requires the same number of matching components. More importantly,
    each position must describe the feature expected by its matching weight.
    """),

    code(r"""
    def dot_product_from_scratch(first_vector, second_vector):
        '''Multiply matching positions and add the products.'''
        first = np.asarray(first_vector, dtype=float)
        second = np.asarray(second_vector, dtype=float)
        if first.ndim != 1 or second.ndim != 1:
            raise ValueError("Dot product inputs must be one-dimensional")
        if first.shape != second.shape:
            raise ValueError(f"Dot product shapes must match: {first.shape} versus {second.shape}")

        total = 0.0
        for first_value, second_value in zip(first, second):
            total += first_value * second_value
        return total


    scratch_dot = dot_product_from_scratch(route_vector, contribution_weights)
    print("scratch dot product:", scratch_dot)

    assert scratch_dot == route_vector @ contribution_weights
    """),

    md(r"""
    ## 6 · Length, distance, and cosine similarity

    ### 6.1 Euclidean length

    The Euclidean norm, written $\lVert\mathbf x\rVert_2$, is the ordinary straight-
    line length of a vector from the origin.

    $$
    \lVert\mathbf x\rVert_2
    = \sqrt{x_1^2+x_2^2+\cdots+x_d^2}
    $$

    For $[3,4]$:

    $$
    \lVert[3,4]\rVert_2
    = \sqrt{3^2+4^2}
    = \sqrt{25}
    = 5
    $$

    ### 6.2 Distance between two points

    Subtract first, then measure the difference vector:

    $$
    \operatorname{distance}(\mathbf a,\mathbf b)
    = \lVert\mathbf a-\mathbf b\rVert_2
    $$

    ### 6.3 Cosine similarity

    Cosine similarity keeps the angle relationship and removes vector length:

    $$
    \operatorname{cosine}(\mathbf a,\mathbf b)
    = \frac{\mathbf a\cdot\mathbf b}
           {\lVert\mathbf a\rVert_2\lVert\mathbf b\rVert_2}
    $$

    Its usual range is −1 to 1:

    - 1: same direction;
    - 0: perpendicular directions;
    - −1: opposite directions.

    Cosine similarity is undefined if either vector has length zero because division
    by zero has no meaning.
    """),

    code(r"""
    length_example = np.array([3.0, 4.0])
    point_a = np.array([1.0, 1.0])
    point_b = np.array([2.0, 0.0])

    length = np.sqrt(np.sum(length_example ** 2))
    distance = np.linalg.norm(point_a - point_b)
    cosine_similarity = (point_a @ point_b) / (
        np.linalg.norm(point_a) * np.linalg.norm(point_b)
    )

    print("length of [3,4]:", length)
    print("distance from a to b:", round(distance, 4))
    print("cosine similarity:", round(cosine_similarity, 4))

    assert length == 5
    assert np.isclose(distance, np.sqrt(2))
    assert np.isclose(cosine_similarity, 1 / np.sqrt(2))
    """),

    code(r"""
    def safe_cosine_similarity(first_vector, second_vector):
        '''Return cosine similarity or reject a zero vector.'''
        first = np.asarray(first_vector, dtype=float)
        second = np.asarray(second_vector, dtype=float)
        if first.shape != second.shape or first.ndim != 1:
            raise ValueError("Cosine inputs must be one-dimensional with matching shapes")

        first_length = np.linalg.norm(first)
        second_length = np.linalg.norm(second)
        if np.isclose(first_length, 0) or np.isclose(second_length, 0):
            raise ValueError("Cosine similarity is undefined for a zero vector")
        return float((first @ second) / (first_length * second_length))


    try:
        safe_cosine_similarity([0, 0], [1, 2])
        raise AssertionError("Expected a zero vector to raise ValueError")
    except ValueError as error:
        print(type(error).__name__ + ":", error)
        assert "zero vector" in str(error)
    """),

    md(r"""
    ### Which comparison should we use?

    | Operation | Output | Main question | Important limitation |
    | --- | --- | --- | --- |
    | Dot product | Scalar | What weighted total or alignment do these vectors produce? | Magnitude affects the result |
    | Euclidean distance | Non-negative scalar | How far apart are these points? | Feature scale affects distance |
    | Cosine similarity | Scalar from −1 to 1 | How similar are their directions? | Ignores magnitude; fails on zero vectors |

    Matching shape does not guarantee matching meaning. Comparing unscaled height,
    income, and temperature as though they share one geometry can be misleading.
    """),

    md(r"""
    ## 7 · Matrices, transpose, and shape compatibility

    ### 7.1 Transpose swaps axes

    For:

    $$
    A=
    \begin{bmatrix}
    1&2&3\\
    4&5&6
    \end{bmatrix}
    $$

    the transpose is:

    $$
    A^\top=
    \begin{bmatrix}
    1&4\\
    2&5\\
    3&6
    \end{bmatrix}
    $$

    $A$ has shape $2\times3$. $A^\top$ has shape $3\times2$. Entry
    $a_{ij}$ moves to position $a_{ji}$.
    """),

    code(r"""
    matrix_a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    matrix_a_transpose = matrix_a.T

    print("A shape:", matrix_a.shape)
    print(matrix_a)
    print("\nA transpose shape:", matrix_a_transpose.shape)
    print(matrix_a_transpose)

    assert matrix_a.shape == (2, 3)
    assert matrix_a_transpose.shape == (3, 2)
    assert matrix_a[0, 2] == matrix_a_transpose[2, 0]
    """),

    md(r"""
    ### 7.2 The inner dimensions must match

    Matrix multiplication follows this shape rule:

    $$
    (m\times n)(n\times p)\longrightarrow(m\times p)
    $$

    The two inner dimensions are both $n$, so they disappear. The outer dimensions
    $m$ and $p$ remain in the output.

    Shape compatibility tells us an operation can be calculated. It does not prove
    the row and column meanings are sensible.
    """),

    md(r"""
    ## 8 · Matrix-vector multiplication scores a batch

    A matrix-vector product takes the dot product of every matrix row with the vector.

    Our route matrix has shape $3\times3$, and the weight vector has three
    components:

    $$
    X\mathbf w=
    \begin{bmatrix}
    4&2&1\\
    7&1&0\\
    3&3&1
    \end{bmatrix}
    \begin{bmatrix}
    2\\3\\5
    \end{bmatrix}
    $$

    The first output is the first-row dot product:

    $$
    4(2)+2(3)+1(5)=19
    $$

    The full result is:

    $$
    X\mathbf w=
    \begin{bmatrix}
    19\\17\\20
    \end{bmatrix}
    $$

    Input shape `(3,3)` multiplied by shape `(3,)` produces output shape `(3,)`: one
    estimated variable-time contribution for each route.
    """),

    code(r"""
    batch_time_contributions = route_matrix @ contribution_weights

    print("route matrix shape:", route_matrix.shape)
    print("weight shape:", contribution_weights.shape)
    print("output shape:", batch_time_contributions.shape)
    print("time contributions:", batch_time_contributions)

    manual_batch = np.array(
        [row @ contribution_weights for row in route_matrix]
    )

    assert np.allclose(batch_time_contributions, [19, 17, 20])
    assert np.allclose(batch_time_contributions, manual_batch)
    """),

    md(r"""
    This is the mathematical core of a linear model: a feature matrix multiplied by
    a weight vector produces one weighted score per row. CML-01 explains how the
    weights can be learned rather than chosen by a planner.
    """),

    md(r"""
    ## 9 · Matrix-matrix multiplication composes work

    Each output cell is one row-by-column dot product.

    Let:

    $$
    A=
    \begin{bmatrix}
    1&2&3\\
    4&5&6
    \end{bmatrix}
    \qquad
    B=
    \begin{bmatrix}
    1&2\\
    0&1\\
    1&0
    \end{bmatrix}
    $$

    Shapes are $2\times3$ and $3\times2$, so the result has shape $2\times2$.

    The top-left output is:

    $$
    1(1)+2(0)+3(1)=4
    $$

    The top-right output is:

    $$
    1(2)+2(1)+3(0)=4
    $$

    Repeating this for every row and column gives:

    $$
    AB=
    \begin{bmatrix}
    4&4\\
    10&13
    \end{bmatrix}
    $$
    """),

    code(r"""
    matrix_b = np.array(
        [
            [1.0, 2.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    matrix_product = matrix_a @ matrix_b
    elementwise_square = matrix_a * matrix_a

    print("A shape:", matrix_a.shape)
    print("B shape:", matrix_b.shape)
    print("A @ B shape:", matrix_product.shape)
    print("A @ B:\n", matrix_product)
    print("\nA * A keeps A's shape:", elementwise_square.shape)

    assert np.allclose(matrix_product, [[4, 4], [10, 13]])
    assert matrix_product.shape == (2, 2)
    assert elementwise_square.shape == matrix_a.shape
    """),

    md(r"""
    ### Multiplication order matters

    In general, $AB\ne BA$. Sometimes $BA$ has a different shape; sometimes it
    cannot be calculated at all.

    Matrix multiplication represents composition. If $A\mathbf x$ applies
    transformation $A$ first and then $B$ acts on the result, the combined
    transformation is:

    $$
    B(A\mathbf x)=(BA)\mathbf x
    $$

    The matrix nearest the vector acts first.
    """),

    code(r"""
    incompatible_left = np.ones((2, 3))
    incompatible_right = np.ones((4, 2))

    try:
        incompatible_left @ incompatible_right
        raise AssertionError("Expected incompatible inner dimensions to fail")
    except ValueError as error:
        print(type(error).__name__ + ": matrix shapes (2,3) and (4,2) cannot multiply")
    """),

    md(r"""
    ## 10 · Matrices as transformations

    A matrix is not only stored numbers. It is a rule that maps one vector to another.

    Common two-dimensional transformations include:

    | Transformation | Matrix | Effect |
    | --- | --- | --- |
    | Scale x by 2 | $\begin{bmatrix}2&0\\0&1\end{bmatrix}$ | Horizontal stretch |
    | Reflect across y-axis | $\begin{bmatrix}-1&0\\0&1\end{bmatrix}$ | Flip left/right |
    | Shear | $\begin{bmatrix}1&1\\0&1\end{bmatrix}$ | Shift x according to y |
    | Rotate by 90° | $\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ | Quarter turn counter-clockwise |

    Apply the 90-degree rotation to $[2,1]$:

    $$
    \begin{bmatrix}
    0&-1\\
    1&0
    \end{bmatrix}
    \begin{bmatrix}
    2\\1
    \end{bmatrix}
    =
    \begin{bmatrix}
    -1\\2
    \end{bmatrix}
    $$
    """),

    code(r"""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    scale_x = np.array([[2.0, 0.0], [0.0, 1.0]])
    reflect_y_axis = np.array([[-1.0, 0.0], [0.0, 1.0]])
    shear = np.array([[1.0, 1.0], [0.0, 1.0]])
    rotate_90 = np.array([[0.0, -1.0], [1.0, 0.0]])

    transformed_sets = {
        "original": points,
        "scale x": points @ scale_x.T,
        "reflect": points @ reflect_y_axis.T,
        "shear": points @ shear.T,
        "rotate 90°": points @ rotate_90.T,
    }

    figure, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for axis, (name, transformed_points) in zip(axes, transformed_sets.items()):
        axis.plot(transformed_points[:, 0], transformed_points[:, 1], marker="o")
        axis.axhline(0, color="gray", linewidth=0.5)
        axis.axvline(0, color="gray", linewidth=0.5)
        axis.set_xlim(-3, 5)
        axis.set_ylim(-3, 4)
        axis.set_aspect("equal")
        axis.set_title(name)

    figure.suptitle("The same points under four linear transformations")
    figure.tight_layout()
    plt.show()

    assert np.allclose(rotate_90 @ np.array([2.0, 1.0]), [-1, 2])
    """),

    md(r"""
    Linear transformations preserve two rules:

    $$
    A(\mathbf u+\mathbf v)=A\mathbf u+A\mathbf v
    $$

    $$
    A(c\mathbf u)=c(A\mathbf u)
    $$

    They preserve addition and scalar multiplication. Ordinary matrix multiplication
    cannot perform a translation by itself because a linear transformation must map
    the zero vector to the zero vector. Models handle translation with an intercept or
    an added column of ones.
    """),

    md(r"""
    ## 11 · Identity matrices and systems of equations

    ### 11.1 Identity leaves a vector unchanged

    The identity matrix $I$ has ones on its main diagonal and zeros elsewhere:

    $$
    I=
    \begin{bmatrix}
    1&0\\
    0&1
    \end{bmatrix}
    $$

    $$
    I\mathbf x=\mathbf x
    $$

    A zero matrix maps every compatible input to a zero vector.

    ### 11.2 A system becomes one matrix equation

    Consider:

    $$
    2x+y=5
    $$

    $$
    x-y=1
    $$

    Matrix form is:

    $$
    \begin{bmatrix}
    2&1\\
    1&-1
    \end{bmatrix}
    \begin{bmatrix}
    x\\y
    \end{bmatrix}
    =
    \begin{bmatrix}
    5\\1
    \end{bmatrix}
    $$

    The solution is $x=2, y=1$. Substitution verifies both equations.
    """),

    code(r"""
    identity = np.eye(2)
    system_matrix = np.array([[2.0, 1.0], [1.0, -1.0]])
    system_target = np.array([5.0, 1.0])
    system_solution = np.linalg.solve(system_matrix, system_target)

    print("identity times [2,1]:", identity @ np.array([2.0, 1.0]))
    print("solution [x,y]:", system_solution)
    print("A @ solution:", system_matrix @ system_solution)

    assert np.allclose(identity @ system_solution, system_solution)
    assert np.allclose(system_solution, [2, 1])
    assert np.allclose(system_matrix @ system_solution, system_target)
    """),

    md(r"""
    An inverse, when it exists, reverses a square transformation. In code, solve
    $A\mathbf x=\mathbf b$ with `np.linalg.solve(A, b)` rather than calculating
    `np.linalg.inv(A) @ b`. `solve` states the real task and uses a direct numerical
    method.

    Not every matrix has an inverse. A non-square matrix has no ordinary two-sided
    inverse, and dependent rows or columns can make a square system non-unique.
    """),

    md(r"""
    ## 12 · Linear combinations, span, independence, basis, and rank

    ### 12.1 Linear combination

    A linear combination scales vectors and adds them:

    $$
    c_1\mathbf v_1+c_2\mathbf v_2
    $$

    For the standard coordinate directions:

    $$
    \mathbf e_1=[1,0]
    \qquad
    \mathbf e_2=[0,1]
    $$

    any two-dimensional vector can be written as:

    $$
    [a,b]=a\mathbf e_1+b\mathbf e_2
    $$

    ### 12.2 Span and basis

    The **span** is every vector reachable through linear combinations. Because
    $\mathbf e_1$ and $\mathbf e_2$ reach every point in the plane, they span
    $\mathbb R^2$.

    A **basis** is a non-redundant set that spans the space. The coordinate vectors
    form one basis for the plane.

    ### 12.3 Dependence and rank

    Vectors are linearly dependent when one can be made from the others. For example:

    $$
    [2,4]=2[1,2]
    $$

    These vectors point along the same line and contribute only one independent
    direction.

    **Rank** counts independent directions in a matrix. A matrix with duplicate or
    scaled-copy columns has less rank than columns.
    """),

    code(r"""
    independent_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    dependent_matrix = np.array([[1.0, 2.0], [2.0, 4.0]])

    independent_rank = np.linalg.matrix_rank(independent_matrix)
    dependent_rank = np.linalg.matrix_rank(dependent_matrix)

    target_vector = np.array([3.0, -2.0])
    basis_coordinates = np.array([3.0, -2.0])
    reconstructed = independent_matrix @ basis_coordinates

    print("independent rank:", independent_rank)
    print("dependent rank:", dependent_rank)
    print("basis reconstruction:", reconstructed)

    assert independent_rank == 2
    assert dependent_rank == 1
    assert np.allclose(reconstructed, target_vector)
    """),

    md(r"""
    In machine learning, two duplicate feature columns provide the same direction.
    A model may still produce predictions, but individual coefficients can become
    non-unique or unstable. CML-01 and later regularization lessons revisit this in a
    concrete model.
    """),

    md(r"""
    ## 13 · Projection: keep the component along a direction

    Projection is the vector version of casting a shadow onto a line.

    To project $\mathbf a$ onto nonzero vector $\mathbf b$:

    $$
    \operatorname{proj}_{\mathbf b}(\mathbf a)
    = \frac{\mathbf a\cdot\mathbf b}
           {\mathbf b\cdot\mathbf b}\mathbf b
    $$

    Use:

    $$
    \mathbf a=[3,1]
    \qquad
    \mathbf b=[1,1]
    $$

    Calculate the scale:

    $$
    \frac{\mathbf a\cdot\mathbf b}{\mathbf b\cdot\mathbf b}
    = \frac{3(1)+1(1)}{1(1)+1(1)}
    = \frac{4}{2}
    = 2
    $$

    Therefore:

    $$
    \operatorname{proj}_{\mathbf b}(\mathbf a)=2[1,1]=[2,2]
    $$

    The leftover residual is:

    $$
    \mathbf r=\mathbf a-\operatorname{proj}_{\mathbf b}(\mathbf a)
    =[3,1]-[2,2]=[1,-1]
    $$

    The residual is perpendicular to the projection direction:

    $$
    \mathbf r\cdot\mathbf b=1(1)+(-1)(1)=0
    $$
    """),

    code(r"""
    vector_to_project = np.array([3.0, 1.0])
    projection_direction = np.array([1.0, 1.0])

    projection_scale = (
        vector_to_project @ projection_direction
    ) / (
        projection_direction @ projection_direction
    )
    projected_vector = projection_scale * projection_direction
    projection_residual = vector_to_project - projected_vector

    print("projection scale:", projection_scale)
    print("projection:", projected_vector)
    print("residual:", projection_residual)
    print("residual dot direction:", projection_residual @ projection_direction)

    assert projection_scale == 2
    assert np.allclose(projected_vector, [2, 2])
    assert np.allclose(projection_residual, [1, -1])
    assert np.isclose(projection_residual @ projection_direction, 0)
    """),

    code(r"""
    figure, axis = plt.subplots(figsize=(6, 5))
    for vector, colour, label in [
        (vector_to_project, "tab:blue", "a"),
        (projection_direction, "tab:orange", "direction b"),
        (projected_vector, "tab:green", "projection"),
    ]:
        axis.quiver(
            0,
            0,
            *vector,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=colour,
            width=0.012,
            label=label,
        )

    axis.plot(
        [vector_to_project[0], projected_vector[0]],
        [vector_to_project[1], projected_vector[1]],
        linestyle="--",
        color="gray",
        label="perpendicular residual",
    )
    axis.set_xlim(-1, 4)
    axis.set_ylim(-2, 4)
    axis.set_aspect("equal")
    axis.axhline(0, color="black", linewidth=0.5)
    axis.axvline(0, color="black", linewidth=0.5)
    axis.set_title("Projection keeps the component along b")
    axis.legend()
    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    Projection onto a zero vector is undefined because the denominator
    $\mathbf b\cdot\mathbf b$ would be zero.

    ### Bridge to later lessons

    - CML-01: least squares projects a target toward the predictions a feature matrix
      can express;
    - FND-02: covariance matrices organise how numerical variables vary together;
    - FND-04: gradients operate on parameter vectors after a loss is defined;
    - MLE-06: eigenvectors, SVD, and PCA identify important directions for
      unsupervised representation.

    These are dependencies, not topics to memorise before their practical problems
    have been introduced.
    """),

    md(r"""
    ## 14 · Mini-project, exercises, and mastery checkpoint

    ### Mini-project: checked route-time calculator

    **Project goal:** calculate one variable-time estimate per route using a feature
    matrix and declared weight vector.

    **Dataset columns:**

    | Column | Meaning | Unit |
    | --- | --- | --- |
    | `route_id` | Stable route label | Identifier |
    | `distance_km` | Travel distance | Kilometres |
    | `stop_count` | Planned stops | Count |
    | `rain_indicator` | Rain present | 0 or 1 |

    **Expected workflow:** validate names and shapes, build $X$, calculate each
    component contribution, calculate $X\mathbf w$, add a fixed four-minute setup
    time, and return a readable report.
    """),

    code(r"""
    import pandas as pd

    route_data = pd.DataFrame(
        {
            "route_id": ["A", "B", "C", "D"],
            "distance_km": [4.0, 7.0, 3.0, 5.0],
            "stop_count": [2.0, 1.0, 3.0, 2.0],
            "rain_indicator": [1.0, 0.0, 1.0, 0.0],
        }
    )

    project_feature_columns = ["distance_km", "stop_count", "rain_indicator"]
    project_weights = np.array([2.0, 3.0, 5.0])
    fixed_setup_minutes = 4.0

    project_matrix = route_data[project_feature_columns].to_numpy()
    contribution_matrix = project_matrix * project_weights
    variable_minutes = project_matrix @ project_weights
    estimated_total_minutes = fixed_setup_minutes + variable_minutes

    project_report = route_data[["route_id"]].copy()
    project_report["distance_minutes"] = contribution_matrix[:, 0]
    project_report["stop_minutes"] = contribution_matrix[:, 1]
    project_report["rain_minutes"] = contribution_matrix[:, 2]
    project_report["fixed_setup_minutes"] = fixed_setup_minutes
    project_report["estimated_total_minutes"] = estimated_total_minutes

    print("feature matrix shape:", project_matrix.shape)
    print("weight shape:", project_weights.shape)
    print("output shape:", estimated_total_minutes.shape)
    print(project_report)

    assert project_matrix.shape == (4, 3)
    assert project_weights.shape == (3,)
    assert estimated_total_minutes.shape == (4,)
    assert np.allclose(variable_minutes, contribution_matrix.sum(axis=1))
    assert project_report["estimated_total_minutes"].tolist() == [23, 21, 24, 20]
    """),

    md(r"""
    Expected output: one row per route and one final time estimate in minutes. A result
    is trustworthy only if the feature order matches the weight order and every
    component product has the same output unit.

    ### Worked example

    For route `[2 km, 1 stop, no rain]` and weights `[2,3,5]`:

    $$
    2(2)+1(3)+0(5)=7\text{ variable minutes}
    $$

    Adding four fixed minutes produces 11 estimated total minutes.

    ### Guided practice

    1. Classify each object as scalar, vector, or matrix: `7`, `[7,2]`, and
       `[[7,2],[3,1]]`.
    2. Calculate `[2,-1]+[3,4]`, `[2,-1]-[3,4]`, and `3[2,-1]` by hand.
    3. Calculate `[2,3,1]·[4,-1,2]` and show every product.
    4. Calculate the length of `[6,8]` and the distance between `[1,2]` and `[4,6]`.
    5. Transpose a $2\times3$ matrix and write its new shape.
    6. Multiply a $2\times2$ matrix by a two-component vector manually.

    ### Independent practice

    7. Multiply a $2\times3$ matrix by a $3\times2$ matrix. Explain every output
       cell as a row-column dot product.
    8. Apply the scale, reflection, shear, and rotation matrices to `[3,2]`.
    9. Solve a new two-equation system with `np.linalg.solve` and verify the result by
       substitution.
    10. Create one rank-2 and one rank-1 matrix. Explain the independent directions.
    11. Project `[4,2]` onto `[1,1]` and prove the residual is perpendicular.

    ### Challenge

    Rebuild the route-time project without copying its code. Then deliberately:

    - reorder two feature columns and detect the contract violation;
    - supply a weight vector with the wrong shape and explain the failure;
    - add one route and predict the new output shape before running;
    - implement matrix-vector multiplication with explicit loops;
    - verify the loop result against NumPy;
    - create six meaningful assertions.

    ### Self-check before reading solutions

    Before every operation, write:

    - each input shape;
    - each axis meaning;
    - the expected output shape;
    - the unit of one output value;
    - whether the operation is elementwise, a dot product, or matrix multiplication.
    """),

    md(r"""
    ### Solution and scoring rubric

    1. Scalar, vector, and matrix, respectively.
    2. The results are `[5,3]`, `[-1,-5]`, and `[6,-3]`.
    3. $2(4)+3(-1)+1(2)=8-3+2=7$.
    4. The length is 10. The difference between the two points is `[3,4]`, so their
       distance is 5.
    5. The transposed shape is $3\times2$; rows and columns swap.
    6. Calculate one row-vector dot product for each output component.
    7. Inner dimensions 3 and 3 match; the output shape is $2\times2$.
    8. Apply each matrix separately and label which geometric property changed.
    9. A valid solution reproduces the target when multiplied by the system matrix.
    10. Rank counts independent row or column directions, not total stored values.
    11. Subtract the projection from `[4,2]`; its dot product with `[1,1]` must be zero.

    Challenge scoring:

    | Skill | Points |
    | --- | ---: |
    | Scalar/vector/matrix meanings and shapes | 2 |
    | Vector arithmetic, length, distance, and cosine | 3 |
    | Dot product and unit explanation | 3 |
    | Matrix-vector and matrix-matrix calculations | 3 |
    | Transformations, systems, and rank | 3 |
    | Projection and perpendicular residual | 2 |
    | Project contract checks and assertions | 4 |

    Maximum: 20 points.

    **Common mistakes:** adding vectors with different meanings, confusing `*` with
    `@`, forgetting that NumPy 1D transpose changes nothing, multiplying incompatible
    inner dimensions, reversing transformation order, treating a matching shape as
    proof of semantic correctness, attempting cosine or projection with a zero vector,
    calculating an inverse when `solve` states the task, and calling duplicate columns
    independent.

    **Readiness threshold:** 16/20, including correct dot product, matrix-vector,
    matrix-matrix, rank, and projection calculations plus explicit shape reasoning.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    You are ready for FND-02 when you can, without copying this notebook:

    - distinguish scalars, vectors, matrices, dimensions, shapes, and orientations;
    - calculate vector arithmetic and explain why component meaning matters;
    - calculate dot products, lengths, distances, and cosine similarity;
    - distinguish elementwise multiplication from `@`;
    - transpose matrices and explain the NumPy 1D transpose trap;
    - predict matrix-product shapes before calculating;
    - calculate matrix-vector and matrix-matrix products by hand;
    - explain matrices as transformations and multiplication as composition;
    - solve and verify a small linear system;
    - explain linear combinations, span, independence, basis, and rank;
    - calculate a projection and prove its residual is perpendicular;
    - complete the mini-project with at least 16/20 points.

    ### Teach it back

    Explain how one route becomes a vector, several routes become a matrix, weights
    become a vector, and matrix-vector multiplication produces one time estimate per
    route. Include every shape, axis meaning, and unit.

    Then explain how dot product, length, cosine, matrix multiplication, rank, and
    projection each answer a different question.

    ### Memory aid

    **Shapes tell us which values can meet; meanings tell us whether they should.**

    FND-02 will use vectors and matrices to organise samples, expectations, variance,
    and covariance without requiring advanced decompositions yet.
    """),
]


build("01_ml_foundations/01_linear_algebra_essentials.ipynb", cells)
