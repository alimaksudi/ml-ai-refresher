"""Builder for Lesson FND-01 — Linear Algebra Essentials.


NOTE on string literals: every cell body is a RAW string (r\"\"\"...\"\"\") so that
LaTeX backslashes (\\frac, \\alpha) and code escapes (\\n) survive verbatim.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    # ---------------------------------------------------------------- Title
    md(r"""
    # FND-01 · Linear Algebra Essentials
    ### Section 01 — Mathematical Foundations · *ML/AI Senior Mastery Curriculum*

    **Prerequisites:** complete PRE-01 through PRE-03, or confirm that you can read
    subscripts, summations, functions, vectors, matrix shapes, and a basic
    derivative. **Estimated time:** 4–6 hours including exercises.

    > **How to read this notebook.** Work top to bottom. Each section answers the
    > same questions a senior engineer must answer about *any* technique: what it
    > is, why it exists, how it works (intuitively, then mathematically), how to
    > build it from scratch, when to use it, when **not** to, the tradeoffs, and
    > how it behaves in production. Don't skip the derivations — being able to
    > rebuild them on a whiteboard is exactly what separates "I used a library"
    > from "I understand the system."

    Linear algebra is the *substrate* of machine learning. A dataset is a matrix.
    A linear model is a dot product. A neural network layer is a matrix multiply
    plus a nonlinearity. An embedding is a vector. Attention is a product of three
    matrices. PCA, least squares, recommender systems, and the geometry of
    similarity search are all linear algebra wearing different hats. Master this
    and the rest of the curriculum stops feeling like a pile of tricks.
    """),

    # ============================================================ 1. Objectives
    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - Read data, models, and transformations *geometrically* — as vectors, spaces,
      and linear maps, not just grids of numbers.
    - The four ideas that recur everywhere: **dot products** (similarity &
      projection), **matrix multiplication** (composition of transformations),
      **eigendecomposition / SVD** (the principal axes of a transformation), and
      **norms** (how we measure size and write regularizers).
    - Derive the **least-squares normal equations** from scratch and explain *why*
      practitioners almost never solve them as written.
    - Implement dot products, matrix multiply, **power iteration**, least squares,
      and **low-rank SVD compression** in pure NumPy.
    - Diagnose **ill-conditioning** and **multicollinearity** — the silent killers
      of linear models — and know the fixes.

    **Why it matters in industry**
    - Every feature matrix, embedding table, and weight tensor is a linear-algebra
      object. Cost (memory, FLOPs, latency) is dictated by their *shapes and ranks*.
    - SVD/PCA underpin dimensionality reduction, recommender systems, and
      compression — directly tied to **storage and serving cost**.
    - Numerical-stability bugs (a model that's wildly accurate offline and garbage
      in prod) are usually conditioning problems in disguise.

    **Typical interview questions**
    - "What does it mean for a matrix to be rank-deficient, and why should an ML
      engineer care?"
    - "Derive the normal equations. Now tell me why we'd use QR or SVD instead."
    - "Explain SVD geometrically. How is it related to PCA?"
    - "Your linear regression coefficients are huge and flip sign when you add a
      row. What's happening and how do you fix it?"
    - "What's the difference between L1 and L2 norms, and why does L1 produce
      sparse solutions?"
    """),

    # =================================================== 2. Historical Motivation
    md(r"""
    ## 2 · Historical Motivation

    **What came before.** Linear algebra grew out of *solving systems of linear
    equations* — Gaussian elimination is ~2000 years old (Chinese *Nine Chapters*),
    formalized by Gauss in the 1800s. The matrix as a first-class object came later
    (Cayley, Sylvester, 1850s). For a long time it was a tool for **exact** solving:
    given $A\mathbf{x}=\mathbf{b}$, find the unique $\mathbf{x}$.

    **The problem that broke the old view.** Real data is **overdetermined and
    noisy** — more equations than unknowns, no exact solution. Legendre and Gauss
    (c. 1805–1809), working on astronomy/geodesy, asked instead for the
    $\mathbf{x}$ that *minimizes the error* $\lVert A\mathbf{x}-\mathbf{b}\rVert^2$.
    That single shift — from "solve exactly" to "minimize a loss" — is the seed of
    all of supervised learning.

    **Why the modern toolkit was invented.**
    - **Eigendecomposition** answered: along which directions does a transformation
      simply *stretch* without rotating? This is the language of stability,
      vibration, PageRank, and covariance structure.
    - **SVD** (Beltrami/Jordan 1870s; numerically practical via Golub–Kahan, 1965)
      generalized eigendecomposition to *any* matrix, even non-square ones. It is
      the most important matrix factorization in ML: it gives the optimal low-rank
      approximation, powers PCA, latent-factor recommenders, and pseudo-inverses.
    - **Numerically stable algorithms** (QR via Householder, 1958) were invented
      because the mathematically obvious method — form $A^\top A$ and invert it —
      *amplifies floating-point error catastrophically* on real hardware.

    The throughline: linear algebra moved from *exact symbolic solving* to *robust,
    approximate, geometry-aware computation on noisy data* — which is exactly what
    ML needs.
    """),

    # ================================================ 3. Intuition & Visual
    md(r"""
    ## 3 · Intuition & Visual Understanding

    Three mental images carry most of the weight. Build them now; the math later
    just makes them precise.

    **(a) A vector is two things at once.** It is a *point* (a row of your dataset:
    one customer, one image, one document embedding) **and** an *arrow* (a direction
    with a magnitude). The arrow picture is what lets us talk about *similarity* and
    *direction of steepest ascent*.

    **(b) The dot product measures alignment.** $\mathbf{a}\cdot\mathbf{b} =
    \lVert\mathbf a\rVert\,\lVert\mathbf b\rVert\cos\theta$. If you remember one
    formula from this notebook, make it this one. It says: the dot product is large
    when two vectors point the same way, zero when they're perpendicular
    (*orthogonal = unrelated*), negative when opposed. **Cosine similarity** — the
    workhorse of embeddings and semantic search — is literally this with the
    magnitudes divided out.

    **(c) A matrix is a verb, not a noun.** Multiplying by a matrix $A$ *transforms*
    space: it stretches, rotates, shears, and projects. Matrix multiplication $BA$
    is *function composition* — "do $A$, then do $B$." A neural network is a stack
    of these verbs interleaved with nonlinearities.

    ```mermaid
    flowchart LR
        D["Raw data<br/>(points in R^n)"] -->|"matrix A<br/>(a linear map)"| T["Transformed space<br/>(rotated / stretched / projected)"]
        T -->|"SVD reveals"| P["Principal axes<br/>(directions of max variance)"]
        P --> U["Use: PCA · compression<br/>least squares · similarity"]
    ```

    Run the next cells and *look* before reading the math.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    print("NumPy", np.__version__)
    """),

    code(r"""
    # Figure 1 — a vector is an arrow; addition is tip-to-tail.
    fig, ax = plt.subplots()
    origin = np.zeros(2)
    u = np.array([3.0, 1.0])
    v = np.array([1.0, 2.0])
    for vec, color, name in [(u, "tab:blue", "u"),
                             (v, "tab:orange", "v"),
                             (u + v, "tab:green", "u+v")]:
        ax.quiver(*origin, *vec, angles="xy", scale_units="xy", scale=1,
                  color=color, width=0.012)
        ax.annotate(name, vec * 1.03, color=color, fontsize=14, fontweight="bold")
    # show the tip-to-tail construction
    ax.plot([u[0], (u + v)[0]], [u[1], (u + v)[1]], "k--", lw=1, alpha=0.6)
    ax.set_xlim(-1, 5); ax.set_ylim(-1, 4); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Figure 1 — Vectors as arrows; u + v is tip-to-tail")
    plt.show()
    """),

    md(r"""
    **Figure 1.** Each vector is an arrow from the origin. Addition slides $\mathbf
    v$'s tail to $\mathbf u$'s tip — the green arrow is the sum. This is the same
    operation as adding two rows of features elementwise; the geometry just makes
    it visible. Scaling a vector ($2\mathbf u$) stretches the arrow without
    changing its direction.
    """),

    code(r"""
    # Figure 2 — the dot product as projection, and cosine similarity.
    def cosine(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    a = np.array([4.0, 1.0])
    b = np.array([1.0, 3.0])
    # vector projection of a onto b:  proj = (a.b / b.b) * b
    proj = (a @ b) / (b @ b) * b

    fig, ax = plt.subplots()
    ax.quiver(0, 0, *a, angles="xy", scale_units="xy", scale=1, color="tab:blue", width=0.012)
    ax.quiver(0, 0, *b, angles="xy", scale_units="xy", scale=1, color="tab:orange", width=0.012)
    ax.quiver(0, 0, *proj, angles="xy", scale_units="xy", scale=1, color="tab:green", width=0.012)
    ax.plot([a[0], proj[0]], [a[1], proj[1]], "k--", lw=1)   # the "shadow" line
    ax.annotate("a", a * 1.03, color="tab:blue", fontsize=14, fontweight="bold")
    ax.annotate("b", b * 1.03, color="tab:orange", fontsize=14, fontweight="bold")
    ax.annotate("proj of a onto b", proj * 1.05, color="tab:green", fontsize=11)
    ax.set_xlim(-1, 5); ax.set_ylim(-1, 4); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"Figure 2 — projection; cos(theta) = {cosine(a, b):.3f}")
    plt.show()

    print(f"a . b              = {a @ b:.3f}")
    print(f"cosine similarity  = {cosine(a, b):.3f}   (1=identical dir, 0=orthogonal, -1=opposite)")
    """),

    md(r"""
    **Figure 2.** The dashed line drops a perpendicular from the tip of $\mathbf a$
    onto the line through $\mathbf b$; the green arrow is the *shadow* of $\mathbf
    a$ on $\mathbf b$ — the **projection**. The dot product $\mathbf a\cdot\mathbf
    b$ is (length of $\mathbf b$) × (length of that shadow). **Cosine similarity**
    keeps only the angle, discarding magnitudes — which is why it is the default
    similarity for embeddings: we care about *direction* (meaning), not *length*
    (which often just tracks document length or token count). Lesson RAG-01
    (Similarity Search) builds directly on this.
    """),

    # ============================================ 4. Mathematical Foundations
    md(r"""
    ## 4 · Mathematical Foundations

    We build up in the order an engineer actually needs: spaces → maps →
    measurement → the two decompositions that matter (eigen and SVD).

    **Notation bridge.** Capital letters such as $A$ name matrices; bold lowercase
    letters such as $\mathbf x$ name vectors; ordinary lowercase letters usually
    name single numbers. $A^\top$ means transpose (swap rows and columns), a
    subscript such as $x_i$ selects item $i$, $\sum$ means add repeated terms,
    $\mathbb R^n$ means vectors containing $n$ real numbers, and $m\times n$
    means `m rows by n columns`. Lesson PRE-02 introduces these ideas with numbers.

    ### 4.1 Vector spaces, span, independence, basis, rank
    A **vector space** is a set closed under addition and scalar multiplication
    (you can add vectors and scale them and stay inside). The **span** of vectors
    $\{\mathbf v_1,\dots,\mathbf v_k\}$ is every linear combination
    $\sum_i c_i\mathbf v_i$ — the subspace they can "reach."

    **Symbols:** $\mathbf v_i$ is vector number $i$; $c_i$ is the ordinary number
    multiplying that vector; $k$ is the number of available vectors; $\dots$ means
    the pattern continues; $\sum_i$ instructs us to add all $c_i\mathbf v_i$.
    For two vectors, this expands to $c_1\mathbf v_1+c_2\mathbf v_2$.

    Vectors are **linearly independent** if none is a combination of the others —
    no redundancy. The **rank** of a matrix is the dimension of the space its
    columns span = the number of *genuinely independent* directions in your data.

    > **Why an ML engineer cares.** Two perfectly correlated features (e.g.
    > `height_cm` and `height_inches`) are linearly dependent. The feature matrix
    > is **rank-deficient**, $A^\top A$ is singular, and your linear model's
    > coefficients become unidentifiable (infinitely many equally-good answers).
    > This is **multicollinearity** — Section 7.

    ### 4.2 Matrices as linear maps; multiplication as composition
    An $m\times n$ matrix $A$ maps a vector in $\mathbb R^n$ to one in $\mathbb
    R^m$ via $\mathbf y = A\mathbf x$. Linearity means $A(\alpha\mathbf x+\beta\mathbf
    z)=\alpha A\mathbf x+\beta A\mathbf z$ — straight lines stay straight, the
    origin stays put. The columns of $A$ are *where the basis vectors land*.

    **Read and symbols:** “A maps the n-value vector x to the m-value vector y.”
    The Greek letters $\alpha$ (alpha) and $\beta$ (beta) are ordinary scaling
    numbers. The equation says that transforming a scaled sum produces the same
    result as transforming first and then taking the scaled sum.

    Matrix multiplication $C=BA$ is defined precisely so that applying $C$ equals
    "apply $A$, then apply $B$": $C\mathbf x = B(A\mathbf x)$. That is the *reason*
    for the row-times-column rule — it is composition of functions, nothing more.

    ### 4.3 Norms — how we measure size (and write regularizers)
    A **norm** $\lVert\cdot\rVert$ measures vector length. The family of
    $\ell_p$ norms:
    $$\lVert\mathbf x\rVert_p = \Big(\sum_i |x_i|^p\Big)^{1/p}.$$
    **Read aloud:** “the p-norm of x is the p-th root of the sum of each absolute
    component raised to p.” **Symbols:** $x_i$ is component $i$; $|x_i|$ is its
    distance from zero; $p$ selects the kind of length; $1/p$ takes the matching
    root. For $\mathbf x=(3,4)$ and $p=2$, the norm is
    $\sqrt{3^2+4^2}=5$.
    - $p=2$ (**Euclidean**): ordinary length; smooth; the geometry of least squares
      and Ridge ($L_2$) regularization.
    - $p=1$ (**Manhattan**): sum of absolute values; its "corners" on the axes are
      *why* Lasso ($L_1$) produces **sparse** solutions (Section 6 visual).
    - $p=\infty$: $\max_i|x_i|$.

    ### 4.4 Least squares — derive the normal equations
    We want $\mathbf x$ minimizing the squared error
    $J(\mathbf x)=\lVert A\mathbf x-\mathbf b\rVert_2^2=(A\mathbf x-\mathbf b)^\top(A\mathbf x-\mathbf b)$.

    **Read and symbols:** $J$ names the error function; $A\mathbf x$ is the
    prediction; $\mathbf b$ is the target; their difference is the residual;
    superscript $\top$ transposes a vector so the multiplication produces one
    number; superscript `2` squares the length.

    Expand and differentiate (gradient of a quadratic):
    $$J(\mathbf x)=\mathbf x^\top A^\top A\,\mathbf x-2\mathbf b^\top A\,\mathbf x+\mathbf b^\top\mathbf b,
    \qquad \nabla_{\mathbf x}J = 2A^\top A\,\mathbf x-2A^\top\mathbf b.$$
    Set the gradient to zero:
    $$\boxed{A^\top A\,\mathbf x = A^\top\mathbf b}\quad\text{(the \emph{normal equations})}.$$

    **Symbols:** $\nabla_{\mathbf x}J$ is the vector of slopes of $J$ with respect
    to every component of $\mathbf x$; $\mathbf0$ is a vector of zeros; the box
    highlights the equation to remember. Lesson PRE-04 explains gradients as
    multi-direction slopes. Setting the gradient to zero finds a flat candidate
    point; because this squared-error problem is convex, that candidate is the
    global minimum when the system has sufficient rank.

    **Geometric reading (the better intuition).** $A\mathbf x$ can only ever live in
    the *column space* of $A$. The closest reachable point to $\mathbf b$ is its
    **orthogonal projection** onto that subspace; the residual $\mathbf b-A\mathbf
    x$ must be perpendicular to every column of $A$, i.e. $A^\top(\mathbf b-A\mathbf
    x)=\mathbf 0$ — the same equation, derived with a picture instead of calculus.
    This *is* linear regression (Lesson CML-01).

    ### 4.5 Eigenvalues & eigenvectors — invariant directions
    For a square $A$, a nonzero $\mathbf v$ with $A\mathbf v=\lambda\mathbf v$ is an
    **eigenvector**: a direction the map only *stretches* (by $\lambda$), never
    rotates. They come from the characteristic equation $\det(A-\lambda I)=0$.

    **Read and symbols:** “A times v equals lambda times v.” $\lambda$ (lambda) is
    the stretch factor; $I$ is the identity matrix, which leaves vectors unchanged;
    $\det$ is the determinant, a number that is zero when a square matrix collapses
    at least one direction. You do not need to calculate determinants by hand here.
    For **symmetric** matrices (like every covariance matrix $X^\top X$), the
    *spectral theorem* guarantees real eigenvalues and an orthonormal eigenbasis —
    the mathematical bedrock of PCA.

    ### 4.6 Singular Value Decomposition — the master factorization
    Every $m\times n$ matrix factors as
    $$A = U\,\Sigma\,V^\top,$$
    with $U$ ($m\times m$) and $V$ ($n\times n$) **orthogonal** (rotations) and
    $\Sigma$ diagonal with non-negative **singular values** $\sigma_1\ge\sigma_2\ge\dots\ge 0$.
    **Read and symbols:** “A equals U times Sigma times V transpose.” $U$ and $V$
    describe rotations or reflections; capital $\Sigma$ (Sigma) is a diagonal
    matrix of stretch amounts; lowercase $\sigma_i$ is stretch amount $i$; $\ge$
    means “greater than or equal to,” so the values are ordered largest first.
    Geometrically: *every linear map is a rotation, then an axis-aligned scaling,
    then another rotation.* The columns of $V$ are the input directions that get
    stretched the most; $U$ are where they land; $\sigma_i$ are the stretch factors.

    The singular values are the square roots of the eigenvalues of $A^\top A$, so
    SVD is "eigendecomposition that works for any matrix." Its killer property is
    the **Eckart–Young theorem**: truncating to the top $k$ singular triplets gives
    the *best possible* rank-$k$ approximation of $A$ — the foundation of PCA,
    compression, and latent-factor models (Section 6).
    """),

    code(r"""
    # Figure 3 — a matrix is a verb. Watch A bend space and reveal its singular axes.
    A = np.array([[2.0, 1.0],
                  [0.5, 1.5]])

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(theta), np.sin(theta)])   # unit circle
    ellipse = A @ circle                                  # its image under A

    U, S, Vt = np.linalg.svd(A)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(circle[0], circle[1], color="tab:blue")
    # right-singular vectors V: the input directions that get stretched most
    for i in range(2):
        axes[0].quiver(0, 0, Vt[i, 0], Vt[i, 1], angles="xy", scale_units="xy",
                       scale=1, color="tab:red", width=0.012)
    axes[0].set_title("Input: unit circle + V directions"); axes[0].set_aspect("equal")
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)

    axes[1].plot(ellipse[0], ellipse[1], color="tab:green")
    # axes of the ellipse = sigma_i * U_i
    for i in range(2):
        axes[1].quiver(0, 0, S[i] * U[0, i], S[i] * U[1, i], angles="xy",
                       scale_units="xy", scale=1, color="tab:red", width=0.012)
    axes[1].set_title(f"Output: ellipse; singular values = {S.round(3)}")
    axes[1].set_aspect("equal"); axes[1].set_xlim(-3, 3); axes[1].set_ylim(-3, 3)
    plt.suptitle("Figure 3 — A maps the unit circle to an ellipse; SVD gives its axes")
    plt.show()
    """),

    md(r"""
    **Figure 3.** The unit circle (left) is squashed and rotated into an *ellipse*
    (right) — that is what "a matrix transforms space" means, concretely. The red
    arrows are the singular directions: the longest ellipse axis has length
    $\sigma_1$, the shortest $\sigma_2$. If $\sigma_2$ were ~0 the ellipse would
    collapse to a line — the matrix would be (nearly) **rank-deficient**, and the
    ratio $\sigma_1/\sigma_2$ (the **condition number**) would blow up. Hold onto
    that: it is the single most important number for numerical stability (Section 7).
    """),

    code(r"""
    # Figure 4 — why L1 is sparse and L2 is smooth: the unit balls have different shapes.
    xs = np.linspace(-1.6, 1.6, 400)
    X, Y = np.meshgrid(xs, xs)
    fig, ax = plt.subplots(figsize=(6, 6))
    for p, color in [(1, "tab:orange"), (2, "tab:blue"), (4, "tab:green")]:
        Z = (np.abs(X) ** p + np.abs(Y) ** p) ** (1 / p)
        ax.contour(X, Y, Z, levels=[1.0], colors=[color])
    Zinf = np.maximum(np.abs(X), np.abs(Y))            # L-infinity
    ax.contour(X, Y, Zinf, levels=[1.0], colors=["tab:red"])
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=c, label=l) for c, l in
                       [("tab:orange", "L1 (diamond)"), ("tab:blue", "L2 (circle)"),
                        ("tab:green", "L4"), ("tab:red", "L-inf (square)")]])
    ax.set_aspect("equal"); ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Figure 4 — unit balls {x : ||x||_p = 1}")
    plt.show()
    """),

    md(r"""
    **Figure 4.** The set of vectors with norm exactly 1 looks different for each
    $p$. The $L_1$ "ball" is a **diamond with sharp corners on the axes**. When an
    optimizer pushes a solution toward this ball (Lasso), it tends to land *on a
    corner* — where some coordinates are exactly **zero**. That is the geometric
    reason $L_1$ regularization yields sparse, feature-selecting models, while the
    smooth round $L_2$ ball (Ridge) just shrinks coefficients toward zero without
    zeroing them. You'll meet both again in CML-01, CML-02, and MLE-03.
    """),

    # ============================================ 5. Scratch implementation
    md(r"""
    ## 5 · Manual Implementation from Scratch

    The point is not to beat NumPy — it is to prove to yourself that these
    operations are *not magic*. We implement four building blocks and verify each
    against NumPy: dot product, matrix multiply, **power iteration** (top
    eigenpair), and **least squares**. Pure Python/NumPy primitives only.
    """),

    code(r"""
    # 5.1 Dot product, matmul, and L2 norm from first principles (explicit loops).
    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def matmul(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        m, n = A.shape
        n2, p = B.shape
        assert n == n2, "inner dimensions must match"
        C = np.zeros((m, p))
        for i in range(m):
            for j in range(p):
                C[i, j] = dot(A[i, :], B[:, j])   # row i of A · column j of B
        return C

    def l2(x):
        return dot(x, x) ** 0.5

    a = np.array([1.0, 2.0, 3.0]); b = np.array([4.0, 5.0, 6.0])
    A = np.random.randn(4, 3); B = np.random.randn(3, 5)
    print("dot  matches numpy:", np.allclose(dot(a, b), a @ b))
    print("matmul matches numpy:", np.allclose(matmul(A, B), A @ B))
    print("l2   matches numpy:", np.allclose(l2(a), np.linalg.norm(a)))
    """),

    code(r"""
    # 5.2 Power iteration: find the dominant eigenpair of a symmetric matrix.
    # Intuition: repeatedly applying A amplifies the component along the largest
    # eigenvector fastest, so a random vector "rotates" toward it. Renormalize each
    # step to avoid overflow/underflow.
    def power_iteration(A, iters=2000, tol=1e-12):
        n = A.shape[0]
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        lam_old = 0.0
        for _ in range(iters):
            w = A @ v
            v = w / np.linalg.norm(w)
            lam = v @ A @ v                       # Rayleigh quotient = eigenvalue estimate
            if abs(lam - lam_old) < tol:
                break
            lam_old = lam
        return lam, v

    M = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    lam, vec = power_iteration(M)
    w, V = np.linalg.eigh(M)                       # ground truth (ascending order)
    print(f"power-iteration eigenvalue : {lam:.6f}")
    print(f"numpy largest eigenvalue   : {w[-1]:.6f}")
    print(f"eigenvector alignment |cos|: {abs(vec @ V[:, -1]):.6f}  (1.0 = same direction)")
    """),

    code(r"""
    # 5.3 Least squares from scratch via the normal equations we derived in 4.4.
    def lstsq_normal(A, b):
        return np.linalg.solve(A.T @ A, A.T @ b)   # solve, never explicit inverse

    # synthetic linear data: y = 2 + 3*x1 - 1.5*x2 + noise
    n = 200
    X = np.random.randn(n, 2)
    Xb = np.hstack([np.ones((n, 1)), X])           # prepend bias column
    true_w = np.array([2.0, 3.0, -1.5])
    y = Xb @ true_w + 0.1 * np.random.randn(n)

    w_scratch = lstsq_normal(Xb, y)
    w_numpy = np.linalg.lstsq(Xb, y, rcond=None)[0]
    print("scratch :", w_scratch.round(3))
    print("numpy   :", w_numpy.round(3))
    print("true    :", true_w)
    print("agree   :", np.allclose(w_scratch, w_numpy, atol=1e-6))
    """),

    # ============================================ 6. Visualization (SVD)
    md(r"""
    ## 6 · Visualization — Low-Rank Structure via SVD

    Here is the single most useful linear-algebra demo for ML: a matrix often
    carries most of its information in a *few* singular directions. Truncating to
    the top $k$ (Eckart–Young) gives the best rank-$k$ approximation — this is
    exactly **PCA** on centered data and the engine behind compression and
    latent-factor recommenders. We compress a structured "image" matrix.
    """),

    code(r"""
    # Build a structured 128x128 matrix (an "image") with clear low-rank structure.
    t = np.linspace(-3, 3, 128)
    Xg, Yg = np.meshgrid(t, t)
    img = (np.sin(Xg) * np.cos(Yg) + 0.6 * np.cos(2 * Xg) - 0.4 * Yg)
    img += 0.05 * np.random.randn(*img.shape)       # a little high-rank "noise"

    U, S, Vt = np.linalg.svd(img, full_matrices=False)

    def rank_k(k):
        return (U[:, :k] * S[:k]) @ Vt[:k, :]

    ks = [1, 3, 8, 30, 128]
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(16, 3))
    axes[0].imshow(img, cmap="viridis"); axes[0].set_title("original"); axes[0].axis("off")
    full = 128 * 128
    for ax, k in zip(axes[1:], ks):
        approx = rank_k(k)
        stored = k * (128 + 128 + 1)               # U_k, V_k, S_k entries
        err = np.linalg.norm(img - approx) / np.linalg.norm(img)
        ax.imshow(approx, cmap="viridis")
        ax.set_title(f"k={k}\n{stored/full:.0%} storage\nerr {err:.1%}")
        ax.axis("off")
    plt.suptitle("Figure 5 — SVD low-rank approximation: most signal lives in a few components")
    plt.show()
    """),

    code(r"""
    # Figure 6 — the singular-value spectrum explains *why* compression works.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].semilogy(S, "o-", ms=3)
    ax[0].set_title("Figure 6a — singular values (log scale)")
    ax[0].set_xlabel("index i"); ax[0].set_ylabel("sigma_i")
    energy = np.cumsum(S**2) / np.sum(S**2)
    ax[1].plot(energy, "o-", ms=3)
    ax[1].axhline(0.99, color="r", ls="--", label="99% energy")
    ax[1].set_title("Figure 6b — cumulative energy captured")
    ax[1].set_xlabel("rank k"); ax[1].set_ylabel("fraction of variance"); ax[1].legend()
    plt.show()
    k99 = int(np.searchsorted(energy, 0.99) + 1)
    print(f"Only {k99} of 128 components capture 99% of the matrix's energy.")
    """),

    md(r"""
    **Figures 5–6.** With $k=8$ components the reconstruction is visually
    indistinguishable from the original while storing a small fraction of the
    numbers. The singular-value spectrum (6a) drops off fast — a *steep spectrum
    means the data is effectively low-dimensional*. The cumulative-energy curve
    (6b) tells you exactly how many components to keep for a target fidelity. This
    is the entire idea of **PCA** (keep top-$k$ directions of variance) and of
    **matrix-factorization recommenders** (users × items ≈ low-rank user-factors ×
    item-factors). The noise we added is what lives in the long flat tail — and
    *dropping it is denoising*.
    """),

    # ============================================ 7. Failure Modes
    md(r"""
    ## 7 · Failure Modes

    Linear algebra fails quietly. The model trains, metrics look plausible offline,
    and then coefficients are nonsense or predictions are unstable in production.

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Multicollinearity** | Huge / sign-flipping coefficients; they swing wildly when you add a row | Near-dependent feature columns ⇒ $A^\top A$ nearly singular | Drop/merge correlated features; **Ridge** ($L_2$) regularization; PCA |
    | **Ill-conditioning** | Tiny input change ⇒ large output change; loss of precision | Large condition number $\sigma_{\max}/\sigma_{\min}$ | Center & **scale** features; solve via **QR/SVD**, not normal equations |
    | **Normal-equations instability** | `lstsq` is fine but `solve(A.T@A, …)` is garbage | Forming $A^\top A$ **squares** the condition number | Never form $A^\top A$; use QR or SVD |
    | **Rank deficiency** | Non-unique solution; `inv` raises / returns junk | $\text{rank}(A) < n$ | Pseudo-inverse / truncated SVD; regularize |

    The next cell makes the "normal equations square the condition number" failure
    *quantitative* using the classic ill-conditioned **Hilbert matrix**.
    """),

    code(r"""
    # Demonstrate: solving via A^T A amplifies error catastrophically.
    n = 12
    idx = np.arange(1, n + 1)
    H = 1.0 / (idx[:, None] + idx[None, :] - 1)     # Hilbert matrix: famously ill-conditioned
    x_true = np.ones(n)
    b = H @ x_true

    x_solve = np.linalg.solve(H, b)                       # direct solve
    x_normal = np.linalg.solve(H.T @ H, H.T @ b)          # via normal equations (bad!)
    x_lstsq = np.linalg.lstsq(H, b, rcond=None)[0]        # SVD-based (good)

    def relerr(x):
        return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    print(f"condition number of H        : {np.linalg.cond(H):.3e}")
    print(f"condition number of H^T H    : {np.linalg.cond(H.T @ H):.3e}   (~ squared!)")
    print()
    print(f"rel. error, direct solve     : {relerr(x_solve):.3e}")
    print(f"rel. error, normal equations : {relerr(x_normal):.3e}   <-- catastrophic")
    print(f"rel. error, lstsq (SVD)      : {relerr(x_lstsq):.3e}   <-- best")
    """),

    md(r"""
    **What you just saw.** $H$ is already ill-conditioned ($\kappa\sim10^{16}$), and
    forming $H^\top H$ *squares* that condition number, pushing the problem past the
    precision of 64-bit floats — so the normal-equations answer can be off by
    orders of magnitude even though the math is "correct." The SVD-based `lstsq` is
    dramatically more accurate on the same data.

    **The senior takeaway:** the textbook formula $\mathbf x=(A^\top A)^{-1}A^\top
    \mathbf b$ is for *derivations*, not *code*. In production you call a routine
    backed by QR or SVD, you **center and scale** your features (which shrinks the
    condition number), and you **regularize**. Ridge adds $\lambda I$ to $A^\top A$,
    which directly lifts the smallest singular values away from zero — fixing
    conditioning *and* overfitting at once.
    """),

    # ============================================ 8. Production Library
    md(r"""
    ## 8 · Production Library Implementation

    NumPy/SciPy don't reimplement the math in Python — they dispatch to **BLAS**
    (Basic Linear Algebra Subprograms) and **LAPACK**, decades-old, heavily tuned
    Fortran/C kernels that exploit cache hierarchies, SIMD, and multiple cores. A
    `A @ B` you write becomes a multi-threaded, blocked `dgemm` call. This is why
    "just write a loop" (our Section 5) is for *learning*, never for *serving*.

    Rules of thumb for **which routine to call**:
    - Need to *solve* $A\mathbf x=\mathbf b$, $A$ square & well-conditioned →
      `np.linalg.solve` (LU). **Never** `inv(A) @ b` — it's slower and less stable.
    - *Overdetermined / least squares* → `np.linalg.lstsq` or `scipy.linalg.lstsq`
      (SVD), or QR for speed. Not the normal equations.
    - *Symmetric / covariance* → `np.linalg.eigh` (faster & stabler than `eig`).
    - *Any factorization for dimensionality reduction* → `np.linalg.svd`, or
      `sklearn.decomposition.TruncatedSVD` / randomized SVD for large, low-rank-ish
      data (you only want the top $k$).
    - *Large & sparse* (text TF-IDF, graphs) → `scipy.sparse` + iterative solvers
      (`scipy.sparse.linalg.cg`, `svds`). Forming a dense $A^\top A$ would OOM.
    """),

    code(r"""
    # Production idioms, all NumPy (SciPy mirrors these with more options).
    A = np.random.randn(500, 500)
    b = np.random.randn(500)

    x1 = np.linalg.solve(A, b)            # GOOD: LU-based solve
    x2 = np.linalg.inv(A) @ b             # BAD pattern: explicit inverse
    print("solve vs inv agree:", np.allclose(x1, x2), "(but solve is faster + stabler)")

    # Least squares the right way:
    M = np.random.randn(1000, 20)
    y = np.random.randn(1000)
    coef, residuals, rank, sv = np.linalg.lstsq(M, y, rcond=None)
    print("lstsq solution shape:", coef.shape, "| design-matrix rank:", rank)

    # QR factorization (the workhorse behind stable least squares):
    Q, R = np.linalg.qr(M)
    coef_qr = np.linalg.solve(R, Q.T @ y)
    print("QR matches lstsq:", np.allclose(coef, coef_qr, atol=1e-8))
    """),

    md(r"""
    **Scratch vs production — what the library buys you:** (1) *speed* — BLAS is
    100–1000× faster than Python loops via blocking, SIMD, and threads; (2)
    *stability* — pivoting, QR, and SVD avoid the error amplification we saw in
    Section 7; (3) *memory* — sparse formats and out-of-core/randomized methods
    handle matrices that don't fit in RAM; (4) *correctness edge cases* — graceful
    handling of rank deficiency via the pseudo-inverse. Your job as a senior
    engineer is to *pick the right routine and prepare the data*, not to reinvent
    `dgemm`.
    """),

    # ============================================ 9. Business Case Study
    md(r"""
    ## 9 · Realistic Business Case Study — Compressing an Embedding Store

    **Scenario.** A B2C company runs semantic search over a catalog of **10 million
    products**. Each product has a **768-dimensional** float32 embedding. The store
    is $10^7 \times 768$ float32 ≈ **30 GB**, held in RAM across the serving fleet
    for low-latency nearest-neighbor search.

    **Business objectives**
    - Cut serving cost (RAM is the dominant line item).
    - Keep p99 query latency < 50 ms and search relevance (recall@10) within 1pt.

    **Cost of mistakes**
    - Over-compress → relevance drops → fewer conversions (direct revenue loss).
    - Under-compress → RAM cost stays high; fleet can't scale at peak.

    **Constraints**: fixed embedding model; nightly rebuild budget; must support
    incremental inserts.

    **Linear-algebra solution.** The 768 dimensions are *not* independent — learned
    embeddings live near a lower-dimensional manifold. Run (truncated) **SVD/PCA**
    on a sample of embeddings; the singular-value spectrum (exactly Figure 6) shows,
    say, 256 components capture 99% of the variance. Project all vectors to 256-d:

    - **Storage:** 30 GB → 10 GB (3× cheaper RAM).
    - **Latency:** distance computations are now in 256-d → ~3× fewer FLOPs/query.
    - **Relevance:** measured drop < 1pt because we discarded near-noise directions.

    **KPIs to watch:** recall@10 vs the uncompressed baseline, p99 latency, RAM/host,
    and *spectrum drift* over time (if new products expand the intrinsic dimension,
    256 may stop being enough — that's a retraining signal). This same math is PCA,
    and it returns in MLE-05, NLP-02, and RAG-01.
    """),

    # ============================================ 10. Production Considerations
    md(r"""
    ## 10 · Production Considerations

    - **Latency / throughput.** Cost scales with matrix *shape* and *rank*. A
      $d$-dim dot product is $O(d)$; reducing $d$ via PCA is a direct latency win.
      Batch operations into matrix–matrix products to saturate BLAS.
    - **Memory.** Dense $n\times n$ is $O(n^2)$; prefer sparse representations for
      text/graph data and *never* materialize $A^\top A$ for tall-skinny $A$.
    - **Numerical stability / monitoring.** Log the **condition number** and
      **singular-value spectrum** of key matrices. A rising condition number or a
      collapsing smallest singular value is an early warning of multicollinearity
      or degenerate features *before* metrics visibly break.
    - **Drift & retraining.** PCA/SVD bases are fit on a data snapshot. As the data
      distribution shifts, the retained components capture less variance — monitor
      *explained variance* and refit on a schedule.
    - **Cost.** Low-rank approximations trade a small accuracy loss for large RAM,
      bandwidth, and FLOP savings — usually the highest-leverage optimization in a
      vector-search or recommender system.
    - **Reproducibility.** SVD sign and the ordering of equal singular values are
      not unique; pin conventions (and random seeds for *randomized* SVD) so a
      rebuilt index matches the previous one.
    """),

    # ============================================ 11. Tradeoff Analysis
    md(r"""
    ## 11 · Tradeoff Analysis

    **Choosing a method to solve least squares $\min\lVert A\mathbf x-\mathbf b\rVert$:**

    | Dimension | Normal equations $(A^\top A)^{-1}A^\top b$ | QR factorization | SVD (`lstsq`) | Iterative (CG/LSQR) |
    |---|---|---|---|---|
    | Speed (dense) | Fastest *to write* | Fast | Slower | Fast for sparse/huge |
    | Numerical stability | **Poor** (squares $\kappa$) | Good | **Best** | Good |
    | Handles rank-deficiency | No | Partially | **Yes** (pseudo-inverse) | With care |
    | Memory | Forms $A^\top A$ ($O(n^2)$) | $O(mn)$ | $O(mn)$ | **Lowest** (matrix-free) |
    | When to use | Derivations / tiny well-conditioned | Default dense LS | Ill-conditioned / need rank | Massive sparse systems |

    **Dimensionality reduction — full vs truncated/randomized SVD:**

    | Dimension | Full SVD | Truncated / randomized SVD |
    |---|---|---|
    | Cost | $O(mn\min(m,n))$ | $O(mnk)$ — only top $k$ |
    | Memory | All factors | Only $k$ components |
    | Accuracy | Exact | Excellent if spectrum decays |
    | Use case | Small matrices, analysis | Large embedding/recommender matrices |

    **The meta-lesson:** the "obviously simplest" formula is rarely the production
    choice. Senior judgement is matching the *numerical method* to the *conditioning
    and scale* of the data.
    """),

    # ============================================ 12. Interview Prep
    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *What is the rank of a matrix and why does it matter for ML?* → Number of
      independent directions; low rank ⇒ redundant features ⇒ unidentifiable models
      and unstable solutions.
    - *Dot product vs cosine similarity — when each?* → Cosine when magnitude is a
      nuisance (text/embedding length); dot product when magnitude carries signal
      (e.g. confidence-weighted vectors, MIPS recommenders).

    **Deep-dive questions**
    - *Derive the normal equations two ways (calculus and projection).* (Section 4.4)
    - *Explain SVD geometrically and relate it to PCA.* → Rotation–scale–rotation;
      PCA = SVD of the centered data matrix; principal components = right-singular
      vectors; explained variance ∝ $\sigma_i^2$.
    - *Why does L1 give sparsity?* → The diamond unit ball has corners on the axes
      (Figure 4).

    **Whiteboard questions**
    - "Code power iteration for the top eigenvector." (Section 5.2)
    - "Given a tall design matrix, write numerically stable least squares." → QR:
      `Q,R=qr(A); x=solve(R, Q.T@b)`. Bonus points for *not* writing the normal
      equations and explaining why.

    **Strong vs weak answers**
    - *"Why not use `inv(A)@b`?"*
      - **Weak:** "It works, I always use it."
      - **Strong:** "`solve` is faster and more stable; explicit inverses
        accumulate error and waste FLOPs. For least squares I'd go further and use
        QR/SVD because the normal equations square the condition number."
    - *"What's the condition number?"*
      - **Weak:** "How invertible a matrix is."
      - **Strong:** "$\sigma_{\max}/\sigma_{\min}$ — the worst-case amplification of
        relative input error to output error. Large $\kappa$ means small data
        perturbations cause large coefficient swings; I'd scale features and
        regularize."

    **Follow-ups they'll push on:** "Now it's 100M rows and sparse — what changes?"
    (iterative/matrix-free, sparse formats). "How do you pick $k$ for PCA?"
    (cumulative explained variance / a downstream metric, not a round number).

    **Common mistakes:** confusing eigen- and singular values; claiming every matrix
    has an eigendecomposition (only diagonalizable squares do — SVD always exists);
    forgetting to **center** before PCA; reporting a model is "broken" when it's
    actually an unscaled, ill-conditioned design.
    """),

    # ============================================ 13. Teach-Back
    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    Say these out loud, as if to an interviewer. If you stall, that's the section
    to reread.

    1. **What is it?** What are vectors, matrices, rank, eigenvectors, and SVD —
       in one sentence each, geometrically.
    2. **Why was it invented?** Why did linear algebra move from "solve exactly" to
       "minimize error," and why was SVD a breakthrough over eigendecomposition?
    3. **How does it work?** Walk through the SVD picture: rotate → scale → rotate.
    4. **Why does it work?** Why is the top-$k$ SVD the *best* rank-$k$
       approximation, and why does that enable compression/PCA?
    5. **When to use it?** Name three ML places dot products / SVD show up.
    6. **When NOT to use it?** When are the normal equations the wrong tool?
    7. **Tradeoffs?** Normal equations vs QR vs SVD vs iterative.
    8. **How would you productionize it?** What do you monitor, and how do you keep
       a least-squares / PCA pipeline numerically stable at scale?
    """),

    # ============================================ 14. Exercises
    md(r"""
    ## 14 · Exercises

    **Estimated time:** 90–150 minutes. Questions 1–2 are guided checks;
    Questions 3–6 are implementation practice; Questions 7–9 use a scoring rubric.

    **Beginner (conceptual)**
    1. By hand, compute $\mathbf a\cdot\mathbf b$ and $\cos\theta$ for $\mathbf
       a=(1,2)$, $\mathbf b=(2,-1)$. What does the result tell you about the angle?
    2. Give a real ML example each of two linearly *dependent* and two linearly
       *independent* features.

    **Beginner → Intermediate (coding)**
    3. Extend `power_iteration` to find the *second* eigenvector (hint: *deflation* —
       subtract the projection onto the first eigenvector each step).
    4. Implement Ridge regression from scratch: solve $(A^\top A+\lambda I)\mathbf
       x=A^\top\mathbf b$. Plot coefficient norm vs $\lambda$ (the "shrinkage path").

    **Intermediate (coding + analysis)**
    5. Take any grayscale image you like, run SVD, and plot reconstruction error vs
       $k$ and vs storage. At what compression ratio does it become visibly lossy?
    6. Construct a design matrix with two nearly-collinear columns. Show how the
       fitted coefficients explode as collinearity increases, and how Ridge tames
       them. Plot it.

    **Senior (interview + production design)**
    7. *Whiteboard:* derive the normal equations from the projection argument (no
       calculus), then explain to a junior why you'd ship QR instead.
    8. *Design:* you must run nearest-neighbor search over 100M 1024-d embeddings
       under a 20 ms p99 budget and a fixed RAM envelope. Propose a linear-algebra
       compression strategy, state how you'd choose the target dimension, and list
       the metrics that would tell you it's safe to ship and when to refit.
    9. *Stability:* a teammate's linear model has coefficients in the millions and
       flips sign across retrains. Diagnose it from first principles and give three
       concrete fixes, ordered by what you'd try first.

    <details>
    <summary><strong>Hints, expected results, and scoring rubric</strong></summary>

    1. $1\times2+2\times(-1)=0$, so the vectors are perpendicular and cosine is
       zero. Show the multiply-then-add step for full credit.
    2. Dependent examples include centimetres/inches or total and duplicated total;
       independent examples measure genuinely different properties. Explain why.
    3. For a symmetric matrix, orthogonalize each iterate against the first
       eigenvector before normalizing. Check that the two returned vectors have dot
       product approximately zero.
    4. Coefficient norm should generally shrink as $\lambda$ grows. Do not
       regularize an intercept column unless that choice is intentional.
    5. Reconstruction error falls as $k$ grows while storage rises. Report both axes
       and identify the knee using an explicit criterion.
    6. As two columns become closer, unregularized coefficients should become less
       stable; Ridge should reduce their magnitude and retrain-to-retrain variation.
    7. Award 2 points for the projection argument, 1 for obtaining the normal
       equations, and 2 for explaining conditioning and why QR avoids forming
       $A^\top A$.
    8. Award one point each for dimension reduction method, offline recall metric,
       latency/RAM estimate, target-dimension selection, and drift/refit trigger.
    9. Full credit identifies multicollinearity or scaling, verifies it with singular
       values/condition number, then proposes scaling, feature removal/grouping, and
       regularization in a justified order.

    A score of 12/15 across Questions 7–9 indicates senior-level readiness; lower
    scores identify which production reasoning to revisit.
    </details>
    """),

    # ---------------------------------------------------------------- Footer
    md(r"""
    ---
    ### Summary
    You now own the four ideas that recur through the entire curriculum —
    **dot products** (similarity/projection), **matrix multiply** (composition),
    **eigen/SVD** (principal axes & optimal low-rank structure), and **norms**
    (size & regularization) — plus the engineering reflexes (condition number,
    QR/SVD over normal equations, center-and-scale) that keep linear models stable
    in production.

    **Related lesson:** `FND-02 · Probability and Statistics` — the other half of ML's
    mathematical substrate: uncertainty, likelihood, and the reasoning that turns
    these vectors and matrices into *learning*.
    """),
]

build("01_ml_foundations/01_linear_algebra_essentials.ipynb", cells)
