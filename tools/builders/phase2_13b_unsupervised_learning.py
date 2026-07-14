"""Builder for Notebook 13B — Unsupervised Learning Foundations."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # 13B · Unsupervised Learning Foundations
    ### Classical ML completion — structure without labelled targets

    **Prerequisites:** 01–02, 10–11, and 36. **Estimated time:** 6–8 hours.
    Supervised learning asks whether predictions match labels. Here labels are absent,
    so assumptions, stability, geometry, and downstream usefulness become central.
    """),
    md(r"""
    ## 1 · Learning Objectives

    Standardize data safely; use PCA/SVD for lower-dimensional representation;
    explain variance retained and reconstruction error; implement k-means; distinguish
    clustering from classification; evaluate clusters without pretending they are
    ground truth; and use anomaly scores with explicit review thresholds.
    """),
    md(r"""
    ## 2 · Historical Motivation

    Many datasets contain measurements but no reliable labels. Dimensionality
    reduction summarizes correlated variables, clustering proposes groups for
    investigation, and anomaly detection prioritizes unusual cases. These techniques
    are exploratory tools, not automatic discoveries of real categories.
    """),
    md(r"""
    ## 3 · Intuition and Visual Understanding

    PCA rotates the coordinate system toward directions with the most variation.
    K-means alternates between assigning points to the nearest center and moving each
    center to its assigned mean. Anomaly detection ranks unusualness; a threshold
    turns ranking into an operational review decision.
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    PCA chooses a unit vector $v$ maximizing projected variance:
    $$v_1=\arg\max_{\lVert v\rVert=1}\operatorname{Var}(Xv).$$
    $X\in\mathbb R^{n\times d}$ is centered data and $Xv$ is one score per row.
    For two perfectly correlated standardized features, the first direction is
    proportional to `(1,1)` and captures nearly all variance. PCA is scale-sensitive
    and high variance does not necessarily mean high task value.

    K-means minimizes $\sum_i\lVert x_i-\mu_{c_i}\rVert^2$, where $c_i$ assigns row
    $i$ to center $\mu$. It prefers roughly spherical clusters under Euclidean distance.
    """),
    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.ensemble import IsolationForest

    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    projection = PCA(n_components=2, random_state=42).fit_transform(X)
    print("shape", X.shape, "projected", projection.shape)
    """),
    md(r"""
    ## 5 · Manual Implementation from Scratch

    K-means needs an initialization, assignment step, update step, and stopping rule.
    Run multiple initializations because the objective is non-convex.
    """),
    code(r"""
    def kmeans(X, k, seed=42, max_iter=100):
        rng = np.random.default_rng(seed)
        centers = X[rng.choice(len(X), size=k, replace=False)].copy()
        for _ in range(max_iter):
            labels = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            updated = np.vstack([X[labels == j].mean(axis=0) for j in range(k)])
            if np.allclose(updated, centers):
                break
            centers = updated
        return labels, centers

    labels, centers = kmeans(projection, 3)
    assert labels.shape == (len(X),)
    """),
    md(r"""
    ## 6 · Visualization

    A two-dimensional PCA plot is a projection, not proof that clusters exist in the
    original space. Use it to form questions, then test stability and domain meaning.
    """),
    code(r"""
    plt.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", s=25)
    plt.xlabel("principal component 1"); plt.ylabel("principal component 2")
    plt.title("K-means assignments shown in a PCA projection")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Fitting scaling/PCA on all rows before an evaluated downstream split.
    - Calling a cluster a customer “type” without external evidence.
    - Selecting k only because one plot looks attractive.
    - Using Euclidean distance on incomparable or categorical features.
    - Treating anomaly score as fraud probability.
    - Ignoring instability across seeds and samples.
    """),
    md(r"""
    ## 8 · Library Implementation

    Compare k values using inertia, silhouette, stability, and usefulness—not one
    metric alone. Preserve the preprocessing and fitted representation together.
    """),
    code(r"""
    for k in range(2, 6):
        candidate = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
        print(k, round(candidate.inertia_, 1), round(silhouette_score(X, candidate.labels_), 3))
    detector = IsolationForest(contamination=0.05, random_state=42).fit(X)
    scores = -detector.score_samples(X)
    print("top anomaly row indices", np.argsort(scores)[-5:][::-1])
    """),
    md(r"""
    ## 9 · Realistic Case Study

    A laboratory uses PCA to identify redundant measurements, clustering to propose
    review cohorts, and anomaly scores to prioritize instrument-quality checks. Domain
    experts review all interpretations; no cluster label becomes a decision target
    without validation.
    """),
    md(r"""
    ## 10 · Production and Learning Considerations

    Version preprocessing, component loadings, centers, thresholds, and feature
    schema. Monitor reconstruction error, assignment distance, cluster-size drift,
    and review yield. Refit only after checking that changed structure is real.
    """),
    md(r"""
    ## 11 · Tradeoff Analysis

    PCA is efficient and interpretable through loadings but linear. K-means scales
    well but assumes Euclidean geometry. Density methods find irregular clusters but
    are sensitive to scale and density. Isolation Forest ranks anomalies efficiently
    but does not explain their operational significance.
    """),
    md(r"""
    ## 12 · Readiness and Interview Preparation

    Explain why clustering has no ordinary accuracy, why scaling changes k-means,
    how PCA differs from feature selection, and how you would determine whether an
    anomaly detector saves reviewer time.
    """),
    md(r"""
    ## 13 · Teach-Back

    Explain PCA without saying “it compresses data,” then explain why a colorful
    cluster plot is not evidence of natural categories. Give one safe and one unsafe
    operational use of anomaly scores.
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Worked:** standardize two differently scaled features and compare distances.
    **Guided (30 min):** reconstruct standardized wine data from 2, 5, and 10 PCA
    components; plot reconstruction error. **Self-check:** error cannot increase as
    components are added. **Independent (45 min):** compare k=2…6 across five seeds
    using silhouette and assignment stability. **Challenge (60 min):** set an anomaly
    review threshold from a fixed review budget and report review yield assumptions.

    <details><summary><strong>Solution and scoring rubric</strong></summary>
    Use `inverse_transform` for reconstruction; report mean squared reconstruction
    error. Stability requires matching or pairwise co-assignment rather than comparing
    arbitrary cluster numbers. Award 3 points for preprocessing/reconstruction, 3 for
    multi-seed evaluation, 2 for threshold reasoning, and 2 for limitations.
    Common mistakes: interpreting cluster IDs as ordered labels and fitting on future
    evaluation data. **Readiness threshold: 8/10.**
    </details>
    """),
]

build("phase2_ml_engineering/13b_unsupervised_learning.ipynb", cells)
