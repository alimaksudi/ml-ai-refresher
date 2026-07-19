"""Build MLE-06: beginner-first unsupervised learning foundations."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # MLE-06 · Unsupervised Learning Foundations

    **Prerequisites:** FND-01, FND-02, MLE-02, MLE-03, and EVAL-01  
    **Estimated study time:** 10–12 hours, including practice  
    **Next lesson:** DL-01 · PyTorch Foundations

    Supervised learning receives a target and asks whether predictions match it.
    Unsupervised learning receives measurements without a trusted target. The algorithm
    can find geometric structure, but geometry alone cannot tell us whether a group is
    real, useful, stable, or safe to act on.

    This lesson treats PCA, clustering, and anomaly detection as tools for forming and
    testing hypotheses—not machines that discover truth automatically.

    ### Scope boundary

    We focus on numerical tabular data. We compare K-means, hierarchical clustering,
    DBSCAN, and Isolation Forest. Mixed-data clustering, spectral clustering, Gaussian
    mixtures, UMAP, t-SNE, autoencoders, and deep anomaly detection are deferred.

    A colorful plot is not validation. A cluster ID is not a class label. An anomaly
    score is not a probability.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - define the operational question before choosing an unsupervised method;
    - explain how feature meaning, missingness, and scale define geometry;
    - calculate standardized values and Euclidean distance manually;
    - explain why distances become less informative in high dimensions;
    - center data and calculate covariance, eigenvectors, PCA scores, and reconstruction;
    - interpret explained variance and loadings without calling them importance;
    - implement guarded K-means with convergence history and empty-cluster recovery;
    - use multiple starts instead of trusting one initialization;
    - compare cluster counts with inertia, silhouette, and stability;
    - compare K-means, hierarchical clustering, and DBSCAN by geometry;
    - use external labels only after exploratory choices are frozen;
    - convert anomaly ranking into a review-budget threshold;
    - preserve preprocessing and clustering as one fitted artifact.

    ### Learning path

    ```mermaid
    flowchart LR
        A[Declare question] --> B[Define feature geometry]
        B --> C[Scale and measure distance]
        C --> D[Learn PCA]
        D --> E[Implement K-means]
        E --> F[Check inertia and silhouette]
        F --> G[Check stability]
        G --> H[Compare geometries]
        H --> I[External meaning]
        I --> J[Anomaly review budget]
        J --> K[Version the artifact]
    ```

    Vectors, means, and variance  
    → required before PCA  
    → because PCA projects centered vectors toward directions of high variance.

    Distance and scaling  
    → required before K-means  
    → because the nearest center is determined entirely by the chosen geometry.

    Cluster assignments  
    → required before silhouette and stability  
    → because those measures evaluate a proposed partition, not objective truth.
    """),

    md(r"""
    ## 2 · Start with a question, not an algorithm

    A laboratory records five continuous sensor measurements from a production line:

    - temperature in degrees Celsius;
    - pressure in kilopascals;
    - vibration in millimetres per second;
    - flow in litres per minute;
    - energy in kilowatts.

    No trusted operating-mode label is available during development. The team asks:

    1. Can correlated measurements be summarized with fewer coordinates?
    2. Are there stable operating cohorts worth expert investigation?
    3. Which new readings should fit within a fixed review budget?

    Those questions map to PCA, clustering, and anomaly ranking. None asks the algorithm
    to invent a business category.

    We fit preprocessing and structure on a trusted historical reference window. A new
    batch is transformed later. The simulation secretly contains reference regimes and
    injected anomalies, but those answers remain unused until the relevant choices are
    frozen.
    """),

    code(r"""
    import itertools

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.datasets import make_moons
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    random_generator = np.random.default_rng(42)
    sensor_names = [
        "temperature_c",
        "pressure_kpa",
        "vibration_mm_s",
        "flow_l_min",
        "energy_kw",
    ]

    normal_sensor_rows = []
    hidden_regime_labels = []
    latent_centers = [(-2.2, -1.0), (0.0, 2.2), (2.3, -0.8)]

    for regime_id, (latent_center_1, latent_center_2) in enumerate(latent_centers):
        latent_1 = random_generator.normal(latent_center_1, 0.55, 220)
        latent_2 = random_generator.normal(latent_center_2, 0.50, 220)
        regime_rows = np.column_stack(
            [
                72 + 7.5 * latent_1 + random_generator.normal(0, 1.2, 220),
                240 + 32 * latent_2 + random_generator.normal(0, 5, 220),
                2.2 + 0.45 * latent_1 - 0.25 * latent_2 + random_generator.normal(0, 0.12, 220),
                105 + 9 * latent_1 + 11 * latent_2 + random_generator.normal(0, 2.5, 220),
                480 + 38 * latent_1 + 24 * latent_2 + random_generator.normal(0, 8, 220),
            ]
        )
        normal_sensor_rows.append(regime_rows)
        hidden_regime_labels.extend([regime_id] * len(regime_rows))

    normal_sensor_table = pd.DataFrame(np.vstack(normal_sensor_rows), columns=sensor_names)
    hidden_regime_labels = np.asarray(hidden_regime_labels)

    reference_table, new_normal_table, hidden_reference_regimes, hidden_new_regimes = train_test_split(
        normal_sensor_table,
        hidden_regime_labels,
        test_size=0.25,
        random_state=42,
    )

    injected_anomalies = pd.DataFrame(
        {
            "temperature_c": random_generator.normal(105, 5, 18),
            "pressure_kpa": random_generator.normal(330, 15, 18),
            "vibration_mm_s": random_generator.normal(5.2, 0.5, 18),
            "flow_l_min": random_generator.normal(65, 8, 18),
            "energy_kw": random_generator.normal(650, 25, 18),
        }
    )
    new_batch_table = pd.concat([new_normal_table, injected_anomalies], ignore_index=True)
    hidden_new_anomaly_flags = np.concatenate(
        [np.zeros(len(new_normal_table), dtype=int), np.ones(len(injected_anomalies), dtype=int)]
    )

    # Introduce a small amount of missingness to make preprocessing realistic.
    for table in (reference_table, new_batch_table):
        missing_mask = random_generator.random(table.shape) < 0.01
        table.iloc[:, :] = table.mask(missing_mask)

    print("reference rows:", len(reference_table))
    print("new-batch rows:", len(new_batch_table))
    print("reference missing cells:", int(reference_table.isna().sum().sum()))
    print("new-batch labels: hidden")

    assert len(reference_table) == 495
    assert len(new_batch_table) == 183
    """),

    md(r"""
    ## 3 · Feature geometry comes before PCA or clustering

    Euclidean distance between rows $a$ and $b$ is:

    $$
    d(a,b)=\sqrt{\sum_{j=1}^{p}(a_j-b_j)^2}
    $$

    **Symbols:** $p$ is feature count; $a_j$ and $b_j$ are feature-$j$ values; and
    $d(a,b)$ is their distance.

    Pressure differences may be measured in tens while vibration differences are
    decimals. Without scaling, pressure can dominate the distance merely because of its
    unit.

    Standardization uses training-reference statistics:

    $$
    z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
    $$

    **Symbols:** $x_{ij}$ is row $i$, feature $j$; $\mu_j$ and $\sigma_j$ are the
    reference mean and standard deviation; and $z_{ij}$ is the standardized value.

    Median imputation and scaling are fitted on reference rows only. New rows reuse the
    fitted values. For categories, arbitrary codes such as red=1 and blue=2 do not create
    meaningful Euclidean distance; use an encoding and method appropriate to mixed data.
    """),

    code(r"""
    reference_imputer = SimpleImputer(strategy="median")
    reference_scaler = StandardScaler()

    reference_imputed = reference_imputer.fit_transform(reference_table)
    reference_scaled = reference_scaler.fit_transform(reference_imputed)
    new_batch_imputed = reference_imputer.transform(new_batch_table)
    new_batch_scaled = reference_scaler.transform(new_batch_imputed)

    first_row_raw = reference_imputed[0]
    second_row_raw = reference_imputed[1]
    first_row_scaled = reference_scaled[0]
    second_row_scaled = reference_scaled[1]

    raw_squared_contributions = (first_row_raw - second_row_raw) ** 2
    scaled_squared_contributions = (first_row_scaled - second_row_scaled) ** 2
    raw_distance = np.sqrt(raw_squared_contributions.sum())
    scaled_distance = np.sqrt(scaled_squared_contributions.sum())

    distance_contributions = pd.DataFrame(
        {
            "feature": sensor_names,
            "raw_squared_contribution": raw_squared_contributions,
            "scaled_squared_contribution": scaled_squared_contributions,
        }
    )
    print(distance_contributions.round(3).to_string(index=False))
    print("raw distance:", round(raw_distance, 3))
    print("standardized distance:", round(scaled_distance, 3))

    assert np.allclose(reference_scaled.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(reference_scaled.std(axis=0), 1, atol=1e-12)
    """),

    md(r"""
    ## 4 · High dimensions can make neighbours less distinct

    Adding dimensions adds nonnegative terms to every squared distance. With many noisy
    dimensions, nearest and farthest points can become similar relative to their total
    distance. This is one form of the **curse of dimensionality**.

    Analogy: two people are easy to compare by height and age. Add hundreds of unrelated
    trivia scores and almost everyone differs from everyone in many ways. “Nearest” may
    stop meaning what the problem needs.

    We simulate standard normal points and report nearest distance divided by farthest
    distance. Values closer to one mean less contrast. This is an illustration, not a
    universal law for every dataset.
    """),

    code(r"""
    dimension_records = []
    geometry_generator = np.random.default_rng(7)

    for dimension_count in [2, 5, 20, 100, 500]:
        simulated_points = geometry_generator.normal(size=(400, dimension_count))
        query_point = simulated_points[0]
        other_distances = np.linalg.norm(simulated_points[1:] - query_point, axis=1)
        dimension_records.append(
            {
                "dimensions": dimension_count,
                "nearest": other_distances.min(),
                "farthest": other_distances.max(),
                "nearest_to_farthest_ratio": other_distances.min() / other_distances.max(),
            }
        )

    distance_concentration = pd.DataFrame(dimension_records)
    print(distance_concentration.round(3).to_string(index=False))

    assert distance_concentration.iloc[-1]["nearest_to_farthest_ratio"] > distance_concentration.iloc[0]["nearest_to_farthest_ratio"]
    """),

    md(r"""
    ## 5 · Build PCA from centering, covariance, and projection

    PCA rotates centered data toward orthogonal directions with high variance.

    For centered matrix $X\in\mathbb R^{n\times p}$, sample covariance is:

    $$
    C=\frac{X^TX}{n-1}
    $$

    An eigenvector $v$ of $C$ satisfies $Cv=\lambda v$. The eigenvalue $\lambda$ is
    variance along that direction. A row's component score is the dot product $Xv$.

    PCA steps:

    1. center each feature;
    2. calculate covariance;
    3. find eigenvectors and eigenvalues;
    4. sort directions by decreasing eigenvalue;
    5. project onto chosen directions;
    6. reconstruct by projecting back and restoring the mean.

    An eigenvector may flip sign without changing the component. Loadings describe a
    direction, not causal importance or supervised predictive value.
    """),

    code(r"""
    small_pca_data = np.array(
        [
            [1.0, 1.2],
            [2.0, 1.9],
            [3.0, 3.2],
            [4.0, 3.9],
        ]
    )
    small_mean = small_pca_data.mean(axis=0)
    small_centered = small_pca_data - small_mean
    small_covariance = small_centered.T @ small_centered / (len(small_centered) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(small_covariance)
    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_order]
    eigenvectors = eigenvectors[:, descending_order]

    first_component_direction = eigenvectors[:, 0]
    first_component_scores = small_centered @ first_component_direction
    one_component_reconstruction = (
        np.outer(first_component_scores, first_component_direction) + small_mean
    )
    reconstruction_mse = np.mean((small_pca_data - one_component_reconstruction) ** 2)

    print("mean:", small_mean.round(3))
    print("covariance:\n", small_covariance.round(3))
    print("eigenvalues:", eigenvalues.round(3))
    print("first direction:", first_component_direction.round(3))
    print("first-component scores:", first_component_scores.round(3))
    print("one-component reconstruction MSE:", round(float(reconstruction_mse), 4))

    assert eigenvalues[0] >= eigenvalues[1]
    assert np.isclose(np.linalg.norm(first_component_direction), 1.0)
    """),

    md(r"""
    ## 6 · Use PCA with explained variance, loadings, and reconstruction

    Explained-variance ratio for component $j$ is:

    $$
    r_j=\frac{\lambda_j}{\sum_{m=1}^{p}\lambda_m}
    $$

    **Symbols:** $\lambda_j$ is component-$j$ variance; $p$ is original feature count;
    and $r_j$ is the fraction of total standardized variance captured.

    Cumulative variance is a compression diagnostic, not a universal rule for choosing
    dimensions. Reconstruction error asks how much standardized information is lost.
    If PCA feeds a supervised model, component count must be selected inside development
    folds—not from test performance.

    We fit PCA on reference rows and reuse it for new rows. A two-component projection
    is used for visualization only; clustering later uses the full standardized space.
    """),

    code(r"""
    full_pca = PCA().fit(reference_scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    component_summary = pd.DataFrame(
        {
            "component": np.arange(1, len(sensor_names) + 1),
            "explained_variance_ratio": full_pca.explained_variance_ratio_,
            "cumulative_variance": cumulative_variance,
        }
    )

    visualization_pca = PCA(n_components=2).fit(reference_scaled)
    reference_projection = visualization_pca.transform(reference_scaled)
    new_batch_projection = visualization_pca.transform(new_batch_scaled)
    reference_reconstruction = visualization_pca.inverse_transform(reference_projection)
    reference_reconstruction_mse = np.mean((reference_scaled - reference_reconstruction) ** 2)

    loading_table = pd.DataFrame(
        visualization_pca.components_.T,
        index=sensor_names,
        columns=["PC1 loading", "PC2 loading"],
    )

    print(component_summary.round(3).to_string(index=False))
    print("\ntwo-component loadings:")
    print(loading_table.round(3))
    print("\ntwo-component reconstruction MSE:", round(float(reference_reconstruction_mse), 4))

    assert np.isclose(full_pca.explained_variance_ratio_.sum(), 1.0)
    assert reference_projection.shape == (len(reference_scaled), 2)
    """),

    md(r"""
    ## 7 · Implement guarded K-means with multiple starts

    K-means minimizes within-cluster squared distance:

    $$
    J=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2
    $$

    **Symbols:** $x_i$ is row $i$; $c_i$ is its assigned cluster; $\mu_{c_i}$ is that
    cluster's center; and $J$ is inertia.

    Each iteration assigns rows to nearest centers, then replaces every center with its
    assigned mean. The objective is non-convex: different starts may reach different
    local minima. If a cluster becomes empty, a naive mean produces `NaN`. Our teaching
    implementation repositions that center to the currently worst-represented point.
    """),

    code(r"""
    def guarded_kmeans_once(feature_matrix, cluster_count, random_seed=0, max_iterations=100, tolerance=1e-5):
        '''Run one guarded K-means initialization and return objective history.'''
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        if feature_matrix.ndim != 2 or not np.isfinite(feature_matrix).all():
            raise ValueError("feature_matrix must be a finite two-dimensional array")
        if not 1 < cluster_count < len(feature_matrix):
            raise ValueError("cluster_count must be between 2 and number of rows minus 1")

        random_generator_local = np.random.default_rng(random_seed)
        initial_indices = random_generator_local.choice(
            len(feature_matrix),
            size=cluster_count,
            replace=False,
        )
        centers = feature_matrix[initial_indices].copy()
        objective_history = []

        for _ in range(max_iterations):
            squared_distances = ((feature_matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            cluster_labels = squared_distances.argmin(axis=1)
            objective_history.append(float(squared_distances[np.arange(len(feature_matrix)), cluster_labels].sum()))
            updated_centers = centers.copy()

            for cluster_id in range(cluster_count):
                cluster_rows = feature_matrix[cluster_labels == cluster_id]
                if len(cluster_rows):
                    updated_centers[cluster_id] = cluster_rows.mean(axis=0)
                else:
                    # Reuse the point farthest from its current nearest center.
                    worst_represented_index = np.argmax(squared_distances.min(axis=1))
                    updated_centers[cluster_id] = feature_matrix[worst_represented_index]

            maximum_center_shift = np.linalg.norm(updated_centers - centers, axis=1).max()
            centers = updated_centers
            if maximum_center_shift <= tolerance:
                break

        final_squared_distances = ((feature_matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        final_labels = final_squared_distances.argmin(axis=1)
        final_inertia = float(final_squared_distances[np.arange(len(feature_matrix)), final_labels].sum())
        return final_labels, centers, final_inertia, objective_history


    manual_runs = []
    for seed in range(10):
        run_labels, run_centers, run_inertia, run_history = guarded_kmeans_once(
            reference_scaled,
            cluster_count=3,
            random_seed=seed,
        )
        manual_runs.append(
            {
                "seed": seed,
                "labels": run_labels,
                "centers": run_centers,
                "inertia": run_inertia,
                "history": run_history,
            }
        )

    best_manual_run = min(manual_runs, key=lambda run: run["inertia"])
    print("inertia by seed:", [round(run["inertia"], 1) for run in manual_runs])
    print("selected seed:", best_manual_run["seed"])
    print("iterations:", len(best_manual_run["history"]))
    print("cluster sizes:", np.bincount(best_manual_run["labels"]))

    assert np.isfinite(best_manual_run["centers"]).all()
    assert best_manual_run["history"][-1] <= best_manual_run["history"][0]
    """),

    md(r"""
    ## 8 · Choose cluster count with several imperfect clues

    **Inertia** always stays the same or decreases as more centers are added. It cannot
    select $k$ alone.

    **Silhouette** compares how close a row is to its own cluster versus its nearest
    other cluster. It ranges roughly from -1 to 1; higher often means better separation.
    It tends to prefer compact, separated clusters and may dislike meaningful varying
    densities or non-convex shapes.

    **Stability** asks whether small changes in initialization or sampled rows preserve
    pairwise grouping. Adjusted Rand Index (ARI) compares partitions while ignoring
    arbitrary cluster-number names. ARI near one means strong agreement; near zero means
    chance-like agreement under its adjustment.

    No metric creates domain meaning. We require reasonable silhouette, multi-run
    stability, usable cluster sizes, and expert-interpretable profiles.
    """),

    code(r"""
    cluster_evaluation_records = []
    stability_generator = np.random.default_rng(42)

    for cluster_count in range(2, 7):
        full_candidate = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            random_state=42,
        ).fit(reference_scaled)

        bootstrap_predictions = []
        for repeat in range(8):
            sampled_indices = stability_generator.integers(
                0,
                len(reference_scaled),
                len(reference_scaled),
            )
            bootstrap_model = KMeans(
                n_clusters=cluster_count,
                n_init=10,
                random_state=repeat,
            ).fit(reference_scaled[sampled_indices])
            bootstrap_predictions.append(bootstrap_model.predict(reference_scaled))

        pairwise_stabilities = [
            adjusted_rand_score(first_labels, second_labels)
            for first_labels, second_labels in itertools.combinations(bootstrap_predictions, 2)
        ]
        cluster_sizes = np.bincount(full_candidate.labels_, minlength=cluster_count)
        cluster_evaluation_records.append(
            {
                "k": cluster_count,
                "inertia": full_candidate.inertia_,
                "silhouette": silhouette_score(reference_scaled, full_candidate.labels_),
                "mean_stability_ARI": np.mean(pairwise_stabilities),
                "smallest_cluster": cluster_sizes.min(),
            }
        )

    cluster_evaluation = pd.DataFrame(cluster_evaluation_records)
    stable_candidates = cluster_evaluation[cluster_evaluation["mean_stability_ARI"] >= 0.80]
    selected_cluster_count = int(
        stable_candidates.sort_values(["silhouette", "smallest_cluster"], ascending=False).iloc[0]["k"]
    )
    final_kmeans = KMeans(
        n_clusters=selected_cluster_count,
        n_init=30,
        random_state=42,
    ).fit(reference_scaled)

    print(cluster_evaluation.round(3).to_string(index=False))
    print("selected k from development evidence:", selected_cluster_count)
    print("hidden simulation regimes: still unused")

    assert selected_cluster_count in cluster_evaluation["k"].to_numpy()
    """),

    md(r"""
    A two-dimensional projection helps humans inspect a partition, but it discards
    information. Cluster assignments below come from the full standardized feature
    space; PCA only draws them.
    """),

    code(r"""
    fig, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        reference_projection[:, 0],
        reference_projection[:, 1],
        c=final_kmeans.labels_,
        cmap="tab10",
        s=24,
        alpha=0.8,
    )
    axis.set_xlabel("principal component 1")
    axis.set_ylabel("principal component 2")
    axis.set_title("Full-space K-means assignments shown in a PCA projection")
    plt.show()
    """),

    md(r"""
    ## 9 · Match the algorithm to the geometry

    K-means prefers compact groups around means. It is not the default for every shape.

    - **Agglomerative clustering** starts with individual rows and repeatedly merges
      groups. A dendrogram can expose hierarchy, but choices of linkage and distance
      matter and prediction for new rows is not built into the basic estimator.
    - **DBSCAN** connects dense neighbourhoods and labels sparse points `-1` as noise.
      It can find curved shapes and avoids choosing $k$, but `eps`, minimum samples,
      scale, and varying density strongly affect results.

    We compare methods on moon-shaped data. The hidden moon labels are not used for
    fitting; the picture exists only to make geometry visible.
    """),

    code(r"""
    moon_features, _ = make_moons(n_samples=450, noise=0.07, random_state=42)
    moon_features_scaled = StandardScaler().fit_transform(moon_features)

    geometry_assignments = {
        "K-means": KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(moon_features_scaled),
        "Agglomerative": AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(moon_features_scaled),
        "DBSCAN": DBSCAN(eps=0.22, min_samples=6).fit_predict(moon_features_scaled),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (method_name, method_labels) in zip(axes, geometry_assignments.items()):
        axis.scatter(
            moon_features_scaled[:, 0],
            moon_features_scaled[:, 1],
            c=method_labels,
            cmap="tab10",
            s=16,
        )
        axis.set_title(method_name)
        axis.set_xticks([])
        axis.set_yticks([])
    plt.suptitle("The same rows produce different partitions under different geometry assumptions")
    plt.tight_layout()
    plt.show()

    print("DBSCAN label counts:", dict(zip(*np.unique(geometry_assignments["DBSCAN"], return_counts=True))))
    """),

    md(r"""
    ## 10 · Add external meaning only after exploratory choices are frozen

    Internal metrics judge geometry. They cannot prove that clusters correspond to
    operating modes, customer needs, biological types, or any other domain concept.

    External evaluation may use:

    - labels collected independently after clustering;
    - expert review blinded to cluster IDs;
    - downstream usefulness on a separate task;
    - repeatability across time, sites, or instruments.

    Our simulation has hidden regime IDs. We reveal them only now, after preprocessing,
    $k$, and the clustering method are selected. ARI measures agreement without requiring
    cluster IDs to use the same numeric names. In a genuinely unlabeled project, this
    number would not exist.
    """),

    code(r"""
    external_regime_ari = adjusted_rand_score(hidden_reference_regimes, final_kmeans.labels_)

    reference_profile_table = pd.DataFrame(reference_imputed, columns=sensor_names)
    reference_profile_table["cluster_id"] = final_kmeans.labels_
    cluster_profiles = reference_profile_table.groupby("cluster_id").agg(
        rows=("temperature_c", "size"),
        temperature_c=("temperature_c", "mean"),
        pressure_kpa=("pressure_kpa", "mean"),
        vibration_mm_s=("vibration_mm_s", "mean"),
        flow_l_min=("flow_l_min", "mean"),
        energy_kw=("energy_kw", "mean"),
    )

    print("external ARI available only because this is a simulation:", round(external_regime_ari, 3))
    print("\ncluster profiles for expert interpretation:")
    print(cluster_profiles.round(2).to_string())

    assert -1 <= external_regime_ari <= 1
    assert cluster_profiles["rows"].sum() == len(reference_table)
    """),

    md(r"""
    ## 11 · Treat anomaly detection as ranking plus a review decision

    Isolation Forest repeatedly partitions rows with random feature and split choices.
    Unusual rows tend to become isolated in fewer splits, producing a stronger anomaly
    score.

    The score is not a fault probability. Without labels, a threshold can come from an
    operational review budget: “inspect the top 5% of this batch.” Domain experts then
    measure review yield and decide whether the detector saves time.

    Our trusted reference window contains ordinary operation. The new simulation batch
    includes hidden injected anomalies. We freeze the 5% budget before revealing them.
    """),

    code(r"""
    anomaly_detector = IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=42,
        n_jobs=1,
    ).fit(reference_scaled)

    new_batch_anomaly_scores = -anomaly_detector.score_samples(new_batch_scaled)
    review_fraction = 0.05
    review_threshold = np.quantile(new_batch_anomaly_scores, 1 - review_fraction)
    review_flags = new_batch_anomaly_scores >= review_threshold

    # Hidden simulation answers are revealed only after score and budget are frozen.
    reviewed_anomalies = int(np.sum(hidden_new_anomaly_flags[review_flags]))
    review_count = int(np.sum(review_flags))
    total_injected_anomalies = int(np.sum(hidden_new_anomaly_flags))
    review_precision = reviewed_anomalies / review_count
    anomaly_recall = reviewed_anomalies / total_injected_anomalies

    print("review budget fraction:", review_fraction)
    print("rows sent to review:", review_count)
    print("injected anomalies found:", reviewed_anomalies, "of", total_injected_anomalies)
    print("review yield in simulation:", round(review_precision, 3))
    print("simulated anomaly recall:", round(anomaly_recall, 3))

    assert review_count == int(np.ceil(review_fraction * len(new_batch_table)))
    """),

    md(r"""
    ## 12 · Preserve preprocessing and assignment as one artifact

    New rows must receive the same median values, scaling parameters, feature order, and
    cluster centers as reference rows. A pipeline prevents accidental refitting during
    prediction.

    Version:

    - feature schema and unit definitions;
    - imputation statistics and scaling parameters;
    - PCA means, loadings, and component count when PCA is deployed;
    - cluster method, centers, $k$, seeds, and library version;
    - anomaly detector and review-budget rule;
    - reference-window dates and stability report.

    Monitor missingness, distance to assigned center, cluster-size mix, PCA reconstruction
    error, anomaly-score distribution, and review yield. A changed cluster number is not
    automatically a new real-world category.
    """),

    code(r"""
    clustering_artifact = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "cluster",
                KMeans(
                    n_clusters=selected_cluster_count,
                    n_init=30,
                    random_state=42,
                ),
            ),
        ]
    )
    clustering_artifact.fit(reference_table)
    new_batch_cluster_assignments = clustering_artifact.predict(new_batch_table)

    reference_cluster_mix = pd.Series(
        clustering_artifact.predict(reference_table)
    ).value_counts(normalize=True).sort_index()
    new_batch_cluster_mix = pd.Series(
        new_batch_cluster_assignments
    ).value_counts(normalize=True).sort_index()

    new_batch_reconstruction = visualization_pca.inverse_transform(new_batch_projection)
    new_batch_reconstruction_mse = np.mean((new_batch_scaled - new_batch_reconstruction) ** 2)

    artifact_report = pd.DataFrame(
        {
            "reference_mix": reference_cluster_mix,
            "new_batch_mix": new_batch_cluster_mix,
        }
    ).fillna(0)

    print(artifact_report.round(3).to_string())
    print("reference two-PC reconstruction MSE:", round(float(reference_reconstruction_mse), 4))
    print("new-batch two-PC reconstruction MSE:", round(float(new_batch_reconstruction_mse), 4))
    print("new rows transformed without refitting preprocessing")

    assert len(new_batch_cluster_assignments) == len(new_batch_table)
    """),

    md(r"""
    ## 13 · Mini-project: investigate laboratory operating structure

    **Goal:** produce a reproducible investigation package, not a claim that clusters are
    natural categories.

    **Dataset columns:** temperature, pressure, vibration, flow, and energy.

    **Expected workflow:**

    1. state the exploratory and operational questions;
    2. audit units, missingness, impossible values, and reference-window quality;
    3. fit imputation and scaling on reference data;
    4. calculate raw and scaled distance contributions;
    5. fit PCA and report loadings, variance, and reconstruction;
    6. implement multi-start K-means and record objective history;
    7. compare $k$ using inertia, silhouette, stability, and minimum cluster size;
    8. compare at least one alternative geometry;
    9. create blinded profiles for domain review;
    10. freeze an anomaly review budget and report review yield assumptions;
    11. package preprocessing and assignments together.

    **Expected output:** evidence table, stability table, two cautious visualizations,
    profile table, review-budget report, artifact contract, and limitations.

    **Evaluation criteria:** correct geometry, no future-data fitting, guarded code,
    stability evidence, domain humility, and reproducibility.
    """),

    md(r"""
    ## 14 · Practice, solutions, and mastery checkpoint

    ### Worked example

    Feature A has mean 100 and standard deviation 20. Value 130 becomes
    $(130-100)/20=1.5$. Feature B has mean 2 and standard deviation 0.5. Value 2.5
    becomes 1. Both now contribute on standardized rather than raw-unit scales.

    ### Guided practice

    1. Calculate Euclidean distance between `[0, 1]` and `[3, 5]`.
    2. Center `[2, 4, 6]` manually.
    3. Explain PCA projection versus feature selection.
    4. Explain why inertia cannot choose $k$ alone.
    5. Explain why ARI can compare differently numbered cluster labels.

    ### Independent practice

    6. Add a robust-scaling comparison for one outlier-heavy feature.
    7. Plot K-means objective history for ten starts.
    8. Repeat stability with time blocks instead of row bootstrap.
    9. Vary DBSCAN `eps` and document cluster/noise sensitivity.
    10. Replace the 5% anomaly budget with an hourly reviewer-capacity rule.

    ### Challenge

    Rebuild the laboratory project without copying. Include manual standardization,
    distance contributions, covariance PCA, reconstruction, guarded K-means, multi-start
    selection, silhouette, bootstrap stability, geometry comparison, delayed external
    evaluation, review-budget anomaly detection, and one fitted artifact.

    ### Self-check

    1. Why do feature units change K-means?
    2. Why must PCA center data?
    3. What does an explained-variance ratio mean?
    4. Why can a component's sign flip?
    5. What problem does multi-start K-means address?
    6. Why is a high silhouette insufficient?
    7. When is ordinary row bootstrap invalid?
    8. Why is an anomaly score not a probability?

    ### Solution and scoring rubric

    1. Distance is $\sqrt{3^2+4^2}=5$.
    2. Mean is 4, so centered values are `[-2, 0, 2]`.
    3. PCA creates weighted combinations; feature selection retains original columns.
    4. Inertia cannot increase when an additional center is available.
    5. ARI compares pairwise grouping and is invariant to cluster ID names.

    Award two points for each self-check and four points for the challenge explanation.
    Full credit requires both mathematical and operational boundaries.

    ### Common mistakes

    - Treating arbitrary category codes as Euclidean quantities.
    - Fitting imputation, scaling, or PCA on future evaluation rows.
    - Reading PCA loadings as causal or supervised importance.
    - Clustering a two-dimensional visualization without admitting information loss.
    - Trusting one K-means initialization.
    - Ignoring empty clusters in a manual implementation.
    - Selecting $k$ from inertia alone.
    - Comparing raw cluster ID numbers across runs.
    - Calling a high silhouette proof of natural categories.
    - Using DBSCAN without scaling or sensitivity analysis.
    - Treating an anomaly score as fault probability.
    - Naming clusters before blinded domain review.

    ### Readiness threshold

    Score at least **16/20** and correctly explain scaling, PCA reconstruction, K-means
    objective, stability, external meaning, review threshold, and artifact boundaries.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Explain this chain without notes:

    operational question  
    → feature geometry  
    → reference-only preprocessing  
    → PCA and reconstruction  
    → guarded multi-start K-means  
    → inertia plus silhouette plus stability  
    → geometry alternatives  
    → delayed external meaning  
    → anomaly review budget  
    → versioned artifact.

    ### Teach it back

    Explain why three stable clusters still may not be three real-world types. Then
    explain why PCA plots, anomaly scores, and internal metrics are useful evidence but
    cannot supply missing domain truth.

    ### Memory aid

    **Define the geometry, test the stability, seek external meaning, and never confuse
    structure with truth.**

    ### Next dependency

    Vectors, fitted transformations, optimization loops, and honest evaluation  
    → required before PyTorch foundations  
    → because tensors and learned representations reuse the same mathematical and
    experimental boundaries at larger scale.
    """),
]


build("03_ml_engineering/06_unsupervised_learning_foundations.ipynb", cells)
