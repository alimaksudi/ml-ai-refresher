"""Builder for Lesson FND-03 — Data Workflow, EDA, Cleaning, Pandas, and SQL."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # FND-03 · Data Workflow, EDA, Cleaning, Pandas, and SQL
    ### Section 01 — From Mathematical Objects to Trustworthy Training Data

    Models do not receive “reality.” They receive rows, columns, types, timestamps,
    missing values, duplicates, and labels produced by a data process. This notebook
    teaches the workflow that must happen before model selection.

    **Estimated time:** 4–6 hours including exercises.
    **Prerequisites:** PRE-01 through PRE-05 and Section 01 lessons FND-01/FND-02. This is
    the first applied-data notebook and must be completed before any model.

    **Mastery path:** define the decision and prediction unit → establish a naive
    baseline → inspect the raw data contract → choose the split strategy → fit
    transformations on training data only. A model is not allowed to begin until
    those decisions are written down.
    """),

    md(r"""
    ## 1 · Learning Objectives

    You will be able to:

    - define the prediction unit, target, prediction time, and feature availability;
    - state the decision the prediction supports and the cost of each error type;
    - construct a naive baseline that every learned model must beat;
    - load a real dataset into pandas without losing its data contract;
    - inspect shape, types, ranges, missingness, duplicates, and class balance;
    - separate descriptive exploration from model validation;
    - split data before fitting imputers, scalers, encoders, or selectors;
    - write basic SQL for selection, grouping, filtering, and validation;
    - package learned preprocessing with the model in one sklearn Pipeline;
    - distinguish a data-quality alarm from model-performance evidence.
    """),

    md(r"""
    ## 2 · Historical Motivation

    Many failed ML systems were not defeated by the choice between two algorithms.
    They were defeated by a shifted column, duplicated entity, future-derived feature,
    changed unit, missing category, or preprocessing step fitted on all data.

    Exploratory data analysis exists to understand what the table actually contains.
    A data contract exists to preserve that meaning when the table changes. A pipeline
    exists to apply the same learned transformations during validation and serving.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    Treat the workflow as a sequence of contracts:

    ```text
    prediction question
        -> row and target definition
        -> raw schema checks
        -> split strategy
        -> train-fitted transformations
        -> model
        -> evaluation on untouched examples
        -> serving schema and monitoring
    ```

    EDA can describe all available raw rows, but any statistic that changes model
    inputs—mean imputation, scaling, category vocabulary, feature selection—must be
    learned from training data only.
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Missing-value rate

    $$r_j=\frac{1}{n}\sum_{i=1}^{n}\mathbb 1[x_{ij}\text{ is missing}].$$

    **Read aloud:** “the missing rate for feature j is the number of rows where
    feature j is missing, divided by the number of rows.”

    **Symbols:** $n$ is row count; $i$ selects a row; $j$ selects a feature;
    $x_{ij}$ is the value in row $i$, column $j$; $\mathbb 1[\cdot]$ is an indicator
    that returns 1 when its condition is true and 0 otherwise.

    **Small example:** 8 missing cells among 100 rows gives $r_j=8/100=0.08=8\%$.
    The rate does not tell us whether missingness is random or informative.

    ### 4.2 Training mean and standard deviation

    $$\mu_j=\frac1n\sum_{i=1}^{n}x_{ij},\qquad
    \sigma_j=\sqrt{\frac1n\sum_{i=1}^{n}(x_{ij}-\mu_j)^2}.$$

    **Read and symbols:** $\mu_j$ is the mean of feature $j$; $\sigma_j$ is its
    population-style standard deviation; deviations measure distance from the mean.
    These values must be calculated on the training partition and reused unchanged
    for validation, test, and serving.

    **Small example:** values `(2, 4, 6)` have mean 4 and standard deviation
    $\sqrt{8/3}\approx1.63$.

    ### 4.3 Standardization

    $$z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.$$

    **Read aloud:** “standardized value z for row i and feature j equals the raw
    value minus the training mean, divided by the training standard deviation.”

    **Symbols:** $z_{ij}$ is unitless; zero is the training mean; `+1` is one
    training standard deviation above it. If $\sigma_j=0$, the feature is constant
    and this division is undefined.

    ### 4.4 Correlation

    $$\rho_{XY}=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}.$$

    **Read and symbols:** $\rho$ (rho) is linear correlation; covariance measures
    co-movement; $\sigma_X,\sigma_Y$ scale it into the range `−1` to `1` when both
    variables vary. Correlation near zero does not rule out nonlinear relationships,
    and correlation never proves causation.

    ### 4.5 Interquartile range

    $$\operatorname{IQR}=Q_{0.75}-Q_{0.25}.$$

    **Read and symbols:** $Q_{0.25}$ is the first quartile, below which 25% of values
    lie; $Q_{0.75}$ is the third quartile; IQR is the width of the middle 50%.
    A common diagnostic flags values below $Q_{0.25}-1.5\,IQR$ or above
    $Q_{0.75}+1.5\,IQR$, but a flagged value is not automatically an error.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    We use scikit-learn's bundled copy of the UCI Wine recognition dataset so the
    notebook is reproducible without network access. The observations are real
    chemical analyses, not synthetically generated rows.
    """),

    code(r"""
    import sqlite3
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_wine

    dataset = load_wine(as_frame=True)
    frame = dataset.frame.rename(
        columns={"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"}
    )
    feature_names = [column for column in frame.columns if column != "target"]

    print("shape:", frame.shape)
    print("target counts:\n", frame["target"].value_counts().sort_index())
    print("missing cells:", int(frame.isna().sum().sum()))
    print("duplicate rows:", int(frame.duplicated().sum()))

    assert frame.shape == (178, 14)
    assert len(feature_names) == 13
    assert frame.isna().sum().sum() == 0
    """),

    md(r"""
    A useful inspection report separates schema facts from learned statistics. The
    next function can run before modeling and returns data that can be logged or
    compared across versions.
    """),

    code(r"""
    def data_quality_report(data, target):
        numeric = data.select_dtypes(include="number")
        return {
            "rows": int(len(data)),
            "columns": int(data.shape[1]),
            "duplicate_rows": int(data.duplicated().sum()),
            "missing_rate": data.isna().mean().to_dict(),
            "numeric_min": numeric.min().to_dict(),
            "numeric_max": numeric.max().to_dict(),
            "target_rate": data[target].value_counts(normalize=True).sort_index().to_dict(),
        }

    report = data_quality_report(frame, "target")
    print("rows:", report["rows"])
    print("class proportions:", report["target_rate"])
    assert abs(sum(report["target_rate"].values()) - 1.0) < 1e-12
    """),

    md(r"""
    To demonstrate cleaning, we inject known problems into a copy. Because the
    original dataset is clean, every detected problem has a known cause.
    """),

    code(r"""
    rng = np.random.default_rng(42)
    dirty = frame.copy()
    missing_rows = rng.choice(dirty.index, size=12, replace=False)
    dirty.loc[missing_rows, "malic_acid"] = np.nan
    dirty = pd.concat([dirty, dirty.iloc[[0]]], ignore_index=True)

    print("injected missing values:", int(dirty["malic_acid"].isna().sum()))
    print("injected duplicate rows:", int(dirty.duplicated().sum()))

    deduplicated = dirty.drop_duplicates().reset_index(drop=True)
    assert deduplicated.shape[0] == frame.shape[0]
    assert deduplicated["malic_acid"].isna().sum() == 12
    """),

    md(r"""
    ## 6 · Visualization

    Distribution plots answer descriptive questions. They do not choose thresholds
    or prove model quality. Always label whether a chart uses all raw data, training
    data, or an evaluation partition.
    """),

    code(r"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    frame["target"].value_counts().sort_index().plot.bar(ax=axes[0], color="tab:blue")
    axes[0].set_title("Class counts — all raw rows")
    axes[0].set_xlabel("cultivar class")
    axes[0].set_ylabel("rows")

    for class_id, group in frame.groupby("target"):
        axes[1].hist(group["alcohol"], bins=12, alpha=0.5, label=f"class {class_id}")
    axes[1].set_title("Alcohol distribution by class")
    axes[1].set_xlabel("alcohol")
    axes[1].legend()

    correlation = frame[feature_names].corr()
    image = axes[2].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axes[2].set_title("Feature correlation — descriptive")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(image, ax=axes[2], shrink=0.8)
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Fitting imputation, scaling, encoding, or selection before splitting.
    - Randomly splitting time-dependent or grouped rows.
    - Removing every statistical outlier without checking domain validity.
    - Deduplicating on all columns when repeated observations are legitimate events.
    - Treating missingness as harmless after imputation.
    - Using a target-derived or future-derived feature.
    - Running dozens of EDA comparisons and then presenting one selected pattern as
      confirmatory evidence.
    - Letting serving accept reordered, renamed, or silently coerced columns.
    """),

    md(r"""
    ## 8 · Production Library Implementation

    The split happens before learned preprocessing. The imputer and scaler live in
    the same Pipeline as the model, so cross-validation and serving reuse identical
    transformations.
    """),

    code(r"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = deduplicated[feature_names]
    y = deduplicated["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print("test macro F1:", round(f1_score(y_test, predictions, average="macro"), 3))

    learned_median = pipeline.named_steps["impute"].statistics_[feature_names.index("malic_acid")]
    training_median = X_train["malic_acid"].median()
    assert np.isclose(learned_median, training_median)
    """),

    md(r"""
    SQL expresses data checks close to storage. This in-memory example uses the
    standard-library SQLite engine; production syntax varies by warehouse.
    """),

    code(r"""
    connection = sqlite3.connect(":memory:")
    frame.to_sql("wine_samples", connection, index=False)

    class_summary = pd.read_sql_query(
        '''
        SELECT target,
               COUNT(*) AS rows,
               AVG(alcohol) AS mean_alcohol,
               MIN(proline) AS min_proline,
               MAX(proline) AS max_proline
        FROM wine_samples
        GROUP BY target
        ORDER BY target
        ''',
        connection,
    )
    print(class_summary)
    assert class_summary["rows"].sum() == 178
    connection.close()
    """),

    md(r"""
    ## 9 · Realistic Case Study — From Dataset to Serving Contract

    The companion capstone uses this dataset to build a cultivar classifier. Its
    contract fixes 13 feature names and their order, records a dataset hash, learns
    scaling only within training folds, evaluates an untouched test split, and
    exports training-reference ranges for input diagnostics.

    The question is deliberately low stakes: the output is an educational cultivar
    classification, not a wine quality, safety, pricing, or regulatory decision.
    """),

    md(r"""
    ## 10 · Production Considerations

    A production data contract should record:

    - prediction unit and event time;
    - target definition and label delay;
    - feature name, type, unit, allowed range, and null policy;
    - source table and transformation lineage;
    - entity/group keys and split policy;
    - category vocabulary and unknown-category behavior;
    - schema owner, version, compatibility rules, and rollback;
    - training-reference statistics and monitoring windows.

    Data-quality checks stop malformed input. They do not establish that the model
    remains accurate; delayed labels and outcome monitoring are still required.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    | Choice | Benefit | Cost or risk |
    |---|---|---|
    | Drop incomplete rows | Simple | Bias and lost data |
    | Median imputation | Stable and cheap | Hides missingness without indicator |
    | Model-based imputation | Captures relationships | Complexity and leakage risk |
    | Global random split | Efficient | Invalid for time/group dependence |
    | Strict schema rejection | Prevents silent corruption | Requires coordinated migrations |
    | Permissive coercion | Fewer rejected requests | Can silently change meaning |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    A strong answer to “how do you prepare data?” starts with prediction time and
    leakage, not `dropna()`. It names the independent evaluation unit, separates raw
    schema checks from train-fitted transformations, packages transformations with
    the model, and describes how train/serve parity is verified.

    **Diagnostic question:** offline metrics suddenly improve after adding a new
    aggregate. First verify when that aggregate becomes available and whether it was
    computed point-in-time correctly.
    """),

    md(r"""
    ## 13 · Teach-Back

    Explain, without code:

    1. the difference between EDA and validation;
    2. why learned preprocessing belongs inside a pipeline;
    3. why a statistical outlier is not automatically bad data;
    4. how a target can leak through an aggregate;
    5. why schema correctness and model correctness are different guarantees.
    """),

    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Estimated time:** 60–90 minutes.

    ### Guided practice

    1. Add unit, null-policy, and owner fields to a feature contract for `alcohol`.
    2. Inject 10% missingness into `magnesium`; verify the imputer statistic equals
       the training median rather than the full-data median.
    3. Write SQL that returns class count and mean `color_intensity` by target.
    4. Add a `FunctionTransformer` or custom validator that rejects an unexpected
       feature order before prediction.

    ### Independent practice

    5. Create a grouped synthetic entity column with repeated rows and demonstrate
       why a random row split leaks entity information.
    6. Compare median imputation with `KNNImputer` using the same untouched split.
    7. Design data-quality and model-quality alerts separately for this capstone.

    <details>
    <summary><strong>Solutions and scoring rubric</strong></summary>

    1. A complete contract includes numeric type, measurement unit, allowed or
       observed range, whether null is permitted, source, and owner.
    2. Split first. Fit `SimpleImputer(strategy="median")` on `X_train`; compare its
       stored statistic with `X_train["magnesium"].median()` using `np.isclose`.
    3. Use `SELECT target, COUNT(*) AS rows, AVG(color_intensity) ... GROUP BY target`.
    4. Full credit validates both names and order and fails before calling the model.
    5. The same entity must remain on one side; use GroupShuffleSplit or GroupKFold.
    6. Compare only after fitting each imputer inside its own pipeline. Report metric
       uncertainty instead of selecting from one point estimate.
    7. Data checks cover schema, missingness, range, volume, and freshness. Model
       checks require predictions, delayed labels, calibration/performance, slices,
       and business outcomes.

    Award two points per question: one for correct implementation or design and one
    for explaining leakage or operational consequences. A score of 11/14 is ready
    for the applied capstone.
    </details>
    """),
]


build("01_ml_foundations/03_data_workflow_eda_and_cleaning.ipynb", cells)
