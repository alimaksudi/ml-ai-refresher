"""Build MLE-04: beginner-first Imbalanced Learning."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # MLE-04 · Imbalanced Learning

    **Prerequisites:** CML-02, MLE-01, MLE-02, and MLE-03  
    **Estimated study time:** 8–10 hours, including practice  
    **Next lesson:** MLE-05 · Explainability with SHAP

    A classification dataset is imbalanced when one class appears much less often
    than another. The main danger is not the percentage by itself. The danger is
    choosing a model, metric, threshold, or sampling method that ignores the rare
    outcome and its real cost.

    The goal is not to “balance everything.” The goal is to define the positive
    event, measure the baseline honestly, choose the smallest justified intervention,
    and preserve the validation boundary.

    ### Scope boundary

    This lesson teaches binary classification with continuous numerical features. It
    covers threshold selection, class weights, random sampling, SMOTE, and fold-safe
    pipelines. It defers:

    - multiclass imbalance;
    - specialized SMOTE variants and ensemble samplers;
    - probability recalibration procedures;
    - monitoring and base-rate drift to PROD-05;
    - feature explanations to MLE-05.

    Accuracy remains useful context, but it is never used alone.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - declare the positive class and prediction decision;
    - count classes and calculate an imbalance ratio;
    - calculate an always-majority baseline;
    - build and interpret a confusion matrix;
    - explain why accuracy alone can hide minority failure;
    - separate ranking, probability estimation, and threshold decisions;
    - choose a threshold from validation costs;
    - calculate inverse-frequency class weights manually;
    - implement random oversampling and undersampling;
    - calculate Euclidean distance and one SMOTE point;
    - explain why neighbour-based synthesis needs compatible scales;
    - implement a guarded SMOTE function for continuous features;
    - place preprocessing and resampling inside each training fold;
    - compare methods using validation evidence only;
    - evaluate one selected configuration once on a sealed test partition.

    ### Learning path

    ```mermaid
    flowchart LR
        A[Define positive event] --> B[Count classes]
        B --> C[Majority baseline]
        C --> D[Confusion matrix]
        D --> E[Probability and threshold]
        E --> F[Validation cost]
        F --> G[Class weights]
        G --> H[Random sampling]
        H --> I[Distance and SMOTE]
        I --> J[Fold-safe pipeline]
        J --> K[One sealed test]
    ```

    Dependency map:

    Classification metrics  
    → required before imbalance interventions  
    → because an intervention cannot be judged by class counts alone.

    Train/validation/test boundaries  
    → required before resampling  
    → because resampling is learned from training rows and must not touch held-out rows.

    Feature scaling  
    → required before SMOTE  
    → because nearest neighbours depend on distances between features.
    """),

    md(r"""
    ## 2 · The practical problem: rare machine defects

    A factory predicts whether a component has a serious defect after six sensor
    measurements become available. Only about 4% of historical components are
    defective.

    **Positive class:** `1` means serious defect.  
    **Negative class:** `0` means no serious defect.  
    **Prediction time:** after sensor measurements, before expensive manual inspection.

    Missing a defect is more costly than inspecting a safe component. We use this
    declared development cost:

    - false negative: 10 cost units;
    - false positive: 1 cost unit.

    These values are a teaching contract, not universal business costs.

    Analogy: a smoke alarm should not be judged only by how often rooms contain no
    fire. The analogy stops because a model emits a score and the decision threshold
    can be changed separately.
    """),

    md(r"""
    ## 3 · Count classes before changing anything

    Let $n_0$ be the number of negative rows and $n_1$ the number of positive rows.
    A simple majority-to-minority imbalance ratio is:

    $$
    \operatorname{ratio}=\frac{n_0}{n_1}
    $$

    If there are 960 negative and 40 positive rows, the ratio is $960/40=24$. There
    are 24 negative rows for each positive row.

    The always-negative baseline would be 96% accurate, but it would miss all 40
    defects. That baseline exposes why accuracy alone is incomplete; it does not prove
    that every ordinary classifier will ignore the minority.

    We split before fitting preprocessing, models, thresholds, or samplers.
    """),

    code(r"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    all_features, all_labels = make_classification(
        n_samples=1200,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        weights=[0.96, 0.04],
        class_sep=1.0,
        flip_y=0.01,
        random_state=42,
    )

    # Seal 20% first. It cannot influence preprocessing, threshold, or method choice.
    development_features, sealed_test_features, development_labels, sealed_test_labels = train_test_split(
        all_features,
        all_labels,
        test_size=0.20,
        stratify=all_labels,
        random_state=42,
    )
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        development_features,
        development_labels,
        test_size=0.25,
        stratify=development_labels,
        random_state=42,
    )

    # Scaling parameters come from training rows only.
    training_scaler = StandardScaler()
    train_features_scaled = training_scaler.fit_transform(train_features)
    validation_features_scaled = training_scaler.transform(validation_features)

    negative_count = int(np.sum(train_labels == 0))
    positive_count = int(np.sum(train_labels == 1))
    imbalance_ratio = negative_count / positive_count

    print("training rows:", len(train_labels))
    print("validation rows:", len(validation_labels))
    print("sealed test rows:", len(sealed_test_labels))
    print("training negative rows:", negative_count)
    print("training positive rows:", positive_count)
    print("majority-to-minority ratio:", round(imbalance_ratio, 2))
    print("test status: sealed")

    assert len(train_labels) + len(validation_labels) + len(sealed_test_labels) == 1200
    assert positive_count < negative_count
    """),

    md(r"""
    ## 4 · Confusion counts reveal what accuracy compresses

    For positive class 1:

    - true positive (TP): defect correctly flagged;
    - false positive (FP): safe component incorrectly flagged;
    - false negative (FN): defect missed;
    - true negative (TN): safe component correctly cleared.

    $$
    \operatorname{accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
    $$

    $$
    \operatorname{precision}=\frac{TP}{TP+FP},\qquad
    \operatorname{recall}=\frac{TP}{TP+FN}
    $$

    The always-negative model has many true negatives, zero recall, and no predicted
    positives. We report its precision as zero by convention rather than dividing by
    zero.
    """),

    code(r"""
    def summarize_binary_decisions(actual_labels, predicted_labels, false_positive_cost=1, false_negative_cost=10):
        '''Return confusion counts, core metrics, and declared decision cost.'''
        actual_labels = np.asarray(actual_labels, dtype=int)
        predicted_labels = np.asarray(predicted_labels, dtype=int)

        true_positive = int(np.sum((actual_labels == 1) & (predicted_labels == 1)))
        false_positive = int(np.sum((actual_labels == 0) & (predicted_labels == 1)))
        false_negative = int(np.sum((actual_labels == 1) & (predicted_labels == 0)))
        true_negative = int(np.sum((actual_labels == 0) & (predicted_labels == 0)))

        accuracy = (true_positive + true_negative) / len(actual_labels)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        decision_cost = false_positive_cost * false_positive + false_negative_cost * false_negative

        return {
            "TN": true_negative,
            "FP": false_positive,
            "FN": false_negative,
            "TP": true_positive,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "cost": decision_cost,
        }


    always_negative_validation = np.zeros(len(validation_labels), dtype=int)
    baseline_summary = summarize_binary_decisions(
        validation_labels,
        always_negative_validation,
    )

    print(pd.Series(baseline_summary))

    assert baseline_summary["TP"] == 0
    assert baseline_summary["recall"] == 0
    assert baseline_summary["accuracy"] > 0.90
    """),

    md(r"""
    ## 5 · Separate ranking, probability, and decision threshold

    Logistic regression first produces a probability estimate $p$. A threshold $t$
    converts it into a decision:

    $$
    \hat y=\begin{cases}
    1 & p\ge t\\
    0 & p<t
    \end{cases}
    $$

    Lowering $t$ normally flags more rows: recall may rise and false positives may
    also rise. The probability model has not changed; only the decision rule changed.

    Three different questions must remain separate:

    1. **Ranking:** do positive rows tend to receive higher scores?
    2. **Probability:** do predicted probabilities match observed frequencies?
    3. **Decision:** which threshold matches the declared costs or capacity?

    Before changing training data, test whether the existing model plus a validated
    threshold already solves the decision problem.
    """),

    code(r"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    plain_logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    plain_logistic_model.fit(train_features_scaled, train_labels)
    plain_validation_probabilities = plain_logistic_model.predict_proba(
        validation_features_scaled
    )[:, 1]

    default_validation_decisions = (plain_validation_probabilities >= 0.5).astype(int)
    default_summary = summarize_binary_decisions(
        validation_labels,
        default_validation_decisions,
    )
    validation_average_precision = average_precision_score(
        validation_labels,
        plain_validation_probabilities,
    )

    print(pd.Series(default_summary))
    print("validation average precision:", round(validation_average_precision, 3))

    assert np.all((plain_validation_probabilities >= 0) & (plain_validation_probabilities <= 1))
    assert 0 <= validation_average_precision <= 1
    """),

    md(r"""
    ## 6 · Choose the threshold on validation costs

    With false-positive cost $C_{FP}$ and false-negative cost $C_{FN}$:

    $$
    C(t)=C_{FP}\,FP(t)+C_{FN}\,FN(t)
    $$

    **Symbols:** $t$ is a candidate threshold; $FP(t)$ and $FN(t)$ are validation
    counts at that threshold.

    We declared $C_{FP}=1$ and $C_{FN}=10$. Candidate thresholds are development
    choices, so validation selects one. Test remains sealed.

    The exact minimum can be noisy when positives are rare. A production process may
    add confidence intervals, review-capacity constraints, or a minimum recall rule.
    """),

    code(r"""
    threshold_records = []
    candidate_thresholds = np.linspace(0.05, 0.95, 19)

    for candidate_threshold in candidate_thresholds:
        candidate_decisions = (
            plain_validation_probabilities >= candidate_threshold
        ).astype(int)
        candidate_summary = summarize_binary_decisions(
            validation_labels,
            candidate_decisions,
        )
        threshold_records.append(
            {"threshold": candidate_threshold, **candidate_summary}
        )

    threshold_table = pd.DataFrame(threshold_records)
    best_threshold_row = threshold_table.loc[threshold_table["cost"].idxmin()]
    selected_plain_threshold = float(best_threshold_row["threshold"])

    print(threshold_table[["threshold", "FP", "FN", "precision", "recall", "cost"]].to_string(index=False))
    print("selected validation threshold:", selected_plain_threshold)
    print("test status: still sealed")

    assert selected_plain_threshold in candidate_thresholds
    assert best_threshold_row["cost"] <= default_summary["cost"]
    """),

    md(r"""
    ## 7 · Class weights change the training objective

    Scikit-learn's balanced weight for class $c$ is:

    $$
    w_c=\frac{n}{K n_c}
    $$

    where $n$ is total training rows, $K$ is number of classes, and $n_c$ is the
    training count for class $c$.

    A positive row receives more weight when positives are rare. In weighted logistic
    loss, its error contributes more strongly to optimization.

    Weighting is related to repeating rows, but it is not universally identical:
    regularization, stochastic optimization, tree construction, and probability
    interpretation can make results differ. Class weighting can also change
    calibration; it does not guarantee honest probabilities.
    """),

    code(r"""
    number_of_training_rows = len(train_labels)
    number_of_classes = 2
    balanced_class_weights = {
        0: number_of_training_rows / (number_of_classes * negative_count),
        1: number_of_training_rows / (number_of_classes * positive_count),
    }

    weighted_logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight=balanced_class_weights,
        random_state=42,
    )
    weighted_logistic_model.fit(train_features_scaled, train_labels)
    weighted_validation_probabilities = weighted_logistic_model.predict_proba(
        validation_features_scaled
    )[:, 1]
    weighted_validation_decisions = (
        weighted_validation_probabilities >= 0.5
    ).astype(int)
    weighted_summary = summarize_binary_decisions(
        validation_labels,
        weighted_validation_decisions,
    )

    print("manual balanced weights:", balanced_class_weights)
    print(pd.Series(weighted_summary))

    assert balanced_class_weights[1] > balanced_class_weights[0]
    """),

    md(r"""
    ## 8 · Random oversampling and undersampling change training rows

    **Random oversampling** repeats minority training rows. It retains majority data
    but exact copies can encourage memorization.

    **Random undersampling** discards majority training rows. It reduces computation
    but may throw away useful boundary examples.

    Neither method belongs on validation or test data. We use a target minority-to-
    majority ratio of 0.5 rather than automatically forcing 50/50 classes.

    Example: with 10 minority and 100 majority rows, ratio 0.5 requires 50 minority
    rows. Oversampling adds 40 minority draws; undersampling keeps 20 majority rows.
    """),

    code(r"""
    def random_oversample(feature_matrix, labels, target_ratio=0.5, random_seed=0):
        '''Repeat minority training rows until the declared minority/majority ratio.'''
        feature_matrix = np.asarray(feature_matrix)
        labels = np.asarray(labels, dtype=int)
        random_generator = np.random.default_rng(random_seed)

        minority_indices = np.flatnonzero(labels == 1)
        majority_indices = np.flatnonzero(labels == 0)
        required_minority_count = int(np.ceil(target_ratio * len(majority_indices)))
        additional_count = required_minority_count - len(minority_indices)

        if additional_count <= 0:
            return feature_matrix.copy(), labels.copy()

        repeated_indices = random_generator.choice(
            minority_indices,
            size=additional_count,
            replace=True,
        )
        selected_indices = np.concatenate(
            [majority_indices, minority_indices, repeated_indices]
        )
        return feature_matrix[selected_indices], labels[selected_indices]


    def random_undersample(feature_matrix, labels, target_ratio=0.5, random_seed=0):
        '''Drop majority training rows until the declared minority/majority ratio.'''
        feature_matrix = np.asarray(feature_matrix)
        labels = np.asarray(labels, dtype=int)
        random_generator = np.random.default_rng(random_seed)

        minority_indices = np.flatnonzero(labels == 1)
        majority_indices = np.flatnonzero(labels == 0)
        majority_to_keep = int(np.floor(len(minority_indices) / target_ratio))
        majority_to_keep = min(majority_to_keep, len(majority_indices))
        kept_majority_indices = random_generator.choice(
            majority_indices,
            size=majority_to_keep,
            replace=False,
        )
        selected_indices = np.concatenate([kept_majority_indices, minority_indices])
        return feature_matrix[selected_indices], labels[selected_indices]


    oversampled_train_features, oversampled_train_labels = random_oversample(
        train_features_scaled,
        train_labels,
    )
    undersampled_train_features, undersampled_train_labels = random_undersample(
        train_features_scaled,
        train_labels,
    )

    print("original counts:", np.bincount(train_labels))
    print("oversampled counts:", np.bincount(oversampled_train_labels))
    print("undersampled counts:", np.bincount(undersampled_train_labels))

    assert len(oversampled_train_labels) > len(train_labels)
    assert len(undersampled_train_labels) < len(train_labels)
    """),

    md(r"""
    ## 9 · Nearest-neighbour distance comes before SMOTE

    For vectors $a$ and $b$, Euclidean distance is:

    $$
    d(a,b)=\sqrt{\sum_{j=1}^{d}(a_j-b_j)^2}
    $$

    SMOTE selects a minority row $x_i$, chooses one of its nearby minority neighbours
    $x_j$, draws $\lambda\in[0,1]$, and interpolates:

    $$
    x_{new}=x_i+\lambda(x_j-x_i)
    $$

    If $x_i=[0,2]$, $x_j=[2,4]$, and $\lambda=0.25$:

    $$
    x_{new}=[0,2]+0.25[2,2]=[0.5,2.5]
    $$

    Distance is scale-sensitive. A feature measured in thousands can dominate one
    measured between zero and one. That is why this lesson fits scaling on training
    data before neighbour search. Ordinary interpolation is also inappropriate for
    raw category codes, sparse indicators, isolated outliers, or mixed-class overlap.
    """),

    code(r"""
    first_minority_point = np.array([0.0, 2.0])
    neighbour_minority_point = np.array([2.0, 4.0])
    interpolation_fraction = 0.25

    neighbour_distance = np.sqrt(
        np.sum((first_minority_point - neighbour_minority_point) ** 2)
    )
    synthetic_point = first_minority_point + interpolation_fraction * (
        neighbour_minority_point - first_minority_point
    )

    print("Euclidean distance:", round(neighbour_distance, 4))
    print("synthetic point:", synthetic_point)

    assert np.isclose(neighbour_distance, np.sqrt(8))
    assert np.allclose(synthetic_point, [0.5, 2.5])
    """),

    md(r"""
    ## 10 · Implement guarded SMOTE for continuous training features

    SMOTE does not know whether a synthetic point is scientifically valid. It assumes
    interpolation between nearby minority rows is meaningful.

    Our educational implementation checks:

    - binary labels 0 and 1;
    - enough minority rows for the requested neighbours;
    - a target ratio that actually requires new rows;
    - finite continuous feature values.

    It still lacks production optimizations and mixed-feature strategies. Use a tested
    library pipeline for practical work.
    """),

    code(r"""
    def smote_continuous_training_data(
        feature_matrix,
        labels,
        target_ratio=0.5,
        number_of_neighbors=5,
        random_seed=0,
    ):
        '''Create interpolated minority rows from already-scaled training features.'''
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        labels = np.asarray(labels, dtype=int)
        random_generator = np.random.default_rng(random_seed)

        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("SMOTE lesson expects binary labels 0 and 1")
        if not np.isfinite(feature_matrix).all():
            raise ValueError("feature matrix must contain finite continuous values")

        minority_features = feature_matrix[labels == 1]
        majority_count = int(np.sum(labels == 0))
        required_minority_count = int(np.ceil(target_ratio * majority_count))
        synthetic_count = required_minority_count - len(minority_features)

        if synthetic_count <= 0:
            return feature_matrix.copy(), labels.copy()
        if len(minority_features) <= number_of_neighbors:
            raise ValueError("number_of_neighbors must be smaller than minority count")

        synthetic_features = np.empty((synthetic_count, feature_matrix.shape[1]))

        for synthetic_index in range(synthetic_count):
            source_index = random_generator.integers(len(minority_features))
            source_point = minority_features[source_index]

            # Compute distances only among real minority training rows.
            distances = np.linalg.norm(minority_features - source_point, axis=1)
            nearest_indices = np.argsort(distances)[1 : number_of_neighbors + 1]
            neighbour_index = random_generator.choice(nearest_indices)
            neighbour_point = minority_features[neighbour_index]

            interpolation_fraction = random_generator.random()
            synthetic_features[synthetic_index] = source_point + interpolation_fraction * (
                neighbour_point - source_point
            )

        resampled_features = np.vstack([feature_matrix, synthetic_features])
        resampled_labels = np.concatenate(
            [labels, np.ones(synthetic_count, dtype=int)]
        )
        return resampled_features, resampled_labels


    smote_train_features, smote_train_labels = smote_continuous_training_data(
        train_features_scaled,
        train_labels,
        target_ratio=0.5,
        number_of_neighbors=5,
        random_seed=42,
    )

    print("training counts before SMOTE:", np.bincount(train_labels))
    print("training counts after SMOTE:", np.bincount(smote_train_labels))
    print("validation counts unchanged:", np.bincount(validation_labels))

    assert len(smote_train_labels) > len(train_labels)
    assert len(validation_labels) == 240
    """),

    md(r"""
    ## 11 · Put preprocessing and resampling inside each training fold

    The correct fold sequence is:

    ```mermaid
    flowchart LR
        A[Original development rows] --> B[Create one CV split]
        B --> C[Fit scaler on fold training]
        C --> D[SMOTE fold training only]
        D --> E[Fit classifier]
        E --> F[Transform and score untouched fold validation]
    ```

    Resampling before splitting allows duplicated or synthetic information derived
    from a held-out row to influence training. Fitting the scaler before the fold also
    leaks held-out distribution information.

    `imblearn.pipeline.Pipeline` understands that a sampler runs during fitting but
    not during prediction. A standard sklearn pipeline does not accept a sampler in
    the same role.
    """),

    code(r"""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
    from sklearn.model_selection import StratifiedKFold, cross_validate

    fold_safe_smote_pipeline = ImbalancedPipeline(
        steps=[
            ("scale", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    stratified_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validation sees development rows only; the final test remains absent.
    cross_validation_results = cross_validate(
        fold_safe_smote_pipeline,
        development_features,
        development_labels,
        cv=stratified_folds,
        scoring={"average_precision": "average_precision", "recall": "recall"},
        return_train_score=False,
    )

    print("fold average precision:", cross_validation_results["test_average_precision"].round(3))
    print("fold recall:", cross_validation_results["test_recall"].round(3))
    print("test status: still sealed")

    assert len(cross_validation_results["test_average_precision"]) == 5
    """),

    md(r"""
    ## 12 · Compare methods on the same validation partition

    We compare six declared configurations:

    - always-negative baseline;
    - plain logistic regression at threshold 0.5;
    - the same plain model at its validation-selected threshold;
    - class-weighted logistic regression at 0.5;
    - random oversampling at ratio 0.5;
    - random undersampling at ratio 0.5;
    - SMOTE at ratio 0.5.

    Class weighting and sampling can change the learned boundary and ranking; threshold
    movement changes only the final decision. They may produce similar confusion
    counts in one example, but they are not the same operation.

    We also print mean predicted probability as a calibration warning. A changed mean
    is not a complete calibration diagnosis; formal recalibration is deferred.
    """),

    code(r"""
    def fit_logistic_on_resampled_training(resampled_features, resampled_labels):
        '''Fit the same logistic model so only the training-row strategy changes.'''
        fitted_model = LogisticRegression(max_iter=1000, random_state=42)
        fitted_model.fit(resampled_features, resampled_labels)
        return fitted_model


    oversampled_model = fit_logistic_on_resampled_training(
        oversampled_train_features,
        oversampled_train_labels,
    )
    undersampled_model = fit_logistic_on_resampled_training(
        undersampled_train_features,
        undersampled_train_labels,
    )
    smote_model = fit_logistic_on_resampled_training(
        smote_train_features,
        smote_train_labels,
    )

    validation_configurations = [
        ("always negative", np.zeros(len(validation_labels)), 0.5, None),
        ("plain @ 0.5", plain_validation_probabilities, 0.5, plain_logistic_model),
        ("plain @ selected", plain_validation_probabilities, selected_plain_threshold, plain_logistic_model),
        ("class weight", weighted_validation_probabilities, 0.5, weighted_logistic_model),
        ("random oversample", oversampled_model.predict_proba(validation_features_scaled)[:, 1], 0.5, oversampled_model),
        ("random undersample", undersampled_model.predict_proba(validation_features_scaled)[:, 1], 0.5, undersampled_model),
        ("SMOTE", smote_model.predict_proba(validation_features_scaled)[:, 1], 0.5, smote_model),
    ]

    comparison_records = []
    configuration_registry = {}

    for configuration_name, probabilities, threshold, fitted_model in validation_configurations:
        decisions = (probabilities >= threshold).astype(int)
        summary = summarize_binary_decisions(validation_labels, decisions)
        comparison_records.append(
            {
                "configuration": configuration_name,
                "threshold": threshold,
                "mean_probability": float(np.mean(probabilities)),
                **summary,
            }
        )
        configuration_registry[configuration_name] = {
            "model": fitted_model,
            "threshold": threshold,
        }

    validation_comparison = pd.DataFrame(comparison_records).sort_values(
        ["cost", "FN", "FP"],
        ignore_index=True,
    )
    selected_configuration_name = validation_comparison.iloc[0]["configuration"]

    print(validation_comparison.to_string(index=False))
    print("selected from validation:", selected_configuration_name)
    print("actual validation positive fraction:", round(validation_labels.mean(), 4))
    print("test status: still sealed")

    assert selected_configuration_name != "always negative"
    assert validation_comparison.iloc[0]["cost"] <= validation_comparison.iloc[-1]["cost"]
    """),

    md(r"""
    ## 13 · Mini-project decision and one sealed test

    **Project goal:** minimize the declared inspection cost while reporting precision,
    recall, and accuracy for context.

    **Dataset columns:** six anonymous continuous sensor measurements. They are
    anonymous because this synthetic dataset demonstrates workflow, not domain truth.

    **Evaluation criteria:**

    - method and threshold selected from validation only;
    - scaler fitted from training only;
    - samplers used on scaled training rows only;
    - test transformed but never resampled;
    - exactly one final test summary;
    - no claim that the winning method is universally best.

    The final test may disagree with validation because rare positive counts are
    small. Report the result; do not reopen method selection on this test.
    """),

    code(r"""
    # Freeze the selected configuration before touching test features.
    selected_configuration = configuration_registry[selected_configuration_name]
    selected_model = selected_configuration["model"]
    selected_threshold = selected_configuration["threshold"]

    sealed_test_features_scaled = training_scaler.transform(sealed_test_features)

    if selected_model is None:
        final_test_probabilities = np.zeros(len(sealed_test_labels))
    else:
        final_test_probabilities = selected_model.predict_proba(
            sealed_test_features_scaled
        )[:, 1]

    final_test_decisions = (
        final_test_probabilities >= selected_threshold
    ).astype(int)
    final_test_summary = summarize_binary_decisions(
        sealed_test_labels,
        final_test_decisions,
    )

    print("selected configuration:", selected_configuration_name)
    print("frozen threshold:", selected_threshold)
    print("final sealed-test rows:", len(sealed_test_labels))
    print(pd.Series(final_test_summary))
    print("this is a final estimate, not a new method-selection table")

    assert len(final_test_decisions) == len(sealed_test_labels)
    assert 0 <= final_test_summary["recall"] <= 1
    assert final_test_summary["cost"] >= 0
    """),

    md(r"""
    ## 14 · Practice, solutions, and mastery checkpoint

    ### Worked example

    With 90 negative and 10 positive rows, the imbalance ratio is 9. An always-negative
    classifier has 90% accuracy and zero recall. If $C_{FN}=10$, its decision cost is
    $10\times10=100$.

    ### Guided practice

    1. Calculate the ratio for 980 negative and 20 positive rows.
    2. Build confusion counts for actual $[0,0,1,1]$ and predicted $[0,1,0,1]$.
    3. Compare costs at two thresholds using $C_{FP}=1,C_{FN}=5$.
    4. Calculate balanced class weights for 900 negative and 100 positive rows.
    5. Interpolate between $[1,2]$ and $[5,6]$ using $\lambda=0.25$.

    ### Independent practice

    6. Add input validation to both random sampling functions.
    7. Compare target sampling ratios 0.25, 0.5, and 1.0 on validation cost.
    8. Prove that SMOTE never changes validation row counts in the project.
    9. Add a review-capacity constraint to threshold selection.
    10. Compare average precision before and after weighting without changing test.

    ### Challenge

    Rebuild the defect project without copying. Include a positive-class contract,
    measured class ratio, always-negative baseline, validation-selected threshold, class
    weights, fold-safe SMOTE pipeline, configuration registry, and exactly one test.

    ### Self-check

    1. Why can high accuracy coexist with zero recall?
    2. What changes when a threshold moves?
    3. What changes when class weights are applied?
    4. What information does undersampling discard?
    5. Why must SMOTE features have compatible scales?
    6. Why must SMOTE run inside a training fold?
    7. Are class weighting and oversampling always identical?
    8. Why should test not choose the sampling ratio?

    ### Solution and scoring rubric

    1. $980/20=49$ negative rows per positive row.
    2. TN=1, FP=1, FN=1, TP=1.
    3. Calculate $FP+5FN$ for each threshold; lower cost wins on validation.
    4. $w_0=1000/(2\times900)\approx0.556$ and $w_1=1000/(2\times100)=5$.
    5. $[1,2]+0.25[4,4]=[2,3]$.

    Score the eight self-check answers at two points each and the challenge at four
    points. Full credit requires both mechanism and leakage reasoning.

    ### Common mistakes

    - Calling imbalance a problem without defining the positive class and costs.
    - Reporting accuracy without confusion counts.
    - Selecting a threshold from test data.
    - Resampling validation or test rows.
    - Fitting scaling before the split or fold.
    - Applying ordinary SMOTE to raw categories or incompatible scales.
    - Treating class weights, oversampling, and thresholding as identical.
    - Assuming balanced 50/50 training data is always optimal.
    - Claiming weighting automatically preserves calibration.
    - Combining every intervention without measuring incremental value.

    ### Readiness threshold

    Score at least **16/20**, including correct confusion counts, threshold cost,
    class weights, SMOTE interpolation, fold boundary, and sealed-test workflow.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Explain this chain without notes:

    positive event  
    → class counts  
    → majority baseline  
    → confusion matrix  
    → validation threshold  
    → class weights or training-only sampling  
    → fold-safe comparison  
    → one sealed test.

    ### Teach it back

    Explain why threshold movement can improve a decision without retraining, while
    class weighting and resampling can change the fitted model. Then explain why none
    may inspect validation or test rows during fitting.

    ### Memory aid

    **Define the rare event, validate the decision, resample training only, and test
    the frozen choice once.**

    ### Next dependency

    Validated tree ensembles and imbalance decisions  
    → required before SHAP explanations  
    → because an explanation cannot rescue a model selected with leaked or misleading
    evidence.
    """),
]


build("03_ml_engineering/04_imbalanced_learning.ipynb", cells)
