"""Build EVAL-01: validation-safe classical ML evaluation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # EVAL-01 · Classical ML Evaluation as a Decision Process

    **Prerequisites:** MLE-01, MLE-02, MLE-04, MLE-05, and PROD-04  
    **Estimated study time:** 10–12 hours, including practice  
    **Next lesson:** MLE-06 · Unsupervised Learning Foundations

    Earlier lessons taught individual metrics, leakage boundaries, imbalance methods,
    explanations, and experiment records. This lesson connects them into one defensible
    evaluation process.

    A metric is not a trophy. It is evidence about a declared model, population,
    prediction time, and decision. Evaluation succeeds when another person can reproduce
    the result and understand what it does—and does not—justify.

    ### Scope boundary

    The main project is binary classification. A separate section covers the regression
    contract. Multiclass reporting appears as an extension. Ranking and retrieval metrics
    move to EVAL-03, where relevance judgments, query sampling, and cutoffs can be taught
    correctly.

    This lesson does not teach deployment monitoring, causal inference, or formal fairness
    assessment. It prepares reliable evidence for those later decisions.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - write an evaluation contract before fitting a model;
    - choose stratified, grouped, or temporal splits from the data-generating process;
    - keep fitting, calibration, selection, and final test responsibilities separate;
    - compare models with the same rows, metrics, and baselines;
    - distinguish discrimination, calibration, and decision quality;
    - select a cost-sensitive threshold on development data only;
    - calculate bootstrap uncertainty and a paired model difference;
    - audit operational slices with support counts;
    - use explanations as diagnostic evidence rather than performance metrics;
    - record the full comparison in a reproducible experiment table;
    - evaluate one frozen model and threshold once on sealed test data;
    - apply a separate, correct evaluation contract to regression.

    ### Learning path

    ```mermaid
    flowchart LR
        A[Declare decision] --> B[Choose split unit]
        B --> C[Fit candidates]
        C --> D[Calibrate probabilities]
        D --> E[Compare on selection rows]
        E --> F[Choose threshold and model]
        F --> G[Quantify uncertainty]
        G --> H[Audit slices and reliance]
        H --> I[Record and freeze]
        I --> J[Open test once]
    ```

    Prediction contract  
    → required before metric choice  
    → because “good” performance depends on the real decision and error costs.

    Validation boundaries  
    → required before model comparison  
    → because repeated selection on test data turns the test into training data.

    Uncertainty  
    → required before declaring a winner  
    → because a small observed difference may be sampling noise.
    """),

    md(r"""
    ## 2 · Declare the decision before choosing metrics

    A subscription service predicts whether an active customer will cancel within the
    next 30 days.

    **Unit of observation:** one active customer at weekly scoring time.  
    **Positive class:** `1` means cancellation within 30 days.  
    **Prediction time:** Monday before any retention contact.  
    **Action:** offer a retention call when predicted positive.  
    **False positive:** call a customer who would stay; cost 3 units.  
    **False negative:** miss a customer who cancels; cost 35 units.

    We predeclare three kinds of evidence:

    - **discrimination:** average precision and ROC AUC;
    - **probability quality:** Brier score and reliability bins;
    - **decision quality:** precision, recall, and cost at the frozen threshold.

    Accuracy is useful context, especially against a baseline. It is incomplete—not
    “almost always wrong.”
    """),

    md(r"""
    ## 3 · Give every data partition exactly one job

    We use four partitions:

    | Partition | Job | Must not do |
    |---|---|---|
    | Training | fit candidate models and preprocessing | calibrate, select, or report final performance |
    | Calibration | learn score-to-probability mappings | select model or threshold |
    | Selection | compare candidates, choose threshold, inspect slices | claim untouched final performance |
    | Sealed test | one final report after freezing | change model, threshold, background, or metric |

    The row split must match the data-generating process:

    - ordinary independent customers: stratified random split;
    - multiple rows per customer, patient, device, or household: grouped split;
    - future deployment: chronological split;
    - both group and time structure: respect both, even if it reduces sample size.

    Random rows are not automatically valid. The independent unit—not the file row—is
    what must remain separated.
    """),

    code(r"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    modeled_feature_names = [
        "recent_usage",
        "support_contacts",
        "tenure_months",
        "payment_variability",
        "plan_changes",
        "engagement_score",
        "billing_delay",
    ]
    modeled_features_array, all_labels = make_classification(
        n_samples=2400,
        n_features=len(modeled_feature_names),
        n_informative=6,
        n_redundant=1,
        weights=[0.87, 0.13],
        class_sep=1.05,
        flip_y=0.02,
        random_state=42,
    )
    all_features = pd.DataFrame(modeled_features_array, columns=modeled_feature_names)
    # Add this column after data generation so its name and meaning are genuinely true.
    random_reference_generator = np.random.default_rng(42)
    all_features["random_reference"] = random_reference_generator.normal(size=len(all_features))
    feature_names = list(all_features.columns)

    development_features, sealed_test_features, development_labels, sealed_test_labels = train_test_split(
        all_features,
        all_labels,
        test_size=0.20,
        stratify=all_labels,
        random_state=42,
    )
    train_features, remaining_features, train_labels, remaining_labels = train_test_split(
        development_features,
        development_labels,
        test_size=0.40,
        stratify=development_labels,
        random_state=42,
    )
    calibration_features, selection_features, calibration_labels, selection_labels = train_test_split(
        remaining_features,
        remaining_labels,
        test_size=0.50,
        stratify=remaining_labels,
        random_state=42,
    )

    partition_summary = pd.DataFrame(
        {
            "partition": ["training", "calibration", "selection", "sealed test"],
            "rows": [len(train_labels), len(calibration_labels), len(selection_labels), len(sealed_test_labels)],
            "positive_rate": [
                train_labels.mean(),
                calibration_labels.mean(),
                selection_labels.mean(),
                sealed_test_labels.mean(),
            ],
        }
    )
    print(partition_summary.round(3).to_string(index=False))
    print("test status: sealed")

    assert sum(partition_summary["rows"]) == len(all_labels)
    assert set(train_features.index).isdisjoint(selection_features.index)
    """),

    md(r"""
    ## 4 · Fit candidates and baselines under the same rules

    A comparison is fair only when candidates receive the same training rows and are
    scored on the same selection rows.

    We compare logistic regression, random forest, and gradient boosting. Two baselines
    keep the numbers honest:

    - **always negative:** a decision baseline with zero recall;
    - **prevalence probability:** every row receives the training positive rate.

    The prevalence baseline is especially important for Brier score and log loss. A
    Brier score of `0.25` is not a universal “random” reference; the correct simple
    reference depends on event prevalence.
    """),

    code(r"""
    candidate_models = {
        "logistic": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        ),
        "gradient boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    for candidate_name, candidate_model in candidate_models.items():
        candidate_model.fit(train_features, train_labels)
        print(candidate_name, "fitted on", len(train_labels), "training rows")

    training_prevalence = float(train_labels.mean())
    prevalence_selection_probabilities = np.full(len(selection_labels), training_prevalence)
    always_negative_selection_decisions = np.zeros(len(selection_labels), dtype=int)

    print("training prevalence baseline:", round(training_prevalence, 4))
    print("always-negative selection accuracy:", round(accuracy_score(selection_labels, always_negative_selection_decisions), 3))
    print("always-negative selection recall:", round(recall_score(selection_labels, always_negative_selection_decisions, zero_division=0), 3))

    assert recall_score(selection_labels, always_negative_selection_decisions, zero_division=0) == 0
    """),

    md(r"""
    ## 5 · Separate ranking from probability calibration

    A model can rank likely churners well while producing poor probabilities.

    - **Average precision (AP):** summarizes the precision–recall ranking curve.
    - **ROC AUC:** probability that a random positive ranks above a random negative.
    - **Brier score:** mean squared probability error.
    - **Log loss:** strongly penalizes confident wrong probabilities.

    We fit a simple sigmoid calibrator on calibration rows only. It maps each candidate's
    clipped log-odds score to an observed probability. Selection rows do not fit this
    mapping.

    Calibration is not a cosmetic correction. It changes the meaning of a number such
    as `0.70`, while a monotonic calibrator normally preserves ranking order.
    """),

    code(r"""
    def probability_to_log_odds(probabilities):
        '''Convert clipped probabilities to finite log-odds for sigmoid calibration.'''
        clipped_probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(clipped_probabilities / (1 - clipped_probabilities))


    calibrated_candidates = {}
    for candidate_name, candidate_model in candidate_models.items():
        calibration_raw_probabilities = candidate_model.predict_proba(calibration_features)[:, 1]
        calibration_log_odds = probability_to_log_odds(calibration_raw_probabilities).reshape(-1, 1)

        sigmoid_calibrator = LogisticRegression(max_iter=1000, random_state=42)
        sigmoid_calibrator.fit(calibration_log_odds, calibration_labels)

        selection_raw_probabilities = candidate_model.predict_proba(selection_features)[:, 1]
        selection_calibrated_probabilities = sigmoid_calibrator.predict_proba(
            probability_to_log_odds(selection_raw_probabilities).reshape(-1, 1)
        )[:, 1]

        calibrated_candidates[candidate_name] = {
            "model": candidate_model,
            "calibrator": sigmoid_calibrator,
            "selection_probabilities": selection_calibrated_probabilities,
        }

    probability_comparison_records = [
        {
            "candidate": "prevalence baseline",
            "average_precision": average_precision_score(selection_labels, prevalence_selection_probabilities),
            "roc_auc": 0.5,
            "brier": brier_score_loss(selection_labels, prevalence_selection_probabilities),
            "log_loss": log_loss(selection_labels, prevalence_selection_probabilities),
        }
    ]

    for candidate_name, candidate_artifacts in calibrated_candidates.items():
        probabilities = candidate_artifacts["selection_probabilities"]
        probability_comparison_records.append(
            {
                "candidate": candidate_name,
                "average_precision": average_precision_score(selection_labels, probabilities),
                "roc_auc": roc_auc_score(selection_labels, probabilities),
                "brier": brier_score_loss(selection_labels, probabilities),
                "log_loss": log_loss(selection_labels, probabilities),
            }
        )

    probability_comparison = pd.DataFrame(probability_comparison_records)
    print(probability_comparison.round(4).to_string(index=False))
    print("test status: still sealed")

    assert probability_comparison.loc[probability_comparison["candidate"] != "prevalence baseline", "average_precision"].max() > training_prevalence
    """),

    md(r"""
    ## 6 · Read reliability bins with support counts

    A reliability bin compares mean predicted probability with observed event rate:

    $$
    \operatorname{gap}_b=\left|\overline p_b-\overline y_b\right|
    $$

    **Symbols:** $b$ is one probability bin; $\overline p_b$ is its mean predicted
    probability; and $\overline y_b$ is its observed positive fraction.

    Expected Calibration Error averages these gaps by bin size. ECE depends on bin
    boundaries and can hide cancellation or sparse regions, so always show the table or
    reliability plot with support. “Observed positive fraction” is not accuracy.
    """),

    code(r"""
    def reliability_table(actual_labels, probabilities, number_of_bins=8):
        '''Return equal-width probability bins with observed event rates and support.'''
        bin_edges = np.linspace(0, 1, number_of_bins + 1)
        bin_ids = np.clip(np.digitize(probabilities, bin_edges[1:-1]), 0, number_of_bins - 1)
        records = []

        for bin_id in range(number_of_bins):
            bin_mask = bin_ids == bin_id
            if not np.any(bin_mask):
                continue
            mean_probability = float(np.mean(probabilities[bin_mask]))
            observed_positive_rate = float(np.mean(actual_labels[bin_mask]))
            records.append(
                {
                    "bin": bin_id,
                    "support": int(np.sum(bin_mask)),
                    "mean_probability": mean_probability,
                    "observed_positive_rate": observed_positive_rate,
                    "absolute_gap": abs(mean_probability - observed_positive_rate),
                }
            )
        return pd.DataFrame(records)


    ranking_leader_name = probability_comparison.loc[
        probability_comparison["candidate"] != "prevalence baseline"
    ].sort_values("average_precision", ascending=False).iloc[0]["candidate"]
    ranking_leader_probabilities = calibrated_candidates[ranking_leader_name]["selection_probabilities"]
    leader_reliability = reliability_table(selection_labels, ranking_leader_probabilities)
    estimated_ece = np.average(
        leader_reliability["absolute_gap"],
        weights=leader_reliability["support"],
    )

    print("candidate:", ranking_leader_name)
    print(leader_reliability.round(4).to_string(index=False))
    print("bin-dependent ECE:", round(float(estimated_ece), 4))

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    axis.scatter(
        leader_reliability["mean_probability"],
        leader_reliability["observed_positive_rate"],
        s=leader_reliability["support"] * 2,
        color="tab:blue",
        alpha=0.75,
        label="selection bins; size = support",
    )
    axis.set_xlabel("mean calibrated probability")
    axis.set_ylabel("observed positive fraction")
    axis.set_title("Reliability needs both gaps and support")
    axis.legend()
    plt.show()
    """),

    md(r"""
    ## 7 · Select the decision threshold on declared costs

    Positive means “will cancel,” so:

    - false positive: unnecessary retention call, cost $C_{FP}=3$;
    - false negative: missed cancellation, cost $C_{FN}=35$.

    At threshold $t$:

    $$
    C(t)=C_{FP}FP(t)+C_{FN}FN(t)
    $$

    Lowering the threshold normally predicts more positives: false negatives may fall
    while false positives rise. The empirical optimum depends on validation evidence,
    score quality, capacity, and declared costs. It is not guaranteed to equal `0.5` or
    a simple analytical formula.
    """),

    code(r"""
    false_positive_cost = 3
    false_negative_cost = 35
    candidate_thresholds = np.linspace(0.05, 0.95, 37)

    def summarize_decisions(actual_labels, probabilities, threshold):
        '''Calculate confusion counts, decision metrics, and declared total cost.'''
        decisions = (probabilities >= threshold).astype(int)
        true_positive = int(np.sum((actual_labels == 1) & (decisions == 1)))
        false_positive = int(np.sum((actual_labels == 0) & (decisions == 1)))
        false_negative = int(np.sum((actual_labels == 1) & (decisions == 0)))
        true_negative = int(np.sum((actual_labels == 0) & (decisions == 0)))
        return {
            "TN": true_negative,
            "FP": false_positive,
            "FN": false_negative,
            "TP": true_positive,
            "accuracy": accuracy_score(actual_labels, decisions),
            "precision": precision_score(actual_labels, decisions, zero_division=0),
            "recall": recall_score(actual_labels, decisions, zero_division=0),
            "cost": false_positive_cost * false_positive + false_negative_cost * false_negative,
        }


    decision_records = []
    frozen_candidate_registry = {}

    for candidate_name, candidate_artifacts in calibrated_candidates.items():
        probabilities = candidate_artifacts["selection_probabilities"]
        threshold_rows = []
        for threshold in candidate_thresholds:
            threshold_rows.append(
                {"threshold": threshold, **summarize_decisions(selection_labels, probabilities, threshold)}
            )
        threshold_table = pd.DataFrame(threshold_rows)
        best_row = threshold_table.sort_values(["cost", "FN", "FP"]).iloc[0]
        decision_records.append({"candidate": candidate_name, **best_row.to_dict()})
        frozen_candidate_registry[candidate_name] = {
            **candidate_artifacts,
            "threshold": float(best_row["threshold"]),
        }

    decision_comparison = pd.DataFrame(decision_records).sort_values(
        ["cost", "FN", "FP"],
        ignore_index=True,
    )
    selected_candidate_name = str(decision_comparison.iloc[0]["candidate"])
    selected_threshold = float(decision_comparison.iloc[0]["threshold"])
    selected_selection_probabilities = calibrated_candidates[selected_candidate_name]["selection_probabilities"]

    print(decision_comparison[["candidate", "threshold", "FP", "FN", "precision", "recall", "cost"]].round(3).to_string(index=False))
    print("selected candidate:", selected_candidate_name)
    print("selected threshold:", round(selected_threshold, 3))
    print("test status: still sealed")

    assert decision_comparison.iloc[0]["cost"] <= decision_comparison.iloc[-1]["cost"]
    """),

    md(r"""
    ## 8 · Quantify uncertainty before declaring a winner

    Metrics are statistics calculated from a sample. If we observed different customers,
    the number would change.

    A pairs bootstrap repeatedly samples selection rows with replacement and recalculates
    the metric. Percentiles form an approximate interval. It assumes rows are independent;
    grouped or temporal data needs group-aware or block resampling.

    We also bootstrap the paired cost difference between the selected model and the next
    candidate. Resampling the same row indices preserves the head-to-head comparison.
    Because selection already chose the winner, treat this interval as descriptive—not
    a fresh confirmatory test.
    """),

    code(r"""
    def bootstrap_metric_interval(
        actual_labels,
        probabilities,
        metric_function,
        repetitions=600,
        random_seed=0,
    ):
        '''Bootstrap rows and return the 2.5%, median, and 97.5% percentiles.'''
        random_generator_local = np.random.default_rng(random_seed)
        metric_values = []
        row_count = len(actual_labels)

        for _ in range(repetitions):
            sampled_indices = random_generator_local.integers(0, row_count, row_count)
            sampled_labels = actual_labels[sampled_indices]
            if len(np.unique(sampled_labels)) < 2:
                continue
            metric_values.append(
                metric_function(sampled_labels, probabilities[sampled_indices])
            )

        return np.percentile(metric_values, [2.5, 50, 97.5])


    selected_ap_interval = bootstrap_metric_interval(
        selection_labels,
        selected_selection_probabilities,
        average_precision_score,
        repetitions=600,
        random_seed=42,
    )

    comparison_candidate_name = str(decision_comparison.iloc[1]["candidate"])
    comparison_probabilities = calibrated_candidates[comparison_candidate_name]["selection_probabilities"]
    comparison_threshold = frozen_candidate_registry[comparison_candidate_name]["threshold"]

    random_generator_bootstrap = np.random.default_rng(42)
    paired_cost_differences = []
    for _ in range(600):
        sampled_indices = random_generator_bootstrap.integers(0, len(selection_labels), len(selection_labels))
        sampled_labels = selection_labels[sampled_indices]
        selected_summary = summarize_decisions(
            sampled_labels,
            selected_selection_probabilities[sampled_indices],
            selected_threshold,
        )
        comparison_summary = summarize_decisions(
            sampled_labels,
            comparison_probabilities[sampled_indices],
            comparison_threshold,
        )
        paired_cost_differences.append(
            100 * (selected_summary["cost"] - comparison_summary["cost"]) / len(sampled_indices)
        )

    paired_difference_interval = np.percentile(paired_cost_differences, [2.5, 50, 97.5])

    print("selected AP 95% bootstrap interval:", selected_ap_interval.round(3))
    print("paired cost difference per 100 rows vs", comparison_candidate_name, ":", paired_difference_interval.round(2))
    print("negative paired values favor the selected candidate")
    print("interval width is evidence about uncertainty, not a pass/fail decoration")
    """),

    md(r"""
    ## 9 · Audit slices with denominators, not just percentages

    Overall performance can hide an operational failure. We predeclare two usage slices
    from a raw feature: lower recent usage and higher recent usage.

    Every slice report includes:

    - row support;
    - positive support and positive rate;
    - precision and recall at the frozen threshold;
    - cost per row.

    A dramatic percentage based on three positives is not stable evidence. Slice analysis
    diagnoses where to investigate; protected-group fairness requires additional metrics,
    governance, legal context, and data-quality review.
    """),

    code(r"""
    usage_cutoff = float(train_features["recent_usage"].median())
    selection_slice_labels = np.where(
        selection_features["recent_usage"].to_numpy() <= usage_cutoff,
        "lower recent usage",
        "higher recent usage",
    )

    slice_records = []
    for slice_name in np.unique(selection_slice_labels):
        slice_mask = selection_slice_labels == slice_name
        slice_summary = summarize_decisions(
            selection_labels[slice_mask],
            selected_selection_probabilities[slice_mask],
            selected_threshold,
        )
        slice_records.append(
            {
                "slice": slice_name,
                "rows": int(np.sum(slice_mask)),
                "positives": int(np.sum(selection_labels[slice_mask])),
                "positive_rate": float(np.mean(selection_labels[slice_mask])),
                "precision": slice_summary["precision"],
                "recall": slice_summary["recall"],
                "cost_per_row": slice_summary["cost"] / np.sum(slice_mask),
            }
        )

    slice_report = pd.DataFrame(slice_records)
    print(slice_report.round(3).to_string(index=False))

    assert slice_report["rows"].sum() == len(selection_labels)
    """),

    md(r"""
    ## 10 · Use explanation as diagnostic evidence

    MLE-05 taught that explanation and evaluation answer different questions.

    We use validation permutation importance to check whether the selected model relies
    on plausible signals or on `random_reference`. This does not increase AP, prove
    causality, or certify fairness. It can reveal a reason to reject or investigate a
    model even when aggregate metrics look good.

    The explanation uses selection rows because test is still sealed. If interpretation
    changes model choice, that is development work and belongs before final testing.
    """),

    code(r"""
    selected_model = frozen_candidate_registry[selected_candidate_name]["model"]
    selected_importance = permutation_importance(
        selected_model,
        selection_features,
        selection_labels,
        scoring="average_precision",
        n_repeats=12,
        random_state=42,
        n_jobs=1,
    )
    reliance_report = pd.Series(
        selected_importance.importances_mean,
        index=selection_features.columns,
        name="mean_AP_decrease",
    ).sort_values(ascending=False)

    print(reliance_report.round(4))
    print("random-reference reliance:", round(float(reliance_report["random_reference"]), 4))

    assert len(reliance_report) == selection_features.shape[1]
    """),

    md(r"""
    ## 11 · Regression needs a different evaluation contract

    Regression has no positive class or classification threshold.

    $$
    \operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
    $$

    $$
    \operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
    $$

    **Symbols:** $n$ is row count; $y_i$ is actual target; and $\hat y_i$ is prediction.
    MAE weights errors linearly; RMSE emphasizes large errors. $R^2$ compares squared
    error with predicting the target mean and can be negative on held-out data.

    Always compare with a simple mean or median baseline, report errors in target units,
    inspect residual slices, and quantify uncertainty. MAPE is unsafe near zero and can
    distort business meaning; never hide that problem with a tiny denominator constant.
    """),

    code(r"""
    regression_random_generator = np.random.default_rng(7)
    regression_features = regression_random_generator.normal(size=(700, 5))
    regression_target = (
        30
        + 8 * regression_features[:, 0]
        - 5 * regression_features[:, 1]
        + 3 * regression_features[:, 0] * regression_features[:, 2]
        + regression_random_generator.normal(0, 4, 700)
    )
    regression_train_features, regression_validation_features, regression_train_target, regression_validation_target = train_test_split(
        regression_features,
        regression_target,
        test_size=0.30,
        random_state=42,
    )

    regression_baseline_predictions = np.full(
        len(regression_validation_target),
        regression_train_target.mean(),
    )
    ridge_regression = Pipeline(
        steps=[("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
    )
    ridge_regression.fit(regression_train_features, regression_train_target)
    ridge_predictions = ridge_regression.predict(regression_validation_features)

    regression_report = pd.DataFrame(
        [
            {
                "candidate": "training-mean baseline",
                "MAE": mean_absolute_error(regression_validation_target, regression_baseline_predictions),
                "RMSE": mean_squared_error(regression_validation_target, regression_baseline_predictions) ** 0.5,
                "R2": r2_score(regression_validation_target, regression_baseline_predictions),
            },
            {
                "candidate": "ridge",
                "MAE": mean_absolute_error(regression_validation_target, ridge_predictions),
                "RMSE": mean_squared_error(regression_validation_target, ridge_predictions) ** 0.5,
                "R2": r2_score(regression_validation_target, ridge_predictions),
            },
        ]
    )
    print(regression_report.round(3).to_string(index=False))

    assert regression_report.loc[regression_report["candidate"] == "ridge", "RMSE"].iloc[0] < regression_report.loc[regression_report["candidate"] == "training-mean baseline", "RMSE"].iloc[0]
    """),

    md(r"""
    ## 12 · Record the evidence and freeze the choice

    An experiment record should answer:

    - which code and data definition produced the result;
    - which partitions fitted model, calibrator, and threshold;
    - which metrics and costs were declared before test;
    - which candidate and threshold were selected;
    - what uncertainty, slice, and reliance warnings remain.

    We create a compact table here. A production tracker may store the same fields in
    MLflow or another system, but the tool does not create the scientific discipline.
    """),

    code(r"""
    experiment_records = probability_comparison.merge(
        decision_comparison[["candidate", "threshold", "precision", "recall", "cost"]],
        on="candidate",
        how="left",
    )
    experiment_records["fit_partition"] = np.where(
        experiment_records["candidate"].eq("prevalence baseline"),
        "training prevalence",
        "training rows",
    )
    experiment_records["calibration_partition"] = np.where(
        experiment_records["candidate"].eq("prevalence baseline"),
        "none",
        "calibration rows",
    )
    experiment_records["selection_partition"] = "selection rows"
    experiment_records["test_used"] = False
    experiment_records["selected"] = experiment_records["candidate"].eq(selected_candidate_name)

    frozen_artifact = {
        "candidate": selected_candidate_name,
        "model": frozen_candidate_registry[selected_candidate_name]["model"],
        "calibrator": frozen_candidate_registry[selected_candidate_name]["calibrator"],
        "threshold": selected_threshold,
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": false_negative_cost,
        "feature_names": feature_names,
    }

    print(experiment_records.round(4).to_string(index=False))
    print("frozen candidate:", frozen_artifact["candidate"])
    print("frozen threshold:", round(frozen_artifact["threshold"], 3))
    print("test status: still sealed")

    assert experiment_records["test_used"].eq(False).all()
    """),

    md(r"""
    ## 13 · Open the sealed test exactly once

    The model, calibrator, threshold, costs, feature schema, and metrics are frozen.
    Test may now estimate performance on untouched rows.

    A disappointing test result is information, not permission to retune on test. Return
    to development with a new version and preserve a genuinely untouched future test or
    prospective evaluation.

    We report multiple predeclared metrics because they answer different questions. This
    is one evaluation event—not repeated selection.
    """),

    code(r"""
    frozen_raw_test_probabilities = frozen_artifact["model"].predict_proba(sealed_test_features)[:, 1]
    frozen_test_probabilities = frozen_artifact["calibrator"].predict_proba(
        probability_to_log_odds(frozen_raw_test_probabilities).reshape(-1, 1)
    )[:, 1]
    final_test_summary = summarize_decisions(
        sealed_test_labels,
        frozen_test_probabilities,
        frozen_artifact["threshold"],
    )

    final_test_report = {
        "candidate": frozen_artifact["candidate"],
        "threshold": frozen_artifact["threshold"],
        "rows": len(sealed_test_labels),
        "positives": int(np.sum(sealed_test_labels)),
        "average_precision": average_precision_score(sealed_test_labels, frozen_test_probabilities),
        "roc_auc": roc_auc_score(sealed_test_labels, frozen_test_probabilities),
        "brier": brier_score_loss(sealed_test_labels, frozen_test_probabilities),
        **final_test_summary,
    }

    print(pd.Series(final_test_report).round(4))
    print("final report only; no test-based model or threshold changes")

    assert final_test_report["rows"] == len(sealed_test_labels)
    assert 0 <= final_test_report["average_precision"] <= 1
    """),

    md(r"""
    ## 14 · Practice, solutions, and mastery checkpoint

    ### Worked example

    Positive means churn. At one threshold, FP=20 and FN=6. With $C_{FP}=3$ and
    $C_{FN}=35$, cost is $3\times20+35\times6=270$ units. Swapping the error labels
    would change the business decision completely.

    ### Guided practice

    1. Assign one job to training, calibration, selection, and test.
    2. Explain discrimination, calibration, and decision quality.
    3. Calculate cost for FP=8, FN=4 under the lesson contract.
    4. Explain why a reliability bin needs support.
    5. Explain why paired bootstrap resamples the same row indices for both models.

    ### Independent practice

    6. Replace the random split with a chronological split and describe the changed claim.
    7. Add group-aware bootstrap resampling for repeated customers.
    8. Compare sigmoid and isotonic calibration using a new development partition.
    9. Add a minimum-recall constraint to threshold selection.
    10. Create a slice report with uncertainty rather than point estimates alone.

    ### Challenge

    Rebuild the churn evaluation without copying. Include an evaluation contract, four
    partitions, three candidates, two baselines, calibration, validation-only threshold,
    bootstrap AP interval, paired difference, slice support, reliance diagnostic,
    experiment record, frozen artifact, and exactly one test report.

    ### Self-check

    1. Why is accuracy useful but incomplete?
    2. Why can good AP coexist with poor Brier score?
    3. Which rows may fit a calibrator?
    4. Which rows may choose a threshold?
    5. Why is a narrow metric difference insufficient by itself?
    6. When is an ordinary row bootstrap invalid?
    7. Can SHAP or permutation importance replace held-out performance?
    8. What must happen before the test is opened?

    ### Solution and scoring rubric

    1. Training fits model; calibration fits mapping; selection chooses; test estimates once.
    2. AP measures ranking, while Brier measures probability error.
    3. Cost is $3\times8+35\times4=164$ units.
    4. A large gap from very few rows is unstable.
    5. Pairing preserves each row's shared difficulty across candidates.

    Award two points for each self-check and four points for the challenge explanation.
    Full credit requires correct data boundaries and error meanings.

    ### Common mistakes

    - Choosing metrics after seeing which model wins.
    - Splitting rows when customers or time are the independent unit.
    - Fitting calibration on selection or test rows.
    - Reversing false-positive and false-negative business meanings.
    - Choosing and reporting a threshold on the same final test.
    - Calling trapezoidal PR area and Average Precision identical.
    - Reporting ECE without bins and support.
    - Declaring a winner without uncertainty or paired evidence.
    - Reporting slices without denominators.
    - Treating explanations as performance, causality, or fairness proof.
    - Hiding MAPE's near-zero failure with a tiny constant.
    - Logging metrics without data, code, threshold, and cost definitions.

    ### Readiness threshold

    Score at least **16/20** and correctly explain the four partitions, three evidence
    types, error costs, bootstrap unit, frozen artifact, and one-test rule.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Explain this chain without notes:

    decision contract  
    → valid split unit  
    → shared candidates and baselines  
    → separate calibration  
    → selection metrics and threshold  
    → uncertainty and slices  
    → diagnostic explanation  
    → experiment record  
    → frozen artifact  
    → one sealed test.

    ### Teach it back

    Explain why the model with highest AP may not minimize business cost, why calibration
    must use separate rows, and why test performance cannot select a new threshold.

    ### Memory aid

    **Declare first, separate every job, compare with uncertainty, freeze the decision,
    and test once.**

    ### Next dependency

    Reproducible supervised evaluation  
    → required before unsupervised learning evaluation  
    → because clustering and dimensionality reduction need even more careful proxy
    objectives when labels are absent.
    """),
]


build("08_evaluation/01_classical_ml_evaluation.ipynb", cells)
