"""Build MLE-05: beginner-first model interpretability and SHAP."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # MLE-05 · Model Interpretability and SHAP

    **Prerequisites:** CML-01 through CML-06, MLE-01, MLE-02, and MLE-03  
    **Estimated study time:** 10–12 hours, including practice  
    **Next lesson:** EVAL-01 · Classical ML Evaluation

    A model can score well and still rely on the wrong signal. Interpretability helps
    us ask what the fitted model learned, where it behaves strangely, and why one
    prediction differs from a reference prediction.

    We will not begin with SHAP. That would be like learning an aircraft dashboard
    before learning what speed and direction mean. We first build an interpretation
    ladder, then use SHAP for the question it is designed to answer.

    ### Scope boundary

    This lesson focuses on supervised tabular models. It covers coefficients, tree
    impurity importance, permutation importance, PDP, ICE, exact small Shapley values,
    and TreeSHAP. It defers causal inference, formal fairness assessment, counterfactual
    recourse, image attribution, and explanation monitoring.

    An explanation describes a fitted model. It does not prove that the model is good,
    fair, causal, or legally compliant.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - decide whether a global, effect-shape, or local question is being asked;
    - explain a standardized linear coefficient without calling it causal;
    - describe how tree impurity importance is accumulated and why it can mislead;
    - calculate validation permutation importance manually;
    - build and read partial-dependence and ICE curves;
    - calculate a two-feature Shapley allocation by hand;
    - explain every symbol in the general Shapley formula;
    - implement exact interventional Shapley values for a tiny model;
    - explain how background data, correlation, and output scale change meaning;
    - use TreeSHAP and verify its additive reconstruction;
    - explain a frozen model without reopening test-based model selection.

    ### Learning path

    ```mermaid
    flowchart LR
        A[Validate model] --> B[Coefficients]
        B --> C[Tree importance]
        C --> D[Permutation importance]
        D --> E[PDP and ICE]
        E --> F[Two-feature game]
        F --> G[Exact Shapley]
        G --> H[Background and dependence]
        H --> I[TreeSHAP]
        I --> J[One held-out explanation]
    ```

    Validation  
    → required before interpretation  
    → because a faithful explanation of a broken model is still misleading.

    Global importance  
    → required before local attribution  
    → because students should separate “what matters overall?” from “why this row?”.

    Background distributions  
    → required before SHAP  
    → because an attribution is always relative to a reference.
    """),

    md(r"""
    ## 2 · Start with a validated model and a precise question

    A delivery company predicts arrival time from distance, traffic, rain, driver
    experience, a correlated distance estimate, and a random tracking signal.

    Before explaining anything, ask:

    1. **Global reliance:** which features affect performance across many rows?
    2. **Effect shape:** how does prediction usually change as one feature changes?
    3. **Local attribution:** why is this prediction above or below a reference?

    These are different questions and need different tools.

    We create training, validation, and sealed-test partitions. Validation confirms
    that the model has useful predictive evidence. Test remains sealed until the mini
    project. Interpretation never repairs leakage or weak evaluation.
    """),

    code(r"""
    import itertools
    from math import factorial

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    random_generator = np.random.default_rng(42)
    number_of_rows = 900

    distance_km = random_generator.uniform(1, 30, number_of_rows)
    traffic_index = random_generator.uniform(0, 10, number_of_rows)
    rain_flag = random_generator.binomial(1, 0.28, number_of_rows)
    driver_experience_years = random_generator.uniform(0, 12, number_of_rows)
    distance_estimate_km = distance_km + random_generator.normal(0, 0.8, number_of_rows)
    random_tracking_signal = random_generator.normal(0, 1, number_of_rows)

    delivery_features = pd.DataFrame(
        {
            "distance_km": distance_km,
            "traffic_index": traffic_index,
            "rain_flag": rain_flag,
            "driver_experience_years": driver_experience_years,
            "distance_estimate_km": distance_estimate_km,
            "random_tracking_signal": random_tracking_signal,
        }
    )
    delivery_minutes = (
        12
        + 1.8 * distance_km
        + 3.8 * traffic_index
        + 7.0 * rain_flag
        - 0.7 * driver_experience_years
        + 0.20 * distance_km * traffic_index
        + random_generator.normal(0, 4, number_of_rows)
    )

    development_features, sealed_test_features, development_target, sealed_test_target = train_test_split(
        delivery_features,
        delivery_minutes,
        test_size=0.20,
        random_state=42,
    )
    train_features, validation_features, train_target, validation_target = train_test_split(
        development_features,
        development_target,
        test_size=0.25,
        random_state=42,
    )

    delivery_model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=1,
    )
    delivery_model.fit(train_features, train_target)
    validation_predictions = delivery_model.predict(validation_features)
    validation_rmse = mean_squared_error(
        validation_target,
        validation_predictions,
    ) ** 0.5

    print("training rows:", len(train_features))
    print("validation rows:", len(validation_features))
    print("sealed test rows:", len(sealed_test_features))
    print("validation RMSE in minutes:", round(validation_rmse, 2))
    print("validation R-squared:", round(r2_score(validation_target, validation_predictions), 3))
    print("test status: sealed")

    assert validation_rmse < np.std(validation_target)
    assert list(train_features.columns) == list(validation_features.columns)
    """),

    md(r"""
    ## 3 · Coefficients are the transparent starting point

    A linear model predicts with a weighted sum:

    $$
    \hat y=b+w_1x_1+w_2x_2+\cdots+w_px_p
    $$

    **Symbols:** $\hat y$ is the prediction; $b$ is the intercept; $x_j$ is feature
    $j$; $w_j$ is its fitted coefficient; and $p$ is the number of features.

    If inputs are standardized, a coefficient describes the prediction change
    associated with a one-standard-deviation increase while other model inputs remain
    fixed. It is global and signed, but correlation makes individual coefficients
    unstable and observational data does not make them causal.

    Example: coefficient `+8` means the model adds about eight predicted minutes for
    a one-standard-deviation increase, all else fixed. It does not mean changing that
    feature will cause eight extra minutes.
    """),

    code(r"""
    standardized_ridge_model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    standardized_ridge_model.fit(train_features, train_target)

    standardized_coefficients = pd.Series(
        standardized_ridge_model.named_steps["ridge"].coef_,
        index=train_features.columns,
        name="standardized_coefficient",
    ).sort_values(key=np.abs, ascending=False)

    ridge_validation_predictions = standardized_ridge_model.predict(validation_features)
    ridge_validation_rmse = mean_squared_error(
        validation_target,
        ridge_validation_predictions,
    ) ** 0.5

    print(standardized_coefficients.round(2))
    print("ridge validation RMSE:", round(ridge_validation_rmse, 2))
    print("distance correlation:", round(train_features["distance_km"].corr(train_features["distance_estimate_km"]), 3))

    assert len(standardized_coefficients) == train_features.shape[1]
    """),

    md(r"""
    ## 4 · Tree impurity importance is fast but not a verdict

    Every tree split reduces squared error. Scikit-learn's impurity importance adds a
    feature's weighted reductions across all trees and normalizes the totals.

    It is cheap because it is produced during fitting. But it reports how the training
    algorithm used splits—not how much a feature improves held-out predictions.
    Continuous or high-cardinality features have more candidate split points, and
    correlated features can divide or substitute for one another.

    Use it as a quick diagnostic, never as proof of business importance, causality, or
    data quality.
    """),

    code(r"""
    impurity_importance = pd.Series(
        delivery_model.feature_importances_,
        index=train_features.columns,
        name="impurity_importance",
    ).sort_values(ascending=False)

    print(impurity_importance.round(3))
    print("importance sum:", round(impurity_importance.sum(), 6))

    assert np.isclose(impurity_importance.sum(), 1.0)
    assert (impurity_importance >= 0).all()
    """),

    md(r"""
    ## 5 · Permutation importance asks a held-out performance question

    Permutation importance breaks one validation column by shuffling it. If validation
    error rises, the fitted model relied on that column for those rows.

    For feature $j$:

    $$
    I_j=\operatorname{RMSE}_{\text{shuffled }j}-\operatorname{RMSE}_{\text{original}}
    $$

    **Symbols:** $I_j$ is importance in minutes; $j$ identifies one feature; and RMSE
    is root mean squared error. Positive values mean shuffling harmed performance.

    We repeat shuffles because one random ordering can be lucky. Correlated substitutes
    can make each feature look weak: breaking one leaves the other available.
    """),

    code(r"""
    def manual_permutation_importance(
        fitted_model,
        feature_table,
        actual_target,
        repeats=10,
        random_seed=0,
    ):
        '''Measure validation RMSE increase after independently shuffling each column.'''
        random_generator_local = np.random.default_rng(random_seed)
        baseline_predictions = fitted_model.predict(feature_table)
        baseline_rmse = mean_squared_error(actual_target, baseline_predictions) ** 0.5
        records = []

        for feature_name in feature_table.columns:
            repeated_increases = []
            for _ in range(repeats):
                shuffled_table = feature_table.copy()
                shuffled_table[feature_name] = random_generator_local.permutation(
                    shuffled_table[feature_name].to_numpy()
                )
                shuffled_predictions = fitted_model.predict(shuffled_table)
                shuffled_rmse = mean_squared_error(actual_target, shuffled_predictions) ** 0.5
                repeated_increases.append(shuffled_rmse - baseline_rmse)

            records.append(
                {
                    "feature": feature_name,
                    "mean_rmse_increase": np.mean(repeated_increases),
                    "standard_deviation": np.std(repeated_increases),
                }
            )

        return pd.DataFrame(records).sort_values("mean_rmse_increase", ascending=False)


    manual_importance = manual_permutation_importance(
        delivery_model,
        validation_features,
        validation_target,
        repeats=12,
        random_seed=42,
    )

    sklearn_importance_result = permutation_importance(
        delivery_model,
        validation_features,
        validation_target,
        scoring="neg_root_mean_squared_error",
        n_repeats=12,
        random_state=42,
        n_jobs=1,
    )
    sklearn_importance = pd.Series(
        sklearn_importance_result.importances_mean,
        index=validation_features.columns,
    ).sort_values(ascending=False)

    print("manual validation importance in RMSE minutes:")
    print(manual_importance.round(3).to_string(index=False))
    print("\nsklearn result:")
    print(sklearn_importance.round(3))

    assert manual_importance.iloc[0]["mean_rmse_increase"] > 0
    """),

    md(r"""
    ## 6 · PDP and ICE show prediction shape, not causal effect

    Permutation asks whether the model relies on a feature. Partial dependence asks
    how predictions change on average when we replace that feature with values on a
    grid:

    $$
    \widehat f_{PDP}(z)=\frac{1}{n}\sum_{i=1}^{n}f(z,x_{i,-j})
    $$

    **Symbols:** $z$ is one chosen value for feature $j$; $n$ is the number of rows;
    $x_{i,-j}$ means all other features from row $i$; and $f$ is the fitted model.

    An ICE curve keeps each row separate. PDP averages those ICE curves. If the curves
    differ, an interaction may be hiding behind the average.

    Analogy: PDP is the class average; ICE is every student's score line. The analogy
    stops because replacing one feature may create combinations that never occur in
    reality, especially when features are correlated.
    """),

    code(r"""
    traffic_grid = np.linspace(
        validation_features["traffic_index"].quantile(0.05),
        validation_features["traffic_index"].quantile(0.95),
        25,
    )
    ice_row_indices = [0, 1, 2, 3, 4]
    ice_predictions = np.empty((len(ice_row_indices), len(traffic_grid)))
    partial_dependence_predictions = []

    for grid_index, traffic_value in enumerate(traffic_grid):
        changed_validation = validation_features.copy()
        changed_validation["traffic_index"] = traffic_value
        grid_predictions = delivery_model.predict(changed_validation)
        partial_dependence_predictions.append(grid_predictions.mean())
        ice_predictions[:, grid_index] = grid_predictions[ice_row_indices]

    fig, axis = plt.subplots(figsize=(8, 5))
    for row_predictions in ice_predictions:
        axis.plot(traffic_grid, row_predictions, color="tab:blue", alpha=0.35)
    axis.plot(
        traffic_grid,
        partial_dependence_predictions,
        color="black",
        linewidth=3,
        label="PDP: average prediction",
    )
    axis.set_xlabel("traffic index inserted into each validation row")
    axis.set_ylabel("predicted delivery minutes")
    axis.set_title("ICE curves and their PDP average")
    axis.legend()
    plt.show()

    assert len(partial_dependence_predictions) == len(traffic_grid)
    """),

    md(r"""
    ## 7 · Calculate a two-feature Shapley value before the general formula

    Imagine two signals helping predict delay: traffic $T$ and rain $R$.

    | Known signals | Model value |
    |---|---:|
    | neither | 10 |
    | traffic only | 14 |
    | rain only | 13 |
    | both | 20 |

    Traffic can join first or second:

    - traffic first: $14-10=4$;
    - traffic after rain: $20-13=7$;
    - average traffic contribution: $(4+7)/2=5.5$.

    Rain contributes $(3+6)/2=4.5$. The baseline is 10 and the contributions sum to
    10, so $10+5.5+4.5=20$.

    The word **fair** here has a narrow mathematical meaning: average marginal credit
    under the declared game. It does not mean social fairness or causal responsibility.
    """),

    code(r"""
    two_feature_values = {
        frozenset(): 10.0,
        frozenset({"traffic"}): 14.0,
        frozenset({"rain"}): 13.0,
        frozenset({"traffic", "rain"}): 20.0,
    }

    traffic_shapley = ((14 - 10) + (20 - 13)) / 2
    rain_shapley = ((13 - 10) + (20 - 14)) / 2

    print("baseline:", two_feature_values[frozenset()])
    print("traffic contribution:", traffic_shapley)
    print("rain contribution:", rain_shapley)
    print("reconstructed value:", 10 + traffic_shapley + rain_shapley)

    assert np.isclose(traffic_shapley, 5.5)
    assert np.isclose(rain_shapley, 4.5)
    """),

    md(r"""
    ## 8 · Generalize the calculation carefully

    For feature $i$ among $M$ features:

    $$
    \phi_i=\sum_{S\subseteq N\setminus\{i\}}
    \frac{|S|!(M-|S|-1)!}{M!}
    \left[v(S\cup\{i\})-v(S)\right]
    $$

    **Symbol reference:**

    | Symbol | Meaning |
    |---|---|
    | $N$ | set of all feature positions |
    | $M$ | total number of features |
    | $i$ | feature receiving credit |
    | $S$ | one subset that does not contain $i$ |
    | $v(S)$ | model value when features in $S$ are treated as known |
    | $v(S\cup\{i\})-v(S)$ | contribution from adding feature $i$ |
    | $\phi_i$ | average weighted contribution assigned to feature $i$ |
    | $!$ | factorial; for example, $3!=3\times2\times1$ |

    The weight is the probability that subset $S$ appears before feature $i$ in a
    random feature ordering. Exact enumeration grows exponentially, so it is only a
    teaching method for a small feature count.

    Classical Shapley allocation is characterized by efficiency, symmetry, dummy,
    and additivity. SHAP applies related additive-attribution ideas to model outputs;
    its meaning still depends on how “unknown” features are represented.
    """),

    md(r"""
    ## 9 · Implement exact interventional Shapley values

    For an interventional explanation, absent columns retain values from background
    rows while present columns are replaced by the row being explained. This can make
    unrealistic combinations, but the operation is explicit and testable.

    We use only three features so every coalition remains visible. The implementation
    is exact for this empirical value function—not a universal truth about the world.
    """),

    code(r"""
    def empirical_coalition_value(predict_function, explained_row, present_features, background_rows):
        '''Average predictions after inserting the declared present feature values.'''
        evaluation_rows = np.asarray(background_rows, dtype=float).copy()
        explained_row = np.asarray(explained_row, dtype=float)
        if present_features:
            present_indices = list(present_features)
            evaluation_rows[:, present_indices] = explained_row[present_indices]
        return float(np.mean(predict_function(evaluation_rows)))


    def exact_interventional_shapley(predict_function, explained_row, background_rows):
        '''Enumerate every coalition for a small educational feature set.'''
        number_of_features = len(explained_row)
        all_feature_indices = tuple(range(number_of_features))
        shapley_values = np.zeros(number_of_features)

        for feature_index in all_feature_indices:
            remaining_features = [
                index for index in all_feature_indices if index != feature_index
            ]
            for subset_size in range(len(remaining_features) + 1):
                for subset in itertools.combinations(remaining_features, subset_size):
                    coalition_weight = (
                        factorial(len(subset))
                        * factorial(number_of_features - len(subset) - 1)
                        / factorial(number_of_features)
                    )
                    value_without_feature = empirical_coalition_value(
                        predict_function,
                        explained_row,
                        subset,
                        background_rows,
                    )
                    value_with_feature = empirical_coalition_value(
                        predict_function,
                        explained_row,
                        subset + (feature_index,),
                        background_rows,
                    )
                    shapley_values[feature_index] += coalition_weight * (
                        value_with_feature - value_without_feature
                    )

        return shapley_values


    def small_delivery_rule(feature_matrix):
        '''Return minutes from distance, traffic, and rain, including one interaction.'''
        distance = feature_matrix[:, 0]
        traffic = feature_matrix[:, 1]
        rain = feature_matrix[:, 2]
        return 10 + 2 * distance + 3 * traffic + 5 * rain + 0.4 * distance * traffic


    small_background = train_features[
        ["distance_km", "traffic_index", "rain_flag"]
    ].iloc[:80].to_numpy()
    small_explained_row = validation_features[
        ["distance_km", "traffic_index", "rain_flag"]
    ].iloc[0].to_numpy()

    exact_values = exact_interventional_shapley(
        small_delivery_rule,
        small_explained_row,
        small_background,
    )
    exact_baseline = small_delivery_rule(small_background).mean()
    exact_prediction = small_delivery_rule(small_explained_row.reshape(1, -1))[0]

    print("background prediction:", round(exact_baseline, 3))
    print("Shapley values:", exact_values.round(3))
    print("reconstructed prediction:", round(exact_baseline + exact_values.sum(), 3))
    print("actual prediction:", round(exact_prediction, 3))

    assert np.isclose(exact_baseline + exact_values.sum(), exact_prediction)
    """),

    md(r"""
    ## 10 · Background choice and correlation change the question

    A SHAP value is not just “the contribution.” It is a contribution relative to:

    - a fitted model;
    - one explained row;
    - a model-output scale;
    - a value function;
    - a background population.

    A city-wide background answers “different from a typical city-wide delivery.” A
    rainy-day background answers “different from a typical rainy-day delivery.” Both
    can be valid, but they answer different questions.

    Correlated features are also substitutes. Permuting one distance column leaves the
    other available, so individual importance may be diluted. Permuting them together
    measures their grouped reliance. Credit does not always split equally; the result
    depends on the model and dependence assumption.
    """),

    code(r"""
    def grouped_permutation_rmse_increase(
        fitted_model,
        feature_table,
        actual_target,
        grouped_columns,
        random_seed=0,
    ):
        '''Shuffle related columns with one shared row order and measure RMSE change.'''
        baseline_rmse = mean_squared_error(
            actual_target,
            fitted_model.predict(feature_table),
        ) ** 0.5
        random_generator_local = np.random.default_rng(random_seed)
        shared_order = random_generator_local.permutation(len(feature_table))
        shuffled_table = feature_table.copy()
        shuffled_table.loc[:, grouped_columns] = feature_table[grouped_columns].to_numpy()[shared_order]
        shuffled_rmse = mean_squared_error(
            actual_target,
            fitted_model.predict(shuffled_table),
        ) ** 0.5
        return shuffled_rmse - baseline_rmse


    grouped_distance_increase = grouped_permutation_rmse_increase(
        delivery_model,
        validation_features,
        validation_target,
        ["distance_km", "distance_estimate_km"],
        random_seed=42,
    )

    all_background = small_background
    rainy_background = train_features.loc[
        train_features["rain_flag"].eq(1),
        ["distance_km", "traffic_index", "rain_flag"],
    ].iloc[:80].to_numpy()
    values_all_background = exact_interventional_shapley(
        small_delivery_rule,
        small_explained_row,
        all_background,
    )
    values_rainy_background = exact_interventional_shapley(
        small_delivery_rule,
        small_explained_row,
        rainy_background,
    )

    print("grouped distance RMSE increase:", round(grouped_distance_increase, 3))
    print("all-delivery baseline:", round(small_delivery_rule(all_background).mean(), 3))
    print("rainy-delivery baseline:", round(small_delivery_rule(rainy_background).mean(), 3))
    print("values using all deliveries:", values_all_background.round(3))
    print("values using rainy deliveries:", values_rainy_background.round(3))

    assert not np.isclose(
        small_delivery_rule(all_background).mean(),
        small_delivery_rule(rainy_background).mean(),
    )
    """),

    md(r"""
    ## 11 · Use TreeSHAP and verify what it adds up to

    TreeSHAP exploits tree structure instead of enumerating every coalition. We state
    the contract explicitly:

    - model: the frozen random forest;
    - explained rows: validation rows;
    - background: 100 training rows;
    - dependence rule: interventional;
    - output: raw regression prediction in minutes.

    For classification, raw output may be a margin or log-odds rather than probability.
    Always inspect `model_output` and verify reconstruction before interpreting signs.
    """),

    code(r"""
    import shap

    explanation_background = train_features.sample(n=100, random_state=42)
    explained_validation_rows = validation_features.iloc[:80]

    tree_explainer = shap.TreeExplainer(
        delivery_model,
        data=explanation_background,
        feature_perturbation="interventional",
        model_output="raw",
    )
    validation_explanations = tree_explainer(explained_validation_rows)

    reconstructed_predictions = (
        np.asarray(validation_explanations.base_values)
        + np.asarray(validation_explanations.values).sum(axis=1)
    )
    direct_predictions = delivery_model.predict(explained_validation_rows)

    print("SHAP version:", shap.__version__)
    print("explanation matrix shape:", validation_explanations.values.shape)
    print("maximum reconstruction difference:", round(float(np.max(np.abs(reconstructed_predictions - direct_predictions))), 8))

    assert validation_explanations.values.shape == explained_validation_rows.shape
    assert np.allclose(reconstructed_predictions, direct_predictions, atol=1e-5)
    """),

    md(r"""
    A local waterfall begins at the background prediction and adds signed feature
    attributions until it reaches one row's prediction. A global bar chart averages
    absolute local magnitudes; it loses sign. A beeswarm keeps sign, magnitude, and
    feature value, but it still does not establish causality.
    """),

    code(r"""
    shap.plots.waterfall(validation_explanations[0], max_display=8, show=False)
    plt.title("One validation delivery: contributions relative to training background")
    plt.tight_layout()
    plt.show()

    shap.plots.bar(validation_explanations, max_display=8, show=False)
    plt.title("Global magnitude across explained validation deliveries")
    plt.tight_layout()
    plt.show()

    shap.plots.beeswarm(validation_explanations, max_display=8, show=False)
    plt.title("Direction and magnitude across validation deliveries")
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 12 · Choose the smallest tool that answers the question

    | Method | Main question | Strength | Main limitation |
    |---|---|---|---|
    | Standardized coefficient | Which direction globally? | Simple and signed | Linear form; correlation |
    | Impurity importance | Which columns created tree splits? | Nearly free | Training-based split bias |
    | Permutation importance | What hurts held-out performance? | Model-agnostic | Correlated substitutes |
    | PDP | What is average prediction shape? | Easy global curve | Can create unrealistic rows |
    | ICE | Do rows have different shapes? | Reveals heterogeneity | Visually crowded |
    | SHAP | Why this prediction relative to a background? | Local, signed, additive | Background and dependence sensitive |

    **Use SHAP when:** a validated model needs local diagnostic attribution and the
    reference population can be documented.

    **Do not use SHAP when:** the real question is causal effect, fairness, guaranteed
    recourse, or whether the model generalizes. Use experiments or causal methods for
    interventions, group fairness metrics for fairness, counterfactual methods plus
    feasibility rules for recourse, and held-out evaluation for generalization.

    In regulated credit, an attribution can support investigation, but it does not
    automatically become a compliant adverse-action reason. Reasons must accurately
    reflect the factors the decision process actually considered, and legal review is
    part of the system—not an optional chart caption.
    """),

    md(r"""
    ## 13 · Mini-project: explain one frozen delivery model

    **Goal:** evaluate the already-frozen model once on sealed test rows, then explain
    one test prediction relative to the already-fixed training background.

    **Columns:** distance, traffic, rain, driver experience, correlated distance
    estimate, and a random tracking signal.

    **Workflow:**

    1. keep the fitted model and background unchanged;
    2. calculate one final test RMSE;
    3. choose the first test row by position, not by interesting outcome;
    4. calculate its TreeSHAP explanation;
    5. verify baseline plus attributions equals prediction;
    6. report association, reference, output scale, and limitations.

    The explanation is not used to retune the model or choose a new background.
    """),

    code(r"""
    final_test_predictions = delivery_model.predict(sealed_test_features)
    final_test_rmse = mean_squared_error(
        sealed_test_target,
        final_test_predictions,
    ) ** 0.5

    selected_test_row = sealed_test_features.iloc[[0]]
    selected_test_explanation = tree_explainer(selected_test_row)
    selected_test_prediction = delivery_model.predict(selected_test_row)[0]
    selected_test_reconstruction = (
        float(np.asarray(selected_test_explanation.base_values).reshape(-1)[0])
        + float(np.asarray(selected_test_explanation.values).sum())
    )

    local_report = pd.DataFrame(
        {
            "feature": selected_test_row.columns,
            "value": selected_test_row.iloc[0].to_numpy(),
            "shap_minutes": selected_test_explanation.values[0],
        }
    ).sort_values("shap_minutes", key=np.abs, ascending=False)

    print("final sealed-test RMSE:", round(final_test_rmse, 2))
    print("selected test prediction:", round(selected_test_prediction, 2))
    print("reconstructed prediction:", round(selected_test_reconstruction, 2))
    print(local_report.round(3).to_string(index=False))
    print("interpretation: model attribution relative to fixed training background; not causal advice")

    assert np.isclose(selected_test_reconstruction, selected_test_prediction, atol=1e-5)
    """),

    md(r"""
    ## 14 · Practice, solutions, and mastery checkpoint

    ### Worked example

    Original validation RMSE is 5 minutes. Shuffling traffic produces RMSE values 9,
    10, and 8. The increases are 4, 5, and 3; mean permutation importance is 4 minutes.

    ### Guided practice

    1. Explain the difference between global reliance and local attribution.
    2. Calculate the rain Shapley value from the two-feature table.
    3. Explain why absolute SHAP values remove direction.
    4. Name every part of a SHAP explanation contract.
    5. Explain why a validated model must come before interpretation.

    ### Independent practice

    6. Implement grouped permutation for traffic and a noisy traffic copy.
    7. Build PDP and ICE for driver experience and describe their limits.
    8. Compare explanations under two documented training backgrounds.
    9. Explain a classification model on raw-margin and probability scales.
    10. Add repeated explanation-stability checks without inspecting test outcomes.

    ### Challenge

    Rebuild the delivery project without copying. Include a model card paragraph,
    coefficient baseline, impurity warning, manual permutation importance, PDP/ICE,
    hand-calculated Shapley game, exact implementation, TreeSHAP reconstruction, and
    one sealed-test explanation.

    ### Self-check

    1. Can impurity importance prove held-out usefulness?
    2. Why can correlated substitutes have low individual permutation importance?
    3. What does PDP average?
    4. How does ICE differ from PDP?
    5. What is the baseline in a local SHAP explanation?
    6. Why does background choice change attributions?
    7. Does mathematical allocation fairness prove social fairness?
    8. Does SHAP prove causality or legal compliance?

    ### Solution and scoring rubric

    1. Global reliance concerns many rows; local attribution concerns one prediction.
    2. Rain is $(3+6)/2=4.5$.
    3. Magnitude remains, but positive and negative directions disappear.
    4. Model, row, output scale, value function, dependence rule, and background.
    5. Interpretation can faithfully describe leakage or a model that does not generalize.

    Award two points for each self-check and four points for the challenge explanation.
    Full credit requires correct boundaries, not just correct plotting code.

    ### Common mistakes

    - Explaining a model before validating it.
    - Treating impurity importance as held-out evidence.
    - Permuting test data repeatedly during method development.
    - Reading PDP or SHAP as causal effect.
    - Ignoring correlated substitutes.
    - Hiding the background population.
    - Mixing raw margins, log-odds, and probabilities.
    - Calling mathematical credit allocation socially fair.
    - Using SHAP as a fairness metric or compliance certificate.
    - Silently catching every library exception.

    ### Readiness threshold

    Score at least **16/20** and correctly explain validation, correlation, background,
    output scale, additive reconstruction, and the causal boundary.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Explain this chain without notes:

    validated model  
    → transparent coefficient baseline  
    → held-out permutation importance  
    → PDP and ICE  
    → two-feature Shapley calculation  
    → declared background and output scale  
    → verified TreeSHAP reconstruction.

    ### Teach it back

    Explain why permutation importance, PDP, and SHAP answer different questions. Then
    explain why a beautiful local explanation cannot prove causality, fairness, or
    generalization.

    ### Memory aid

    **Validate first, name the question, declare the reference, and explain only what
    the model—not the world—has shown.**

    ### Next dependency

    Validated predictions and bounded explanations  
    → required before integrated classical-ML evaluation  
    → because model selection must combine performance, robustness, and interpretation
    without confusing their evidence.
    """),
]


build("03_ml_engineering/05_explainability_with_shap.ipynb", cells)
