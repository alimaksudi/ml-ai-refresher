"""Build CML-05: beginner-first Gradient Boosting Foundations."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # CML-05 · Gradient Boosting Foundations

    **Prerequisites:** FND-04, CML-03, CML-04, MLE-02, and MLE-03  
    **Estimated study time:** 8–10 hours, including practice  
    **Next lesson:** CML-06 · XGBoost Mechanics and Regularization

    A random forest trains trees independently and averages them. Gradient boosting
    takes a different path: it adds small trees in sequence, and each new tree tries
    to correct what the current model still gets wrong.

    The goal is not to memorize a boosting API. The goal is to calculate the first
    corrections by hand, watch the loss change, understand why a learning rate exists,
    and select the number of rounds without touching the final test.

    ### Scope boundary

    This lesson teaches squared-loss gradient boosting for regression, then provides
    a small binary-classification bridge. It intentionally defers:

    - Hessians and second-order Taylor approximations;
    - XGBoost leaf weights and split-gain formulas;
    - histogram, sparsity-aware, and distributed split search;
    - L1/L2 leaf regularization and XGBoost-specific parameters;
    - SHAP to MLE-05 and monitoring to PROD-05;
    - imbalanced-learning methods to MLE-04.

    Those topics need the residual-correction loop taught here.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - explain how a regression-tree leaf predicts a mean;
    - calculate squared error for one candidate regression split;
    - explain why the training-target mean is a useful first prediction;
    - calculate residuals and one boosted correction manually;
    - trace three boosting rounds in a table;
    - explain the roles of the weak learner, learning rate, and number of rounds;
    - connect ordinary residuals to the negative squared-loss gradient;
    - implement a regression-stump learner from scratch;
    - implement the boosting loop from scratch;
    - select a round using validation loss only;
    - explain why validation loss can eventually worsen without claiming it must;
    - use scikit-learn after understanding the manual mechanism;
    - explain the binary-classification residual $y-p$ at a high level;
    - evaluate one selected model once on a sealed test partition.

    ### Learning path

    ```mermaid
    flowchart LR
        A[Regression leaf mean] --> B[Initial mean prediction]
        B --> C[Residuals]
        C --> D[Small correction tree]
        D --> E[Learning-rate step]
        E --> F[Repeat rounds]
        F --> G[Loss gradient connection]
        G --> H[Validation chooses round]
        H --> I[One sealed test]
    ```

    Dependency map:

    Loss and gradient descent  
    → required before gradient boosting  
    → because boosting chooses corrections that reduce a declared loss.

    Decision trees  
    → required before boosted trees  
    → because every correction model is still a tree.

    Validation boundaries  
    → required before early stopping  
    → because the stopping round is a model-development choice.
    """),

    md(r"""
    ## 2 · The practical problem: one simple model leaves structure behind

    Suppose a delivery team predicts travel time. Predicting the training mean gives
    a stable baseline, but it misses patterns such as:

    - long routes need a positive correction;
    - short routes need a negative correction;
    - heavy traffic changes the correction again.

    One large tree could try to learn everything at once. Boosting instead makes a
    rough prediction and adds several small corrections.

    Analogy: revise a draft in focused passes. The first pass fixes the largest
    structural issue; later passes fix what remains. The analogy stops when a learner
    starts fitting accidental noise—another revision is not automatically helpful.

    ```mermaid
    flowchart LR
        P0[Current predictions] --> R[Calculate residuals]
        R --> T[Fit a small tree]
        T --> C[Scale its correction]
        C --> P1[Updated predictions]
        P1 --> R
    ```
    """),

    md(r"""
    ## 3 · Regression-tree bridge: leaves predict means

    CML-03 used class counts in leaves. A regression tree predicts a numerical value,
    so each leaf commonly stores the mean target of its training rows.

    For targets $[2,4,6]$:

    $$
    \bar y=\frac{2+4+6}{3}=4
    $$

    The squared errors around that leaf prediction are:

    $$
    (2-4)^2+(4-4)^2+(6-4)^2=4+0+4=8
    $$

    A candidate split is useful when the total squared error in its two child leaves
    is smaller than the parent's squared error.

    For feature values $[1,2,3,4]$, targets $[2,2,8,8]$, and threshold 2.5:

    - left mean is 2 and left squared error is 0;
    - right mean is 8 and right squared error is 0;
    - the split perfectly separates this tiny training pattern.

    This is the regression counterpart of Gini reduction.
    """),

    code(r"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    def sum_of_squared_errors(values):
        '''Return squared error around the mean of one candidate leaf.'''
        values = np.asarray(values, dtype=float)
        leaf_mean = values.mean()
        return float(np.sum((values - leaf_mean) ** 2))


    split_feature = np.array([1.0, 2.0, 3.0, 4.0])
    split_targets = np.array([2.0, 2.0, 8.0, 8.0])
    candidate_threshold = 2.5

    # Route rows into the two candidate leaves.
    left_mask = split_feature <= candidate_threshold
    right_mask = ~left_mask

    parent_squared_error = sum_of_squared_errors(split_targets)
    left_squared_error = sum_of_squared_errors(split_targets[left_mask])
    right_squared_error = sum_of_squared_errors(split_targets[right_mask])
    error_reduction = parent_squared_error - left_squared_error - right_squared_error

    print("parent squared error:", parent_squared_error)
    print("left leaf mean:", split_targets[left_mask].mean())
    print("right leaf mean:", split_targets[right_mask].mean())
    print("child squared error:", left_squared_error + right_squared_error)
    print("squared-error reduction:", error_reduction)

    assert np.isclose(parent_squared_error, 36.0)
    assert np.isclose(left_squared_error + right_squared_error, 0.0)
    assert np.isclose(error_reduction, 36.0)
    """),

    md(r"""
    ## 4 · Begin with the training-target mean

    With no feature-based rules yet, squared-loss boosting starts from one constant:

    $$
    F_0(x)=\bar y
    $$

    $F_0$ means the model before any correction trees. For targets $[2,4,6]$, every
    first prediction is 4.

    The residual is actual minus predicted:

    $$
    r_i=y_i-F_0(x_i)
    $$

    So the residuals are $[-2,0,2]$. Negative means the model predicted too high;
    positive means it predicted too low.

    The mean is not arbitrary. Among all single constant predictions, it minimizes
    squared error. CML-01 established the same baseline for regression.
    """),

    code(r"""
    manual_targets = np.array([2.0, 4.0, 6.0])
    initial_prediction = manual_targets.mean()
    initial_predictions = np.full(len(manual_targets), initial_prediction)
    initial_residuals = manual_targets - initial_predictions
    initial_mean_squared_error = np.mean(initial_residuals ** 2)

    print("targets:", manual_targets)
    print("initial prediction for every row:", initial_predictions)
    print("initial residuals:", initial_residuals)
    print("initial MSE:", round(initial_mean_squared_error, 4))

    assert np.isclose(initial_prediction, 4.0)
    assert np.array_equal(initial_residuals, [-2.0, 0.0, 2.0])
    assert np.isclose(initial_mean_squared_error, 8 / 3)
    """),

    md(r"""
    ## 5 · Apply one small correction

    Suppose a tiny tree predicts residual corrections $[-1,0,1]$. With learning rate
    $\nu=0.5$, only half of each proposed correction is added:

    $$
    F_1(x_i)=F_0(x_i)+0.5h_1(x_i)
    $$

    New predictions are:

    $$
    [4,4,4]+0.5[-1,0,1]=[3.5,4,4.5]
    $$

    New residuals become $[-1.5,0,1.5]$, and MSE falls from approximately 2.667 to
    1.5. The tree does not replace the current model; it contributes one scaled
    correction.

    **Symbols:** $F_1$ is the model after one round; $h_1$ is the first correction
    tree; $\nu$ is the learning rate.
    """),

    code(r"""
    proposed_correction = np.array([-1.0, 0.0, 1.0])
    learning_rate = 0.5

    updated_predictions = initial_predictions + learning_rate * proposed_correction
    updated_residuals = manual_targets - updated_predictions
    updated_mean_squared_error = np.mean(updated_residuals ** 2)

    correction_table = pd.DataFrame(
        {
            "target": manual_targets,
            "prediction_before": initial_predictions,
            "tree_correction": proposed_correction,
            "scaled_correction": learning_rate * proposed_correction,
            "prediction_after": updated_predictions,
            "residual_after": updated_residuals,
        }
    )

    print(correction_table.to_string(index=False))
    print("MSE before:", round(initial_mean_squared_error, 4))
    print("MSE after:", round(updated_mean_squared_error, 4))

    assert np.allclose(updated_predictions, [3.5, 4.0, 4.5])
    assert updated_mean_squared_error < initial_mean_squared_error
    """),

    md(r"""
    ## 6 · Trace three rounds before using gradient notation

    Each round repeats the same loop:

    1. calculate current residuals;
    2. fit a small tree to those residuals;
    3. multiply the tree output by the learning rate;
    4. add the scaled correction;
    5. calculate loss again.

    The next cell uses predetermined correction outputs so every arithmetic step is
    visible. A real tree learns these outputs from feature-based leaves.
    """),

    code(r"""
    visible_predictions = initial_predictions.copy()
    visible_learning_rate = 0.5
    visible_tree_corrections = [
        np.array([-1.0, 0.0, 1.0]),
        np.array([-0.75, 0.0, 0.75]),
        np.array([-0.375, 0.0, 0.375]),
    ]
    round_records = []

    # Record round zero so the student can compare every later loss with the baseline.
    round_records.append(
        {
            "round": 0,
            "predictions": visible_predictions.round(3).tolist(),
            "residuals": (manual_targets - visible_predictions).round(3).tolist(),
            "MSE": np.mean((manual_targets - visible_predictions) ** 2),
        }
    )

    for round_number, tree_correction in enumerate(visible_tree_corrections, start=1):
        # Boosting adds the new learner; it never discards earlier predictions.
        visible_predictions = (
            visible_predictions + visible_learning_rate * tree_correction
        )
        remaining_residuals = manual_targets - visible_predictions

        round_records.append(
            {
                "round": round_number,
                "predictions": visible_predictions.round(3).tolist(),
                "residuals": remaining_residuals.round(3).tolist(),
                "MSE": np.mean(remaining_residuals ** 2),
            }
        )

    manual_round_table = pd.DataFrame(round_records)
    print(manual_round_table.to_string(index=False))

    assert manual_round_table["MSE"].is_monotonic_decreasing
    assert manual_round_table.iloc[-1]["MSE"] < manual_round_table.iloc[0]["MSE"]
    """),

    md(r"""
    ## 7 · The additive model summarizes the loop

    After $M$ rounds:

    $$
    F_M(x)=F_0(x)+\nu\sum_{m=1}^{M}h_m(x)
    $$

    Read it as:

    > final prediction = initial prediction + the sum of all scaled tree corrections.

    **Symbols:**

    - $x$: one feature row;
    - $F_0(x)$: initial mean prediction;
    - $F_M(x)$: prediction after $M$ rounds;
    - $h_m(x)$: correction predicted by tree $m$;
    - $\nu$: learning rate;
    - $M$: total boosting rounds;
    - $\sum$: add the contributions from rounds 1 through $M$.

    Earlier trees are frozen. Tree $m+1$ sees residuals left after trees 1 through
    $m$ have already contributed.
    """),

    md(r"""
    ## 8 · Why “gradient” appears in the name

    For one row, use half squared loss:

    $$
    L(y,F)=\frac{1}{2}(y-F)^2
    $$

    Here $y$ is the actual target and $F$ is the current prediction. Differentiate
    loss with respect to the prediction:

    $$
    \frac{\partial L}{\partial F}=F-y
    $$

    The negative gradient is:

    $$
    -\frac{\partial L}{\partial F}=y-F
    $$

    That is exactly the ordinary residual. Therefore, fitting a tree to residuals is
    fitting a tree to an approximate downhill direction for squared loss.

    Unlike ordinary gradient descent, we are not directly changing one fixed vector
    of coefficients. We add a new function—a tree—that predicts a useful correction
    for both current and future rows. This is the intuition behind “gradient descent
    in function space.”
    """),

    code(r"""
    gradient_targets = np.array([2.0, 4.0, 6.0])
    gradient_predictions = np.array([4.0, 4.0, 4.0])

    squared_loss_gradient = gradient_predictions - gradient_targets
    negative_gradient = -squared_loss_gradient
    ordinary_residual = gradient_targets - gradient_predictions

    print("loss gradient F - y:", squared_loss_gradient)
    print("negative gradient:", negative_gradient)
    print("ordinary residual y - F:", ordinary_residual)

    assert np.array_equal(negative_gradient, ordinary_residual)
    """),

    md(r"""
    ## 9 · Learning rate and rounds work together

    The learning rate $\nu$ controls how much of each tree's correction is accepted.

    - A large value makes faster, larger updates.
    - A small value makes slower, smaller updates and usually needs more rounds.
    - A value that is too small with too few rounds underfits.
    - A value that is too large can follow noisy corrections aggressively.

    “Halve the learning rate and double the rounds” is a rough experiment idea, not
    a law. Tree depth, data, loss, and noise also matter.

    Training loss normally decreases as more correction trees are added. Validation
    loss may improve, flatten, fluctuate, or worsen. Early stopping uses validation
    evidence to choose a round; it must never choose the round from the final test.
    """),

    md(r"""
    ## 10 · Implement a regression stump from scratch

    A **stump** is a tree with one split and two leaves. It is a useful weak learner
    because one round can make only a simple correction.

    The fitter below:

    1. enumerates each numerical feature;
    2. tries midpoint thresholds;
    3. predicts the mean residual in each child;
    4. chooses the split with the smallest child squared error.

    This is deliberately readable rather than optimized.
    """),

    code(r"""
    def fit_regression_stump(feature_matrix, correction_targets, minimum_leaf_rows=2):
        '''Fit one regression stump by minimizing child squared error.'''
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        correction_targets = np.asarray(correction_targets, dtype=float)

        best_stump = None
        best_child_error = np.inf
        number_of_rows, number_of_features = feature_matrix.shape

        for feature_index in range(number_of_features):
            unique_values = np.unique(feature_matrix[:, feature_index])
            candidate_thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in candidate_thresholds:
                left_mask = feature_matrix[:, feature_index] <= threshold
                right_mask = ~left_mask

                # Reject a split that creates a leaf too small to be trustworthy.
                if left_mask.sum() < minimum_leaf_rows or right_mask.sum() < minimum_leaf_rows:
                    continue

                left_value = float(correction_targets[left_mask].mean())
                right_value = float(correction_targets[right_mask].mean())

                left_error = np.sum((correction_targets[left_mask] - left_value) ** 2)
                right_error = np.sum((correction_targets[right_mask] - right_value) ** 2)
                child_error = float(left_error + right_error)

                if child_error < best_child_error:
                    best_child_error = child_error
                    best_stump = {
                        "feature_index": feature_index,
                        "threshold": float(threshold),
                        "left_value": left_value,
                        "right_value": right_value,
                        "child_squared_error": child_error,
                    }

        if best_stump is None:
            # Constant features or tiny data cannot form a valid split.
            constant_value = float(correction_targets.mean())
            return {
                "feature_index": None,
                "threshold": None,
                "left_value": constant_value,
                "right_value": constant_value,
                "child_squared_error": sum_of_squared_errors(correction_targets),
            }

        return best_stump


    def predict_regression_stump(stump, feature_matrix):
        '''Predict the stored left or right correction for every row.'''
        feature_matrix = np.asarray(feature_matrix, dtype=float)

        if stump["feature_index"] is None:
            return np.full(len(feature_matrix), stump["left_value"])

        left_mask = feature_matrix[:, stump["feature_index"]] <= stump["threshold"]
        predictions = np.full(len(feature_matrix), stump["right_value"], dtype=float)
        predictions[left_mask] = stump["left_value"]
        return predictions


    tiny_features = np.array([[1.0], [2.0], [3.0], [4.0]])
    tiny_residuals = np.array([-2.0, -2.0, 2.0, 2.0])
    tiny_stump = fit_regression_stump(tiny_features, tiny_residuals)
    tiny_corrections = predict_regression_stump(tiny_stump, tiny_features)

    print("learned stump:", tiny_stump)
    print("predicted corrections:", tiny_corrections)

    assert np.isclose(tiny_stump["threshold"], 2.5)
    assert np.array_equal(tiny_corrections, tiny_residuals)
    """),

    md(r"""
    ## 11 · Implement the boosting loop from scratch

    The next class handles the new responsibilities:

    - store the training-target mean;
    - calculate residuals each round;
    - fit one stump to those residuals;
    - update training predictions with shrinkage;
    - preserve every stump so future predictions can replay the same sum.

    We use a synthetic nonlinear regression problem and create train, validation, and
    sealed test partitions before fitting anything.
    """),

    code(r"""
    from sklearn.datasets import make_friedman1
    from sklearn.model_selection import train_test_split

    all_features, all_targets = make_friedman1(
        n_samples=600,
        n_features=8,
        noise=1.0,
        random_state=42,
    )

    # Seal 20% first. This partition is not used in round or learning-rate selection.
    development_features, sealed_test_features, development_targets, sealed_test_targets = train_test_split(
        all_features,
        all_targets,
        test_size=0.20,
        random_state=42,
    )
    train_features, validation_features, train_targets, validation_targets = train_test_split(
        development_features,
        development_targets,
        test_size=0.25,
        random_state=42,
    )

    print("training rows:", len(train_targets))
    print("validation rows:", len(validation_targets))
    print("sealed test rows:", len(sealed_test_targets))
    print("test status: sealed — not used for model development")

    assert len(train_targets) + len(validation_targets) + len(sealed_test_targets) == 600
    """),

    code(r"""
    class LearningGradientBoostingRegressor:
        '''Educational squared-loss booster using regression stumps.'''

        def __init__(self, number_of_rounds=150, learning_rate=0.05, minimum_leaf_rows=5):
            self.number_of_rounds = number_of_rounds
            self.learning_rate = learning_rate
            self.minimum_leaf_rows = minimum_leaf_rows

        def fit(self, feature_matrix, targets):
            '''Fit stumps sequentially to the residuals left by the current model.'''
            feature_matrix = np.asarray(feature_matrix, dtype=float)
            targets = np.asarray(targets, dtype=float)

            self.initial_prediction_ = float(targets.mean())
            current_predictions = np.full(len(targets), self.initial_prediction_)
            self.stumps_ = []
            self.training_mse_by_round_ = []

            for _ in range(self.number_of_rounds):
                # For squared loss, residuals are the negative loss gradients.
                residuals = targets - current_predictions
                stump = fit_regression_stump(
                    feature_matrix,
                    residuals,
                    minimum_leaf_rows=self.minimum_leaf_rows,
                )
                stump_corrections = predict_regression_stump(stump, feature_matrix)

                # Shrink the proposed correction before adding it to the ensemble.
                current_predictions = (
                    current_predictions + self.learning_rate * stump_corrections
                )

                self.stumps_.append(stump)
                self.training_mse_by_round_.append(
                    float(np.mean((targets - current_predictions) ** 2))
                )

            return self

        def staged_predict(self, feature_matrix):
            '''Yield predictions after each successive boosting round.'''
            feature_matrix = np.asarray(feature_matrix, dtype=float)
            current_predictions = np.full(len(feature_matrix), self.initial_prediction_)

            for stump in self.stumps_:
                current_predictions = current_predictions + self.learning_rate * (
                    predict_regression_stump(stump, feature_matrix)
                )
                yield current_predictions.copy()

        def predict(self, feature_matrix, number_of_rounds=None):
            '''Replay either all fitted stumps or a declared prefix of them.'''
            feature_matrix = np.asarray(feature_matrix, dtype=float)
            rounds_to_use = len(self.stumps_) if number_of_rounds is None else number_of_rounds
            current_predictions = np.full(len(feature_matrix), self.initial_prediction_)

            for stump in self.stumps_[:rounds_to_use]:
                current_predictions = current_predictions + self.learning_rate * (
                    predict_regression_stump(stump, feature_matrix)
                )

            return current_predictions


    learning_booster = LearningGradientBoostingRegressor(
        number_of_rounds=160,
        learning_rate=0.05,
        minimum_leaf_rows=5,
    )
    learning_booster.fit(train_features, train_targets)

    print("initial prediction:", round(learning_booster.initial_prediction_, 3))
    print("first training MSE:", round(learning_booster.training_mse_by_round_[0], 3))
    print("final training MSE:", round(learning_booster.training_mse_by_round_[-1], 3))

    assert len(learning_booster.stumps_) == 160
    assert learning_booster.training_mse_by_round_[-1] < learning_booster.training_mse_by_round_[0]
    """),

    md(r"""
    ## 12 · Validation chooses the round; the test stays sealed

    Calculate validation MSE after every fitted round:

    $$
    \operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
    $$

    **Symbols:** $n$ is the number of validation rows; $y_i$ is an actual target;
    $\hat y_i$ is its current prediction.

    The round with the lowest validation MSE is a development choice. A practical
    system may require a patience rule and minimum improvement so tiny fluctuations
    do not control stopping. Here we use the exact minimum to make the first mechanism
    visible.

    Training MSE usually keeps falling because each round targets training residuals.
    Validation MSE measures whether those corrections transfer to unseen development
    rows. It may or may not show a clear rise in a particular finite example.
    """),

    code(r"""
    validation_mse_by_round = []

    # Evaluate every prefix on validation only. The sealed test is still untouched.
    for staged_validation_predictions in learning_booster.staged_predict(validation_features):
        validation_mse = np.mean(
            (validation_targets - staged_validation_predictions) ** 2
        )
        validation_mse_by_round.append(float(validation_mse))

    selected_round = int(np.argmin(validation_mse_by_round)) + 1
    selected_validation_mse = validation_mse_by_round[selected_round - 1]

    print("selected round from validation:", selected_round)
    print("selected validation MSE:", round(selected_validation_mse, 3))
    print("test status: still sealed")

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        range(1, len(validation_mse_by_round) + 1),
        learning_booster.training_mse_by_round_,
        label="training MSE",
    )
    axis.plot(
        range(1, len(validation_mse_by_round) + 1),
        validation_mse_by_round,
        label="validation MSE",
    )
    axis.axvline(selected_round, color="black", linestyle="--", label="selected round")
    axis.set_xlabel("boosting round")
    axis.set_ylabel("mean squared error")
    axis.set_title("Validation—not test—chooses the boosting round")
    axis.legend()
    axis.grid(alpha=0.3)
    plt.show()

    assert 1 <= selected_round <= learning_booster.number_of_rounds
    assert np.isclose(selected_validation_mse, min(validation_mse_by_round))
    """),

    md(r"""
    ## 13 · Use scikit-learn after mastering the loop

    `GradientBoostingRegressor` implements the same high-level sequence with deeper
    regression trees, several loss options, row subsampling, and optimized split
    search.

    Important controls:

    - `n_estimators`: number of sequential correction trees;
    - `learning_rate`: scale applied to each new tree;
    - `max_depth`: complexity of each correction tree;
    - `min_samples_leaf`: minimum rows in each leaf;
    - `subsample`: fraction of training rows available to each round;
    - `loss`: the objective whose negative gradient becomes the correction target.

    Unlike random-forest trees, boosted trees depend on all earlier rounds and cannot
    be trained independently. Scaling is usually unnecessary for ordinary tree
    splits, but preprocessing and feature availability must still respect the data
    boundary.
    """),

    code(r"""
    from sklearn.ensemble import GradientBoostingRegressor

    sklearn_booster = GradientBoostingRegressor(
        n_estimators=160,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        loss="squared_error",
        random_state=42,
    )
    sklearn_booster.fit(train_features, train_targets)

    sklearn_validation_mse_by_round = [
        float(np.mean((validation_targets - predictions) ** 2))
        for predictions in sklearn_booster.staged_predict(validation_features)
    ]
    sklearn_selected_round = int(np.argmin(sklearn_validation_mse_by_round)) + 1

    print("scratch selected validation round:", selected_round)
    print("sklearn selected validation round:", sklearn_selected_round)
    print("scratch selected validation MSE:", round(selected_validation_mse, 3))
    print(
        "sklearn selected validation MSE:",
        round(sklearn_validation_mse_by_round[sklearn_selected_round - 1], 3),
    )
    print("test status: still sealed")

    assert len(sklearn_validation_mse_by_round) == 160
    assert 1 <= sklearn_selected_round <= 160
    """),

    md(r"""
    ## 14 · Classification bridge, project, and mastery checkpoint

    ### Binary-classification bridge

    Classification boosting follows the same correction idea with a different loss.
    The running model produces a score $F(x)$, and sigmoid converts it to probability:

    $$
    p=\frac{1}{1+e^{-F}}
    $$

    For binary log loss, the negative gradient simplifies to:

    $$
    r=y-p
    $$

    A positive residual asks the next tree to increase the score; a negative residual
    asks it to decrease the score. CML-06 will build on this with second-order
    information. For now, master squared-loss regression before implementing the
    classification booster.

    ### Mini-project: diabetes progression regression

    **Goal:** predict a numerical disease-progression target from ten standardized
    baseline measurements in scikit-learn's diabetes dataset.

    **Workflow:** training-majority is not meaningful for regression, so use the
    training-target mean as baseline; create train/validation/test partitions; fit a
    pre-declared booster; choose its round on validation; evaluate once on test.

    **Success contract:** the selected booster must beat the mean baseline on
    validation; no test prediction may be calculated before the selected round is
    frozen.
    """),

    code(r"""
    from sklearn.datasets import load_diabetes

    diabetes = load_diabetes()
    diabetes_features = diabetes.data
    diabetes_targets = diabetes.target

    # Seal the project test before calculating the training baseline or fitting a model.
    project_development_features, project_test_features, project_development_targets, project_test_targets = train_test_split(
        diabetes_features,
        diabetes_targets,
        test_size=0.20,
        random_state=23,
    )
    project_train_features, project_validation_features, project_train_targets, project_validation_targets = train_test_split(
        project_development_features,
        project_development_targets,
        test_size=0.25,
        random_state=23,
    )

    # Freeze a baseline using only training targets.
    project_baseline_value = float(project_train_targets.mean())
    baseline_validation_predictions = np.full(
        len(project_validation_targets),
        project_baseline_value,
    )
    baseline_validation_mse = float(
        np.mean((project_validation_targets - baseline_validation_predictions) ** 2)
    )

    # Fit one pre-declared maximum number of rounds, then select a prefix on validation.
    project_booster = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=8,
        loss="squared_error",
        random_state=23,
    )
    project_booster.fit(project_train_features, project_train_targets)

    project_validation_mse_by_round = [
        float(np.mean((project_validation_targets - predictions) ** 2))
        for predictions in project_booster.staged_predict(project_validation_features)
    ]
    project_selected_round = int(np.argmin(project_validation_mse_by_round)) + 1
    project_selected_validation_mse = project_validation_mse_by_round[
        project_selected_round - 1
    ]

    partition_manifest = pd.DataFrame(
        {
            "partition": ["train", "validation", "test"],
            "rows": [
                len(project_train_targets),
                len(project_validation_targets),
                len(project_test_targets),
            ],
            "used_for_round_selection": [False, True, False],
        }
    )

    print(partition_manifest.to_string(index=False))
    print("baseline validation MSE:", round(baseline_validation_mse, 2))
    print("selected booster round:", project_selected_round)
    print("selected booster validation MSE:", round(project_selected_validation_mse, 2))
    print("test status: sealed — no test predictions calculated yet")

    assert len(project_train_targets) + len(project_validation_targets) + len(project_test_targets) == len(diabetes_targets)
    assert project_selected_validation_mse < baseline_validation_mse
    assert 1 <= project_selected_round <= 300
    """),

    code(r"""
    # Replay the staged predictions and keep only the validation-selected prefix on test.
    project_test_stage_iterator = project_booster.staged_predict(project_test_features)
    selected_test_predictions = None

    for round_number, stage_predictions in enumerate(project_test_stage_iterator, start=1):
        if round_number == project_selected_round:
            selected_test_predictions = stage_predictions
            break

    final_test_mse = float(
        np.mean((project_test_targets - selected_test_predictions) ** 2)
    )
    baseline_test_mse = float(
        np.mean((project_test_targets - project_baseline_value) ** 2)
    )

    print("final sealed-test rows:", len(project_test_targets))
    print("mean-baseline test MSE:", round(baseline_test_mse, 2))
    print("selected booster test MSE:", round(final_test_mse, 2))
    print("this is a final estimate, not permission to select a different round")

    assert selected_test_predictions is not None
    assert len(selected_test_predictions) == len(project_test_targets)
    assert np.isfinite(final_test_mse)
    """),

    md(r"""
    ### Worked example

    Targets are $[2,4,6]$, current predictions are $[4,4,4]$, a stump proposes
    $[-1,0,1]$, and $\nu=0.5$. The updated predictions are $[3.5,4,4.5]$ and MSE
    falls from $8/3$ to $1.5$.

    ### Guided practice

    1. Calculate a leaf mean and squared error for targets $[1,3,5]$.
    2. Calculate residuals for targets $[5,7]$ and predictions $[6,6]$.
    3. Apply learning rate 0.2 to proposed corrections $[-2,3]$.
    4. Explain why a correction tree predicts numerical residuals.
    5. Explain why the selected round belongs to validation, not test.

    ### Independent practice

    6. Rebuild the stump search with descriptive variable names and input checks.
    7. Add `minimum_improvement` to the scratch booster.
    8. Compare learning rates 0.1 and 0.03 using the same validation partition.
    9. Implement patience-based early stopping without using test data.
    10. Plot residuals before and after one learned stump.

    ### Challenge

    Rebuild the diabetes project without copying. Include a mean baseline, partition
    manifest, validation-selected round, exactly one final test evaluation, at least
    eight assertions, and no XGBoost, SHAP, monitoring, or test-based tuning.

    ### Self-check

    1. Why does a regression-tree leaf predict a mean?
    2. What does a positive residual mean?
    3. Does a new tree replace the current ensemble?
    4. What does the learning rate scale?
    5. Why is an ordinary squared-loss residual a negative gradient?
    6. Why can training and validation loss move differently?
    7. What evidence selects the round?
    8. What remains for CML-06?

    ### Solution and scoring rubric

    1. For $[1,3,5]$, mean is 3 and squared error is $4+0+4=8$.
    2. Residuals are $[-1,1]$.
    3. Scaled corrections are $[-0.4,0.6]$.
    4. A correction tree is a regressor because residuals are numerical.
    5. Validation selects the round because selecting it is a development decision.

    Score the eight self-check answers at two points each and the challenge at four
    points. Full credit requires arithmetic, mechanism, and data-boundary reasoning.

    ### Common mistakes

    - Fitting the next tree to original targets instead of current residuals.
    - Replacing predictions instead of adding a correction.
    - Forgetting to multiply by the learning rate.
    - Assuming a smaller learning rate is automatically better.
    - Treating a rough learning-rate/rounds heuristic as a law.
    - Using a classification tree to predict numerical residuals.
    - Selecting the stopping round from test loss.
    - Claiming validation loss must rise in every finite dataset.
    - Jumping to Hessians before understanding $y-F$.
    - Reporting only training loss.

    ### Readiness threshold

    Score at least **16/20**, including a correct manual correction, stump explanation,
    gradient connection, validation-selected round, and sealed-test workflow.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Explain this chain without notes:

    training-target mean  
    → residuals  
    → regression stump  
    → scaled correction  
    → repeat  
    → validation selects the round  
    → test once.

    ### Teach it back

    Explain why fitting $y-F$ is both ordinary error correction and a negative-gradient
    step for squared loss. Then explain why the tree must learn a reusable function
    instead of updating each training prediction independently.

    ### Memory aid

    **Gradient boosting adds small trees that predict what the current model still
    misses; validation decides when to stop.**

    ### Next dependency

    Gradient boosting and negative gradients  
    → required before XGBoost second-order mechanics  
    → because Hessians refine a correction loop that must already be understood.
    """),
]


build("02_classical_ml/05_gradient_boosting_foundations.ipynb", cells)
