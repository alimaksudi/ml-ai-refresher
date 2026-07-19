"""Build CML-06: XGBoost mechanics and regularization from first principles."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # CML-06 · XGBoost Mechanics and Regularization

    **Prerequisites:** CML-05, FND-04, CML-02, MLE-02, and MLE-03  
    **Estimated study time:** 9–12 hours, including practice  
    **Next lesson:** MLE-04 · Imbalanced Learning

    CML-05 built gradient boosting from residual corrections. XGBoost keeps the
    sequential additive idea and changes how a new tree is scored: it uses both the
    slope and curvature of the loss, then explicitly penalizes complex or extreme
    corrections.

    The goal is not to copy XGBoost parameters. The goal is to calculate gradients,
    Hessians, leaf weights, and one split gain manually before using the library.

    ### Scope boundary

    This lesson teaches binary logistic boosting, second-order leaf scoring, L2 leaf
    regularization, split penalties, shrinkage, histogram intuition, and missing-value
    routing. It defers:

    - class-imbalance interventions to MLE-04;
    - feature importance and SHAP to MLE-05;
    - monitoring and retraining policy to PROD-05 and PROD-06;
    - distributed/GPU kernel implementation details;
    - ranking objectives and custom-objective engineering.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - connect CML-05's negative gradients to binary log loss;
    - calculate sigmoid probabilities, gradients, and Hessians manually;
    - explain curvature without treating it as mysterious notation;
    - derive the regularized optimal value of one leaf;
    - calculate one candidate split gain;
    - explain how `lambda` shrinks leaf values and `gamma` rejects weak splits;
    - distinguish learning-rate shrinkage from L2 regularization;
    - build one second-order stump from scratch;
    - assemble several Newton-style boosting rounds;
    - explain exact and histogram split search conceptually;
    - explain learned missing-value directions;
    - map mathematical quantities to XGBoost parameters;
    - use validation for early stopping and test exactly once;
    - identify what XGBoost does not repair: leakage, bad labels, or invalid splits.

    ### Learning path

    ```mermaid
    flowchart LR
        A[Scores and probabilities] --> B[Gradients]
        B --> C[Hessians]
        C --> D[Leaf objective]
        D --> E[Optimal leaf weight]
        E --> F[Split gain]
        F --> G[Regularized stump]
        G --> H[Sequential rounds]
        H --> I[Histogram and missing routing]
        I --> J[Validated XGBoost]
    ```

    Dependency map:

    Gradient boosting  
    → required before XGBoost  
    → because second-order scoring refines the same additive correction loop.

    Logistic regression  
    → required before binary XGBoost  
    → because raw scores become probabilities through sigmoid and log loss.
    """),

    md(r"""
    ## 2 · The practical problem: not every correction needs the same confidence

    CML-05 fitted trees to first-order correction targets. A slope tells which
    direction reduces loss, but it does not describe how sharply loss bends nearby.

    Analogy: when walking downhill, slope tells the downhill direction. Curvature
    tells whether the ground is a broad gentle bowl or a narrow steep channel. The
    analogy stops at intuition—XGBoost aggregates curvature across examples in each
    candidate leaf.

    XGBoost asks three questions for every candidate leaf or split:

    1. What direction do the gradients request?
    2. How much curvature supports that correction?
    3. Is the improvement large enough after regularization penalties?

    This produces corrections that depend on both error direction and local loss
    shape.
    """),

    md(r"""
    ## 3 · Start with binary scores, probabilities, and log loss

    A binary boosted model maintains a raw score $F_i$ for row $i$. Sigmoid converts
    the score to a class-1 probability:

    $$
    p_i=\sigma(F_i)=\frac{1}{1+e^{-F_i}}
    $$

    At score zero, probability is 0.5. Binary log loss for one row is:

    $$
    L_i=-\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
    $$

    **Symbols:** $F_i$ is raw score; $p_i$ is predicted probability; $y_i$ is 0 or
    1; $L_i$ is row loss; $\log$ is natural logarithm.

    A probability near the correct label has small loss. A confident probability
    near the wrong label has large loss.
    """),

    code(r"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    def sigmoid(raw_scores):
        '''Convert raw scores to probabilities with clipping for numerical safety.'''
        raw_scores = np.asarray(raw_scores, dtype=float)
        clipped_scores = np.clip(raw_scores, -30, 30)
        return 1.0 / (1.0 + np.exp(-clipped_scores))


    manual_labels = np.array([1.0, 0.0, 1.0])
    manual_scores = np.zeros(3)
    manual_probabilities = sigmoid(manual_scores)
    clipped_probabilities = np.clip(manual_probabilities, 1e-12, 1 - 1e-12)
    row_log_losses = -(
        manual_labels * np.log(clipped_probabilities)
        + (1 - manual_labels) * np.log(1 - clipped_probabilities)
    )

    print("raw scores:", manual_scores)
    print("probabilities:", manual_probabilities)
    print("row log losses:", row_log_losses.round(4))

    assert np.allclose(manual_probabilities, 0.5)
    assert np.allclose(row_log_losses, -np.log(0.5))
    """),

    md(r"""
    ## 4 · Gradient gives direction; Hessian gives curvature

    For binary log loss with respect to raw score $F_i$:

    $$
    g_i=\frac{\partial L_i}{\partial F_i}=p_i-y_i
    $$

    $$
    h_i=\frac{\partial^2 L_i}{\partial F_i^2}=p_i(1-p_i)
    $$

    **Symbols:** $g_i$ is the gradient; $h_i$ is the Hessian, or second derivative.

    At probability 0.5:

    - a positive label has gradient $0.5-1=-0.5$, requesting a higher score;
    - a negative label has gradient $0.5-0=0.5$, requesting a lower score;
    - Hessian is $0.5(1-0.5)=0.25$ for either row.

    CML-05 used the negative gradient $y-p$. XGBoost often stores $g=p-y$ and places
    the minus sign in the optimal leaf-weight formula. These are consistent sign
    conventions, not different algorithms.
    """),

    code(r"""
    manual_gradients = manual_probabilities - manual_labels
    manual_hessians = manual_probabilities * (1 - manual_probabilities)

    gradient_table = pd.DataFrame(
        {
            "label": manual_labels.astype(int),
            "score": manual_scores,
            "probability": manual_probabilities,
            "gradient_p_minus_y": manual_gradients,
            "hessian_p_times_1_minus_p": manual_hessians,
        }
    )

    print(gradient_table.to_string(index=False))

    assert np.allclose(manual_gradients, [-0.5, 0.5, -0.5])
    assert np.allclose(manual_hessians, 0.25)
    """),

    md(r"""
    ## 5 · Derive one regularized leaf value

    A second-order approximation makes one leaf's objective a simple quadratic:

    $$
    \widetilde L(w)=Gw+\frac{1}{2}(H+\lambda)w^2
    $$

    where:

    - $w$ is the leaf's score correction;
    - $G=\sum_i g_i$ is the leaf's gradient sum;
    - $H=\sum_i h_i$ is the leaf's Hessian sum;
    - $\lambda\ge0$ is L2 regularization on the leaf value.

    Differentiate with respect to $w$ and set the slope to zero:

    $$
    \frac{\partial\widetilde L}{\partial w}=G+(H+\lambda)w=0
    $$

    Therefore:

    $$
    w^*=-\frac{G}{H+\lambda}
    $$

    For the three manual rows, $G=-0.5$, $H=0.75$, and $\lambda=1$:

    $$
    w^*=\frac{0.5}{1.75}\approx0.286
    $$

    The positive correction makes sense because two of three labels are positive.
    """),

    code(r"""
    def optimal_leaf_weight(gradient_sum, hessian_sum, l2_regularization):
        '''Return the regularized score correction for one leaf.'''
        return -gradient_sum / (hessian_sum + l2_regularization)


    manual_gradient_sum = manual_gradients.sum()
    manual_hessian_sum = manual_hessians.sum()
    manual_l2 = 1.0
    manual_leaf_weight = optimal_leaf_weight(
        manual_gradient_sum,
        manual_hessian_sum,
        manual_l2,
    )

    print("gradient sum G:", manual_gradient_sum)
    print("Hessian sum H:", manual_hessian_sum)
    print("L2 lambda:", manual_l2)
    print("optimal leaf weight:", round(manual_leaf_weight, 4))

    assert np.isclose(manual_leaf_weight, 0.5 / 1.75)
    """),

    md(r"""
    ## 6 · Calculate one split gain completely

    For feature values $[1,2,3,4]$, labels $[0,0,1,1]$, and initial probability 0.5:

    - gradients are $[0.5,0.5,-0.5,-0.5]$;
    - every Hessian is 0.25;
    - threshold 2.5 separates the two negative and two positive rows.

    The regularized score of one leaf is:

    $$
    S(G,H)=\frac{G^2}{H+\lambda}
    $$

    Split gain is:

    $$
    \operatorname{Gain}
    =\frac{1}{2}\left[
    \frac{G_L^2}{H_L+\lambda}
    +\frac{G_R^2}{H_R+\lambda}
    -\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
    \right]-\gamma
    $$

    $L$ and $R$ mean left and right child. $\gamma$ is the minimum split penalty.
    A split is accepted only when its gain is positive after the penalty.
    """),

    code(r"""
    def regularized_split_gain(
        left_gradient_sum,
        left_hessian_sum,
        right_gradient_sum,
        right_hessian_sum,
        l2_regularization,
        split_penalty,
    ):
        '''Calculate second-order improvement after splitting one parent leaf.'''
        parent_gradient_sum = left_gradient_sum + right_gradient_sum
        parent_hessian_sum = left_hessian_sum + right_hessian_sum

        left_score = left_gradient_sum ** 2 / (left_hessian_sum + l2_regularization)
        right_score = right_gradient_sum ** 2 / (right_hessian_sum + l2_regularization)
        parent_score = parent_gradient_sum ** 2 / (
            parent_hessian_sum + l2_regularization
        )

        return 0.5 * (left_score + right_score - parent_score) - split_penalty


    left_gradient_sum, left_hessian_sum = 1.0, 0.5
    right_gradient_sum, right_hessian_sum = -1.0, 0.5
    split_l2, split_gamma = 1.0, 0.0

    manual_gain = regularized_split_gain(
        left_gradient_sum,
        left_hessian_sum,
        right_gradient_sum,
        right_hessian_sum,
        split_l2,
        split_gamma,
    )
    left_weight = optimal_leaf_weight(left_gradient_sum, left_hessian_sum, split_l2)
    right_weight = optimal_leaf_weight(right_gradient_sum, right_hessian_sum, split_l2)

    print("left leaf weight:", round(left_weight, 4))
    print("right leaf weight:", round(right_weight, 4))
    print("split gain:", round(manual_gain, 4))

    assert np.isclose(left_weight, -2 / 3)
    assert np.isclose(right_weight, 2 / 3)
    assert np.isclose(manual_gain, 2 / 3)
    """),

    md(r"""
    ## 7 · Regularization controls different parts of the update

    Three controls are easy to confuse:

    | Control | Acts on | Main effect |
    |---|---|---|
    | Learning rate $\eta$ | Every tree contribution | Shrinks the entire round's update |
    | L2 $\lambda$ | Leaf weight denominator | Pulls extreme leaf corrections toward zero |
    | Split penalty $\gamma$ | Candidate split gain | Rejects splits whose improvement is too small |

    With $G=1$ and $H=0.5$:

    - $\lambda=0$ gives $w=-2$;
    - $\lambda=1$ gives $w\approx-0.667$;
    - $\lambda=4$ gives $w\approx-0.222$.

    Larger regularization is not automatically better. Too much can erase useful
    corrections and underfit. These controls are chosen with validation evidence,
    never final-test evidence.
    """),

    code(r"""
    l2_values = np.array([0.0, 1.0, 4.0])
    regularized_weights = np.array([
        optimal_leaf_weight(1.0, 0.5, l2_value)
        for l2_value in l2_values
    ])

    regularization_table = pd.DataFrame(
        {"lambda": l2_values, "leaf_weight": regularized_weights}
    )
    print(regularization_table.to_string(index=False))

    assert abs(regularized_weights[0]) > abs(regularized_weights[1])
    assert abs(regularized_weights[1]) > abs(regularized_weights[2])
    """),

    md(r"""
    ## 8 · Build one second-order stump from scratch

    The stump search combines the pieces:

    1. convert current scores to probabilities;
    2. calculate one gradient and Hessian per row;
    3. test feature-threshold candidates;
    4. reject children with too little total Hessian;
    5. select the largest positive regularized gain;
    6. store optimal left and right leaf weights.

    `min_child_weight` in XGBoost is a minimum Hessian-sum requirement, not simply a
    minimum number of rows. For logistic loss, uncertain predictions have larger
    Hessians than extremely confident predictions.
    """),

    code(r"""
    def fit_second_order_stump(
        feature_matrix,
        labels,
        raw_scores,
        l2_regularization=1.0,
        split_penalty=0.0,
        minimum_child_hessian=1.0,
    ):
        '''Fit one binary logistic stump using gradients, Hessians, and gain.'''
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        labels = np.asarray(labels, dtype=float)
        probabilities = sigmoid(raw_scores)
        gradients = probabilities - labels
        hessians = probabilities * (1 - probabilities)

        best_stump = None
        best_gain = 0.0
        number_of_features = feature_matrix.shape[1]

        for feature_index in range(number_of_features):
            unique_values = np.unique(feature_matrix[:, feature_index])
            candidate_thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in candidate_thresholds:
                left_mask = feature_matrix[:, feature_index] <= threshold
                right_mask = ~left_mask

                left_hessian_sum = float(hessians[left_mask].sum())
                right_hessian_sum = float(hessians[right_mask].sum())
                if (
                    left_hessian_sum < minimum_child_hessian
                    or right_hessian_sum < minimum_child_hessian
                ):
                    continue

                left_gradient_sum = float(gradients[left_mask].sum())
                right_gradient_sum = float(gradients[right_mask].sum())
                gain = regularized_split_gain(
                    left_gradient_sum,
                    left_hessian_sum,
                    right_gradient_sum,
                    right_hessian_sum,
                    l2_regularization,
                    split_penalty,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_stump = {
                        "feature_index": feature_index,
                        "threshold": float(threshold),
                        "left_weight": optimal_leaf_weight(
                            left_gradient_sum,
                            left_hessian_sum,
                            l2_regularization,
                        ),
                        "right_weight": optimal_leaf_weight(
                            right_gradient_sum,
                            right_hessian_sum,
                            l2_regularization,
                        ),
                        "gain": float(gain),
                    }

        return best_stump


    def predict_second_order_stump(stump, feature_matrix):
        '''Return each row's raw-score correction from a fitted stump.'''
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        left_mask = feature_matrix[:, stump["feature_index"]] <= stump["threshold"]
        corrections = np.full(len(feature_matrix), stump["right_weight"], dtype=float)
        corrections[left_mask] = stump["left_weight"]
        return corrections


    tiny_features = np.array([[1.0], [2.0], [3.0], [4.0]])
    tiny_labels = np.array([0, 0, 1, 1])
    tiny_scores = np.zeros(4)
    tiny_stump = fit_second_order_stump(
        tiny_features,
        tiny_labels,
        tiny_scores,
        minimum_child_hessian=0.5,
    )

    print("learned second-order stump:", tiny_stump)

    assert np.isclose(tiny_stump["threshold"], 2.5)
    assert np.isclose(tiny_stump["gain"], 2 / 3)
    """),

    md(r"""
    ## 9 · Assemble several Newton-style boosting rounds

    The ensemble still follows CML-05:

    $$
    F_m(x)=F_{m-1}(x)+\eta h_m(x)
    $$

    The difference is how $h_m$ chooses its structure and leaf values. Each round
    recalculates probabilities, gradients, and Hessians from the updated scores.

    The educational model below uses stumps. Production XGBoost grows deeper trees,
    applies more constraints, and uses faster split search, but the score update is
    the same additive mechanism.
    """),

    code(r"""
    class LearningSecondOrderBooster:
        '''Educational binary booster using regularized second-order stumps.'''

        def __init__(
            self,
            number_of_rounds=40,
            learning_rate=0.2,
            l2_regularization=1.0,
            split_penalty=0.0,
            minimum_child_hessian=2.0,
        ):
            self.number_of_rounds = number_of_rounds
            self.learning_rate = learning_rate
            self.l2_regularization = l2_regularization
            self.split_penalty = split_penalty
            self.minimum_child_hessian = minimum_child_hessian

        def fit(self, feature_matrix, labels):
            '''Fit sequential stumps and preserve their raw-score corrections.'''
            feature_matrix = np.asarray(feature_matrix, dtype=float)
            labels = np.asarray(labels, dtype=int)

            # Initial log-odds match the training positive-label frequency.
            positive_fraction = np.clip(labels.mean(), 1e-6, 1 - 1e-6)
            self.initial_score_ = float(
                np.log(positive_fraction / (1 - positive_fraction))
            )
            current_scores = np.full(len(labels), self.initial_score_)
            self.stumps_ = []

            for _ in range(self.number_of_rounds):
                stump = fit_second_order_stump(
                    feature_matrix,
                    labels,
                    current_scores,
                    l2_regularization=self.l2_regularization,
                    split_penalty=self.split_penalty,
                    minimum_child_hessian=self.minimum_child_hessian,
                )
                if stump is None:
                    break

                # Shrink the second-order correction before changing raw scores.
                current_scores = current_scores + self.learning_rate * (
                    predict_second_order_stump(stump, feature_matrix)
                )
                self.stumps_.append(stump)

            return self

        def staged_predict_positive_probability(self, feature_matrix):
            '''Yield probabilities after each fitted stump.'''
            current_scores = np.full(len(feature_matrix), self.initial_score_)
            for stump in self.stumps_:
                current_scores = current_scores + self.learning_rate * (
                    predict_second_order_stump(stump, feature_matrix)
                )
                yield sigmoid(current_scores)

        def predict_positive_probability(self, feature_matrix, number_of_rounds=None):
            '''Replay a selected prefix of stumps and return probabilities.'''
            rounds_to_use = len(self.stumps_) if number_of_rounds is None else number_of_rounds
            current_scores = np.full(len(feature_matrix), self.initial_score_)
            for stump in self.stumps_[:rounds_to_use]:
                current_scores = current_scores + self.learning_rate * (
                    predict_second_order_stump(stump, feature_matrix)
                )
            return sigmoid(current_scores)
    """),

    md(r"""
    ## 10 · Select rounds with validation log loss

    We use a synthetic binary dataset so the scratch mechanism stays visible. Split
    before fitting: training learns stumps, validation selects a round, and test stays
    sealed.

    Log loss is used for round selection because the training objective operates on
    probabilities. A later decision threshold solves a different problem and must not
    be confused with the objective.
    """),

    code(r"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss

    all_features, all_labels = make_classification(
        n_samples=520,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        class_sep=1.0,
        flip_y=0.05,
        random_state=42,
    )

    # Seal test first, then create a separate validation partition.
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

    learning_xgb = LearningSecondOrderBooster(
        number_of_rounds=50,
        learning_rate=0.2,
        l2_regularization=1.0,
        split_penalty=0.0,
        minimum_child_hessian=2.0,
    )
    learning_xgb.fit(train_features, train_labels)

    validation_log_loss_by_round = [
        log_loss(validation_labels, probabilities, labels=[0, 1])
        for probabilities in learning_xgb.staged_predict_positive_probability(
            validation_features
        )
    ]
    selected_round = int(np.argmin(validation_log_loss_by_round)) + 1

    print("stumps fitted:", len(learning_xgb.stumps_))
    print("round selected by validation:", selected_round)
    print("selected validation log loss:", round(validation_log_loss_by_round[selected_round - 1], 4))
    print("test status: sealed — no test predictions calculated")

    assert len(learning_xgb.stumps_) > 0
    assert 1 <= selected_round <= len(learning_xgb.stumps_)
    """),

    md(r"""
    ## 11 · Histogram split search and missing-value routing

    The scratch search tries every midpoint. Production datasets may have millions of
    distinct values, so XGBoost can use histogram-based search:

    1. summarize numerical values into ordered bins;
    2. accumulate gradient and Hessian sums per bin;
    3. scan bin boundaries instead of every distinct value;
    4. reuse compact summaries across candidate splits.

    This trades some split precision for major speed and memory savings. “Histogram”
    refers to quantized feature bins, not a target-frequency chart.

    For missing values, a candidate split evaluates a default direction: send missing
    rows left or send them right, then keep the direction with better training gain.
    This is learned routing, not proof that missingness is harmless. The feature must
    have the same missing-value meaning at prediction time.

    ```mermaid
    flowchart TD
        A[Feature values] --> B[Quantile-style bins]
        B --> C[Sum G and H per bin]
        C --> D[Scan candidate boundaries]
        M[Missing rows] --> L[Try missing left]
        M --> R[Try missing right]
        L --> E[Keep better regularized gain]
        R --> E
    ```
    """),

    md(r"""
    ## 12 · Map the mechanism to XGBoost parameters

    | Parameter | Mechanism | Beginner interpretation |
    |---|---|---|
    | `objective` | Defines gradients and Hessians | What loss is being optimized? |
    | `eta` | Learning-rate shrinkage | How much of each tree is added? |
    | `max_depth` | Tree complexity | How many interactions may one round capture? |
    | `min_child_weight` | Minimum child Hessian sum | Does a child have enough curvature support? |
    | `lambda` | L2 leaf penalty | How strongly are leaf values pulled toward zero? |
    | `gamma` | Split penalty | How much gain must justify another split? |
    | `subsample` | Row sampling per round | How much training-row randomness is added? |
    | `colsample_bytree` | Feature sampling per tree | How many features may one tree inspect? |
    | `tree_method="hist"` | Histogram split search | Should split candidates use bins? |

    Parameters interact. For example, a smaller `eta` usually needs more rounds;
    stronger `lambda` may require more evidence for useful corrections. Change a
    small declared set using validation, record the experiment, and leave test sealed.
    """),

    md(r"""
    ## 13 · Mini-project: validated breast-cancer classifier

    **Goal:** predict malignant versus benign diagnosis from numerical measurements.

    **Workflow:**

    1. declare the positive label;
    2. create train, validation, and sealed test partitions;
    3. freeze a training-majority baseline;
    4. use validation for native early stopping;
    5. freeze `best_iteration`;
    6. evaluate probability loss and wrong-row count once on test.

    **Success contract:** test is absent from the XGBoost evaluation list; no feature
    importance or SHAP claim is made; early stopping uses validation only.
    """),

    code(r"""
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    project_features = cancer.data
    # sklearn labels benign as 1; redefine positive=1 to mean malignant explicitly.
    project_labels = (cancer.target == 0).astype(int)

    project_development_features, project_test_features, project_development_labels, project_test_labels = train_test_split(
        project_features,
        project_labels,
        test_size=0.20,
        stratify=project_labels,
        random_state=19,
    )
    project_train_features, project_validation_features, project_train_labels, project_validation_labels = train_test_split(
        project_development_features,
        project_development_labels,
        test_size=0.25,
        stratify=project_development_labels,
        random_state=19,
    )

    # Freeze the training-majority baseline before model fitting.
    project_baseline_class = int(project_train_labels.mean() >= 0.5)
    baseline_validation_wrong = int(
        np.sum(project_validation_labels != project_baseline_class)
    )

    training_matrix = xgb.DMatrix(project_train_features, label=project_train_labels)
    validation_matrix = xgb.DMatrix(
        project_validation_features,
        label=project_validation_labels,
    )

    project_parameters = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 3,
        "min_child_weight": 2.0,
        "lambda": 1.0,
        "gamma": 0.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": 19,
    }

    # Validation is the only evaluation set used for early stopping.
    project_model = xgb.train(
        project_parameters,
        training_matrix,
        num_boost_round=300,
        evals=[(validation_matrix, "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    validation_probabilities = project_model.predict(
        validation_matrix,
        iteration_range=(0, project_model.best_iteration + 1),
    )
    validation_decisions = (validation_probabilities >= 0.5).astype(int)
    project_validation_wrong = int(
        np.sum(validation_decisions != project_validation_labels)
    )

    partition_manifest = pd.DataFrame(
        {
            "partition": ["train", "validation", "test"],
            "rows": [
                len(project_train_labels),
                len(project_validation_labels),
                len(project_test_labels),
            ],
            "used_for_early_stopping": [False, True, False],
        }
    )

    print(partition_manifest.to_string(index=False))
    print("baseline validation wrong:", baseline_validation_wrong)
    print("XGBoost validation wrong:", project_validation_wrong)
    print("best zero-based iteration:", project_model.best_iteration)
    print("test status: sealed — absent from early stopping")

    assert project_validation_wrong < baseline_validation_wrong
    assert project_model.best_iteration < 300
    """),

    code(r"""
    # The test matrix is created only after best_iteration has been frozen.
    project_test_matrix = xgb.DMatrix(project_test_features, label=project_test_labels)
    final_test_probabilities = project_model.predict(
        project_test_matrix,
        iteration_range=(0, project_model.best_iteration + 1),
    )
    final_test_decisions = (final_test_probabilities >= 0.5).astype(int)
    final_test_wrong = int(np.sum(final_test_decisions != project_test_labels))
    final_test_log_loss = float(
        log_loss(project_test_labels, final_test_probabilities, labels=[0, 1])
    )

    print("final sealed-test rows:", len(project_test_labels))
    print("final sealed-test wrong rows:", final_test_wrong)
    print("final sealed-test log loss:", round(final_test_log_loss, 4))
    print("this final estimate is not a new early-stopping signal")

    assert len(final_test_probabilities) == len(project_test_labels)
    assert np.all((final_test_probabilities >= 0) & (final_test_probabilities <= 1))
    assert np.isfinite(final_test_log_loss)
    """),

    md(r"""
    ## 14 · Practice, solutions, and mastery checkpoint

    ### Worked example

    At $p=0.5$, labels $[0,0,1,1]$ give gradients $[0.5,0.5,-0.5,-0.5]$
    and Hessians $[0.25,0.25,0.25,0.25]$. Threshold 2.5 gives child weights
    $-2/3$ and $2/3$ with $\lambda=1$, and gain $2/3$ when $\gamma=0$.

    ### Guided practice

    1. Calculate $g$ and $h$ for $y=1,p=0.8$.
    2. Calculate $w^*$ for $G=-2,H=1,\lambda=1$.
    3. Recalculate that weight with $\lambda=3$ and explain the change.
    4. Explain why `gamma` affects split acceptance but not sigmoid.
    5. Explain why validation, not test, belongs in the early-stopping list.

    ### Independent practice

    6. Rebuild the leaf-weight and gain functions with denominator checks.
    7. Apply one stump update with $\eta=0.3$ and print scores and probabilities.
    8. Compare scratch models with $\lambda=0$ and $\lambda=4$ on the same validation set.
    9. Add a positive `split_penalty` and count how many stumps still fit.
    10. Bin one numerical feature manually and compare candidate bin boundaries.

    ### Challenge

    Rebuild the breast-cancer project with a partition manifest, validation-only early
    stopping, frozen best iteration, one final test, and assertions proving test is
    absent from development. Do not use SHAP, feature importance, or imbalance methods.

    ### Self-check

    1. Why does XGBoost need both $g$ and $h$?
    2. Where does the minus sign enter the leaf correction?
    3. What does $\lambda$ shrink?
    4. What does $\gamma$ reject?
    5. How is `min_child_weight` different from row count?
    6. Why does `eta` remain necessary after regularized leaf scoring?
    7. What does histogram search approximate?
    8. How is a missing-value direction chosen?

    ### Solution and scoring rubric

    1. For $y=1,p=0.8$, $g=-0.2$ and $h=0.16$.
    2. $w^*=-(-2)/(1+1)=1$.
    3. With $\lambda=3$, $w^*=0.5$; stronger L2 shrinks the correction.
    4. `gamma` is subtracted from split gain; sigmoid only maps scores to probabilities.
    5. Test evidence estimates the frozen procedure and cannot choose its stopping round.

    Score the eight self-check answers at two points each and the challenge at four
    points. Full credit requires a manual leaf weight, split gain, and honest boundary.

    ### Common mistakes

    - Confusing gradient $p-y$ with negative gradient $y-p$.
    - Calling the Hessian another residual.
    - Omitting $\lambda$ from the leaf-weight denominator.
    - Applying $\gamma$ directly to probabilities.
    - Treating `min_child_weight` as an ordinary row count.
    - Using the final test in `evals` for early stopping.
    - Assuming histogram bins are target histograms.
    - Treating learned missing routing as a repair for bad data collection.
    - Tuning many interacting parameters without a baseline and experiment record.
    - Using feature importance before learning its biases.

    ### Readiness threshold

    Score at least **16/20**, including correct $g$, $h$, leaf weight, split gain,
    regularization explanation, and validation-only early stopping.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    Explain this chain without notes:

    score  
    → sigmoid probability  
    → gradient and Hessian  
    → regularized leaf value  
    → split gain  
    → learning-rate-scaled update  
    → validation early stopping  
    → one sealed test.

    ### Teach it back

    Derive $w^*=-G/(H+\lambda)$ from the quadratic leaf objective, then explain in
    plain language why larger curvature and stronger L2 both reduce an extreme update.

    ### Memory aid

    **XGBoost adds regularized trees using gradient direction, Hessian curvature, and
    validation-controlled rounds.**

    ### Next dependency

    Validated probability models  
    → required before imbalanced learning  
    → because MLE-04 changes sampling, weighting, and thresholds only after the base
    model and evaluation boundary are understood.
    """),
]


build("02_classical_ml/06_xgboost_mechanics_and_regularization.ipynb", cells)
