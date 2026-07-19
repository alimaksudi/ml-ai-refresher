"""Build CML-02: Logistic Regression."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # CML-02 · Logistic Regression

    **Prerequisites:** FND-02, CML-01, and FND-04  
    **Estimated study time:** 8–10 hours, including practice  
    **Next lesson:** CML-03 · Decision Trees

    Linear regression predicts a numerical quantity. Logistic regression predicts the
    probability of one class in a binary classification task.

    The name is historically confusing: logistic regression is a **classifier**. It
    builds a linear score, passes that score through the sigmoid function, and fits
    the coefficients using binary cross-entropy.

    ### Scope boundary

    This lesson teaches one binary classifier deeply. It deliberately defers:

    - accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and calibration to MLE-01;
    - imbalance strategies and cost-sensitive evaluation to MLE-04;
    - cross-validation and threshold selection protocols to MLE-02;
    - Ridge/Lasso-style regularization to the later regularization extension;
    - softmax and multiclass cross-entropy to the neural-classification path.

    We will count decision errors here, but we will not compress them into unexplained
    evaluation metrics.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - distinguish regression from binary classification;
    - define a positive class and encode labels as 0 and 1;
    - create a training class-rate probability baseline;
    - keep a linear score, probability, and class decision separate;
    - explain odds and log-odds with small numbers;
    - calculate sigmoid probabilities manually;
    - derive binary cross-entropy from a Bernoulli likelihood;
    - explain why confidently wrong probabilities receive a large loss;
    - derive the logistic-regression gradient;
    - fit a one-feature classifier with gradient descent;
    - use stable sigmoid and logarithm calculations;
    - apply a threshold only after probability prediction;
    - count true/false positive and negative decisions;
    - fit sklearn `LogisticRegression` after the manual implementation;
    - interpret coefficients as changes in log-odds and odds;
    - explain linear-boundary, extrapolation, separation, and causal limitations;
    - complete a split-safe binary-classification mini-project.

    ### Dependency map

    ```mermaid
    flowchart LR
        A[Linear feature score] --> B[Sigmoid probability]
        B --> C[Bernoulli cross-entropy]
        C --> D[Gradient descent fitting]
        B --> E[Decision threshold]
        E --> F[Positive or negative decision]
    ```

    The loss fits probabilities. The threshold creates decisions. Changing a threshold
    does not refit the probability model.
    """),

    md(r"""
    ## 2 · The practical problem: estimate late-delivery probability

    A dispatcher wants an early warning that a route may arrive late. We begin with
    one feature: route distance.

    | Route | Distance (km) | Late label |
    | --- | ---: | ---: |
    | A | 1.0 | 0 |
    | B | 1.5 | 0 |
    | C | 2.0 | 0 |
    | D | 2.5 | 0 |
    | E | 3.0 | 1 |
    | F | 3.5 | 0 |
    | G | 4.0 | 1 |
    | H | 4.5 | 1 |

    Here:

    - 1 is the **positive class**: late;
    - 0 is the **negative class**: not late;
    - the label is known only after delivery;
    - the prediction is requested before departure;
    - the output should be a probability, not a guarantee.

    A probability allows the dispatcher to choose a later action threshold based on
    staffing and error costs. The model does not decide those costs.
    """),

    code(r"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    route_distance_km = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    late_label = np.array([0, 0, 0, 0, 1, 0, 1, 1], dtype=float)

    training_positive_rate = late_label.mean()
    baseline_probabilities = np.full(late_label.shape, training_positive_rate)

    print("labels:", late_label.astype(int))
    print("positive labels:", int(late_label.sum()))
    print("training positive rate:", training_positive_rate)
    print("constant baseline probabilities:", baseline_probabilities)

    assert np.isclose(training_positive_rate, 3 / 8)
    """),

    md(r"""
    ## 3 · Begin with a probability baseline

    Before using features, predict the training positive-class proportion for every
    row:

    $$
    \hat p_{base}=\frac{1}{n}\sum_{i=1}^{n}y_i
    $$

    **Symbols:** $y_i$ is a binary training label; $n$ is the number of training rows;
    $\hat p_{base}$ is the estimated positive rate.

    Here:

    $$
    \hat p_{base}=\frac{3}{8}=0.375
    $$

    The baseline predicts 37.5% late probability for every route. It ignores distance.
    A fitted classifier must produce better validation evidence than this frozen
    training estimate.

    This is a probability baseline, not an “always negative” decision baseline. A
    probability and a thresholded class are different objects.
    """),

    md(r"""
    ## 4 · Score, probability, and decision are three separate objects

    Logistic regression first creates an unrestricted linear score:

    $$
    z=b+wx
    $$

    The score $z$ can be any real number, so it is not yet a probability. The sigmoid
    converts it to a value strictly between 0 and 1:

    $$
    p=\sigma(z)=\frac{1}{1+e^{-z}}
    $$

    A threshold $t$ then creates a decision:

    $$
    \hat y=
    \begin{cases}
    1,&p\ge t\\
    0,&p<t
    \end{cases}
    $$

    **Symbols:** $b$ is an intercept; $w$ is a coefficient; $x$ is a feature; $z$ is
    a score; $e$ is the exponential constant; $p$ is predicted probability; $t$ is a
    chosen threshold; $\hat y$ is a class decision.

    | Score $z$ | Sigmoid probability | Decision at $t=0.5$ |
    | ---: | ---: | ---: |
    | -2 | about 0.119 | 0 |
    | 0 | 0.500 | 1 |
    | 2 | about 0.881 | 1 |

    A different threshold changes decisions but leaves scores and probabilities
    unchanged.
    """),

    code(r"""
    def stable_sigmoid(scores):
        '''Convert real-valued scores to probabilities without avoidable overflow.'''
        score_array = np.asarray(scores, dtype=float)
        probabilities = np.empty_like(score_array)
        nonnegative = score_array >= 0
        probabilities[nonnegative] = 1 / (1 + np.exp(-score_array[nonnegative]))
        negative_exponential = np.exp(score_array[~nonnegative])
        probabilities[~nonnegative] = negative_exponential / (1 + negative_exponential)
        return probabilities


    example_scores = np.array([-2.0, 0.0, 2.0, -1_000.0, 1_000.0])
    example_probabilities = stable_sigmoid(example_scores)
    example_decisions = (example_probabilities >= 0.5).astype(int)

    print("scores:", example_scores)
    print("probabilities:", example_probabilities)
    print("decisions at threshold 0.5:", example_decisions)

    assert np.all((example_probabilities >= 0) & (example_probabilities <= 1))
    assert np.isclose(example_probabilities[1], 0.5)
    assert np.array_equal(example_decisions[:3], [0, 1, 1])
    """),

    md(r"""
    ## 5 · Odds and log-odds explain the sigmoid

    Probability $p$ can be rewritten as **odds**:

    $$
    \operatorname{odds}=\frac{p}{1-p}
    $$

    If $p=0.75$, odds are:

    $$
    \frac{0.75}{0.25}=3
    $$

    Read this as three-to-one odds for the positive class—not as 300% probability.

    Taking a logarithm creates log-odds:

    $$
    \operatorname{logit}(p)=\log\left(\frac{p}{1-p}\right)
    $$

    Logistic regression assumes log-odds are linear:

    $$
    \log\left(\frac{p}{1-p}\right)=b+wx
    $$

    Solving this equation for $p$ produces the sigmoid. The sigmoid is not an arbitrary
    curve attached to a linear model; it is the inverse transformation from log-odds
    back to probability.

    Probability 0.5 has odds 1 and log-odds 0. Probabilities below 0.5 have negative
    log-odds; probabilities above 0.5 have positive log-odds.
    """),

    code(r"""
    probability_examples = np.array([0.25, 0.50, 0.75])
    odds_examples = probability_examples / (1 - probability_examples)
    log_odds_examples = np.log(odds_examples)
    recovered_probabilities = stable_sigmoid(log_odds_examples)

    odds_table = pd.DataFrame(
        {
            "probability": probability_examples,
            "odds": odds_examples,
            "log_odds": log_odds_examples,
            "recovered_probability": recovered_probabilities,
        }
    )

    print(odds_table)

    assert np.allclose(recovered_probabilities, probability_examples)
    assert np.isclose(odds_examples[-1], 3.0)
    """),

    md(r"""
    ## 6 · Bernoulli likelihood leads to binary cross-entropy

    A binary label follows a Bernoulli data story. For one label $y\in\{0,1\}$ and
    predicted probability $p$:

    $$
    P(y\mid p)=p^y(1-p)^{1-y}
    $$

    If $y=1$, this becomes $p$. If $y=0$, it becomes $1-p$.

    The negative log-likelihood for one row is binary cross-entropy:

    $$
    \ell(y,p)=-\left[y\log(p)+(1-y)\log(1-p)\right]
    $$

    Across $n$ rows:

    $$
    L=-\frac{1}{n}\sum_{i=1}^{n}
    \left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
    $$

    **Symbols:** $\ell$ is one-row loss; $L$ is mean training loss; $y_i$ is the
    actual binary label; $p_i$ is predicted positive probability.

    For a positive label $y=1$:

    - $p=0.8$ gives $-\log(0.8)\approx0.223$;
    - $p=0.1$ gives $-\log(0.1)\approx2.303$.

    The confidently wrong prediction receives much more loss. We clip probabilities
    away from exactly 0 and 1 before taking logs because $\log(0)$ is undefined.
    """),

    code(r"""
    def binary_cross_entropy(labels, probabilities, epsilon=1e-12):
        '''Return mean binary cross-entropy after validating labels and shapes.'''
        label_array = np.asarray(labels, dtype=float)
        probability_array = np.asarray(probabilities, dtype=float)
        if label_array.shape != probability_array.shape:
            raise ValueError("labels and probabilities must have matching shapes")
        if not np.all(np.isin(label_array, [0.0, 1.0])):
            raise ValueError("labels must contain only 0 and 1")

        clipped = np.clip(probability_array, epsilon, 1 - epsilon)
        row_losses = -(
            label_array * np.log(clipped)
            + (1 - label_array) * np.log(1 - clipped)
        )
        return float(row_losses.mean())


    positive_good_loss = binary_cross_entropy([1], [0.8])
    positive_bad_loss = binary_cross_entropy([1], [0.1])
    negative_good_loss = binary_cross_entropy([0], [0.1])

    print("positive label, p=0.8:", positive_good_loss)
    print("positive label, p=0.1:", positive_bad_loss)
    print("negative label, p=0.1:", negative_good_loss)

    assert positive_bad_loss > positive_good_loss
    assert np.isclose(positive_good_loss, -np.log(0.8))
    assert np.isfinite(binary_cross_entropy([1, 0], [1.0, 0.0]))
    """),

    md(r"""
    ## 7 · Derive the logistic-regression gradient

    For one feature:

    $$
    z_i=b+wx_i,
    \qquad
    p_i=\sigma(z_i)
    $$

    The sigmoid derivative is:

    $$
    \frac{dp_i}{dz_i}=p_i(1-p_i)
    $$

    After applying the chain rule to binary cross-entropy, the score derivative
    simplifies to:

    $$
    \frac{\partial\ell_i}{\partial z_i}=p_i-y_i
    $$

    Therefore:

    $$
    \frac{\partial L}{\partial b}=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)
    $$

    $$
    \frac{\partial L}{\partial w}=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)x_i
    $$

    At $b=0,w=0$, every score is 0 and every probability is 0.5. For two rows
    $(x,y)=(1,0),(2,1)$:

    $$
    \frac{\partial L}{\partial b}
    =\frac{(0.5-0)+(0.5-1)}{2}=0
    $$

    $$
    \frac{\partial L}{\partial w}
    =\frac{(0.5)(1)+(-0.5)(2)}{2}=-0.25
    $$

    A positive learning rate increases $w$, making the larger feature more likely to
    receive a larger positive-class probability.
    """),

    code(r"""
    def logistic_loss_and_gradient(features, labels, intercept, coefficient):
        '''Calculate binary cross-entropy and gradients for one-feature logistic regression.'''
        feature_array = np.asarray(features, dtype=float)
        label_array = np.asarray(labels, dtype=float)
        if feature_array.ndim != 1 or label_array.ndim != 1:
            raise ValueError("features and labels must be one-dimensional")
        if feature_array.shape != label_array.shape:
            raise ValueError("features and labels must have matching shapes")
        if not np.all(np.isin(label_array, [0.0, 1.0])):
            raise ValueError("labels must contain only 0 and 1")

        scores = intercept + coefficient * feature_array
        probabilities = stable_sigmoid(scores)
        loss = binary_cross_entropy(label_array, probabilities)
        probability_errors = probabilities - label_array
        intercept_gradient = probability_errors.mean()
        coefficient_gradient = np.mean(probability_errors * feature_array)
        gradient = np.array([intercept_gradient, coefficient_gradient])
        return loss, gradient, probabilities


    manual_loss, manual_gradient, manual_probabilities = logistic_loss_and_gradient(
        [1.0, 2.0],
        [0.0, 1.0],
        intercept=0.0,
        coefficient=0.0,
    )

    print("probabilities:", manual_probabilities)
    print("loss:", manual_loss)
    print("gradient [intercept, coefficient]:", manual_gradient)

    assert np.allclose(manual_probabilities, [0.5, 0.5])
    assert np.allclose(manual_gradient, [0.0, -0.25])
    assert np.isclose(manual_loss, np.log(2))
    """),

    md(r"""
    ## 8 · Fit logistic regression from scratch

    The FND-04 update rule does not change:

    $$
    \boldsymbol\theta_{t+1}
    =\boldsymbol\theta_t-\eta\nabla L(\boldsymbol\theta_t)
    $$

    Here $\boldsymbol\theta=[b,w]$. What changed is the prediction rule and loss.

    We standardize distance using training values before fitting. This makes the
    learning-rate behavior easier to control. The final coefficient will therefore
    describe one training standard deviation of distance, not one kilometre.
    """),

    code(r"""
    distance_mean = route_distance_km.mean()
    distance_std = route_distance_km.std(ddof=0)
    standardized_distance = (route_distance_km - distance_mean) / distance_std

    def fit_logistic_regression_from_scratch(
        features,
        labels,
        learning_rate=0.20,
        steps=2_000,
        start=(0.0, 0.0),
    ):
        '''Fit one-feature logistic regression and return parameters and history.'''
        parameters = np.asarray(start, dtype=float).copy()
        if parameters.shape != (2,):
            raise ValueError("start must contain [intercept, coefficient]")
        if learning_rate <= 0 or steps < 1:
            raise ValueError("learning_rate and steps must be positive")

        records = []
        for step in range(steps + 1):
            loss, gradient, probabilities = logistic_loss_and_gradient(
                features,
                labels,
                intercept=parameters[0],
                coefficient=parameters[1],
            )
            if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("optimization became non-finite")

            records.append(
                {
                    "step": step,
                    "loss": loss,
                    "intercept": parameters[0],
                    "coefficient": parameters[1],
                    "gradient_norm": np.linalg.norm(gradient),
                }
            )
            if step < steps:
                parameters = parameters - learning_rate * gradient

        return parameters, pd.DataFrame(records), probabilities


    fitted_parameters, logistic_history, fitted_probabilities = (
        fit_logistic_regression_from_scratch(
            standardized_distance,
            late_label,
            learning_rate=0.20,
            steps=2_000,
        )
    )

    print("parameters [intercept, standardized-distance coefficient]:", fitted_parameters)
    print("starting cross-entropy:", logistic_history.iloc[0]["loss"])
    print("final cross-entropy:", logistic_history.iloc[-1]["loss"])
    print("fitted probabilities:", fitted_probabilities.round(3))

    assert logistic_history.iloc[-1]["loss"] < logistic_history.iloc[0]["loss"]
    assert np.all((fitted_probabilities > 0) & (fitted_probabilities < 1))
    assert fitted_parameters[1] > 0
    """),

    code(r"""
    probability_grid_distance = np.linspace(0.5, 5.0, 300)
    probability_grid_standardized = (
        probability_grid_distance - distance_mean
    ) / distance_std
    probability_curve = stable_sigmoid(
        fitted_parameters[0]
        + fitted_parameters[1] * probability_grid_standardized
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(route_distance_km, late_label, color="tab:blue", label="training labels")
    axes[0].plot(probability_grid_distance, probability_curve, color="tab:red", label="probability curve")
    axes[0].set_xlabel("distance (km)")
    axes[0].set_ylabel("predicted late probability")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("A linear score becomes an S-shaped probability")
    axes[0].legend()

    axes[1].plot(logistic_history["step"], logistic_history["loss"], color="tab:green")
    axes[1].set_xlabel("gradient-descent step")
    axes[1].set_ylabel("training cross-entropy")
    axes[1].set_title("Cross-entropy falls during fitting")

    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 9 · Apply a threshold after probability prediction

    The probability model is now frozen. A threshold converts probabilities to
    decisions:

    $$
    \hat y_i=\mathbb 1[p_i\ge t]
    $$

    **Symbols:** $\mathbb 1$ is an indicator that returns 1 when the condition is true
    and 0 otherwise; $t$ is the decision threshold.

    Four decision outcomes are possible:

    | Actual label | Decision | Name |
    | ---: | ---: | --- |
    | 1 | 1 | True positive |
    | 0 | 0 | True negative |
    | 0 | 1 | False positive |
    | 1 | 0 | False negative |

    A threshold of 0.5 is a convention, not a universal law. Lowering it usually
    creates more positive decisions, which can reduce false negatives while increasing
    false positives. The correct trade-off depends on costs and must be chosen using
    development evidence, not the final test.
    """),

    code(r"""
    def count_binary_decisions(labels, probabilities, threshold):
        '''Return decisions and the four raw error counts.'''
        label_array = np.asarray(labels, dtype=int)
        probability_array = np.asarray(probabilities, dtype=float)
        decisions = (probability_array >= threshold).astype(int)
        counts = {
            "true_positive": int(np.sum((label_array == 1) & (decisions == 1))),
            "true_negative": int(np.sum((label_array == 0) & (decisions == 0))),
            "false_positive": int(np.sum((label_array == 0) & (decisions == 1))),
            "false_negative": int(np.sum((label_array == 1) & (decisions == 0))),
        }
        return decisions, counts


    for threshold in [0.30, 0.50, 0.70]:
        decisions, counts = count_binary_decisions(
            late_label,
            fitted_probabilities,
            threshold,
        )
        print(f"threshold={threshold:.2f}")
        print(" decisions:", decisions)
        print(" counts:", counts)

    decisions_at_half, counts_at_half = count_binary_decisions(
        late_label,
        fitted_probabilities,
        threshold=0.50,
    )

    assert sum(counts_at_half.values()) == len(late_label)
    """),

    md(r"""
    ## 10 · Use scikit-learn after the manual implementation

    We use `penalty=None` so the library solves the same unregularized problem as our
    scratch implementation. Later lessons add coefficient penalties deliberately.

    Both implementations receive the same standardized feature values and labels.
    We compare probabilities and cross-entropy—not only thresholded decisions.
    """),

    code(r"""
    from sklearn.linear_model import LogisticRegression

    sklearn_logistic_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10_000,
    )
    sklearn_logistic_model.fit(standardized_distance.reshape(-1, 1), late_label.astype(int))
    sklearn_probabilities = sklearn_logistic_model.predict_proba(
        standardized_distance.reshape(-1, 1)
    )[:, 1]

    sklearn_loss = binary_cross_entropy(late_label, sklearn_probabilities)
    scratch_loss = binary_cross_entropy(late_label, fitted_probabilities)

    print("sklearn probabilities:", sklearn_probabilities.round(3))
    print("scratch probabilities:", fitted_probabilities.round(3))
    print("sklearn cross-entropy:", sklearn_loss)
    print("scratch cross-entropy:", scratch_loss)

    assert np.allclose(sklearn_probabilities, fitted_probabilities, atol=2e-3)
    assert np.isclose(sklearn_loss, scratch_loss, atol=1e-5)
    """),

    md(r"""
    ## 11 · Interpret coefficients without claiming causation

    The model says:

    $$
    \log\left(\frac{p}{1-p}\right)=b+wx
    $$

    Increasing $x$ by one unit adds $w$ to log-odds. Exponentiating gives an odds
    multiplier:

    $$
    \text{odds multiplier}=e^w
    $$

    If $w=0.4$, then $e^{0.4}\approx1.49$: odds are multiplied by about 1.49 for a
    one-unit increase, holding other included features fixed.

    In our scratch model, $x$ is standardized distance. One unit means one training
    standard deviation of kilometres. Raw and standardized coefficients therefore
    have different units.

    Odds multipliers are not probability-point changes. The probability change from
    one unit depends on the starting score because the sigmoid is curved.

    A coefficient describes a fitted association. It does not prove longer routes
    cause lateness; weather, traffic, route type, and loading may be missing.
    """),

    code(r"""
    standardized_odds_multiplier = np.exp(fitted_parameters[1])

    low_score_probability = stable_sigmoid(np.array([-1.0]))[0]
    low_score_plus_one = stable_sigmoid(np.array([-1.0 + 0.4]))[0]
    high_score_probability = stable_sigmoid(np.array([1.0]))[0]
    high_score_plus_one = stable_sigmoid(np.array([1.0 + 0.4]))[0]

    print("fitted standardized-distance coefficient:", fitted_parameters[1])
    print("fitted odds multiplier:", standardized_odds_multiplier)
    print("probability change from score -1 with +0.4:", low_score_plus_one - low_score_probability)
    print("probability change from score +1 with +0.4:", high_score_plus_one - high_score_probability)

    assert standardized_odds_multiplier > 1
    assert not np.isclose(
        low_score_plus_one - low_score_probability,
        high_score_plus_one - high_score_probability,
    )
    """),

    md(r"""
    ## 12 · When logistic regression helps—and when it does not

    ### Use it when

    - the target has two declared classes;
    - a linear boundary is a reasonable baseline;
    - probability output is useful;
    - fast training and inference matter;
    - coefficient direction and units support inspection.

    ### Do not rely on it unchanged when

    - the feature relationship needs strong curves or interactions;
    - labels or prediction time are unclear;
    - classes can be perfectly separated and coefficients grow without a finite
      unregularized optimum;
    - probabilities are used outside the feature range without evidence;
    - repeated entities or time ordering make the split dishonest;
    - decisions require a causal effect rather than a prediction association.

    ### Failure modes

    | Symptom | Likely cause | Safe next step |
    | --- | --- | --- |
    | Loss does not fall | Wrong gradient, poor scale, or unstable rate | Check first update and finite differences |
    | Probabilities are exactly 0 or 1 in logs | Unstable numerical calculation | Use stable sigmoid and clipped/logit-based loss |
    | Coefficients grow continually | Perfect or near-perfect separation | Add later regularization and inspect data |
    | One threshold gives costly errors | Threshold ignores decision costs | Preserve probabilities; tune later on development evidence |
    | Validation is poor but training is good | Overfit, shift, leakage, or nonlinear boundary | Inspect split and error rows before complexity |
    | Coefficient is called causal | Observational association was overinterpreted | Require experimental or causal design |

    ### Topics deliberately deferred

    - MLE-01 turns raw error counts into task-aligned evaluation metrics.
    - MLE-02 explains cross-validation and controlled threshold/model selection.
    - MLE-04 handles imbalanced outcomes.
    - Later regularization material controls coefficient growth.
    - Decision trees provide nonlinear tabular boundaries next.
    """),

    md(r"""
    ## 13 · Mini-project: binary Wine classification with a sealed test

    **Goal:** estimate whether a Wine sample belongs to cultivar class 0 using two
    chemical features. This is educational only; it is not a quality or safety tool.

    **Task frame:**

    | Field | Definition |
    | --- | --- |
    | Prediction unit | One laboratory-tested wine sample |
    | Positive class | Original cultivar target equals 0 |
    | Features | `alcohol` and `color_intensity` |
    | Prediction time | After those measurements, before cultivar confirmation |
    | Identifier | Generated `sample_id`, excluded from features |
    | Evidence | Training fit, validation probability loss, sealed final test |

    The split happens before the baseline, scaler, or model learns any value.
    """),

    code(r"""
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    wine_dataset = load_wine(as_frame=True)
    wine_frame = wine_dataset.frame.copy()
    wine_frame.insert(
        0,
        "sample_id",
        [f"wine_{row_number:03d}" for row_number in range(len(wine_frame))],
    )
    wine_frame["is_class_zero"] = (wine_frame["target"] == 0).astype(int)

    project_features = ["alcohol", "color_intensity"]
    project_X = wine_frame[project_features]
    project_y = wine_frame["is_class_zero"]
    project_ids = wine_frame["sample_id"]

    X_development, X_test, y_development, y_test, id_development, id_test = train_test_split(
        project_X,
        project_y,
        project_ids,
        test_size=0.20,
        random_state=42,
        stratify=project_y,
    )
    X_train, X_validation, y_train, y_validation, id_train, id_validation = train_test_split(
        X_development,
        y_development,
        id_development,
        test_size=0.25,
        random_state=42,
        stratify=y_development,
    )

    print("training rows:", len(X_train))
    print("validation rows:", len(X_validation))
    print("sealed test rows:", len(X_test))

    assert len(X_train) == 106
    assert len(X_validation) == 36
    assert len(X_test) == 36
    assert set(id_train).isdisjoint(id_validation)
    assert set(id_train).isdisjoint(id_test)
    assert set(id_validation).isdisjoint(id_test)
    """),

    code(r"""
    frozen_baseline_probability = float(y_train.mean())
    validation_baseline_probabilities = np.full(
        len(y_validation),
        frozen_baseline_probability,
    )
    validation_baseline_loss = binary_cross_entropy(
        y_validation,
        validation_baseline_probabilities,
    )

    project_pipeline = Pipeline(
        steps=[
            ("standard_scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000),
            ),
        ]
    )
    project_pipeline.fit(X_train, y_train)
    validation_probabilities = project_pipeline.predict_proba(X_validation)[:, 1]
    validation_model_loss = binary_cross_entropy(y_validation, validation_probabilities)

    print("training class-rate baseline:", round(frozen_baseline_probability, 4))
    print("validation baseline cross-entropy:", round(validation_baseline_loss, 4))
    print("validation model cross-entropy:", round(validation_model_loss, 4))

    assert validation_model_loss < validation_baseline_loss
    assert np.all((validation_probabilities > 0) & (validation_probabilities < 1))
    """),

    code(r"""
    validation_decisions, validation_error_counts = count_binary_decisions(
        y_validation,
        validation_probabilities,
        threshold=0.50,
    )

    project_partition_manifest = pd.concat(
        [
            pd.DataFrame({"sample_id": id_train, "partition": "train"}),
            pd.DataFrame({"sample_id": id_validation, "partition": "validation"}),
            pd.DataFrame({"sample_id": id_test, "partition": "test"}),
        ],
        ignore_index=True,
    )

    project_result = {
        "features": project_features,
        "positive_class": "original cultivar target equals 0",
        "baseline_probability": frozen_baseline_probability,
        "validation_baseline_loss": validation_baseline_loss,
        "validation_model_loss": validation_model_loss,
        "validation_error_counts_at_0.5": validation_error_counts,
        "partition_manifest": project_partition_manifest,
        "test_status": "sealed — no transformation, probability, decision, or score calculated",
    }

    print("validation error counts at threshold 0.5:", validation_error_counts)
    print("partition counts:\n", project_partition_manifest["partition"].value_counts())
    print("test status:", project_result["test_status"])

    assert sum(validation_error_counts.values()) == len(y_validation)
    assert project_partition_manifest["sample_id"].is_unique
    assert len(project_partition_manifest) == len(wine_frame)
    assert project_result["test_status"].startswith("sealed")
    """),

    md(r"""
    ## 14 · Practice, solutions, and mastery checkpoint

    ### Worked example

    For score $z=\log(3)$:

    $$
    p=\frac{1}{1+e^{-\log(3)}}=\frac{3}{4}=0.75
    $$

    If $y=1$, cross-entropy is $-\log(0.75)\approx0.288$. At threshold 0.8,
    the class decision is still 0. Probability and decision are separate.

    ### Guided practice

    1. Explain why a categorical 0/1 target is not an ordinary numerical regression target.
    2. Calculate sigmoid probabilities for scores $[-2,0,2]$.
    3. Convert probabilities 0.2 and 0.8 to odds and log-odds.
    4. Calculate one-row cross-entropy for $(y,p)=(1,0.9)$ and $(1,0.1)$.
    5. At $b=0,w=0$, calculate the gradient for rows $(1,0)$ and $(2,1)$.
    6. Apply thresholds 0.3 and 0.7 to probabilities $[0.2,0.4,0.8]$.

    ### Independent practice

    7. Rebuild stable sigmoid and test scores -1,000 and 1,000.
    8. Rebuild binary cross-entropy and prove confidently wrong predictions cost more.
    9. Fit one-feature logistic regression without copying the training loop.
    10. Verify its gradient with finite differences using FND-04.
    11. Compare scratch probabilities with sklearn probabilities.
    12. Create a raw four-count error table at three thresholds and explain the trade-off
        without using precision, recall, or F1 yet.

    ### Challenge

    Rebuild the Wine mini-project without copying. Include:

    - a declared positive class and prediction time;
    - a frozen training class-rate baseline;
    - disjoint training, validation, and sealed test partitions;
    - a training-only scaling Pipeline;
    - validation cross-entropy calculated manually;
    - validation error counts at three thresholds;
    - coefficient units, odds interpretation, and causal caution;
    - a partition manifest and at least eight assertions;
    - no final-test result and no unexplained evaluation metric.

    ### Self-check

    For every output, identify whether it is:

    - a linear score;
    - a probability;
    - a cross-entropy loss;
    - a class decision;
    - an error count;
    - a coefficient association.

    Then name which rows were allowed to determine it.
    """),

    md(r"""
    ### Solution and scoring rubric

    1. The labels name classes; arithmetic distance between class codes has no target meaning.
    2. Approximately $[0.119,0.500,0.881]$.
    3. For 0.2, odds are 0.25 and log-odds about -1.386. For 0.8, odds are 4 and
       log-odds about 1.386.
    4. $-\log(0.9)\approx0.105$ and $-\log(0.1)\approx2.303$.
    5. Intercept gradient 0; coefficient gradient -0.25.
    6. At 0.3 decisions are $[0,1,1]$; at 0.7 they are $[0,0,1]$.
    7. Both extreme scores must return finite probabilities without overflow warnings.
    8. Loss must be finite after clipping and larger for confident mistakes.
    9. A useful learning rate should lower training cross-entropy.
    10. Analytical and numerical gradients should match within a small tolerance.
    11. Scratch and sklearn probabilities should be close on the same scaled rows.
    12. Lower thresholds usually increase positive decisions, false positives, and
        true positives; higher thresholds usually do the reverse.

    Challenge scoring:

    | Skill | Points |
    | --- | ---: |
    | Task frame, labels, and baseline | 3 |
    | Score, sigmoid, and cross-entropy calculations | 4 |
    | Correct gradient and training loop | 4 |
    | Split and scaling boundaries | 3 |
    | Threshold and raw error-count reasoning | 2 |
    | Coefficient interpretation and limitations | 2 |
    | Assertions and sealed final test | 2 |
    | **Total** | **20** |

    ### Common mistakes

    - Calling the linear score a probability.
    - Calling a probability a final decision.
    - Treating 0.5 as a mandatory threshold.
    - Reversing the positive and negative class meaning.
    - Using MSE because labels are stored as numbers.
    - Taking `log(0)` without numerical protection.
    - Reversing $p-y$ to $y-p$ without changing the update sign.
    - Fitting the scaler on validation or test rows.
    - Choosing a threshold from the final test.
    - Interpreting an odds multiplier as a probability-point increase.
    - Treating a coefficient as a causal effect.
    - Reporting only thresholded decisions and discarding useful probabilities.

    ### Readiness threshold

    Score at least **16/20**, including correct sigmoid, cross-entropy, gradient,
    split boundaries, and score/probability/decision distinctions.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    1. Why is logistic regression a classifier?
    2. What is the difference between score, probability, and decision?
    3. Why does sigmoid produce values between 0 and 1?
    4. How are probability, odds, and log-odds related?
    5. Why does Bernoulli likelihood lead to cross-entropy?
    6. Why is a confidently wrong probability penalized strongly?
    7. Why does the gradient contain $p-y$?
    8. What changes when a decision threshold moves?
    9. What does $e^w$ mean, and what does it not mean?
    10. Why can a falling training loss still fail to justify deployment?

    ### Teach it back

    Explain the complete chain without using sklearn:

    **binary label → training-rate baseline → linear score → sigmoid probability →
    Bernoulli cross-entropy → gradient descent → frozen probability → threshold →
    raw error counts.**

    Name which steps learn parameters and which step expresses a decision policy.

    ### Memory aid

    **Logistic regression fits a linear score through sigmoid and cross-entropy;
    a separate threshold turns its probability into a decision.**

    ### Next dependency

    CML-03 introduces decision trees as a nonlinear alternative. Evaluation metrics
    remain deferred until the student understands what classification predictions and
    errors actually are.
    """),
]


build("02_classical_ml/02_logistic_regression.ipynb", cells)
