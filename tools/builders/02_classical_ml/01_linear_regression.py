"""Builder for Lesson CML-01 — Linear Regression."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # CML-01 · Linear Regression

    *Build the first prediction model one calculation at a time*

    | Lesson detail | Value |
    | --- | --- |
    | Prerequisites | FND-01, FND-02, and FND-03 |
    | Estimated study time | 7–9 hours across two or more sessions |
    | Main outcome | Fit, inspect, and explain an ordinary linear-regression model |
    | Next lesson | FND-04 · Optimization and Gradient Descent |

    Linear regression is the first complete learning algorithm in this curriculum.
    The goal is not to memorise `LinearRegression().fit(...)`. The goal is to know
    where every prediction, residual, squared error, slope, and intercept comes from.

    We deliberately stop before gradient descent, Ridge, Lasso, cross-validation,
    and a large catalogue of metrics. Those ideas make more sense after ordinary
    least squares is familiar.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end of this lesson, you will be able to:

    - recognise a regression task with a numerical target;
    - define the prediction unit, feature, target, and prediction time;
    - create a training-mean baseline before fitting a line;
    - explain slope and intercept using their real units;
    - calculate predictions, residuals, squared errors, and mean squared error by hand;
    - explain why residuals are calculated before they are squared;
    - fit a one-feature least-squares line using centred deviations;
    - implement that calculation with small, readable NumPy code;
    - explain why a constant feature cannot determine a slope;
    - connect one-feature regression to a multi-feature weighted sum;
    - use `np.linalg.lstsq` without explicitly inverting a matrix;
    - fit sklearn `LinearRegression` after the manual version;
    - compare a fixed model with its frozen baseline using validation data only;
    - read residual, outlier, interpolation, and extrapolation plots;
    - interpret coefficients as model associations rather than causal effects;
    - complete a small Wine-data project while leaving final test data sealed.

    MSE is the only required score here because it is also the loss the model fits.
    MAE, RMSE, $R^2$, metric selection, and uncertainty belong to MLE-01.
    """),

    md(r"""
    ## 2 · The practical problem

    A local delivery team wants a rough estimate of travel time from route distance.
    We have four completed routes:

    | Route | Distance (km) | Travel time (minutes) |
    | --- | ---: | ---: |
    | A | 1 | 3 |
    | B | 2 | 5 |
    | C | 3 | 6 |
    | D | 4 | 8 |

    Our task frame is:

    | Field | Definition |
    | --- | --- |
    | Decision | Provide a rough planning estimate for a dispatcher |
    | Prediction unit | One delivery route |
    | Feature | Route distance in kilometres |
    | Target | Travel time in minutes |
    | Prediction time | After route distance is known, before the trip starts |
    | Evaluation unit | One future route |

    Distance cannot explain traffic, weather, turns, or loading time. A line will be
    a useful first approximation, not a complete map of reality.

    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #4c78a8; border-radius: 10px; padding: 12px 16px; background: #eef5ff; color: #172b4d; text-align: center;"><strong>Distance</strong><br>known feature</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 12px 16px; background: #fff4e8; color: #4a2b0b; text-align: center;"><strong>Prediction rule</strong><br>intercept + slope × distance</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 12px 16px; background: #eef8ec; color: #173d17; text-align: center;"><strong>Estimated time</strong><br>numerical target</div>
    </div>

    This is a **regression** problem because the target is a numerical quantity for
    which arithmetic differences are meaningful.
    """),

    md(r"""
    ## 3 · Begin with the simplest baseline

    Before using distance, predict the average training travel time for every route.
    This is the **training-mean baseline**.

    $$
    \bar y = \frac{1}{n}\sum_{i=1}^{n} y_i
    $$

    Symbols:

    - $y_i$ is the actual target for route $i$;
    - $n$ is the number of training routes;
    - $\bar y$ is the arithmetic mean of the training targets.

    For our four routes:

    $$
    \bar y = \frac{3+5+6+8}{4} = \frac{22}{4} = 5.5\text{ minutes}
    $$

    The baseline predicts 5.5 minutes for every route. It ignores distance. That is
    exactly why it is useful: a fitted line must earn its added complexity.
    """),

    code(r"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    route_distance_km = np.array([1.0, 2.0, 3.0, 4.0])
    travel_time_minutes = np.array([3.0, 5.0, 6.0, 8.0])

    training_mean_minutes = travel_time_minutes.mean()
    baseline_predictions = np.full_like(travel_time_minutes, training_mean_minutes)

    print("training target mean:", training_mean_minutes, "minutes")
    print("baseline predictions:", baseline_predictions)

    assert training_mean_minutes == 5.5
    """),

    md(r"""
    A baseline must be learned from training targets and then frozen. For future
    validation rows, we reuse the training mean; we never replace it with the
    validation mean.
    """),

    md(r"""
    ## 4 · A line turns a feature into a prediction

    A one-feature linear model is:

    $$
    \hat y = b_0 + b_1x
    $$

    Read it as:

    > Predicted travel time equals the intercept plus the slope multiplied by route
    > distance.

    Symbols and units:

    - $x$ is route distance, measured in kilometres;
    - $\hat y$ is predicted travel time, measured in minutes;
    - $b_0$ is the intercept, measured in minutes;
    - $b_1$ is the slope, measured in minutes per kilometre.

    The hat on $\hat y$ means “estimated.” It distinguishes a prediction from the
    observed target $y$.

    Suppose we try:

    $$
    \hat y = 1.5 + 1.6x
    $$

    The slope says the predicted time rises by 1.6 minutes when distance rises by one
    kilometre, within the range supported by our data. The intercept predicts 1.5
    minutes at zero kilometres. That may represent fixed preparation time, but the
    data contains no zero-kilometre route, so this interpretation is uncertain.
    """),

    code(r"""
    intercept_minutes = 1.5
    slope_minutes_per_km = 1.6

    line_predictions = intercept_minutes + slope_minutes_per_km * route_distance_km

    print("distance (km):", route_distance_km)
    print("predicted time (minutes):", line_predictions)

    assert np.allclose(line_predictions, [3.1, 4.7, 6.3, 7.9])
    """),

    md(r"""
    “Linear” means linear in the learned parameters $b_0$ and $b_1$. The prediction
    changes by the same amount for every additional kilometre. Later, transformed
    features can create curves while remaining linear in their coefficients, but the
    first mental model should be one ordinary line.
    """),

    md(r"""
    ## 5 · Residuals come before squared loss

    A **residual** is the signed difference between the observed target and prediction:

    $$
    r_i = y_i - \hat y_i
    $$

    Symbols:

    - $r_i$ is the residual for route $i$;
    - $y_i$ is its actual travel time;
    - $\hat y_i$ is its predicted travel time.

    A positive residual means the actual trip took longer than predicted. A negative
    residual means it took less time than predicted. Residuals keep the target unit:
    minutes.

    For the proposed line:

    | Distance | Actual | Predicted | Residual: actual − predicted | Squared residual |
    | ---: | ---: | ---: | ---: | ---: |
    | 1 | 3.0 | 3.1 | −0.1 | 0.01 |
    | 2 | 5.0 | 4.7 | 0.3 | 0.09 |
    | 3 | 6.0 | 6.3 | −0.3 | 0.09 |
    | 4 | 8.0 | 7.9 | 0.1 | 0.01 |

    If we simply add residuals, positive and negative misses can cancel. Squaring
    makes every contribution non-negative and makes large misses matter more.
    """),

    code(r"""
    residuals_minutes = travel_time_minutes - line_predictions
    squared_residuals = residuals_minutes ** 2

    calculation_table = pd.DataFrame(
        {
            "distance_km": route_distance_km,
            "actual_minutes": travel_time_minutes,
            "predicted_minutes": line_predictions,
            "residual_minutes": residuals_minutes,
            "squared_residual_minutes2": squared_residuals,
        }
    )

    print(calculation_table)
    print("residual sum:", residuals_minutes.sum())
    print("squared residual sum:", squared_residuals.sum())

    assert np.allclose(residuals_minutes, [-0.1, 0.3, -0.3, 0.1])
    assert np.isclose(squared_residuals.sum(), 0.2)
    """),

    md(r"""
    ### 5.1 Mean squared error

    Mean squared error, abbreviated MSE, averages the squared residuals:

    $$
    \operatorname{MSE}
    = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
    $$

    For the proposed line:

    $$
    \operatorname{MSE}
    = \frac{0.01+0.09+0.09+0.01}{4}
    = \frac{0.20}{4}
    = 0.05\text{ minutes}^2
    $$

    MSE has squared target units. Here, that is squared minutes. This makes its raw
    value less intuitive, but squaring gives us a smooth objective and strongly
    penalises large misses.

    A **loss function** is the quantity the learning algorithm tries to make small.
    Ordinary least squares uses the sum or mean of squared residuals. Both choose the
    same line because dividing every candidate's sum by the same positive number does
    not change which candidate is smallest.

    The formula can play two different roles:

    | Role | Rows used | Purpose |
    | --- | --- | --- |
    | Training loss | Training rows | Choose slope and intercept |
    | Validation metric | Validation rows after fitting | Judge the frozen model |

    Reusing one formula does not make the roles interchangeable. Validation MSE must
    never be used to refit the same model parameters.
    """),

    code(r"""
    def mean_squared_loss(actual_values, predicted_values):
        '''Return the average squared residual.'''
        actual_array = np.asarray(actual_values, dtype=float)
        predicted_array = np.asarray(predicted_values, dtype=float)
        if actual_array.shape != predicted_array.shape:
            raise ValueError("Actual and predicted values must have the same shape")
        return float(np.mean((actual_array - predicted_array) ** 2))


    baseline_training_mse = mean_squared_loss(travel_time_minutes, baseline_predictions)
    line_training_mse = mean_squared_loss(travel_time_minutes, line_predictions)

    print("baseline training MSE:", baseline_training_mse)
    print("line training MSE:", line_training_mse)

    assert np.isclose(baseline_training_mse, 3.25)
    assert np.isclose(line_training_mse, 0.05)
    """),

    md(r"""
    The line fits these four training routes much better than the constant baseline.
    That does not prove it will predict new routes well. Generalisation requires
    validation rows that did not determine the line.
    """),

    md(r"""
    ## 6 · Fit the least-squares line by hand

    So far, we tried a line whose parameters were already given. Learning means
    choosing the slope and intercept from the training data.

    ### 6.1 Find the slope

    The least-squares slope is:

    $$
    b_1 =
    \frac{\sum_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)}
         {\sum_{i=1}^{n}(x_i-\bar x)^2}
    $$

    Read the numerator as “how distance and time move together.” Read the denominator
    as “how much distance varies by itself.”

    Symbols:

    - $x_i$ and $y_i$ are the feature and target for route $i$;
    - $\bar x$ and $\bar y$ are their training means;
    - $b_1$ is the fitted slope.

    Our means are:

    $$
    \bar x = 2.5\text{ km}
    \qquad
    \bar y = 5.5\text{ minutes}
    $$

    The centred calculation is:

    | $x_i-\bar x$ | $y_i-\bar y$ | Product | Squared feature deviation |
    | ---: | ---: | ---: | ---: |
    | −1.5 | −2.5 | 3.75 | 2.25 |
    | −0.5 | −0.5 | 0.25 | 0.25 |
    | 0.5 | 0.5 | 0.25 | 0.25 |
    | 1.5 | 2.5 | 3.75 | 2.25 |

    Therefore:

    $$
    b_1 = \frac{3.75+0.25+0.25+3.75}{2.25+0.25+0.25+2.25}
        = \frac{8}{5}
        = 1.6\text{ minutes per km}
    $$

    ### 6.2 Find the intercept

    The fitted line passes through the point of training means:

    $$
    b_0 = \bar y - b_1\bar x
    $$

    $$
    b_0 = 5.5 - (1.6)(2.5) = 1.5\text{ minutes}
    $$

    We have now derived the line used earlier rather than guessing it.
    """),

    code(r"""
    distance_mean = route_distance_km.mean()
    time_mean = travel_time_minutes.mean()

    centered_distance = route_distance_km - distance_mean
    centered_time = travel_time_minutes - time_mean

    slope_numerator = np.sum(centered_distance * centered_time)
    slope_denominator = np.sum(centered_distance ** 2)
    fitted_slope = slope_numerator / slope_denominator
    fitted_intercept = time_mean - fitted_slope * distance_mean

    print("slope numerator:", slope_numerator)
    print("slope denominator:", slope_denominator)
    print("fitted slope:", fitted_slope, "minutes per km")
    print("fitted intercept:", fitted_intercept, "minutes")

    assert np.isclose(fitted_slope, 1.6)
    assert np.isclose(fitted_intercept, 1.5)
    """),

    md(r"""
    If every $x$ value is identical, the denominator is zero. Distance did not vary,
    so the data cannot tell us how time changes with distance. This is a data problem,
    not a division trick to work around.
    """),

    code(r"""
    def fit_simple_linear_regression(feature_values, target_values):
        '''Fit one slope and intercept by ordinary least squares.'''
        feature_array = np.asarray(feature_values, dtype=float)
        target_array = np.asarray(target_values, dtype=float)

        if feature_array.ndim != 1 or target_array.ndim != 1:
            raise ValueError("Simple regression expects one-dimensional inputs")
        if feature_array.shape != target_array.shape:
            raise ValueError("Feature and target must contain the same number of rows")
        if len(feature_array) < 2:
            raise ValueError("At least two training rows are required")

        feature_deviation = feature_array - feature_array.mean()
        denominator = np.sum(feature_deviation ** 2)
        if np.isclose(denominator, 0):
            raise ValueError("Cannot fit a slope because the feature is constant")

        target_deviation = target_array - target_array.mean()
        slope = np.sum(feature_deviation * target_deviation) / denominator
        intercept = target_array.mean() - slope * feature_array.mean()
        return float(intercept), float(slope)


    def predict_simple_linear_regression(feature_values, intercept, slope):
        '''Apply a fitted one-feature line without changing its parameters.'''
        feature_array = np.asarray(feature_values, dtype=float)
        return intercept + slope * feature_array


    manual_intercept, manual_slope = fit_simple_linear_regression(
        route_distance_km,
        travel_time_minutes,
    )
    manual_predictions = predict_simple_linear_regression(
        route_distance_km,
        manual_intercept,
        manual_slope,
    )

    print("manual intercept:", manual_intercept)
    print("manual slope:", manual_slope)
    print("manual predictions:", manual_predictions)

    assert np.isclose(manual_intercept, 1.5)
    assert np.isclose(manual_slope, 1.6)
    assert np.allclose(manual_predictions, line_predictions)
    """),

    code(r"""
    try:
        fit_simple_linear_regression([2, 2, 2], [3, 4, 5])
        raise AssertionError("Expected a constant feature to raise ValueError")
    except ValueError as error:
        print(type(error).__name__ + ":", error)
        assert "constant" in str(error)
    """),

    md(r"""
    ## 7 · The loss landscape prepares us for optimization

    Every possible slope and intercept defines a candidate line and therefore an MSE.
    For any chosen slope in this one-feature example, the best matching intercept is:

    $$
    b_0 = \bar y - b_1\bar x
    $$

    We can evaluate many candidate slopes and draw their losses. This is a visual
    search only. FND-04 will explain how gradient descent uses slope information from
    the loss itself to move efficiently through much larger parameter spaces.
    """),

    code(r"""
    candidate_slopes = np.linspace(-0.5, 3.5, 161)
    candidate_losses = []

    for candidate_slope in candidate_slopes:
        candidate_intercept = time_mean - candidate_slope * distance_mean
        candidate_prediction = candidate_intercept + candidate_slope * route_distance_km
        candidate_losses.append(
            mean_squared_loss(travel_time_minutes, candidate_prediction)
        )

    best_grid_position = int(np.argmin(candidate_losses))

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(candidate_slopes, candidate_losses, color="tab:blue")
    axis.scatter(
        [candidate_slopes[best_grid_position]],
        [candidate_losses[best_grid_position]],
        color="tab:red",
        zorder=3,
        label="smallest checked loss",
    )
    axis.axvline(manual_slope, color="tab:green", linestyle="--", label="fitted slope = 1.6")
    axis.set_title("Training MSE for different candidate slopes")
    axis.set_xlabel("slope (minutes per km)")
    axis.set_ylabel("training MSE (minutes²)")
    axis.legend()
    figure.tight_layout()
    plt.show()

    print("best slope on this grid:", candidate_slopes[best_grid_position])
    """),

    md(r"""
    The curve has one clear bottom near 1.6. Ordinary least squares finds that bottom
    directly for this problem. FND-04 will begin from this concrete loss and explain
    iterative optimization without requiring us to guess many values.
    """),

    md(r"""
    ## 8 · From one feature to several features

    With several features, the line becomes a flat surface called a hyperplane. The
    prediction remains a weighted sum:

    $$
    \hat y_i = b + w_1x_{i1} + w_2x_{i2} + \cdots + w_dx_{id}
    $$

    - $i$ identifies one row;
    - $j$ identifies one feature;
    - $x_{ij}$ is feature $j$ for row $i$;
    - $w_j$ is that feature's fitted coefficient;
    - $d$ is the number of features;
    - $b$ is the intercept.

    In matrix form:

    $$
    \hat{\mathbf y} = X\mathbf w + b
    $$

    If $X$ has shape $(n,d)$, then $\mathbf w$ has shape $(d,)$, and the
    prediction vector has shape $(n,)$.

    We can add a column of ones to $X$ and include the intercept inside the weight
    vector. `np.linalg.lstsq` then finds weights that minimise squared residuals using
    stable numerical methods. We do not calculate an explicit matrix inverse.
    """),

    code(r"""
    delivery_features = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [5.0, 1.0],
        ]
    )
    delivery_targets = np.array([3.0, 6.0, 6.0, 9.0, 10.0])

    # Columns: distance_km and rain_indicator.
    design_matrix = np.column_stack(
        [np.ones(len(delivery_features)), delivery_features]
    )
    fitted_weights, residual_sum, matrix_rank, singular_values = np.linalg.lstsq(
        design_matrix,
        delivery_targets,
        rcond=None,
    )
    matrix_predictions = design_matrix @ fitted_weights

    print("design shape:", design_matrix.shape)
    print("weights [intercept, distance, rain]:", fitted_weights.round(3))
    print("predictions:", matrix_predictions.round(3))
    print("matrix rank:", matrix_rank)

    assert design_matrix.shape == (5, 3)
    assert fitted_weights.shape == (3,)
    assert matrix_predictions.shape == (5,)
    """),

    md(r"""
    The distance coefficient has units of minutes per kilometre. The rain-indicator
    coefficient is the predicted difference in minutes between otherwise identical
    rows whose rain value changes from 0 to 1.

    “Otherwise identical” is a model statement. With observational data, a coefficient
    is an adjusted association, not proof that rain or distance caused the outcome.

    ### Optional mathematical bridge

    FND-01 showed projections. Least squares projects the target vector onto the space
    of predictions the feature columns can express. At the solution, the residual
    vector is perpendicular to every design-matrix column:

    $$
    X^\top(\mathbf y-X\mathbf w)=\mathbf 0
    $$

    These are the normal equations. We use `lstsq` rather than forming
    $(X^\top X)^{-1}$, which can make numerical instability worse. The projection
    view is helpful for deeper study, but the manual slope calculation remains the
    required core.
    """),

    md(r"""
    ## 9 · Use scikit-learn after the manual implementation

    The library follows the same sequence:

    1. construct a model object;
    2. call `fit` on training features and targets;
    3. inspect the learned intercept and coefficient;
    4. call `predict` on new feature rows;
    5. calculate declared evidence without refitting.

    First verify sklearn against our hand-fitted delivery line.
    """),

    code(r"""
    from sklearn.linear_model import LinearRegression

    sklearn_delivery_model = LinearRegression()
    sklearn_delivery_model.fit(route_distance_km.reshape(-1, 1), travel_time_minutes)
    sklearn_delivery_predictions = sklearn_delivery_model.predict(
        route_distance_km.reshape(-1, 1)
    )

    print("sklearn intercept:", sklearn_delivery_model.intercept_)
    print("sklearn slope:", sklearn_delivery_model.coef_[0])
    print("sklearn predictions:", sklearn_delivery_predictions)

    assert np.isclose(sklearn_delivery_model.intercept_, manual_intercept)
    assert np.isclose(sklearn_delivery_model.coef_[0], manual_slope)
    assert np.allclose(sklearn_delivery_predictions, manual_predictions)
    """),

    md(r"""
    `reshape(-1, 1)` turns four scalar feature values into a two-dimensional table
    with four rows and one feature column. sklearn expects feature tables to have
    shape `(rows, features)` even when only one feature exists.

    The library did not discover a different algorithm. It reproduced the same least-
    squares calculation through a tested interface.
    """),

    md(r"""
    ## 10 · Diagnose what the line can and cannot express

    ### 10.1 Residual plot

    A residual plot places predictions on the horizontal axis and signed residuals on
    the vertical axis. A useful linear model usually leaves residuals scattered around
    zero without a strong curve.
    """),

    code(r"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(route_distance_km, travel_time_minutes, color="tab:blue", label="observed")
    axes[0].plot(route_distance_km, manual_predictions, color="tab:red", label="fitted line")
    for distance, actual, predicted in zip(
        route_distance_km,
        travel_time_minutes,
        manual_predictions,
    ):
        axes[0].plot([distance, distance], [actual, predicted], color="gray", alpha=0.7)
    axes[0].set_title("Observed routes, fitted line, and residuals")
    axes[0].set_xlabel("distance (km)")
    axes[0].set_ylabel("travel time (minutes)")
    axes[0].legend()

    axes[1].scatter(manual_predictions, residuals_minutes, color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Residuals around zero")
    axes[1].set_xlabel("predicted travel time (minutes)")
    axes[1].set_ylabel("residual: actual − predicted (minutes)")

    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    Four rows are not enough for a serious diagnostic. The figure teaches how to read
    the objects, not how to certify a production model.

    ### 10.2 Interpolation and extrapolation

    - **Interpolation** predicts inside the observed feature range.
    - **Extrapolation** predicts outside it.

    A fitted line continues forever, but the real relationship may not.
    """),

    code(r"""
    plotted_distances = np.linspace(0, 8, 200)
    plotted_predictions = predict_simple_linear_regression(
        plotted_distances,
        manual_intercept,
        manual_slope,
    )

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(plotted_distances, plotted_predictions, color="tab:red", label="linear prediction")
    axis.scatter(route_distance_km, travel_time_minutes, color="tab:blue", label="training routes")
    axis.axvspan(1, 4, color="tab:green", alpha=0.12, label="observed distance range")
    axis.axvspan(4, 8, color="tab:orange", alpha=0.10, label="extrapolation region")
    axis.set_title("The line continues beyond the evidence")
    axis.set_xlabel("distance (km)")
    axis.set_ylabel("predicted travel time (minutes)")
    axis.legend()
    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    A prediction at 3.5 km is interpolation. A prediction at 8 km is extrapolation.
    The arithmetic works in both cases, but only the first lies inside the observed
    distance range.

    ### 10.3 Squared loss is sensitive to outliers

    One extreme target creates a very large squared residual and can pull the line
    toward itself. Investigate unusual rows before deleting them; they may be errors,
    unit problems, or rare valid cases.
    """),

    code(r"""
    time_with_outlier = travel_time_minutes.copy()
    time_with_outlier[-1] = 30.0

    outlier_intercept, outlier_slope = fit_simple_linear_regression(
        route_distance_km,
        time_with_outlier,
    )

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(route_distance_km, time_with_outlier, color="tab:blue", label="data with outlier")
    axis.plot(
        route_distance_km,
        manual_intercept + manual_slope * route_distance_km,
        color="tab:green",
        label=f"original slope {manual_slope:.1f}",
    )
    axis.plot(
        route_distance_km,
        outlier_intercept + outlier_slope * route_distance_km,
        color="tab:red",
        linestyle="--",
        label=f"outlier slope {outlier_slope:.1f}",
    )
    axis.set_title("One extreme target pulls the least-squares line")
    axis.set_xlabel("distance (km)")
    axis.set_ylabel("travel time (minutes)")
    axis.legend()
    figure.tight_layout()
    plt.show()

    print("original slope:", manual_slope)
    print("slope after extreme target:", outlier_slope)
    """),

    md(r"""
    ## 11 · When to use linear regression—and when not to

    ### Use it when

    - the target is numerical;
    - a straight or additive relationship is a reasonable first approximation;
    - a transparent, fast baseline is valuable;
    - data is limited and a highly flexible model is not justified;
    - coefficient direction and units help people inspect the model.

    ### Do not rely on it unchanged when

    - the target is a category rather than a numerical quantity;
    - the relationship contains strong curves or interactions the features do not
      represent;
    - a few uninvestigated extreme rows dominate squared loss;
    - future inputs lie far outside the training range;
    - repeated entities or time dependence make an ordinary random split dishonest;
    - the decision requires a causal effect rather than a prediction association.

    ### Assumptions at beginner depth

    | Question | What to inspect | If it fails |
    | --- | --- | --- |
    | Is the target numerical? | Target meaning and dtype | Use a task suited to the target |
    | Is an additive line plausible? | Scatter and residual plots | Add justified features or later use a nonlinear model |
    | Are rows evaluated independently? | Entity and time keys | Use group- or time-based boundaries |
    | Are extreme rows understood? | Source records and units | Repair documented errors; consider robust methods later |
    | Are prediction inputs in range? | Training-reference ranges | Flag extrapolation or gather evidence |
    | Are coefficients being called causal? | Data-collection design | Use experiments or causal methods for causal claims |

    Detailed regression inference—standard errors, confidence intervals,
    homoscedasticity tests, and Gauss–Markov theory—is an advanced statistics path.
    It is not required to understand how the prediction line is fitted.

    ### Related methods deferred deliberately

    - FND-04: gradient descent on this squared-loss objective;
    - MLE-01: MAE, RMSE, $R^2$, and task-aligned metric selection;
    - MLE-02: cross-validation and model-selection discipline;
    - later regression extension: Ridge, Lasso, Elastic Net, and robust regression;
    - CML-03 and beyond: nonlinear tree-based alternatives.
    """),

    md(r"""
    ## 12 · Advanced intuition after the core is secure

    This section is a bridge, not part of the required first pass.

    ### 12.1 Why Gaussian noise leads to squared loss

    FND-02 introduced probability distributions. Write the data story as:

    $$
    y_i=\hat y_i+\varepsilon_i,
    \qquad
    \varepsilon_i\sim\mathcal N(0,\sigma^2)
    $$

    **Symbols:** $y_i$ is an observed target; $\hat y_i$ is the model prediction;
    $\varepsilon_i$ is unexplained error; $\mathcal N(0,\sigma^2)$ is a normal
    distribution centred at zero with fixed variance $\sigma^2$.

    Under independent errors, the negative log-likelihood is:

    $$
    -\log L
    =\frac{n}{2}\log(2\pi\sigma^2)
    +\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i-\hat y_i)^2
    $$

    **Read it in two pieces:** the first term is constant when $n$ and $\sigma^2$
    are fixed. The second is a positive constant multiplied by the sum of squared
    residuals. Therefore, the coefficients that minimise negative log-likelihood are
    the same coefficients that minimise squared residuals.

    With residuals $[-1,2]$ and fixed variance $\sigma^2=1$:

    $$
    \sum_i r_i^2=(-1)^2+2^2=5
    $$

    A candidate with residuals $[-1,1]$ has squared sum 2 and therefore a smaller
    Gaussian negative log-likelihood under the same fixed variance.

    This does not make Gaussian noise universally true. It explains why squared loss
    is principled under one declared data-generating assumption. Outliers and changing
    residual spread are reasons to question that assumption.

    ### 12.2 Why the matrix solution is a projection

    The feature columns define all predictions a linear model can express. Least
    squares chooses the expressible prediction vector closest to the observed target
    vector. The remaining residual is perpendicular to every feature column.

    ### 12.3 What regularization will add later

    Ordinary least squares only minimises prediction residuals. Ridge and Lasso add a
    penalty on coefficients. That introduces a new choice—the penalty strength—which
    must be selected without using the final test. It therefore belongs after both
    optimization and validation foundations.

    Deep mastery means seeing how these ideas connect while keeping the learning order
    honest: ordinary line first, then optimization, then controlled extensions.
    """),

    md(r"""
    ## 13 · Mini-project: predict a Wine measurement without touching final test

    We reuse the Wine dataset from FND-03, but change the task:

    | Field | Definition |
    | --- | --- |
    | Decision | Estimate `proline` for an educational laboratory review |
    | Prediction unit | One wine sample |
    | Target | `proline`, a numerical chemical measurement |
    | Features | The other 12 chemical measurements |
    | Excluded columns | Cultivar target and generated sample ID |
    | Prediction time | After the 12 feature measurements, before `proline` is read |
    | Split | Training, validation, and sealed final test |

    The bundled metadata does not document all physical units, so coefficient
    interpretation remains limited. We will not invent units.
    """),

    code(r"""
    from sklearn.datasets import load_wine
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    wine_dataset = load_wine(as_frame=True)
    wine_frame = wine_dataset.frame.rename(
        columns={"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"}
    )
    wine_frame.insert(
        0,
        "sample_id",
        [f"wine_{row_number:03d}" for row_number in range(len(wine_frame))],
    )

    project_target_column = "proline"
    project_excluded_columns = {"sample_id", "target", project_target_column}
    project_feature_columns = [
        column for column in wine_frame.columns
        if column not in project_excluded_columns
    ]

    project_X = wine_frame[project_feature_columns]
    project_y = wine_frame[project_target_column]
    project_ids = wine_frame["sample_id"]

    X_development, X_test, y_development, y_test, id_development, id_test = train_test_split(
        project_X,
        project_y,
        project_ids,
        test_size=0.20,
        random_state=42,
    )
    X_train, X_validation, y_train, y_validation, id_train, id_validation = train_test_split(
        X_development,
        y_development,
        id_development,
        test_size=0.25,
        random_state=42,
    )

    print("training rows:", len(X_train))
    print("validation rows:", len(X_validation))
    print("sealed test rows:", len(X_test))
    print("feature count:", len(project_feature_columns))

    assert len(X_train) == 106
    assert len(X_validation) == 36
    assert len(X_test) == 36
    assert set(id_train).isdisjoint(id_validation)
    assert set(id_train).isdisjoint(id_test)
    assert set(id_validation).isdisjoint(id_test)
    """),

    md(r"""
    We create the split before learning the baseline, imputation values, or regression
    coefficients. The final test partition is now sealed and will not be transformed,
    predicted, or scored in this lesson.

    The baseline learns only the training-target mean.
    """),

    code(r"""
    frozen_baseline_proline = float(y_train.mean())
    validation_baseline_predictions = np.full(
        len(y_validation),
        frozen_baseline_proline,
    )
    validation_baseline_mse = mean_squared_loss(
        y_validation,
        validation_baseline_predictions,
    )

    print("training-mean baseline:", round(frozen_baseline_proline, 3))
    print("validation baseline MSE:", round(validation_baseline_mse, 3))
    """),

    md(r"""
    The Pipeline includes median imputation to make the input contract explicit. The
    current dataset has no missing values, so the imputer changes nothing. We do not
    add a scaler: ordinary unregularized least squares does not require equal feature
    scales to define its predictions. Scaling becomes important when coefficient
    penalties or numerical conditioning require it.
    """),

    code(r"""
    project_pipeline = Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
            ("linear_regression", LinearRegression()),
        ]
    )

    project_pipeline.fit(X_train, y_train)
    validation_predictions = project_pipeline.predict(X_validation)
    validation_linear_mse = mean_squared_loss(y_validation, validation_predictions)

    print("validation baseline MSE:", round(validation_baseline_mse, 3))
    print("validation linear-regression MSE:", round(validation_linear_mse, 3))
    print("MSE reduction:", round(validation_baseline_mse - validation_linear_mse, 3))

    assert len(validation_predictions) == len(y_validation)
    assert validation_linear_mse < validation_baseline_mse
    """),

    md(r"""
    The fixed linear model has a smaller validation MSE than the frozen mean baseline
    for this split. Unlike the training MSE comparison, these rows did not determine
    either model. This provides evidence about generalisation to similar unseen rows.
    It is still not a final performance claim, and we made no model choice from the
    final test.
    """),

    code(r"""
    validation_residuals = y_validation.to_numpy() - validation_predictions
    project_regression = project_pipeline.named_steps["linear_regression"]
    coefficient_table = pd.DataFrame(
        {
            "feature": project_feature_columns,
            "coefficient": project_regression.coef_,
        }
    ).sort_values("coefficient")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(validation_predictions, validation_residuals, color="tab:blue")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Validation residuals — fixed linear model")
    axes[0].set_xlabel("predicted proline")
    axes[0].set_ylabel("actual − predicted proline")

    axes[1].barh(coefficient_table["feature"], coefficient_table["coefficient"], color="tab:purple")
    axes[1].set_title("Fitted coefficients — associations, not causes")
    axes[1].set_xlabel("coefficient in undocumented source units")

    figure.tight_layout()
    plt.show()

    print("validation residual mean:", round(float(validation_residuals.mean()), 3))
    """),

    md(r"""
    Coefficient signs describe the fitted association after accounting for the other
    included features. Different feature units make raw coefficient magnitudes
    incomparable as “importance.” Correlated features can also share or exchange
    coefficient weight.

    The validation residual plot can reveal curvature, changing spread, and extreme
    misses. It cannot prove future stability or causation.
    """),

    code(r"""
    project_partition_manifest = pd.concat(
        [
            pd.DataFrame({"sample_id": id_train, "partition": "train"}),
            pd.DataFrame({"sample_id": id_validation, "partition": "validation"}),
            pd.DataFrame({"sample_id": id_test, "partition": "test"}),
        ],
        ignore_index=True,
    )

    project_result = {
        "target": project_target_column,
        "features": project_feature_columns,
        "training_mean_baseline": frozen_baseline_proline,
        "validation_baseline_mse": validation_baseline_mse,
        "validation_linear_mse": validation_linear_mse,
        "pipeline": project_pipeline,
        "partition_manifest": project_partition_manifest,
        "test_status": "sealed — not transformed, predicted, or scored",
    }

    print("project target:", project_result["target"])
    print("partition counts:\n", project_partition_manifest["partition"].value_counts())
    print("test status:", project_result["test_status"])

    assert project_partition_manifest["sample_id"].is_unique
    assert len(project_partition_manifest) == len(wine_frame)
    assert project_result["test_status"].startswith("sealed")

    print("\nLinear-regression mini-project checks passed.")
    """),

    md(r"""
    ## 14 · Exercises, self-check, and solutions

    **Estimated practice time:** 2–3 hours.

    ### Worked example

    For the rule $\hat y=2+3x$ and one row $x=4, y=15$:

    - prediction: $2+3(4)=14$;
    - residual: $15-14=1$;
    - squared residual: $1^2=1$.

    The slope has target units per feature unit. The residual has target units. The
    squared residual has squared target units.

    ### Guided practice

    1. For `x=[1,2,3]`, `y=[2,4,7]`, calculate the target mean and baseline MSE.
    2. Using the candidate line $\hat y=-0.5+2.5x$, calculate every prediction,
       residual, squared residual, and MSE.
    3. Calculate the least-squares slope numerator and denominator for the same data.
    4. Calculate the intercept and prove the fitted line passes through
       $(\bar x,\bar y)$.
    5. Fit the line with `fit_simple_linear_regression` and compare your hand values.

    ### Independent practice

    6. Create eight rows of advertising spend and sales. Keep units visible, fit a
       training-mean baseline and a one-feature line, then compare validation MSE.
    7. Add a second feature and explain every design-matrix shape and coefficient unit.
    8. Draw a residual plot and identify one visible pattern the line cannot represent.
    9. Make predictions just inside and far outside the training feature range. Label
       interpolation and extrapolation and explain why arithmetic certainty is not
       evidence certainty.

    ### Challenge

    Rebuild the Wine `proline` project without copying its code. Include:

    - a written task frame and excluded-column explanation;
    - disjoint training, validation, and sealed final-test partitions;
    - a training-mean baseline;
    - a Pipeline containing imputation and `LinearRegression`;
    - validation MSE calculated with your own function;
    - a residual table and plot;
    - coefficient units and association caveats;
    - a partition manifest and at least six assertions;
    - no Ridge, Lasso, gradient descent, cross-validation, $R^2$, or test score.

    ### Self-check before reading solutions

    For every number, ask:

    - Which rows taught it?
    - What unit does it have?
    - Is it an actual value, prediction, residual, squared residual, or average loss?
    - Is the feature inside the observed range?
    - Am I making a prediction claim or an unsupported causal claim?
    """),

    md(r"""
    ### Solution and scoring rubric

    1. The target mean is $13/3\approx4.333$. Calculate the baseline by subtracting
       this training mean from every target, squaring, and averaging.
    2. Predictions are `[2, 4.5, 7]`; residuals are `[0, -0.5, 0]`; squared residuals
       are `[0, 0.25, 0]`; MSE is $0.25/3\approx0.0833$.
    3. Centre both arrays first. The slope numerator is 5 and denominator is 2, so
       the slope is 2.5.
    4. The intercept is $\bar y-b_1\bar x=4.333-2.5(2)\approx-0.667$. Note that the
       fitted intercept differs from the candidate line in Question 2.
    5. The function should return intercept about −0.667 and slope 2.5. Its fitted
       MSE is smaller than the candidate line's MSE.
    6. Freeze the mean and line after fitting on training rows. Validation values must
       not change either fit.
    7. For shape `(n,2)`, two coefficients produce one prediction per row. State each
       coefficient as target units per corresponding feature unit.
    8. A curve or trend means the additive straight-line representation is incomplete.
    9. Extrapolated arithmetic can be precise while the relationship is unsupported.

    Challenge scoring:

    | Skill | Points |
    | --- | ---: |
    | Correct task frame and feature availability | 2 |
    | Manual baseline, predictions, residuals, and MSE | 3 |
    | Manual slope and intercept calculation | 3 |
    | Leak-free partitions and Pipeline | 3 |
    | Honest validation comparison | 3 |
    | Residual and extrapolation diagnosis | 2 |
    | Coefficient units and causal caution | 2 |
    | Assertions and sealed final test | 2 |

    Maximum: 20 points.

    **Common mistakes:** fitting before defining the target, learning the baseline from
    validation, reversing the residual sign midway, forgetting to square residuals,
    reporting MSE in target units rather than squared units, using a constant feature,
    treating an identifier as a feature, comparing raw coefficient magnitudes across
    different units, selecting choices from final-test results, and interpreting a
    coefficient as causal.

    **Readiness threshold:** 16/20, including correct hand calculations, a baseline,
    split-before-fit code, a validation-only comparison, and a sealed final test.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    You are ready for FND-04 when you can, without copying this notebook:

    - distinguish a numerical regression target from a categorical target;
    - define the prediction unit, feature, target, and prediction time;
    - calculate and freeze a training-mean baseline;
    - explain slope and intercept with units;
    - calculate predictions, residuals, squared residuals, and MSE manually;
    - derive the one-feature least-squares slope and intercept from centred values;
    - explain why a constant feature cannot determine a slope;
    - implement simple regression and reproduce it with sklearn;
    - trace shapes through a multi-feature weighted sum;
    - compare a fixed model with its baseline using validation data only;
    - diagnose residual structure, outlier sensitivity, and extrapolation;
    - explain why coefficients are associations rather than causal effects;
    - complete the mini-project with at least 16/20 points.

    ### Teach it back

    Starting from the four delivery routes, explain the entire chain:

    **training mean → baseline → line → prediction → residual → squared residual →
    MSE → fitted slope and intercept → validation comparison.**

    Explain the unit of every quantity and name exactly which rows are allowed to
    determine it.

    ### Memory aid

    **Linear regression fits a weighted sum by choosing coefficients that make the
    training squared residuals as small as possible.**

    FND-04 will use this exact squared-loss objective to explain gradient descent.
    """),
]


build("02_classical_ml/01_linear_regression.ipynb", cells)
