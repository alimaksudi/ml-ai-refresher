"""Build FND-04: Optimization and Gradient Descent."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # FND-04 · Optimization and Gradient Descent

    **Prerequisites:** FND-01, FND-02, and CML-01  
    **Estimated study time:** 8–10 hours, including practice  
    **Next lesson:** CML-02 · Logistic Regression

    CML-01 gave us a concrete objective: choose coefficients that make mean squared
    error small. Its closed-form calculation worked for ordinary linear regression.
    Many later models do not have a convenient direct solution.

    This lesson builds the alternative one step at a time: measure the local slope of
    loss, move parameters in the downhill direction, check the new loss, and repeat.

    ### Scope boundary

    The required core is plain gradient descent. We deliberately defer:

    - momentum, RMSProp, Adam, AdamW, schedules, and gradient clipping to DL-04;
    - logistic-loss gradients to CML-02;
    - backpropagation through networks to DL-03;
    - Hessians, eigenvalue convergence rates, and formal condition numbers to an
      advanced numerical-optimization path.

    Those topics are important, but they become understandable only after the basic
    update rule is no longer mysterious.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - distinguish a model, parameter, prediction, residual, loss, and optimizer;
    - calculate a derivative and interpret its sign and unit;
    - explain a partial derivative and gradient vector;
    - derive the gradient-descent update from a local slope;
    - perform several parameter updates manually;
    - implement scalar gradient descent without a library;
    - derive intercept and slope gradients for linear-regression MSE;
    - fit the CML-01 delivery line with gradient descent;
    - explain learning rates that are too small, useful, or too large;
    - explain why feature scaling can make one learning rate easier to use;
    - distinguish batch, stochastic, and mini-batch updates;
    - explain convex and non-convex loss landscapes at beginner depth;
    - stop using declared training and validation evidence;
    - check a hand-derived gradient using finite differences;
    - compare the manual implementation with `SGDRegressor`.

    ### The complete loop

    ```mermaid
    flowchart LR
        A[Choose starting parameters] --> B[Make training predictions]
        B --> C[Calculate training loss]
        C --> D[Calculate gradient]
        D --> E[Update parameters]
        E --> F{Stop rule met?}
        F -- No --> B
        F -- Yes --> G[Freeze and evaluate]
    ```

    The optimizer can only minimise the objective it receives. A falling loss does
    not prove the target, features, split, or objective represents the real problem.
    """),

    md(r"""
    ## 2 · The practical problem: find the delivery line iteratively

    CML-01 used four routes:

    | Distance $x$ (km) | Travel time $y$ (minutes) |
    | ---: | ---: |
    | 1 | 3 |
    | 2 | 5 |
    | 3 | 6 |
    | 4 | 8 |

    The model is:

    $$
    \hat y_i=b+wx_i
    $$

    The training objective is mean squared error:

    $$
    L(b,w)=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
    $$

    **Symbols:** $b$ is the intercept in minutes; $w$ is the slope in minutes per
    kilometre; $x_i$ and $y_i$ are route $i$'s feature and target; $\hat y_i$ is its
    prediction; $n$ is the number of training routes; $L$ is loss.

    CML-01 directly calculated $b=1.5$ and $w=1.6$. Our new question is:

    > Can we start from poor coefficients and reach nearly the same line using only
    > repeated local slope information?

    Analogy: standing on a foggy hill, we cannot see the entire landscape. We can feel
    the local slope and take a small downhill step. The analogy stops there: a model
    may have millions of parameter directions, not two physical directions.
    """),

    code(r"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    route_distance_km = np.array([1.0, 2.0, 3.0, 4.0])
    travel_time_minutes = np.array([3.0, 5.0, 6.0, 8.0])

    def mean_squared_loss(actual_values, predicted_values):
        '''Return mean squared error after validating matching shapes.'''
        actual = np.asarray(actual_values, dtype=float)
        predicted = np.asarray(predicted_values, dtype=float)
        if actual.shape != predicted.shape:
            raise ValueError("actual and predicted values must have matching shapes")
        return float(np.mean((predicted - actual) ** 2))


    starting_intercept = 0.0
    starting_slope = 0.0
    starting_predictions = starting_intercept + starting_slope * route_distance_km
    starting_loss = mean_squared_loss(travel_time_minutes, starting_predictions)

    print("starting predictions:", starting_predictions)
    print("starting MSE:", starting_loss)

    assert np.isclose(starting_loss, 33.5)
    """),

    md(r"""
    ## 3 · Name every object before optimizing

    | Object | Meaning | Delivery example |
    | --- | --- | --- |
    | Model | Rule that maps features to predictions | $\hat y=b+wx$ |
    | Parameter | Value learned during fitting | $b$ and $w$ |
    | Prediction | Model output for one row | Estimated minutes |
    | Residual | Actual minus predicted | $y-\hat y$ |
    | Per-row loss | Penalty from one row | $(\hat y-y)^2$ |
    | Objective | Quantity minimised across training rows | Mean squared error |
    | Gradient | Local loss slopes for all parameters | $[\partial L/\partial b,\partial L/\partial w]$ |
    | Optimizer | Rule that updates parameters | Gradient descent |
    | Learning rate | Size multiplier for an update | $\eta$ |

    A metric and a loss can use the same formula while serving different roles.
    Training MSE guides parameter updates. Validation MSE judges frozen parameters.
    Validation values must not be used inside the update loop.

    Optimization answers:

    > Which parameter values make this declared training objective small?

    It does not answer whether the objective is fair, causal, business-relevant, or
    evaluated on an honest split.
    """),

    md(r"""
    ## 4 · From one derivative to a gradient

    ### 4.1 One parameter: a derivative is a local slope

    Let:

    $$
    f(w)=(w-3)^2
    $$

    Its derivative is:

    $$
    \frac{df}{dw}=2(w-3)
    $$

    **Symbols:** $w$ is the parameter; $f(w)$ is its loss; $df/dw$ reads “the local
    change in loss with respect to $w$.”

    At $w=1$:

    $$
    \frac{df}{dw}=2(1-3)=-4
    $$

    The negative slope means increasing $w$ a little should reduce loss. At $w=5$,
    the slope is positive 4, so decreasing $w$ should reduce loss.

    ### 4.2 Several parameters: partial derivatives form a gradient

    If loss depends on intercept $b$ and slope $w$, a **partial derivative** changes
    one parameter while temporarily holding the other fixed. The gradient collects
    both slopes:

    $$
    \nabla L(b,w)=
    \begin{bmatrix}
    \frac{\partial L}{\partial b}\\[4pt]
    \frac{\partial L}{\partial w}
    \end{bmatrix}
    $$

    **Symbols:** $\nabla$ reads “gradient”; $\partial$ marks a partial derivative.
    The gradient has the same number of components as the parameter vector.

    The gradient points toward steepest local increase. Its negative points toward
    steepest local decrease.
    """),

    code(r"""
    def bowl_loss(parameter):
        return (parameter - 3.0) ** 2


    def bowl_derivative(parameter):
        return 2.0 * (parameter - 3.0)


    for parameter in [1.0, 3.0, 5.0]:
        print(
            f"w={parameter:.1f}, loss={bowl_loss(parameter):.1f}, "
            f"derivative={bowl_derivative(parameter):.1f}"
        )

    assert bowl_derivative(1.0) == -4.0
    assert bowl_derivative(3.0) == 0.0
    assert bowl_derivative(5.0) == 4.0
    """),

    md(r"""
    ## 5 · Derive one gradient-descent update by hand

    Gradient descent updates one parameter with:

    $$
    w_{t+1}=w_t-\eta\frac{dL}{dw}(w_t)
    $$

    **Symbols:** $t$ is the current step; $t+1$ is the next step; $w_t$ is the current
    parameter; $\eta$ (eta) is a positive learning rate; $dL/dw$ is the current slope.

    Why subtract? If the derivative is positive, subtraction moves $w$ lower. If the
    derivative is negative, subtracting a negative moves $w$ higher. Both move against
    the local uphill direction.

    Use $L(w)=(w-3)^2$, start at $w_0=1$, and choose $\eta=0.1$:

    $$
    \frac{dL}{dw}(1)=-4
    $$

    $$
    w_1=1-0.1(-4)=1.4
    $$

    Check the loss:

    $$
    L(1)=4,
    \qquad
    L(1.4)=(1.4-3)^2=2.56
    $$

    One update helped. We verify rather than assuming every step helps.
    """),

    code(r"""
    current_parameter = 1.0
    learning_rate = 0.10

    current_gradient = bowl_derivative(current_parameter)
    next_parameter = current_parameter - learning_rate * current_gradient

    print("current parameter:", current_parameter)
    print("current gradient:", current_gradient)
    print("update amount:", -learning_rate * current_gradient)
    print("next parameter:", next_parameter)
    print("loss before:", bowl_loss(current_parameter))
    print("loss after:", bowl_loss(next_parameter))

    assert np.isclose(next_parameter, 1.4)
    assert bowl_loss(next_parameter) < bowl_loss(current_parameter)
    """),

    md(r"""
    ## 6 · Repeat the update: scalar gradient descent from scratch

    An implementation needs:

    1. a derivative function;
    2. a starting parameter;
    3. a learning rate;
    4. a maximum number of steps;
    5. recorded parameter and loss history;
    6. checks for non-finite values.

    The first implementation stays deliberately small. It does not hide the loop
    inside a class or optimizer framework.
    """),

    code(r"""
    def scalar_gradient_descent(loss_function, derivative_function, start, learning_rate, steps):
        '''Minimise a one-parameter loss and return an auditable history table.'''
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if steps < 1:
            raise ValueError("steps must be at least one")

        parameter = float(start)
        records = []

        for step in range(steps + 1):
            loss = float(loss_function(parameter))
            gradient = float(derivative_function(parameter))
            if not np.isfinite(loss) or not np.isfinite(gradient):
                raise FloatingPointError("loss or gradient became non-finite")

            records.append(
                {"step": step, "parameter": parameter, "loss": loss, "gradient": gradient}
            )
            if step < steps:
                parameter = parameter - learning_rate * gradient

        return pd.DataFrame(records)


    scalar_history = scalar_gradient_descent(
        bowl_loss,
        bowl_derivative,
        start=1.0,
        learning_rate=0.10,
        steps=25,
    )

    print(scalar_history.head(4).to_string(index=False))
    print("\nfinal row:\n", scalar_history.tail(1).to_string(index=False))

    assert scalar_history.iloc[-1]["loss"] < scalar_history.iloc[0]["loss"]
    assert abs(scalar_history.iloc[-1]["parameter"] - 3.0) < 0.01
    """),

    code(r"""
    plotted_parameters = np.linspace(0, 6, 200)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(plotted_parameters, bowl_loss(plotted_parameters), color="tab:blue")
    axes[0].scatter(
        scalar_history["parameter"],
        scalar_history["loss"],
        color="tab:red",
        s=22,
        label="updates",
    )
    axes[0].set_xlabel("parameter w")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Gradient descent walks down the loss curve")
    axes[0].legend()

    axes[1].plot(scalar_history["step"], scalar_history["loss"], color="tab:green")
    axes[1].set_xlabel("update step")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss history reveals progress")

    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 7 · Learning rate controls step size

    The learning rate $\eta$ does not change the gradient. It changes how much of the
    gradient becomes one parameter update.

    | Learning rate behavior | Typical loss history |
    | --- | --- |
    | Too small | Falls safely but very slowly |
    | Useful | Falls steadily toward a low value |
    | Too large | Oscillates, increases, or becomes non-finite |

    For $L(w)=(w-3)^2$, the update becomes:

    $$
    w_{t+1}=w_t-2\eta(w_t-3)
    $$

    With $\eta=1$, starting at 1 jumps to 5, then back to 1. The loss never improves:

    $$
    w_1=1-2(1)(1-3)=5
    $$

    Smaller is not always better, and larger is not always faster. We judge the loss
    curve and final evidence.
    """),

    code(r"""
    learning_rate_histories = {}

    for candidate_rate in [0.01, 0.10, 0.60, 1.00, 1.10]:
        history = scalar_gradient_descent(
            bowl_loss,
            bowl_derivative,
            start=1.0,
            learning_rate=candidate_rate,
            steps=18,
        )
        learning_rate_histories[candidate_rate] = history

    figure, axis = plt.subplots(figsize=(8, 4))
    for candidate_rate, history in learning_rate_histories.items():
        axis.plot(history["step"], history["loss"], label=f"eta={candidate_rate}")
    axis.set_yscale("log")
    axis.set_xlabel("update step")
    axis.set_ylabel("loss — logarithmic scale")
    axis.set_title("The same loss with five learning rates")
    axis.legend()
    figure.tight_layout()
    plt.show()

    print("eta=0.01 final loss:", learning_rate_histories[0.01].iloc[-1]["loss"])
    print("eta=0.10 final loss:", learning_rate_histories[0.10].iloc[-1]["loss"])
    print("eta=1.00 final loss:", learning_rate_histories[1.00].iloc[-1]["loss"])
    print("eta=1.10 final loss:", learning_rate_histories[1.10].iloc[-1]["loss"])

    assert learning_rate_histories[0.10].iloc[-1]["loss"] < learning_rate_histories[0.01].iloc[-1]["loss"]
    assert np.isclose(learning_rate_histories[1.00].iloc[-1]["loss"], 4.0)
    assert learning_rate_histories[1.10].iloc[-1]["loss"] > 4.0
    """),

    md(r"""
    ## 8 · Derive the linear-regression gradient

    Return to:

    $$
    \hat y_i=b+wx_i
    $$

    Define prediction error inside the loss as:

    $$
    e_i=\hat y_i-y_i
    $$

    This is the negative of CML-01's residual, but squaring makes the loss identical.
    Keeping one sign convention throughout the derivation prevents mistakes.

    The MSE is:

    $$
    L(b,w)=\frac{1}{n}\sum_{i=1}^{n}e_i^2
    $$

    Apply the chain rule:

    $$
    \frac{\partial L}{\partial b}=\frac{2}{n}\sum_{i=1}^{n}e_i
    $$

    $$
    \frac{\partial L}{\partial w}=\frac{2}{n}\sum_{i=1}^{n}e_ix_i
    $$

    **Why:** changing $b$ changes every prediction by 1; changing $w$ changes
    prediction $i$ by $x_i$.

    At $b=0,w=0$, predictions are all zero and errors are $[-3,-5,-6,-8]$:

    $$
    \frac{\partial L}{\partial b}
    =\frac{2}{4}(-3-5-6-8)=-11
    $$

    $$
    \frac{\partial L}{\partial w}
    =\frac{2}{4}\left[(-3)(1)+(-5)(2)+(-6)(3)+(-8)(4)\right]
    =-31.5
    $$

    With $\eta=0.01$, the first update is $b=0.11$ and $w=0.315$.
    """),

    code(r"""
    def linear_regression_loss_and_gradient(features, targets, intercept, slope):
        '''Calculate MSE and gradients for one-feature linear regression.'''
        feature_array = np.asarray(features, dtype=float)
        target_array = np.asarray(targets, dtype=float)
        if feature_array.ndim != 1 or target_array.ndim != 1:
            raise ValueError("features and targets must be one-dimensional")
        if feature_array.shape != target_array.shape:
            raise ValueError("features and targets must have matching shapes")

        predictions = intercept + slope * feature_array
        errors = predictions - target_array
        loss = np.mean(errors**2)
        intercept_gradient = 2 * np.mean(errors)
        slope_gradient = 2 * np.mean(errors * feature_array)
        gradient = np.array([intercept_gradient, slope_gradient])
        return float(loss), gradient, predictions


    initial_loss, initial_gradient, initial_predictions = linear_regression_loss_and_gradient(
        route_distance_km,
        travel_time_minutes,
        intercept=0.0,
        slope=0.0,
    )

    first_parameters = np.array([0.0, 0.0]) - 0.01 * initial_gradient

    print("initial predictions:", initial_predictions)
    print("initial loss:", initial_loss)
    print("gradient [intercept, slope]:", initial_gradient)
    print("first parameters [intercept, slope]:", first_parameters)

    assert np.allclose(initial_gradient, [-11.0, -31.5])
    assert np.allclose(first_parameters, [0.11, 0.315])
    """),

    md(r"""
    ## 9 · Fit the CML-01 line with gradient descent

    We now repeat the two-parameter update. The history stores parameters, training
    loss, and gradient size so the run can be inspected rather than trusted blindly.

    Gradient size uses the Euclidean norm from FND-01:

    $$
    \lVert\nabla L\rVert_2
    =\sqrt{\left(\frac{\partial L}{\partial b}\right)^2
    +\left(\frac{\partial L}{\partial w}\right)^2}
    $$

    A small gradient means the local surface is flat. It does not by itself prove
    useful validation performance.
    """),

    code(r"""
    def fit_linear_regression_with_gradient_descent(
        features,
        targets,
        learning_rate=0.05,
        steps=1_000,
        start=(0.0, 0.0),
    ):
        '''Fit intercept and slope with full-batch gradient descent.'''
        parameters = np.asarray(start, dtype=float).copy()
        if parameters.shape != (2,):
            raise ValueError("start must contain [intercept, slope]")
        if learning_rate <= 0 or steps < 1:
            raise ValueError("learning_rate and steps must be positive")

        records = []
        for step in range(steps + 1):
            loss, gradient, _ = linear_regression_loss_and_gradient(
                features,
                targets,
                intercept=parameters[0],
                slope=parameters[1],
            )
            if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("optimization became non-finite")

            records.append(
                {
                    "step": step,
                    "intercept": parameters[0],
                    "slope": parameters[1],
                    "training_mse": loss,
                    "gradient_norm": np.linalg.norm(gradient),
                }
            )
            if step < steps:
                parameters = parameters - learning_rate * gradient

        return parameters, pd.DataFrame(records)


    gradient_parameters, regression_history = fit_linear_regression_with_gradient_descent(
        route_distance_km,
        travel_time_minutes,
        learning_rate=0.05,
        steps=1_000,
    )

    closed_form_parameters = np.array([1.5, 1.6])

    print("gradient-descent [intercept, slope]:", gradient_parameters)
    print("closed-form [intercept, slope]:", closed_form_parameters)
    print("final training MSE:", regression_history.iloc[-1]["training_mse"])

    assert np.allclose(gradient_parameters, closed_form_parameters, atol=1e-4)
    assert regression_history.iloc[-1]["training_mse"] < starting_loss
    """),

    code(r"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(regression_history["step"], regression_history["training_mse"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("training MSE — log scale")
    axes[0].set_title("Training loss falls during fitting")

    axes[1].scatter(route_distance_km, travel_time_minutes, label="training routes")
    axes[1].plot(
        route_distance_km,
        gradient_parameters[0] + gradient_parameters[1] * route_distance_km,
        color="tab:red",
        label="gradient-descent line",
    )
    axes[1].set_xlabel("distance (km)")
    axes[1].set_ylabel("travel time (minutes)")
    axes[1].set_title("The iterative solution matches CML-01")
    axes[1].legend()

    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 10 · Feature scale changes optimization geometry

    Suppose one feature uses kilometres and another uses metres. The represented
    information may be equivalent, but a one-unit coefficient change has very
    different effects on predictions. Gradient components can then have very
    different sizes.

    Standardization uses training mean and standard deviation:

    $$
    z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
    $$

    **Symbols:** $x_{ij}$ is row $i$, feature $j$; $\mu_j$ and $\sigma_j$ are that
    feature's training mean and standard deviation; $z_{ij}$ is the standardized value.

    Scaling does not add information, repair invalid data, or guarantee better
    predictions. It can make one learning rate behave more similarly across parameter
    directions. Learn scaling values from training data only.
    """),

    code(r"""
    distance_km = np.array([1.0, 2.0, 3.0, 4.0])
    distance_m = 1_000 * distance_km
    target = travel_time_minutes

    _, km_gradient, _ = linear_regression_loss_and_gradient(
        distance_km, target, intercept=0.0, slope=0.0
    )
    _, metre_gradient, _ = linear_regression_loss_and_gradient(
        distance_m, target, intercept=0.0, slope=0.0
    )

    training_mean = distance_m.mean()
    training_std = distance_m.std(ddof=0)
    standardized_distance = (distance_m - training_mean) / training_std
    _, standardized_gradient, _ = linear_regression_loss_and_gradient(
        standardized_distance, target, intercept=0.0, slope=0.0
    )

    print("gradient using kilometres:", km_gradient)
    print("gradient using metres:", metre_gradient)
    print("gradient after standardization:", standardized_gradient)

    assert abs(metre_gradient[1]) > 100 * abs(km_gradient[1])
    assert np.isclose(standardized_distance.mean(), 0.0)
    assert np.isclose(standardized_distance.std(ddof=0), 1.0)
    """),

    md(r"""
    ## 11 · Batch, stochastic, and mini-batch gradients

    The gradient formulas average row contributions. We can choose how many rows
    contribute to one update:

    | Method | Rows per update | Behavior |
    | --- | --- | --- |
    | Batch gradient descent | All training rows | Stable update; expensive on huge data |
    | Stochastic gradient descent | One row | Cheap, frequent, noisy updates |
    | Mini-batch gradient descent | A small group | Common balance of speed and noise |

    An **epoch** is one pass through all training rows. With four rows, batch gradient
    descent makes one update per epoch; mini-batches of two make two updates; stochastic
    descent makes four.

    “Noisy” means a mini-batch gradient may point differently from the full gradient.
    Across well-shuffled representative batches, it still provides useful progress.
    """),

    code(r"""
    def fit_with_mini_batches(features, targets, learning_rate, epochs, batch_size, seed=7):
        '''Fit one-feature regression with shuffled mini-batch updates.'''
        feature_array = np.asarray(features, dtype=float)
        target_array = np.asarray(targets, dtype=float)
        if not 1 <= batch_size <= len(feature_array):
            raise ValueError("batch_size must lie between 1 and the training row count")

        generator = np.random.default_rng(seed)
        parameters = np.zeros(2, dtype=float)
        epoch_losses = []

        for _ in range(epochs):
            shuffled_indices = generator.permutation(len(feature_array))
            for start_index in range(0, len(feature_array), batch_size):
                batch_indices = shuffled_indices[start_index:start_index + batch_size]
                _, gradient, _ = linear_regression_loss_and_gradient(
                    feature_array[batch_indices],
                    target_array[batch_indices],
                    intercept=parameters[0],
                    slope=parameters[1],
                )
                parameters = parameters - learning_rate * gradient

            epoch_predictions = parameters[0] + parameters[1] * feature_array
            epoch_losses.append(mean_squared_loss(target_array, epoch_predictions))

        return parameters, np.array(epoch_losses)


    batch_parameters, batch_losses = fit_with_mini_batches(
        route_distance_km, travel_time_minutes, learning_rate=0.05, epochs=300, batch_size=4
    )
    mini_parameters, mini_losses = fit_with_mini_batches(
        route_distance_km, travel_time_minutes, learning_rate=0.02, epochs=300, batch_size=2
    )
    stochastic_parameters, stochastic_losses = fit_with_mini_batches(
        route_distance_km, travel_time_minutes, learning_rate=0.01, epochs=300, batch_size=1
    )

    print("batch parameters:", batch_parameters.round(3))
    print("mini-batch parameters:", mini_parameters.round(3))
    print("stochastic parameters:", stochastic_parameters.round(3))

    assert batch_losses[-1] < batch_losses[0]
    assert mini_losses[-1] < mini_losses[0]
    assert stochastic_losses[-1] < stochastic_losses[0]
    """),

    md(r"""
    ## 12 · Loss landscapes, stopping, and failure modes

    ### Convex and non-convex at beginner depth

    A convex bowl has no misleading local valley: any local minimum is also global.
    Linear-regression MSE is convex in its coefficients. Neural-network loss is
    generally non-convex, so starting point and optimization path can matter.

    Convex does not mean every learning rate works, and non-convex does not mean
    learning is impossible.

    ### Stop rules

    Use more than one safeguard:

    - a maximum number of steps or epochs;
    - non-finite loss/gradient checks;
    - loss improvement smaller than a declared tolerance;
    - gradient norm smaller than a declared tolerance;
    - validation loss that stops improving.

    Training loss normally falls during optimization. Validation loss estimates how
    the frozen parameters behave on unseen development rows. If training improves
    while validation worsens, continuing to optimize training loss is not useful
    evidence of generalisation.

    ### Common failure patterns

    | Symptom | Likely cause | First check |
    | --- | --- | --- |
    | Loss rises immediately | Learning rate too large or gradient sign reversed | Print first gradient and update |
    | Loss falls extremely slowly | Learning rate too small or feature scales differ greatly | Compare gradient components and scaled inputs |
    | Loss becomes NaN/Inf | Invalid input, overflow, or unstable step | Validate inputs and inspect first non-finite step |
    | Gradient is always zero | Wrong derivative, constant feature, or flat point | Compare with finite differences |
    | Training falls, validation rises | Overfitting or distribution mismatch | Stop using validation evidence; inspect split |
    | Different result each run | Random batches or start not controlled | Record seed, shuffle, and initialization |
    """),

    code(r"""
    landscape_x = np.linspace(-3, 3, 400)
    convex_loss = landscape_x**2
    nonconvex_loss = 0.2 * landscape_x**4 - landscape_x**2 + 1

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(landscape_x, convex_loss)
    axes[0].set_title("Convex bowl: one global minimum")
    axes[0].set_xlabel("parameter")
    axes[0].set_ylabel("loss")

    axes[1].plot(landscape_x, nonconvex_loss, color="tab:orange")
    axes[1].set_title("Non-convex example: several valleys")
    axes[1].set_xlabel("parameter")
    axes[1].set_ylabel("loss")

    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 13 · Check the gradient, then use a library

    ### 13.1 Finite-difference gradient check

    A centred finite difference approximates a derivative:

    $$
    \frac{dL}{dw}
    \approx
    \frac{L(w+h)-L(w-h)}{2h}
    $$

    **Symbols:** $h$ is a small positive change. If the analytical and numerical
    slopes disagree substantially, inspect the derivation and code before training.

    Very large $h$ is not local. Extremely tiny $h$ can suffer floating-point
    cancellation. Values around $10^{-5}$ are often useful for small checks.
    """),

    code(r"""
    def finite_difference_gradient(function, parameters, step_size=1e-5):
        '''Approximate each partial derivative with a centred finite difference.'''
        parameter_array = np.asarray(parameters, dtype=float)
        numerical_gradient = np.zeros_like(parameter_array)

        for position in range(parameter_array.size):
            plus = parameter_array.copy()
            minus = parameter_array.copy()
            plus[position] += step_size
            minus[position] -= step_size
            numerical_gradient[position] = (function(plus) - function(minus)) / (2 * step_size)

        return numerical_gradient


    check_parameters = np.array([0.7, 1.2])
    analytical_loss, analytical_gradient, _ = linear_regression_loss_and_gradient(
        route_distance_km,
        travel_time_minutes,
        intercept=check_parameters[0],
        slope=check_parameters[1],
    )

    def loss_from_parameter_vector(parameters):
        predictions = parameters[0] + parameters[1] * route_distance_km
        return mean_squared_loss(travel_time_minutes, predictions)


    numerical_gradient = finite_difference_gradient(
        loss_from_parameter_vector,
        check_parameters,
    )

    print("loss:", analytical_loss)
    print("analytical gradient:", analytical_gradient)
    print("finite-difference gradient:", numerical_gradient)
    print("absolute difference:", np.abs(analytical_gradient - numerical_gradient))

    assert np.allclose(analytical_gradient, numerical_gradient, atol=1e-7)
    """),

    md(r"""
    ### 13.2 Scikit-learn version

    `SGDRegressor` packages iterative linear-regression training. We still control
    scaling, random state, learning-rate behavior, and training boundaries. The
    library removes loop bookkeeping; it does not remove the reasoning.
    """),

    code(r"""
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feature_table = route_distance_km.reshape(-1, 1)
    sgd_pipeline = Pipeline(
        steps=[
            ("standard_scaler", StandardScaler()),
            (
                "sgd_regression",
                SGDRegressor(
                    loss="squared_error",
                    penalty=None,
                    learning_rate="constant",
                    eta0=0.01,
                    max_iter=20_000,
                    tol=1e-10,
                    random_state=42,
                ),
            ),
        ]
    )

    sgd_pipeline.fit(feature_table, travel_time_minutes)
    library_predictions = sgd_pipeline.predict(feature_table)
    library_training_mse = mean_squared_loss(travel_time_minutes, library_predictions)

    print("library predictions:", library_predictions.round(3))
    print("library training MSE:", round(library_training_mse, 6))
    print("manual gradient-descent MSE:", round(regression_history.iloc[-1]["training_mse"], 6))

    assert library_training_mse < starting_loss
    assert np.allclose(library_predictions, [3.1, 4.7, 6.3, 7.9], atol=0.05)
    """),

    md(r"""
    ## 14 · Mini-project, practice, and mastery checkpoint

    ### Mini-project: optimize a delivery-time model with a validation stop rule

    **Goal:** fit a one-feature line iteratively while keeping validation rows outside
    parameter updates.

    **Dataset columns:**

    | Column | Meaning | Role |
    | --- | --- | --- |
    | `route_id` | Stable route label | Identifier |
    | `distance_km` | Known route distance | Feature |
    | `travel_minutes` | Observed duration | Numerical target |
    | `partition` | `train` or `validation` | Evidence boundary |

    **Required workflow:** validate the schema; calculate a frozen training-mean
    baseline; fit intercept and slope from training rows only; record training and
    validation MSE; preserve the best validation parameters; compare with the baseline;
    report the update settings and limitations.
    """),

    code(r"""
    project_data = pd.DataFrame(
        {
            "route_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "distance_km": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
            "travel_minutes": [3.0, 5.0, 6.0, 8.0, 4.0, 5.5, 7.0, 9.0],
            "partition": ["train"] * 4 + ["validation"] * 4,
        }
    )

    expected_columns = {"route_id", "distance_km", "travel_minutes", "partition"}
    if set(project_data.columns) != expected_columns:
        raise ValueError("project columns do not match the declared schema")
    if project_data["route_id"].duplicated().any():
        raise ValueError("route_id must be unique")
    if set(project_data["partition"]) != {"train", "validation"}:
        raise ValueError("both train and validation partitions are required")

    train_rows = project_data[project_data["partition"] == "train"]
    validation_rows = project_data[project_data["partition"] == "validation"]

    project_baseline = float(train_rows["travel_minutes"].mean())
    baseline_validation_predictions = np.full(len(validation_rows), project_baseline)
    baseline_validation_mse = mean_squared_loss(
        validation_rows["travel_minutes"],
        baseline_validation_predictions,
    )

    parameters = np.zeros(2, dtype=float)
    best_parameters = parameters.copy()
    best_validation_mse = np.inf
    project_records = []
    project_learning_rate = 0.05

    for step in range(501):
        training_loss, training_gradient, _ = linear_regression_loss_and_gradient(
            train_rows["distance_km"],
            train_rows["travel_minutes"],
            intercept=parameters[0],
            slope=parameters[1],
        )
        validation_predictions = (
            parameters[0] + parameters[1] * validation_rows["distance_km"].to_numpy()
        )
        validation_mse = mean_squared_loss(
            validation_rows["travel_minutes"],
            validation_predictions,
        )
        project_records.append(
            {
                "step": step,
                "training_mse": training_loss,
                "validation_mse": validation_mse,
                "intercept": parameters[0],
                "slope": parameters[1],
            }
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_parameters = parameters.copy()
        if step < 500:
            parameters = parameters - project_learning_rate * training_gradient

    project_history = pd.DataFrame(project_records)

    print("training-mean baseline:", project_baseline)
    print("baseline validation MSE:", round(baseline_validation_mse, 4))
    print("best [intercept, slope]:", best_parameters.round(4))
    print("best validation MSE:", round(best_validation_mse, 4))
    print("learning rate:", project_learning_rate)

    assert len(train_rows) == 4 and len(validation_rows) == 4
    assert best_validation_mse < baseline_validation_mse
    assert np.all(np.isfinite(best_parameters))
    assert project_history["training_mse"].iloc[-1] < project_history["training_mse"].iloc[0]
    """),

    md(r"""
    ### Worked example

    For $L(w)=(w-4)^2$, start at $w=1$ with $\eta=0.25$:

    $$
    \frac{dL}{dw}=2(w-4)=2(1-4)=-6
    $$

    $$
    w_1=1-0.25(-6)=2.5
    $$

    Loss falls from 9 to $(2.5-4)^2=2.25$.

    ### Guided practice

    1. For $L(w)=(w-5)^2$, calculate the derivative at $w=2$.
    2. Use learning rate 0.1 to calculate two updates manually.
    3. Explain why subtracting a negative gradient increases the parameter.
    4. At $b=0,w=0$, calculate both regression gradients for two rows
       $(x,y)=(1,2),(2,4)$.
    5. Predict the shape of a gradient for seven parameters.

    ### Independent practice

    6. Rebuild scalar gradient descent and record step, parameter, loss, and gradient.
    7. Compare three learning rates on one loss and explain their curves.
    8. Derive and implement the intercept and slope gradients without copying.
    9. Compare batch sizes 1, 2, and all rows using a fixed seed.
    10. Standardize a training feature manually and explain why validation uses the
        frozen training mean and standard deviation.
    11. Break one analytical gradient deliberately and prove the finite-difference
        check detects it.

    ### Challenge

    Rebuild the mini-project without copying its code. Add:

    - a maximum-step rule;
    - a non-finite-value rule;
    - a small-improvement patience rule;
    - a gradient-norm rule;
    - recorded learning rate and starting parameters;
    - plots of training and validation loss;
    - comparison with the frozen baseline and `SGDRegressor`;
    - at least eight meaningful assertions.

    ### Self-check

    Before trusting an optimization run, answer:

    - What exact quantity is being minimized?
    - Which rows contribute to each update?
    - Does the gradient shape match the parameter shape?
    - What does the first gradient sign predict?
    - Did the first update lower training loss?
    - Is validation used only for evidence and stopping?
    - Which stop rule ended the run?
    - Could a falling loss still optimize the wrong problem?
    """),

    md(r"""
    ### Solution and scoring rubric

    1. $dL/dw=2(2-5)=-6$.
    2. First update: $2-0.1(-6)=2.6$. New gradient is $2(2.6-5)=-4.8$;
       second update is $2.6-0.1(-4.8)=3.08$.
    3. Subtracting a negative value is addition, which moves toward a locally lower loss.
    4. Errors are $[-2,-4]$. Intercept gradient is $2(-3)=-6$; slope gradient is
       $2[(-2)(1)+(-4)(2)]/2=-10$.
    5. The gradient has seven components, one local slope per parameter.
    6. A valid history shows loss decreasing for a useful rate and records the starting row.
    7. A tiny rate crawls; a useful rate falls steadily; a large rate oscillates or grows.
    8. The implementation should reach approximately $b=1.5,w=1.6$ on the delivery rows.
    9. Smaller batches create noisier paths but should reduce loss with suitable rates.
    10. Validation must use training statistics because it may not teach preprocessing.
    11. The analytical and numerical gradients should disagree after the deliberate error.

    Challenge scoring:

    | Skill | Points |
    | --- | ---: |
    | Manual derivative and two updates | 3 |
    | Correct regression gradient derivation | 4 |
    | Auditable implementation and stop rules | 4 |
    | Learning-rate and scaling explanation | 3 |
    | Batch comparison and reproducibility | 2 |
    | Validation boundary and baseline | 2 |
    | Gradient check and assertions | 2 |
    | **Total** | **20** |

    ### Common mistakes

    - Optimizing before defining the loss.
    - Reversing actual-minus-predicted and predicted-minus-actual during one derivation.
    - Adding the gradient instead of stepping against it.
    - Forgetting the chain-rule factor $x_i$ in the slope gradient.
    - Comparing gradients whose parameter order differs.
    - Choosing a learning rate without checking the loss history.
    - Treating a small gradient as proof of good validation performance.
    - Learning scaling values from validation or test rows.
    - Using validation rows inside gradient updates.
    - Reporting only the final loss and hiding divergence or oscillation.
    - Assuming a library optimizer repairs invalid data or a dishonest split.
    - Jumping to Adam before plain gradient descent can be explained manually.

    ### Readiness threshold

    Score at least **16/20**, including correct manual updates, regression gradients,
    validation boundaries, and a passing finite-difference gradient check.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    1. What is the difference between loss and optimizer?
    2. What does the sign of a derivative tell us?
    3. Why does gradient descent subtract the gradient?
    4. What does the learning rate control?
    5. Why does the slope gradient contain $x_i$?
    6. Why can scaled features be easier to optimize?
    7. How do batch, stochastic, and mini-batch updates differ?
    8. Why can training loss fall while validation loss rises?
    9. What does a finite-difference check test?
    10. Why can successful optimization still solve the wrong problem?

    ### Teach it back

    Starting from the CML-01 delivery MSE, explain:

    **prediction → error → loss → partial derivatives → gradient → learning-rate
    update → repeated training steps → frozen validation evidence.**

    Calculate the first intercept and slope update without code. Then point to the
    exact place where validation data must remain outside the learning loop.

    ### Memory aid

    **Define the loss, measure its local slopes, step against them, and verify both
    training progress and validation evidence.**

    ### Next dependency

    CML-02 will define a classification loss before deriving its gradient. The update
    mechanism stays the same; only the prediction rule and loss derivative change.
    """),
]


build("01_ml_foundations/04_optimization_and_gradient_descent.ipynb", cells)
