"""Beginner-first mastery material for the route through classical ML.

The existing builders retain their deeper refresher material.  This module adds
the small, fully worked core that a new learner must complete before that
material.  Values are plain data so ``nbbuild`` remains the only notebook
assembly implementation.
"""

TOPIC_GUIDES = {
    "MLE-01": {
        "problem": "Measure model errors in a way that matches the real cost of false alarms and missed cases.",
        "analogy": "A metric is a scoreboard: different scoreboards reward different behavior. No single score explains every kind of mistake.",
        "use": "Choose metrics after defining the target, decision, class balance, and error costs.",
        "avoid": "Do not choose a model from accuracy alone or evaluate on its training examples.",
        "alternative": "Inspect the confusion matrix and per-example errors before compressing them into one number.",
        "memory": "Choose the error that matters before choosing the metric.",
    },
    "MLE-02": {
        "problem": "Estimate performance on future unseen cases without allowing answers to leak into training.",
        "analogy": "A test set is a sealed examination. Looking at its answers while studying turns it into training material.",
        "use": "Use held-out validation that matches time, entity, and serving boundaries.",
        "avoid": "Do not randomly split dependent rows or fit preprocessing before the split.",
        "alternative": "Use a simple train-test split before cross-validation; use group or time splits when rows are related.",
        "memory": "Anything learned from held-out data makes the evaluation less held out.",
    },
    "PROD-04": {
        "problem": "Know exactly what changed between experiments and reproduce the result later.",
        "analogy": "An experiment record is a laboratory notebook: hypothesis, materials, procedure, and result belong together. Recording a run does not make its evaluation valid.",
        "use": "Track data version, split, code, parameters, seed, metric, artifact, and conclusion for every serious run.",
        "avoid": "Do not tune on the final test set or compare runs whose data and metric definitions differ.",
        "alternative": "A small CSV or JSON log is enough before adopting a tracking platform.",
        "memory": "A result without its data, code, configuration, and metric definition is not reproducible.",
    },
    "MLE-03": {
        "problem": "Represent raw information in a form whose meaning and scale a chosen model can use.",
        "analogy": "Feature engineering is translation: the facts stay the same, but their representation becomes understandable to the model. A better translation cannot repair missing or false facts.",
        "use": "Choose transformations from the model's assumptions and fit them on training data only.",
        "avoid": "Do not encode categories as arbitrary ordered numbers or use target information outside a training fold.",
        "alternative": "Start with raw validated features and a baseline before adding transformations.",
        "memory": "A feature is useful when its representation matches the model and remains available at prediction time.",
    },
}


ADVANCED_SECTION_NOTES = {
    "FND-01": "Eigenvalues, SVD, conditioning, and compression are the advanced path. Master the vector, matrix, shape, dot-product, and projection bridge first.",
    "PROD-04": "Gaussian processes, Expected Improvement, and Bayesian optimization are optional advanced optimization material—not prerequisites for experiment tracking.",
}


# Each bridge is deliberately small.  It supplies the missing hand calculation
# before the existing deeper derivation and implementation.
CORE_MASTERY = {
    "MLE-01": r"""
## Beginner Core · Calculate every classification metric from one table

For 10 cases, suppose $TP=3$, $TN=4$, $FP=2$, and $FN=1$. These counts sum to 10.

$$
\text{Accuracy}=\frac{3+4}{10}=0.70,
\quad
\text{Precision}=\frac{3}{3+2}=0.60,
\quad
\text{Recall}=\frac{3}{3+1}=0.75.
$$

Accuracy asks “what fraction was correct?” Precision asks “among predicted
positives, what fraction was positive?” Recall asks “among actual positives, what
fraction did we find?” Specificity is $4/(4+2)=0.667$. F1 is the harmonic mean of
precision and recall: $2(0.60)(0.75)/(0.60+0.75)=0.667$.

If a denominator is zero, the metric is undefined; choose and document a library
policy rather than hiding it. Inspect counts before a single score. Threshold curves
compare possible decision policies; AUC measures ranking, not calibration or a
deployed threshold. For numerical targets, begin with MAE and RMSE in target units
and compare against a mean or median baseline.
""",
    "MLE-02": r"""
## Beginner Core · Split before cross-validation

Start with three roles. Training data fits parameters and transformations.
Validation data chooses models, features, thresholds, and hyperparameters. Test data
is used once for the final scoped estimate.

```text
all eligible historical data
        |
        +-- training: learn model and preprocessing
        +-- validation: make development choices
        +-- test: final untouched estimate
```

For 10 ordered time rows, training on rows 1–6, validating on 7–8, and testing on
9–10 respects time. A random split might train on row 10 and evaluate row 3, which
answers the wrong future-prediction question. If several rows belong to one patient,
keep the entire patient on one side.

Cross-validation repeats the train-validation boundary; it does not remove the need
for a final test set or correct grouping. Every learned step—imputation, scaling,
feature selection, target encoding—must fit inside each training fold. Nested CV is
an advanced option when the tuned procedure itself needs an unbiased estimate.
""",
    "PROD-04": r"""
## Beginner Core · Track one honest experiment before tuning

An experiment record must include: question and hypothesis; data snapshot and row
eligibility; split strategy; code version; feature and model configuration; random
seed; metric definition; result and uncertainty; artifact location; conclusion and
limitations.

Begin with a JSON or CSV record. Compare only runs that share the same data boundary
and metric definition. Change one declared factor when the goal is to learn its
effect. A run ID is a label, not evidence.

```text
hypothesis -> frozen data/split -> configuration -> run -> metric -> conclusion
                    |                              |
                    +---------- recorded ----------+
```

Grid search and random search are model-selection procedures and belong inside the
validation design. The final test set is not a tuning dashboard. Bayesian optimization
is an optional later method for choosing configurations efficiently; Gaussian-process
formulas are not required to understand experiment tracking.
""",
    "MLE-03": r"""
## Beginner Core · Choose a representation for the model

Suppose color has values red, blue, and green. Encoding them as 1, 2, and 3 invents
an order and distances. One-hot encoding uses three binary columns instead. For an
unseen category at prediction time, use an explicit unknown policy such as all-zero
with `handle_unknown="ignore"`, and monitor its rate.

Standardization uses training mean and standard deviation. It is helpful for
distance-based, gradient-based, and regularized linear models; it is usually
unnecessary for ordinary tree splits. It is not universally required.

For hour-of-day, 23 and 0 are neighbors. Sine and cosine encode that circular
relationship. Interaction features allow a linear model to represent “the effect of
one feature depends on another.” Each feature must be available with the same meaning
at prediction time.

Compare every transformation against a raw-feature baseline using the same validation
design. Target encoding and feature selection must occur inside folds; keep them on
the advanced path until one-hot and train-only preprocessing are mastered.
""",
}


CORE_CODE = {
    "PRE-03": """import numpy as np\n\nvalues = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\ncolumn_means = np.array([2.5, 3.5, 4.5])\ncentered = values - column_means\n\nprint(\"values shape:\", values.shape)\nprint(\"means shape :\", column_means.shape)\nprint(\"centered:\\n\", centered)\nassert centered.shape == (2, 3)\nassert np.allclose(centered.mean(axis=0), 0.0)""",
    "FND-01": """import numpy as np\n\nmatrix = np.array([[1, 2], [3, 4]])\nvector = np.array([5, 6])\nmanual_output = np.array([1 * 5 + 2 * 6, 3 * 5 + 4 * 6])\nlibrary_output = matrix @ vector\n\nprint(\"matrix shape:\", matrix.shape)\nprint(\"vector shape:\", vector.shape)\nprint(\"manual output:\", manual_output)\nprint(\"NumPy output :\", library_output)\nassert np.array_equal(manual_output, library_output)""",
    "CML-01": """import numpy as np\n\nhours = np.array([1.0, 2.0, 3.0])\nscores = np.array([3.0, 5.0, 7.0])\npredictions = 2.0 * hours + 1.0\nresiduals = scores - predictions\nmean_squared_error = np.mean(residuals ** 2)\n\nprint(\"predictions:\", predictions)\nprint(\"residuals  :\", residuals)\nprint(\"MSE        :\", mean_squared_error)\nassert mean_squared_error == 0.0""",
    "MLE-01": """true_positive, true_negative = 3, 4\nfalse_positive, false_negative = 2, 1\n\naccuracy = (true_positive + true_negative) / 10\nprecision = true_positive / (true_positive + false_positive)\nrecall = true_positive / (true_positive + false_negative)\nf1 = 2 * precision * recall / (precision + recall)\n\nprint(f\"accuracy={accuracy:.3f}\")\nprint(f\"precision={precision:.3f}\")\nprint(f\"recall={recall:.3f}\")\nprint(f\"f1={f1:.3f}\")""",
}
