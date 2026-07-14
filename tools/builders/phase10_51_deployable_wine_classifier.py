"""Builder for Notebook 51 — Deployable Real-Data ML Vertical Slice."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # 51 · Deployable Real-Data ML Vertical Slice
    ### Phase 10 — Applied Capstone

    This notebook connects the curriculum to a real project under
    `projects/wine_classifier`. It trains on scikit-learn's bundled UCI Wine
    recognition dataset, persists a versioned artifact, serves strict API inputs,
    records monitoring diagnostics, and verifies contracts with tests.

    **Estimated time:** 5–8 hours including project exercises.
    **Prerequisites:** Notebook 03A, Notebooks 04–13, and production Phases 8–9.
    """),

    md(r"""
    ## 1 · Learning Objectives

    You will be able to:

    - trace one prediction from raw feature contract to API response;
    - explain why preprocessing and the model are one artifact;
    - tune only inside training data and evaluate once on an untouched holdout;
    - report multiclass accuracy, balanced accuracy, macro F1, log loss, confusion
      matrix, and uncertainty;
    - save model version, data hash, features, metrics, and reference statistics;
    - enforce a serving schema and produce range warnings without silently clipping;
    - distinguish in-process diagnostics from durable production monitoring;
    - run tests and package the service in a container.
    """),

    md(r"""
    ## 2 · Historical Motivation

    A notebook prediction is not a deployed ML system. Between model fitting and a
    reliable request path are contracts: input names and units, transformation state,
    artifact version, evaluation evidence, serialization, API validation, monitoring,
    rollback, and tests.

    This capstone keeps the system intentionally small so every boundary is visible.
    It uses a classical model because deployment quality should not be hidden behind
    model complexity.
    """),

    md(r"""
    ## 3 · Intuition and Visual Understanding

    ```mermaid
    flowchart LR
        D[Bundled real dataset] --> S[Stratified holdout]
        S --> CV[CV on training only]
        CV --> P[Scaler + logistic model]
        P --> E[Untouched test evaluation]
        E --> A[Model + metadata artifact]
        A --> API[FastAPI schema and prediction]
        API --> M[Diagnostics and delayed-label monitoring boundary]
        A --> T[Contract tests]
        API --> C[Container]
    ```

    The saved sklearn Pipeline owns both scaling and prediction. The API owns input
    validation. Metadata explains what was trained. Monitoring reports what happens
    after deployment. Tests protect the boundaries between them.
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Standardization

    $$z_j=\frac{x_j-\mu_j}{\sigma_j}.$$

    **Read and symbols:** raw feature $x_j$ becomes standardized value $z_j$ using
    training mean $\mu_j$ and training standard deviation $\sigma_j$. The scaler is
    fitted inside cross-validation, never on test or serving data.

    **Small example:** if training alcohol mean is 13 and standard deviation is 1,
    a value of 14 becomes `z = 1`.

    ### 4.2 Multiclass probabilities

    $$P(y=k\mid\mathbf x)=\frac{e^{s_k}}{\sum_{c=1}^{K}e^{s_c}}.$$

    **Read and symbols:** $s_k$ is the model score for class $k$; $K$ is class count;
    $e$ is the exponential base; the denominator normalizes all positive exponentials
    to probabilities summing to one. A large score difference produces a confident
    probability, but confidence still requires calibration evidence.

    ### 4.3 Multiclass log loss

    $$\operatorname{LogLoss}=-\frac1n\sum_{i=1}^{n}\log p_{i,y_i}.$$

    **Read and symbols:** $n$ is test-row count; $y_i$ is true class for row $i$;
    $p_{i,y_i}$ is probability assigned to that true class. The logarithm strongly
    penalizes confident wrong predictions. Lower is better.

    ### 4.4 Macro F1

    $$F_{1,\mathrm{macro}}=\frac1K\sum_{k=1}^{K}
    \frac{2\,\mathrm{Precision}_k\,\mathrm{Recall}_k}
    {\mathrm{Precision}_k+\mathrm{Recall}_k}.$$

    **Read and symbols:** calculate F1 separately for each of $K$ classes, then give
    every class equal weight. Macro F1 prevents the largest class from dominating
    the summary.

    ### 4.5 Bootstrap interval

    $$\hat\theta^{(b)}=M\!\left(y^{(b)},\hat y^{(b)}\right).$$

    **Read and symbols:** $b$ identifies a bootstrap resample; $M$ is a metric;
    $y$ and $\hat y$ are paired labels and predictions; $\hat\theta^{(b)}$ is one
    resampled estimate. Percentiles across many resamples approximate uncertainty.
    With only 36 test rows, the interval should be expected to remain wide.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    Inspect the real dataset and its contract before training. The target has three
    cultivar classes; the 13 inputs are chemical measurements.
    """),

    code(r"""
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    project_src = Path("projects/wine_classifier/src").resolve()
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

    from wine_classifier.training import FEATURE_NAMES, load_dataset, train_and_save
    from wine_classifier.service import ModelService

    X, y, class_names = load_dataset()
    print("shape:", X.shape)
    print("features:", FEATURE_NAMES)
    print("classes:", class_names)
    print("class counts:\n", y.value_counts().sort_index())

    assert X.shape == (178, 13)
    assert not X.isna().any().any()
    """),

    md(r"""
    Train into a temporary artifact directory so executing the notebook does not
    overwrite the repository's reviewed artifact. The project function performs the
    split, inner CV, refit, test evaluation, metadata creation, and atomic save.
    """),

    code(r"""
    artifact_temp = tempfile.TemporaryDirectory(prefix="wine-capstone-")
    artifact_dir = Path(artifact_temp.name)
    metadata = train_and_save(artifact_dir, random_state=42)

    print("model version:", metadata["model_version"])
    print("best parameters:", metadata["best_parameters"])
    for name, value in metadata["metrics"].items():
        print(name, value)

    assert metadata["metrics"]["test_macro_f1"] >= 0.85
    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "metadata.json").exists()
    """),

    md(r"""
    ## 6 · Visualization

    The confusion matrix identifies which classes are confused. The accuracy
    interval communicates sampling uncertainty instead of presenting 0.972 as an
    exact population fact.
    """),

    code(r"""
    confusion = np.asarray(metadata["metrics"]["confusion_matrix"])
    ci_low, ci_high = metadata["metrics"]["test_accuracy_ci95"]
    accuracy = metadata["metrics"]["test_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    image = axes[0].imshow(confusion, cmap="Blues")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axes[0].text(column, row, str(confusion[row, column]), ha="center", va="center")
    axes[0].set_xlabel("predicted class")
    axes[0].set_ylabel("true class")
    axes[0].set_title("Untouched-test confusion matrix")
    fig.colorbar(image, ax=axes[0], shrink=0.8)

    axes[1].errorbar(
        [0], [accuracy],
        yerr=[[accuracy - ci_low], [ci_high - accuracy]],
        fmt="o", capsize=8,
    )
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(0.75, 1.02)
    axes[1].set_xticks([0], ["accuracy"])
    axes[1].set_title("Bootstrap 95% interval")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | What this project does | Remaining production requirement |
    |---|---|---|
    | Test leakage | CV runs inside training partition | Temporal/group split when data requires it |
    | Train/serve skew | Saves scaler and model together | Versioned upstream transformations |
    | Feature mismatch | Rejects missing and extra fields | Contract registry and migration policy |
    | Silent range shift | Returns range warnings | Windowed multivariate drift and alerts |
    | Artifact ambiguity | Saves metadata and dataset hash | Signed artifacts and external registry |
    | Process restart | Declares counter reset limitation | Durable metrics backend |
    | Model quality decay | Exposes diagnostics only | Delayed-label performance and calibration |
    | Unsafe use | Model card limits intended use | Authentication, authorization, governance |
    """),

    md(r"""
    ## 8 · Production Library Implementation

    Load the saved bundle and make one framework-independent prediction. The service
    validates the exact feature set, preserves feature order, returns all class
    probabilities, and records diagnostic warnings.
    """),

    code(r"""
    service = ModelService.from_artifact_dir(artifact_dir)
    payload = {name: float(X.iloc[0][name]) for name in FEATURE_NAMES}
    prediction = service.predict(payload)

    print(prediction)
    print(service.monitor.snapshot())
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-9
    assert prediction.model_version == metadata["model_version"]
    """),

    md(r"""
    The FastAPI application is created from the same artifact. `TestClient` exercises
    routing, Pydantic schema validation, and JSON serialization without opening a
    network port.
    """),

    code(r"""
    from fastapi.testclient import TestClient
    from wine_classifier.app import create_app

    client = TestClient(create_app(artifact_dir))
    health = client.get("/health")
    response = client.post("/v1/predict", json=payload)

    print("health:", health.json())
    print("prediction:", response.json())
    assert health.status_code == 200
    assert response.status_code == 200
    """),

    md(r"""
    ## 9 · Realistic Business Case Study

    This project deliberately avoids claiming business value from the cultivar label.
    Its realistic value is engineering evidence: a reviewer can reproduce training,
    inspect metrics, load the exact artifact, call a strict API, observe warnings,
    run tests, and build a container.

    In a real business case, the outcome definition, intervention, cost matrix,
    fairness, privacy, and human workflow must be specified before model selection.
    """),

    md(r"""
    ## 10 · Production Considerations

    Before internet-facing deployment, add:

    - authentication, authorization, TLS, rate limits, and payload-size limits;
    - structured request IDs, traces, durable metrics, and privacy-safe logs;
    - artifact signing, vulnerability scanning, SBOM, and pinned dependencies;
    - multiple workers with an external metrics system;
    - canary/shadow rollout, rollback, and readiness/liveness separation;
    - delayed-label joins and slice-aware quality monitoring;
    - load, failure-injection, and schema-compatibility tests;
    - ownership, incident response, retention, and governance controls.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    | Decision | Benefit | Tradeoff |
    |---|---|---|
    | Logistic regression | Fast and interpretable baseline | Limited nonlinear interactions |
    | Bundled small dataset | Offline reproducibility | Weak external validity and wide intervals |
    | Strict schema | Prevents silent corruption | Requires versioned client migrations |
    | In-process monitor | Easy to understand | Resets and cannot aggregate replicas |
    | One model artifact | Train/serve parity | Requires careful compatibility/versioning |
    | Synchronous API | Simple request path | Not ideal for high-volume batch scoring |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    Be ready to explain:

    - why the test set is outside GridSearchCV;
    - why accuracy and macro F1 tell different stories;
    - what the bootstrap interval resamples and why rows must be independent;
    - how the service prevents feature-order bugs;
    - why range warnings are not drift proof;
    - how you would promote `v2`, observe it, and roll it back;
    - which controls change in a financial, medical, or employment domain.
    """),

    md(r"""
    ## 13 · Teach-Back

    Draw the system from raw rows to API response. At every arrow, name:

    1. the input contract;
    2. the state being learned or transformed;
    3. the evidence or test protecting the boundary;
    4. the monitoring signal available after deployment;
    5. one failure that remains possible.
    """),

    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Estimated time:** 2–4 hours.

    ### Guided implementation

    1. Run the training CLI twice with the same seed. Verify dataset hash, selected
       hyperparameter, predictions, and metrics match; explain why timestamp differs.
    2. Send a request missing `alcohol`, then one with `unknown_feature`. Confirm both
       return 422 before inference.
    3. Send `alcohol=100`. Confirm prediction succeeds with a diagnostic warning.
    4. Add a `/ready` endpoint that verifies model and metadata versions agree.

    ### Independent implementation

    5. Add a batch endpoint capped at 100 rows and test the cap.
    6. Add an API test proving probability keys exactly match metadata class names.
    7. Implement a rolling prediction-distribution monitor with a minimum sample size;
       label it as an unlabeled diagnostic rather than performance.
    8. Create `wine-logreg-v2` with one justified change. Compare paired predictions
       on the same holdout using a bootstrap interval and define a promotion rule.

    ### Senior design

    9. Design the production path for 2,000 requests/second with p99 under 50 ms,
       including load test, replicas, autoscaling, telemetry, canary, and rollback.
    10. Threat-model the API: malformed values, abuse, dependency compromise, artifact
        tampering, model extraction, and sensitive logging.

    <details>
    <summary><strong>Expected checks and scoring rubric</strong></summary>

    - Questions 1–4: tests must be executable and deterministic; award 2 points each.
    - Questions 5–8: award one point for implementation, one for a contract test, and
      one for explaining the statistical or operational limitation.
    - Question 9: full credit includes measured per-replica capacity, headroom,
      percentile latency, readiness, canary gates, and an automated rollback signal.
    - Question 10: full credit maps each threat to prevention, detection, and response
      rather than listing security products.

    **Readiness:** 16/20 across implementation and design indicates the learner can
    extend the vertical slice responsibly. A lower score should identify a specific
    boundary—data, evaluation, artifact, serving, monitoring, or operations—to revisit.
    </details>
    """),
]


build("phase10_applied_capstone/51_deployable_wine_classifier.ipynb", cells)
