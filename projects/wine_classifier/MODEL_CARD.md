# Model Card — Wine Cultivar Classifier

## Intended use

Educational demonstration of a complete tabular-ML lifecycle: data contract,
leak-safe validation, preprocessing pipeline, model artifact, API contract,
monitoring hooks, tests, and container packaging.

It is not intended for wine quality, safety, authenticity, pricing, purchasing,
or regulatory decisions.

## Data

The model uses scikit-learn's copy of the UCI Wine recognition dataset: 178
chemical-analysis samples, 13 numeric features, and three cultivar classes.
The training command records a SHA-256 hash, split details, feature order, and
train-reference statistics in `artifacts/metadata.json`.

## Model and validation

- StandardScaler and multinomial logistic regression in one sklearn Pipeline.
- Hyperparameter selection occurs only inside the training portion using
  stratified five-fold cross-validation.
- The untouched stratified test split reports accuracy, balanced accuracy,
  macro F1, log loss, confusion matrix, and a paired-row bootstrap interval for
  accuracy.

## Limitations

- The dataset is small and old; confidence intervals are correspondingly wide.
- The samples may not represent present-day cultivars, regions, instruments, or
  laboratory procedures.
- Training-range warnings are input diagnostics, not statistical drift proof.
- The in-process metrics endpoint is reset by restart and does not aggregate
  across replicas.
- There is no delayed-label performance monitor because this local educational
  service has no real outcome stream.
- Joblib artifacts use Python object deserialization. Load only artifacts produced
  and integrity-checked by a trusted build pipeline.

## Safe extension path

Before adapting this pattern to a consequential domain, add data governance,
privacy review, representative temporal validation, slice metrics, durable
telemetry, authentication, rate limits, signed artifacts, rollback, and human
oversight appropriate to the decision.
