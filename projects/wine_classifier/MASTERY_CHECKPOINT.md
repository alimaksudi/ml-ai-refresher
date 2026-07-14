# Classical ML Mastery Checkpoint

This is the assessed gate after the revised Foundations → Classical ML path. The
goal is not merely to obtain a high score. The learner must demonstrate that the
score comes from a valid, reproducible decision process.

## Scenario

Build a service that predicts the cultivar class of a wine from 13 chemical
measurements. The prediction supports laboratory triage and dataset exploration;
it must not be presented as a quality, price, authenticity, or safety decision.

## Required evidence

| Competency | Required artifact | Pass condition |
|---|---|---|
| Problem framing | Model metadata and model card | Prediction unit, target, decision, intended use, prohibited use, and error implications are explicit. |
| Baseline | Training metadata | Majority-class baseline is recorded and the learned model beats it on the untouched test set. |
| Data contract | Feature schema and dataset hash | Names, order, sample counts, and immutable data fingerprint are recorded. |
| Leakage control | Training code | Test set is split once; scaling is inside the pipeline; CV/tuning sees training data only. |
| Metric selection | Evaluation metadata | Accuracy, macro F1, balanced accuracy, log loss, confusion matrix, and an uncertainty interval are reported. |
| Reproducibility | Training metadata | Seed, split strategy, candidate hyperparameters, best parameters, model version, and timestamp are recorded. |
| Model comparison | Experiment record | Baseline and candidate are compared; improvement must be practically meaningful. |
| Explainability | Model card or report | Coefficients and their limits are explained; no causal claims are made. |
| Software quality | Tests and package | Training/artifact/API contracts pass; invalid schemas fail safely. |
| Production reasoning | API, monitoring, model card | Health, metadata, metrics, range warnings, and limitations are explicit. |

## Assessment

Run from the repository root:

```bash
make capstone-train
make capstone-test
make mastery-checkpoint
```

The automated gate verifies objective contracts. A human reviewer scores these
teach-back questions from 0–2 each:

1. Why is the split valid for the prediction unit?
2. Why is macro F1 reported alongside accuracy?
3. What does log loss reveal that accuracy hides?
4. Why does scaling belong inside the pipeline?
5. What does the confidence interval mean, and what does it not mean?
6. Why do coefficients not establish causality?
7. What real-world change would invalidate this model?
8. What monitoring signal should trigger investigation rather than automatic retraining?

Pass requires all automated tests, at least **12/16** on the review, and no zero on
questions 1, 4, or 6. Remediate failed items before proceeding to deep learning.
