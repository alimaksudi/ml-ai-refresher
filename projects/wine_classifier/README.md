# Deployable Wine Classifier Capstone

This project is the concrete vertical slice behind Notebook 51. It uses a real
dataset bundled with scikit-learn and keeps preprocessing, training, evaluation,
artifact metadata, serving, monitoring, and tests in one reviewable path.

It also serves as the first assessed gate after Classical ML. Complete the
[mastery checkpoint](MASTERY_CHECKPOINT.md) before continuing to Deep Learning.

## Run locally

From the repository root:

```bash
PYTHONPATH=projects/wine_classifier/src python -m wine_classifier.training \
  --artifact-dir projects/wine_classifier/artifacts

PYTHONPATH=projects/wine_classifier/src uvicorn wine_classifier.app:app \
  --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated API schema.

## Test

```bash
PYTHONPATH=projects/wine_classifier/src \
  pytest projects/wine_classifier/tests -q
```

## Request example

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315_of_diluted_wines": 3.92,
    "proline": 1065
  }'
```

## Production boundaries

The service intentionally exposes what a notebook-only demo usually omits:

- strict feature names and order;
- versioned model and metadata artifacts;
- immutable public model metadata;
- training-range warnings;
- an explicit monitoring limitation statement;
- health, prediction, metadata, and metrics endpoints;
- API and artifact contract tests;
- a container entry point.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and limitations.
