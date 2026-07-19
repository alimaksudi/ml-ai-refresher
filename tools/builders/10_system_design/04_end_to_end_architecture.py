"""SYS-04 — End-to-End AI Architecture (Capstone) builder."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nbbuild import md, code, build

cells = [

md(r"""
# SYS-04 — End-to-End AI Architecture (Capstone)

## 1. Learning Objectives

This capstone integrates every concept from the 50-notebook curriculum into one coherent picture.

By the end of this notebook you will be able to:

- Draw and explain the six-layer reference architecture for production ML systems
- Compare Lambda, Kappa, and micro-batch architectures and choose between them
- Build a complete mini ML system end-to-end in pure Python/NumPy
- Apply the five-step system design interview framework to any ML problem
- Design the ML subsystems for a ride-sharing platform (forecasting, ETA, surge, matching)
- Articulate the key trade-offs: consistency vs availability, latency vs accuracy, cost vs quality
- Identify which notebook covers each component of the architecture
"""),

md(r"""
## 2. Historical Motivation

### The Path from Prototype to Platform

In 2013, most ML at tech companies lived in Jupyter notebooks run by scientists.
By 2015, Netflix, Uber, and Airbnb independently built bespoke ML platforms.
By 2020, all major companies had converged on similar architectural patterns —
which is why the ML system design interview became a standard senior engineer screen.

**Timeline of ML infrastructure maturity**:

| Era | Pattern | Key Insight |
|-----|---------|------------|
| 2013 | Notebooks + SQL | Science was the bottleneck; infra was an afterthought |
| 2015 | Feature stores emerge (Uber Michelangelo) | Feature reuse across models saves months |
| 2017 | ML pipelines (Airflow, TFX) | Reproducibility requires orchestrated pipelines |
| 2019 | Model registries (MLflow, SageMaker) | Deployment without versioning is chaos |
| 2021 | LLMOps, prompt registries | LLMs introduce new operational primitives |
| 2023 | RAG + agents production patterns | Knowledge + reasoning require new infra |

### What Makes ML Systems Hard

**Ten things that make ML systems different from traditional software**:

1. Code AND data are both first-class artifacts that must be versioned
2. Models degrade silently — there is no crash, just wrong answers
3. Training and serving environments can diverge (train-serve skew)
4. Feedback loops: model affects data which retrains model
5. Ground truth arrives late (labels delayed by days/weeks)
6. Scale of compute: training can cost millions, serving can cost pennies or vice versa
7. Fairness and regulatory requirements on model outputs
8. Reproducibility requires pinning not just code but data snapshots
9. A/B testing model changes is more complex than feature flags for code
10. Debugging requires understanding statistics, not just stack traces
"""),

md(r"""
## 3. The Six-Layer Reference Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 6: MONITORING & GOVERNANCE                             │
│  Drift detection · Error budgets · Fairness · Audit trail    │
├──────────────────────────────────────────────────────────────┤
│  LAYER 5: EVALUATION                                          │
│  Offline eval · A/B testing · Shadow mode · Human review     │
├──────────────────────────────────────────────────────────────┤
│  LAYER 4: SERVING                                             │
│  Model server · Load balancer · Feature cache · Circuit bkr  │
├──────────────────────────────────────────────────────────────┤
│  LAYER 3: TRAINING                                            │
│  Pipeline · Hyperparameter search · Model registry · CI/CD   │
├──────────────────────────────────────────────────────────────┤
│  LAYER 2: FEATURE ENGINEERING                                 │
│  Feature store · Point-in-time joins · Drift monitoring      │
├──────────────────────────────────────────────────────────────┤
│  LAYER 1: DATA                                                │
│  Ingestion · Validation · Storage · Lineage · Catalog        │
└──────────────────────────────────────────────────────────────┘
```

### Curriculum → Architecture Mapping

| Layer | Lessons |
|-------|----------|
| Data | FND-02 (Probability), MLE-02 (Validation), PROD-03 (Feature Stores) |
| Feature Engineering | MLE-03 (Feature Engineering), PROD-03 (Feature Stores) |
| Training | CML-01 through CML-06, DL-01 through DL-08, PROD-04, PROD-01 |
| Serving | SYS-01 (Scalable ML), PROD-02 (LLMOps), SYS-02 (Production RAG) |
| Evaluation | EVAL-01 through EVAL-05, including EVAL-03 for RAG |
| Monitoring | PROD-05 (Drift), PROD-06 (Retraining), SYS-03 (Reliability) |
"""),

md(r"""
## 4. Mathematical Foundations

### 4.1 Lambda vs Kappa Architecture

**Lambda**: Batch layer (accurate, slow) + Speed layer (fast, approximate) + Serving layer (merge)
- Pros: high accuracy from batch; freshness from speed layer
- Cons: two code paths → maintenance burden (the "lambda trap")

**Kappa**: Stream-only; reprocess historical data by replaying the stream
- Pros: one code path; simpler ops
- Cons: historical reprocessing is expensive; harder to debug

**Micro-batch** (Spark Structured Streaming): periodic batch at 1–60s intervals
- Pros: nearly real-time; batch semantics; easy to reason about
- Cons: minimum latency ≈ batch interval

### 4.2 System Design Scale Estimation

| Metric | Formula |
|--------|---------|
| RPS → daily requests | RPS × 86,400 |
| Storage (1 year) | requests/day × record_size_bytes × 365 |
| Bandwidth | RPS × avg_payload_bytes |
| Replicas needed | ceil(RPS / capacity_per_instance × (1 + headroom)) |

At 100k RPS, 1KB payload: 100 MB/s inbound bandwidth.

### 4.3 Feedback Loop Stability

If model output influences data (recommendation → click → training label):

$$P(\text{label} | x, t+1) = f(P(\text{model output} | x, t))$$

This is a dynamical system. If the feedback gain exceeds 1, it diverges (popularity collapse,
filter bubbles). Monitor for **exposure bias** and diversity metrics.
"""),

md(r"""
## 5. Manual Implementation from Scratch — Mini End-to-End System
"""),

code(r"""
import numpy as np
import math
import hashlib
import time
import collections
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

rng_global = np.random.default_rng(42)

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1: DATA GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_dataset(n: int = 2000, drift_at: int = 1500, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        drifted = i >= drift_at
        # Feature distribution shifts at drift_at
        f1 = rng.normal(1.0 if drifted else 0.0, 1.0)
        f2 = rng.normal(0.0, 1.5 if drifted else 1.0)
        f3 = rng.normal(0.5, 1.0)
        # Label: P(y=1) changes at drift (concept drift)
        if drifted:
            logit = 0.5 * f1 - 1.2 * f2 + 0.3 * f3
        else:
            logit = 1.2 * f1 + 0.8 * f2 - 0.4 * f3
        prob = 1 / (1 + math.exp(-logit))
        y = int(rng.random() < prob)
        records.append({"id": i, "f1": f1, "f2": f2, "f3": f3, "y": y,
                        "timestamp": 1_700_000_000 + i * 60})
    return records

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2: FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
class SimpleFeatureStore:
    def __init__(self):
        self._features: Dict[str, np.ndarray] = {}
        self._stats: Dict[str, Dict] = {}

    def fit(self, records: List[Dict], feature_cols: List[str]):
        for col in feature_cols:
            vals = np.array([r[col] for r in records])
            self._stats[col] = {"mean": float(vals.mean()), "std": float(vals.std())}

    def transform(self, records: List[Dict], feature_cols: List[str]) -> np.ndarray:
        rows = []
        for r in records:
            row = []
            for col in feature_cols:
                v = r[col]
                mu = self._stats[col]["mean"]
                sigma = self._stats[col]["std"]
                row.append((v - mu) / max(sigma, 1e-9))
            rows.append(row)
        return np.array(rows)


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3: TRAINING + MODEL REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def train_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 50) -> Dict:
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        p   = sigmoid(X @ w + b)
        err = p - y
        w  -= lr * (X.T @ err) / len(y)
        b  -= lr * err.mean()
    return {"w": w, "b": b}

def predict(model: Dict, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ model["w"] + model["b"])

def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    total = len(pos) * len(neg)
    return float(sum(p > n for p in pos for n in neg) / total)

class ModelRegistry:
    def __init__(self):
        self._models: List[Dict] = []
        self._current_idx: Optional[int] = None

    def register(self, model: Dict, metrics: Dict, name: str) -> int:
        version = len(self._models)
        entry = {"version": version, "name": name, "model": model, "metrics": metrics}
        self._models.append(entry)
        return version

    def promote(self, version: int):
        self._current_idx = version
        print(f"  [Registry] Promoted model v{version}: {self._models[version]['metrics']}")

    def current(self) -> Optional[Dict]:
        if self._current_idx is None:
            return None
        return self._models[self._current_idx]["model"]

    def rollback(self):
        if self._current_idx and self._current_idx > 0:
            self._current_idx -= 1
            print(f"  [Registry] Rolled back to v{self._current_idx}")


# ──────────────────────────────────────────────────────────────────────────────
# RUN THE TRAINING LAYER
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = ["f1", "f2", "f3"]
records   = generate_dataset(n=2000, drift_at=1500)
train_recs = records[:1200]
val_recs   = records[1200:1500]
prod_recs  = records[1500:]   # drifted data

fs = SimpleFeatureStore()
fs.fit(train_recs, FEATURES)

X_train = fs.transform(train_recs, FEATURES)
y_train = np.array([r["y"] for r in train_recs])
X_val   = fs.transform(val_recs, FEATURES)
y_val   = np.array([r["y"] for r in val_recs])

model_v1 = train_logistic(X_train, y_train)
auc_v1   = compute_auc(predict(model_v1, X_val), y_val)

registry = ModelRegistry()
v = registry.register(model_v1, {"auc_val": round(auc_v1, 4)}, "fraud_model")
registry.promote(v)
print(f"Model v{v} registered and promoted: val AUC = {auc_v1:.4f}")
"""),

code(r"""
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 4: SERVING + LOAD BALANCER
# ──────────────────────────────────────────────────────────────────────────────
class ServingLayer:
    def __init__(self, registry: ModelRegistry, feature_store: SimpleFeatureStore,
                 feature_cols: List[str]):
        self._registry = registry
        self._fs       = feature_store
        self._cols     = feature_cols
        self._latencies: List[float] = []

    def predict_batch(self, records: List[Dict]) -> np.ndarray:
        import time
        t0 = time.perf_counter()
        model = self._registry.current()
        if model is None:
            return np.full(len(records), 0.3)
        X   = self._fs.transform(records, self._cols)
        scores = predict(model, X)
        self._latencies.append((time.perf_counter() - t0) * 1000)
        return scores

    def p99_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        arr = sorted(self._latencies)
        return arr[min(int(math.ceil(0.99 * len(arr))) - 1, len(arr) - 1)]


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 5: DRIFT MONITORING + RETRAINING TRIGGER
# ──────────────────────────────────────────────────────────────────────────────
def compute_psi_simple(ref: np.ndarray, cur: np.ndarray, n_bins: int = 8) -> float:
    eps = 1e-6
    combined_min, combined_max = min(ref.min(), cur.min()), max(ref.max(), cur.max())
    edges = np.linspace(combined_min - eps, combined_max + eps, n_bins + 1)
    ref_c, _ = np.histogram(ref, bins=edges)
    cur_c, _ = np.histogram(cur, bins=edges)
    ref_f = np.maximum(ref_c / ref_c.sum(), eps)
    cur_f = np.maximum(cur_c / cur_c.sum(), eps)
    return float(((cur_f - ref_f) * np.log(cur_f / ref_f)).sum())

class DriftTrigger:
    def __init__(self, psi_threshold: float = 0.2):
        self._threshold = psi_threshold
        self.triggered  = False

    def check(self, ref_scores: np.ndarray, cur_scores: np.ndarray) -> bool:
        psi = compute_psi_simple(ref_scores, cur_scores)
        print(f"  [Drift] PSI = {psi:.4f}  {'⚠ TRIGGER RETRAIN' if psi > self._threshold else 'OK'}")
        self.triggered = psi > self._threshold
        return self.triggered


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 6: EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
class EvaluationGate:
    def __init__(self, min_auc: float = 0.70, min_improvement: float = 0.005):
        self._min_auc = min_auc
        self._min_imp = min_improvement

    def approve(self, new_auc: float, current_auc: float) -> bool:
        if new_auc < self._min_auc:
            print(f"  [Gate] REJECT: new AUC {new_auc:.4f} < minimum {self._min_auc}")
            return False
        if new_auc < current_auc - self._min_imp:
            print(f"  [Gate] REJECT: new AUC {new_auc:.4f} worse than current {current_auc:.4f}")
            return False
        print(f"  [Gate] APPROVE: new AUC {new_auc:.4f} vs current {current_auc:.4f}")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# FULL END-TO-END RUN
# ──────────────────────────────────────────────────────────────────────────────
serving = ServingLayer(registry, fs, FEATURES)
drift_trigger = DriftTrigger(psi_threshold=0.2)
eval_gate     = EvaluationGate(min_auc=0.70)

# Score pre-drift data (reference)
ref_scores = serving.predict_batch(val_recs)

print("\n=== Month 1-5: Normal serving ===")
normal_scores = serving.predict_batch(records[900:1200])
drift_trigger.check(ref_scores, normal_scores)

print("\n=== Month 6: Drift detected in production ===")
drifted_scores = serving.predict_batch(prod_recs[:200])
triggered = drift_trigger.check(ref_scores, drifted_scores)

if triggered:
    print("\n=== Retraining pipeline triggered ===")
    # Retrain on recent data (sliding window)
    recent_train  = records[1200:1800]
    X_new   = fs.transform(recent_train, FEATURES)
    y_new   = np.array([r["y"] for r in recent_train])
    X_val2  = fs.transform(prod_recs[200:400], FEATURES)
    y_val2  = np.array([r["y"] for r in prod_recs[200:400]])

    model_v2 = train_logistic(X_new, y_new)
    auc_v2   = compute_auc(predict(model_v2, X_val2), y_val2)
    auc_v1_on_new = compute_auc(predict(model_v1, X_val2), y_val2)

    approved = eval_gate.approve(auc_v2, auc_v1_on_new)
    if approved:
        v2 = registry.register(model_v2, {"auc_val": round(auc_v2, 4)}, "fraud_model_retrain")
        registry.promote(v2)

print(f"\nFinal serving layer p99 latency: {serving.p99_latency_ms():.3f}ms")
"""),

md(r"""
## 6. Visualization — Architecture Overview
"""),

code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

rng_v = np.random.default_rng(2024)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("End-to-End AI Architecture — Capstone", fontsize=15, fontweight='bold')

# ── (a) Architecture layers diagram ───────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
layers = [
    ("Data Layer",          "Ingestion · Validation · Storage",       "#2196F3", 1.0),
    ("Feature Layer",       "Feature Store · PIT Joins · Drift",       "#4CAF50", 2.8),
    ("Training Layer",      "Pipeline · HP Search · Model Registry",   "#FF9800", 4.6),
    ("Serving Layer",       "Model Server · LB · Cache · CB",          "#9C27B0", 6.4),
    ("Evaluation Layer",    "Offline Eval · A/B · Shadow Mode",        "#F44336", 8.2),
    ("Monitoring Layer",    "Drift · Error Budget · Fairness",         "#009688", 10.0),
]
for label, sub, color, y in layers:
    rect = mpatches.FancyBboxPatch((0.3, y), 9.4, 1.5, boxstyle="round,pad=0.1",
                                   facecolor=color, alpha=0.25, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(5.0, y + 0.85, label, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.0, y + 0.35, sub,  ha='center', va='center', fontsize=7.5, color='#333333')

ax.set_title("Reference Architecture\n(6 Layers)", fontsize=11)

# ── (b) End-to-end AUC timeline with retrain event ────────────────────────────
ax = axes[1]
months = list(range(1, 13))
# Simulate AUC: stable, drops at drift (month 6), recovers after retrain (month 7)
auc_curve = [0.89, 0.88, 0.89, 0.87, 0.88, 0.75, 0.85, 0.87, 0.88, 0.89, 0.88, 0.89]
ax.plot(months, auc_curve, 'o-', color='steelblue', linewidth=2, markersize=6, label='Model AUC')
ax.axhline(0.80, color='tomato', linestyle='--', linewidth=1.5, label='Min AUC threshold')
ax.axvspan(6, 7, alpha=0.15, color='tomato', label='Drift window')
ax.annotate('Drift detected\n→ retrain triggered', xy=(6, 0.75), xytext=(7.5, 0.72),
            arrowprops=dict(arrowstyle='->', color='grey'), fontsize=8)
ax.annotate('New model\npromoted', xy=(7, 0.85), xytext=(8.5, 0.82),
            arrowprops=dict(arrowstyle='->', color='grey'), fontsize=8)
ax.set_xlabel("Month")
ax.set_ylabel("AUC")
ax.set_title("Production AUC Timeline:\nDrift → Retrain → Recovery", fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(0.65, 0.95)
# AUC drops at month 6 (concept drift from competitor launch); recovers after retrain at month 7.

# ── (c) System design trade-off radar ────────────────────────────────────────
ax = axes[2]
dimensions = ['Latency', 'Throughput', 'Accuracy', 'Cost\nEfficiency', 'Operational\nSimplicity']
N = len(dimensions)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

profiles = {
    "Single server":     [0.6, 0.3, 0.9, 0.9, 1.0],
    "Distilled + INT8":  [0.9, 0.9, 0.7, 0.8, 0.6],
    "Full RAG + GPT-4":  [0.5, 0.4, 1.0, 0.3, 0.5],
}
colors_r = ['steelblue', 'tomato', 'green']

for (name, vals), color in zip(profiles.items(), colors_r):
    vals_closed = vals + vals[:1]
    ax.plot(angles, vals_closed, 'o-', color=color, linewidth=1.5, label=name)
    ax.fill(angles, vals_closed, color=color, alpha=0.07)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dimensions, fontsize=8)
ax.set_ylim(0, 1)
ax.set_title("Architecture Trade-off Radar\n(5 dimensions)", fontsize=11)
ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.35, 1.1))
# Each profile trades off differently; no single architecture wins on all dimensions.

plt.tight_layout()
plt.savefig('/tmp/nb04_end_to_end_architecture.png', dpi=80, bbox_inches='tight')
plt.show()
print("Capstone architecture figure saved.")
"""),

md(r"""
## 7. The System Design Interview Framework

### Five Steps for Any ML System Design Question

**Step 1 — Clarify Requirements (5 min)**
- Functional: what predictions? what latency? online or batch?
- Non-functional: scale (RPS), accuracy target, data freshness
- Constraints: budget, team size, latency SLO

**Step 2 — Estimate Scale (3 min)**
- Users → requests/day → RPS
- Data volume → storage, bandwidth
- Model size × batch size → memory per instance

**Step 3 — Design Core Components (15 min)**
- Data layer: sources, validation, storage
- Feature layer: online vs offline, point-in-time
- Training: algorithm, pipeline, cadence
- Serving: batch vs real-time, compression
- Evaluation: metrics, A/B test design

**Step 4 — Handle Failures (5 min)**
- What breaks? Fallback? Circuit breaker?
- What drifts? Monitoring + retraining trigger
- SLO/error budget?

**Step 5 — Evolution (2 min)**
- MVP → V2 improvements
- What would you do with 10x more data?
- What changes at 100x scale?
"""),

md(r"""
## 8. Five Canonical ML System Designs

### 1. Recommendation Engine
**Components**: user/item embeddings (NLP-02), collaborative filtering (CML-04),
feature store (PROD-03), candidate generation → ranking → re-ranking pipeline,
A/B testing (PROD-04), feedback loop monitoring

**Key trade-off**: diversity vs relevance; exploration vs exploitation

### 2. Fraud Detection
**Components**: real-time features (PROD-03), gradient boosting (CML-05), model serving (SYS-01),
circuit breaker + fallback (SYS-03), drift monitoring (PROD-05), retraining on delay-labelled data (PROD-06)

**Key trade-off**: precision vs recall; latency (real-time) vs accuracy (batch)

### 3. Search Ranking
**Components**: BM25 (SYS-02), dense retrieval (RAG-01), re-ranking (RAG-07),
query understanding (NLP-04), learning-to-rank (MLE-01), click-through rate prediction

**Key trade-off**: freshness vs quality; exploration vs exploitation in serving

### 4. Content Moderation
**Components**: multi-modal classification (DL-05 CNN for images, NLP-03 for text),
human evaluation loop (EVAL-04), guardrails (NLP-05), appeal pipeline, fairness monitoring

**Key trade-off**: false positives (over-moderate) vs false negatives (miss harmful content)

### 5. Demand Forecasting
**Components**: time-series (DL-06 LSTM, classical ARIMA), feature engineering (MLE-03),
evaluation (EVAL-01 RMSE/MAPE), uncertainty quantification, hierarchical reconciliation

**Key trade-off**: model complexity vs interpretability; global vs local models
"""),

md(r"""
## 9. Business Case Study — Ride-Sharing Platform ML Systems

**Systems to design**: Demand Forecasting, ETA, Surge Pricing, Driver Matching
"""),

code(r"""
import numpy as np
import math

rng_rs = np.random.default_rng(7)

# ── Demand Forecasting ────────────────────────────────────────────────────────
def forecast_demand(hour: int, day_of_week: int, weather_rain: bool, event: bool) -> float:
    base     = 120.0
    hour_mul = 1.8 if 7 <= hour <= 9 or 17 <= hour <= 19 else (0.4 if hour < 6 else 1.0)
    wknd_mul = 1.3 if day_of_week >= 5 else 1.0
    rain_mul = 1.4 if weather_rain else 1.0
    event_mul = 2.5 if event else 1.0
    noise = rng_rs.normal(0, 8)
    return max(0, base * hour_mul * wknd_mul * rain_mul * event_mul + noise)

# ── ETA Model ─────────────────────────────────────────────────────────────────
def predict_eta(distance_km: float, hour: int, demand: float) -> float:
    traffic_factor = 1.0 + 0.8 * max(0, demand / 200 - 0.5)  # congestion
    speed_kmh = 40.0 / traffic_factor
    base_min = distance_km / speed_kmh * 60
    return base_min + rng_rs.exponential(1.5)

# ── Surge Pricing ─────────────────────────────────────────────────────────────
def surge_multiplier(demand: float, supply: float) -> float:
    ratio = demand / max(supply, 1)
    if ratio < 0.8:
        return 1.0
    elif ratio < 1.2:
        return 1.0 + (ratio - 0.8) * 1.25
    elif ratio < 2.0:
        return 1.5 + (ratio - 1.2) * 0.6
    else:
        return min(3.5, 2.0 + (ratio - 2.0) * 0.25)

# ── Driver Matching ───────────────────────────────────────────────────────────
def match_score(driver_eta: float, driver_rating: float,
                rider_pref_premium: bool, driver_is_premium: bool) -> float:
    eta_score    = math.exp(-driver_eta / 5.0)
    rating_score = (driver_rating - 4.0) / 1.0
    pref_score   = 0.2 if (rider_pref_premium == driver_is_premium) else -0.1
    return eta_score + 0.3 * rating_score + pref_score


# ── Simulate one hour of operations ──────────────────────────────────────────
N_RIDERS  = 50
N_DRIVERS = 40
hour, dow  = 8, 1  # Monday 8am

demand = forecast_demand(hour, dow, weather_rain=True, event=False)
supply = N_DRIVERS * 0.75  # 75% available
surge  = surge_multiplier(demand, supply)

print(f"Demand forecast: {demand:.0f} rides/hr  (hour={hour}, rain=True)")
print(f"Supply (drivers): {supply:.0f}  →  surge multiplier: {surge:.2f}x")

# Simulate matching
driver_etas     = rng_rs.exponential(4, N_DRIVERS)     # minutes to pickup
driver_ratings  = rng_rs.uniform(4.0, 5.0, N_DRIVERS)
driver_premium  = rng_rs.random(N_DRIVERS) < 0.3

matched, unmatched = 0, 0
wait_times = []
for rider in range(N_RIDERS):
    rider_pref = rng_rs.random() < 0.25  # 25% prefer premium
    scores = [match_score(driver_etas[d], driver_ratings[d], rider_pref, driver_premium[d])
              for d in range(N_DRIVERS)]
    best_driver = int(np.argmax(scores))
    wait_times.append(driver_etas[best_driver])
    matched += 1

print(f"\nMatching results ({N_RIDERS} riders, {N_DRIVERS} drivers):")
print(f"  Matched:     {matched}")
print(f"  Avg wait:    {np.mean(wait_times):.1f} min")
print(f"  p95 wait:    {np.percentile(wait_times, 95):.1f} min")

avg_eta = predict_eta(distance_km=4.5, hour=hour, demand=demand)
base_fare = 8.00
surge_fare = base_fare * surge
print(f"\nETA for 4.5km trip:  {avg_eta:.1f} min")
print(f"Base fare:           ${base_fare:.2f}")
print(f"Surge fare ({surge:.2f}x): ${surge_fare:.2f}")
"""),

md(r"""
## 10. Production Considerations

### The ML Platform Maturity Model

| Level | Description | What you have |
|-------|-------------|--------------|
| 0 | Notebooks in production | Jupyter running as cron jobs |
| 1 | Scripted pipelines | Python scripts, manual deploy |
| 2 | Automated training | Triggered retraining, basic registry |
| 3 | Automated deployment | Shadow mode, canary, rollback |
| 4 | Continuous everything | Drift triggers retrain, auto-promote, full observability |
| 5 | Self-healing | Adaptive sampling, online learning, auto-architecture search |

Most companies are at level 2–3. Level 4+ requires significant investment.
Don't over-engineer: right-size the platform to team size and model count.

### The 10 Non-Negotiables for Production ML

1. Version everything: code, data, features, models
2. Monitor everything: latency, accuracy, data quality, business metrics
3. Fallback for everything: every ML system must degrade gracefully
4. Test data pipelines, not just models
5. Shadow-test every model before promoting
6. Measure business impact, not just ML metrics
7. Document model cards and data sheets
8. Alert on burn rate, not just error rate
9. Practice chaos engineering quarterly
10. Delete models that aren't used — stale models become liabilities
"""),

md(r"""
## 11. Tradeoff Analysis — The Three Great Trade-offs

### Consistency vs Availability (data layer)

| Choice | Consistency | Availability | Example |
|--------|------------|--------------|---------|
| Synchronous writes | High | Lower | Bank transaction features |
| Async replication | Lower | Higher | Recommendation features |
| Eventual consistency | Low | Highest | Ad click features |

### Latency vs Accuracy (serving layer)

| Choice | Latency | Accuracy | Example |
|--------|---------|---------|---------|
| Cached prediction | < 1ms | Lower (stale) | Static recommendation |
| Pre-computed (near-real-time) | 5ms | Medium | Daily-updated ranking |
| Real-time inference | 50–200ms | Highest | Fraud detection |

### Cost vs Quality (model tier)

| Choice | Cost/query | Quality | When |
|--------|-----------|---------|------|
| Rule-based | $0.000001 | Low | Simple/high-volume |
| Small fine-tuned LM | $0.00005 | Good | Standard tasks |
| GPT-4o / Sonnet 4.6 | $0.005 | Excellent | Complex/low-volume |
| Ensemble | $0.015 | Best | Highest-stakes |
"""),

md(r"""
## 12. Senior-Level Interview Preparation

**Q1**: Design a recommendation system for 100M users and 10M items.

> Candidate generation: matrix factorisation / two-tower model → 1000 candidates (fast, approximate). Ranking: gradient boosting on user-item features → top 50. Re-ranking: diversity constraints + business rules → top 10. Feature store: user history (offline batch), real-time events (streaming). Monitoring: exposure bias, diversity metrics, A/B test on CTR and dwell time.

**Q2**: What is the "hidden technical debt" in ML systems?

> (1) Unstable data dependencies — upstream schema changes silently break features. (2) Entangled model pipelines — features shared across models create coupling. (3) Correction cascades — downstream models trained on upstream model outputs. (4) Undeclared consumers — other teams consume your model outputs without contracts. (5) Dead experimental code paths — abandoned feature flags accumulate.

**Q3**: How do you debug a model whose accuracy dropped 10% in production overnight?

> Triage in order: (1) data pipeline — did features arrive? Correct schema? (2) serving — same model version as yesterday? (3) input distribution — PSI on key features. (4) label quality — correct labelling logic? (5) seasonality — is yesterday the same day-of-week as a week ago? (6) upstream model changes — any dependency retrained?

**Q4**: What is the lambda architecture and what is the lambda trap?

> Lambda = batch layer (accurate, nightly) + speed layer (fast, approximate, real-time) + serving layer (merge both). The trap: two separate code paths for the same logic. The batch layer recomputes "correct" results; the speed layer computes approximate results. Any bug must be fixed in two places. Kappa architecture solves this by using only a stream that can be replayed.

**Q5**: How would you A/B test a new ML model in production?

> (1) Define the business outcome and minimum useful effect before launch. (2) Use
> a power calculation matching that outcome—for binary conversion, a two-proportion
> design. (3) Randomise by user ID, not request. (4) Run long enough to cover known
> seasonality without optional peeking. (5) Check sample-ratio mismatch, guardrails,
> confidence interval, and effect size. (6) Promote only when evidence clears both
> the statistical and practical thresholds.

**Q6**: What are the five components of a model card?

> (1) Model description (task, algorithm). (2) Training data (source, size, date range). (3) Performance metrics (overall and by demographic slice). (4) Intended use and limitations. (5) Ethical considerations and bias assessment. Model cards enable audit and regulatory compliance.

**Q7**: How do you handle a feedback loop in a recommendation system?

> Exposure bias: items never shown → never clicked → never in training data → never recommended (popularity collapse). Mitigations: (1) explore-exploit (epsilon-greedy, Thompson sampling). (2) Inverse-propensity weighting using logged exposure probabilities—rare exposures receive larger weights, with clipping to control variance. (3) Monitor diversity metrics (catalog coverage, ILS). (4) Periodically explore under-served items.

**Q8**: A customer reports that your fraud model is flagging their transactions more than competitors' customers. How do you investigate?

> Fairness audit: (1) compute FPR by demographic slice (if available). (2) Check feature proxies that correlate with protected attributes (zip code, spend patterns). (3) Run disparate impact analysis (80% rule or statistical parity). (4) Compute equalised odds: equate TPR and FPR across groups. Fix: fairness constraints in training (adversarial de-biasing, re-weighting), or post-processing threshold adjustment per group.
"""),

md(r"""
## 13. Teach-Back Section

1. Name the six layers of the ML reference architecture and one key component from each.
2. What is the lambda trap and how does Kappa architecture avoid it?
3. Explain the five-step system design interview framework in 2 minutes.
4. For each canonical ML system (recommendation, fraud, search, moderation, forecasting), name the most important evaluation metric.
5. What is the difference between a train-serve skew and data drift?
6. At 100M users and 1ms SLO, why can't you run a full model inference per request?
7. Define a feedback loop in ML. Give an example from recommendation systems.
8. What is a model card and why do senior ML engineers care about them?
"""),

md(r"""
## 14. Exercises

### Beginner
1. Draw the six-layer architecture for a spam detection system. For each layer, name the specific technology or component you would use.
2. Compute the number of instances needed to serve 200k RPS with each instance handling 5k RPS, 30% headroom.
3. What is the error budget (in minutes/month) for a 99.95% SLO?

### Intermediate
4. Extend the mini end-to-end system with two separate comparisons: (a) offline,
   score both models on the same untouched examples and use PROD-04's paired bootstrap
   for the AUC difference; (b) online, randomise users 50/50 and compare a binary
   business outcome with a two-proportion confidence interval. Do not compare one
   AUC value per arm with a t-test.
5. Add **calibration monitoring** to the serving layer: track the fraction of predictions in each decile bucket and alert if the distribution shifts > 20% from the training distribution.
6. Implement a **retraining scheduler** that combines drift-based triggering (PSI > 0.2) with schedule-based triggering (weekly) using an EWMA controller. The weekly trigger fires even if PSI is fine.

### Senior
7. Design the complete ML system for a **content moderation platform**: 10M posts/day, < 200ms classification, 95% recall on harmful content, < 1% FPR, human review queue for edge cases. Include all six architecture layers.
8. Implement an **online A/B test power calculator**: given historical conversion
   rate, minimum detectable effect, allocation, and eligible users per day, compute
   duration for 80% power using PROD-01's two-proportion approximation. State the
   independence assumption and adjust the unit when users generate repeated requests.
9. Implement a **model lineage tracker**: for each model version in the registry, record the training data hash, parent model version, hyperparameters, and evaluation metrics. Given a production incident, trace back through the lineage to find which data version or parent model may have introduced the regression.

---

## System Design Complete — Capstone Next

You have completed the architecture sequence. Finish **CAP-01** next to demonstrate
the full workflow in a deployable vertical slice.

### What You Have Built

| Section | Topic | Lesson IDs |
|---------|-------|------------|
| 00 | Prerequisites | PRE-01 through PRE-06 |
| 01 | ML Foundations | FND-01 through FND-04 |
| 02 | Classical Machine Learning | CML-01 through CML-06 |
| 03 | ML Engineering | MLE-01 through MLE-06 |
| 04 | Deep Learning | DL-01 through DL-08 |
| 05 | NLP and LLMs | NLP-01 through NLP-05 |
| 06 | Retrieval-Augmented Generation | RAG-01 through RAG-08 |
| 07 | AI Agents | AGT-01 through AGT-05 |
| 08 | Evaluation | EVAL-01 through EVAL-05 |
| 09 | Production ML | PROD-01 through PROD-06 |
| 10 | AI System Design | SYS-01 through SYS-04 |
| 11 | Capstone | CAP-01 |

### What's Next

- **Practice system design interviews**: use the five-step framework on new problems
- **Implement in a real stack**: take this curriculum's scratch implementations and replicate them with PyTorch, scikit-learn, and cloud services
- **Contribute back**: identify gaps, extend exercises, add domain-specific notebooks
- **Complete CAP-01**: prove the end-to-end workflow with tests and deployment contracts
- **Apply to real problems**: pick one system from each section and deploy it
"""),

]  # end cells

if __name__ == "__main__":
    build("10_system_design/04_end_to_end_architecture.ipynb", cells)
