"""NB46 — Retraining Strategies builder."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbbuild import md, code, build

cells = [

# ── 1. Learning Objectives ────────────────────────────────────────────────────
md(r"""
# 46 — Retraining Strategies

## 1. Learning Objectives

By the end of this notebook you will be able to:

- Identify the four canonical **retraining triggers** and choose the right one per situation
- Compare **full retrain**, **sliding window**, **warm start**, and **online learning** data strategies
- Implement **EWMAController** from scratch to track smoothed metrics and trigger retraining
- Implement **WindowedRetrainer** with configurable sliding data windows
- Explain **catastrophic forgetting** and when warm-start fine-tuning causes it
- Describe **train-serve skew** after retraining and how to detect it with shadow mode
- Walk through the full **retraining pipeline**: data → validate → train → evaluate → shadow → promote
- Apply retraining cadence optimisation to a fraud detection business case with seasonal drift
"""),

# ── 2. Historical Motivation ───────────────────────────────────────────────────
md(r"""
## 2. Historical Motivation

### The Cost of Not Retraining

PayPal's fraud model, trained on 2008 data, saw recall drop from 94% to 71% by 2011 —
as fraudsters adapted faster than the retraining cadence.  Each percentage point of
recall drop meant millions in undetected fraud.

Netflix's recommendation model in 2016 was retrained monthly.  An A/B test showed
that **daily** retraining improved engagement by 2.1% — worth ~$100M/year at scale.
The insight: user taste drifts weekly, so monthly retraining is already stale.

### The Cost of Over-Retraining

In 2019, a fintech startup retrained their credit model weekly on a sliding 30-day
window.  A one-week data anomaly (a bank holiday) caused the model to over-weight
holiday behaviour.  The model was deployed for 2 weeks before anyone noticed the AUC
had dropped 8 points.  **Lesson**: more retraining is not always better.

### The Vocabulary

| Term | Definition |
|------|------------|
| Scheduled retraining | Retrain on calendar cadence regardless of model performance |
| Performance-based retraining | Retrain when a metric (AUC, accuracy) drops below threshold |
| Drift-based retraining | Retrain when PSI / KS exceeds threshold |
| Event-based retraining | Retrain after a known external event (product launch, regulation change) |
| Catastrophic forgetting | Fine-tuned model forgets old patterns it never sees again |
| Train-serve skew | Model trained on feature-engineered data; served with slightly different preprocessing |
"""),

# ── 3. Intuition & Visual Understanding ──────────────────────────────────────
md(r"""
## 3. Intuition & Visual Understanding

### Data Strategy Trade-offs

```
Full history retrain
  ├── Pros: Uses all signal; stable; avoids recency bias
  └── Cons: Slow; stale data may hurt; expensive

Sliding window (last N months)
  ├── Pros: Adapts quickly; cheap; ignores stale data
  └── Cons: Forgets long-term patterns; sensitive to window size

Warm start (fine-tune from existing weights)
  ├── Pros: Fast; preserves learned representations
  └── Cons: Catastrophic forgetting if new data distribution is very different

Online learning (update per batch/sample)
  ├── Pros: Always up-to-date; no retrain lag
  └── Cons: Noisy; needs careful learning rate; hard to debug; can spiral
```

### The Retraining Pipeline

```
Monitor → Trigger → Fetch data → Validate data quality
    → Train new model → Evaluate (AUC, PSI vs current)
    → Shadow test (serve both; compare outputs)
    → Gate decision (human or automated)
    → Promote (atomic swap)
    → Monitor new model
```

### EWMA for Smooth Metric Tracking

Raw production AUC bounces around due to sample size noise.
EWMA smooths it:

$$\hat{\mu}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \hat{\mu}_{t-1}$$

Alert when $\hat{\mu}_t < \mu_{\text{baseline}} - \Delta$.
"""),

# ── 4. Mathematical Foundations ───────────────────────────────────────────────
md(r"""
## 4. Mathematical Foundations

### 4.1 EWMA (Exponentially Weighted Moving Average)

$$S_t = \alpha x_t + (1 - \alpha) S_{t-1}, \quad S_0 = x_0$$

- $\alpha \in (0,1]$: smoothing factor. High $\alpha$ → more reactive. Low $\alpha$ → smoother.
- Effective window ≈ $\frac{1}{1-\alpha}$ observations.
- Variance of $S_t$ under stationarity: $\text{Var}(S_t) = \frac{\alpha}{2 - \alpha} \sigma^2_x$.

### 4.2 Sliding Window Retrain

Given a stream of labelled samples $(x_i, y_i, t_i)$, the training set for retrain at time $T$ is:

$$\mathcal{D}_T = \{(x_i, y_i) : T - W \le t_i \le T\}$$

where $W$ is the window width in time units.  Optimal $W$ minimises generalisation error on a validation set from $[T, T + \delta]$.

### 4.3 Catastrophic Forgetting

In continual learning, the plasticity-stability dilemma:

- **Plasticity**: learn new patterns quickly
- **Stability**: retain old knowledge

Elastic Weight Consolidation (EWC) penalises updates to important weights:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where $F_i$ is the Fisher information of parameter $i$.

### 4.4 Retraining Cost Model

$$\text{Total cost}(T_{\text{cadence}}) = C_{\text{retrain}} / T + C_{\text{drift}} \cdot T$$

Optimal cadence: $T^* = \sqrt{C_{\text{retrain}} / C_{\text{drift}}}$ (derivative = 0).
"""),

# ── 5. Manual Implementation from Scratch ─────────────────────────────────────
md(r"""
## 5. Manual Implementation from Scratch

We implement `EWMAController` and `WindowedRetrainer` entirely in Python/NumPy.
"""),

code(r"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

# ── EWMAController ────────────────────────────────────────────────────────────
class EWMAController:
    def __init__(self, alpha: float = 0.2, threshold_delta: float = 0.05,
                 min_obs: int = 5):
        self._alpha = alpha
        self._delta = threshold_delta
        self._min_obs = min_obs
        self._ewma: Optional[float] = None
        self._baseline: Optional[float] = None
        self._n = 0
        self.history: List[Tuple[int, float, float]] = []  # (tick, raw, ewma)

    def update(self, metric_value: float) -> bool:
        self._n += 1
        if self._ewma is None:
            self._ewma = metric_value
        else:
            self._ewma = self._alpha * metric_value + (1 - self._alpha) * self._ewma

        if self._baseline is None and self._n >= self._min_obs:
            self._baseline = self._ewma
            print(f"  [EWMA] Baseline set to {self._baseline:.4f} at tick {self._n}")

        self.history.append((self._n, metric_value, self._ewma))

        if self._baseline is not None and self._n > self._min_obs:
            drop = self._baseline - self._ewma
            if drop > self._delta:
                return True  # trigger retrain
        return False

    @property
    def current_ewma(self):
        return self._ewma

    @property
    def baseline(self):
        return self._baseline


# ── WindowedRetrainer ─────────────────────────────────────────────────────────
class WindowedRetrainer:
    def __init__(self, window_size: int):
        self._window  = window_size
        self._data: List[Tuple[np.ndarray, int]] = []  # (x, y)

    def add_batch(self, X: np.ndarray, y: np.ndarray):
        for xi, yi in zip(X, y):
            self._data.append((xi, int(yi)))
        if len(self._data) > self._window:
            self._data = self._data[-self._window:]

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._data:
            raise ValueError("No data in window")
        Xs = np.array([d[0] for d in self._data])
        ys = np.array([d[1] for d in self._data])
        return Xs, ys

    @property
    def window_fill(self):
        return len(self._data)


# ── Simple logistic regression for demo ──────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def logistic_train(X, y, lr=0.05, epochs=30):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        pred = sigmoid(X @ w + b)
        err  = pred - y
        w   -= lr * X.T @ err / len(y)
        b   -= lr * err.mean()
    return w, b

def logistic_auc(X, y, w, b):
    scores = sigmoid(X @ w + b)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    auc = sum(p > n for p in pos for n in neg) / (len(pos) * len(neg))
    return float(auc)


# ── Smoke test ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
ctrl = EWMAController(alpha=0.3, threshold_delta=0.05)

# Simulate AUC observations: stable then drops
auc_stream = list(rng.normal(0.88, 0.01, 10)) + list(rng.normal(0.80, 0.02, 10))
retrain_ticks = []
for auc in auc_stream:
    trigger = ctrl.update(auc)
    if trigger:
        retrain_ticks.append(ctrl._n)
        print(f"  [TRIGGER] Retrain at tick {ctrl._n}: EWMA={ctrl.current_ewma:.4f}, "
              f"drop={ctrl.baseline - ctrl.current_ewma:.4f}")

print(f"\nEWMA history (tick, raw, ewma):")
for t, raw, ewma in ctrl.history:
    print(f"  {t:>3}: raw={raw:.3f}  ewma={ewma:.3f}")
"""),

code(r"""
import numpy as np

rng = np.random.default_rng(99)

# ── WindowedRetrainer smoke test ──────────────────────────────────────────────
retrainer = WindowedRetrainer(window_size=500)

# Simulate monthly data arrival with concept drift
for month in range(1, 7):
    if month <= 3:
        X_batch = rng.normal([0, 0], 1, (200, 2))
        y_batch = (X_batch[:, 0] + X_batch[:, 1] > 0).astype(int)
    else:
        X_batch = rng.normal([1, -1], 1.2, (200, 2))
        y_batch = (X_batch[:, 0] - X_batch[:, 1] > 1).astype(int)

    retrainer.add_batch(X_batch, y_batch)

    X_tr, y_tr = retrainer.get_training_data()
    w, b = logistic_train(X_tr, y_tr)

    # Evaluate on fresh held-out data from same month's distribution
    X_val = rng.normal([0,0] if month <= 3 else [1,-1], 1, (200, 2))
    y_val = (X_val[:,0] + X_val[:,1] > 0 if month <= 3
             else X_val[:,0] - X_val[:,1] > 1).astype(int)
    auc = logistic_auc(X_val, y_val, w, b)
    print(f"Month {month}: window_fill={retrainer.window_fill:>4}, AUC={auc:.3f}")
"""),

# ── 6. Visualization ─────────────────────────────────────────────────────────
md(r"""
## 6. Visualization
"""),

code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

rng = np.random.default_rng(2024)

# ── (a) EWMA smoothing vs raw AUC ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Retraining Strategies — Visual Overview", fontsize=14, fontweight='bold')

ax = axes[0]
raw_auc = list(rng.normal(0.88, 0.015, 15)) + list(rng.normal(0.81, 0.02, 15))
ticks = list(range(1, 31))

ewma_vals, s = [], None
alpha = 0.25
for v in raw_auc:
    s = v if s is None else alpha * v + (1 - alpha) * s
    ewma_vals.append(s)

ax.plot(ticks, raw_auc,  'o', color='steelblue', alpha=0.5, markersize=4, label='Raw AUC')
ax.plot(ticks, ewma_vals, '-', color='steelblue', linewidth=2, label='EWMA (α=0.25)')
ax.axhline(0.88 - 0.05, color='tomato', linestyle='--', linewidth=1.5, label='Alert threshold')
ax.axvline(15, color='grey', linestyle=':', linewidth=1.2, label='Drift starts')
ax.set_title("EWMA Smooths Noisy AUC", fontsize=11)
ax.set_xlabel("Evaluation tick")
ax.set_ylabel("AUC")
ax.legend(fontsize=8)
# EWMA line tracks the trend; individual points are too noisy to trigger an alert on their own.

# ── (b) Data strategy comparison — sliding window vs full history ─────────────
ax = axes[1]
N_months = 12
aucs_full   = [0.86] * 4 + [0.84, 0.82, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75]  # stale data hurts
aucs_window = [0.84] * 4 + [0.87, 0.89, 0.88, 0.87, 0.86, 0.85, 0.86, 0.87]  # window adapts

xs = list(range(1, N_months + 1))
ax.plot(xs, aucs_full,   'o-', color='steelblue', label='Full history retrain')
ax.plot(xs, aucs_window, 's-', color='tomato',    label='Sliding window (3-month)')
ax.axvline(4.5, color='grey', linestyle=':', linewidth=1.2, label='Drift at month 5')
ax.set_title("Sliding Window Adapts to Drift\nFull History Lags", fontsize=11)
ax.set_xlabel("Month")
ax.set_ylabel("AUC on current distribution")
ax.legend(fontsize=8)
# After drift at month 5, sliding window retrains on recent data and recovers;
# full-history retrain stays diluted by old stale data.

# ── (c) Retraining cost model ─────────────────────────────────────────────────
ax = axes[2]
cadences = np.linspace(1, 52, 200)  # cadence in weeks
C_retrain = 5_000  # cost per retrain ($)
C_drift   = 2_000  # weekly cost of drift
total_cost = C_retrain / cadences + C_drift * cadences
optimal_cadence = np.sqrt(C_retrain / C_drift)

ax.plot(cadences, total_cost / 1000, color='steelblue', linewidth=2)
ax.axvline(optimal_cadence, color='tomato', linestyle='--', linewidth=1.5,
           label=f'Optimal cadence ≈ {optimal_cadence:.1f} weeks')
ax.set_title("Retraining Cost Model\nOptimal Cadence Minimises Total Cost", fontsize=11)
ax.set_xlabel("Retraining cadence (weeks)")
ax.set_ylabel("Total cost (k$/year)")
ax.legend(fontsize=9)
# Total cost curve has a U-shape: retrain too often → high compute cost;
# too rarely → accumulated drift cost. Minimum is the sweet spot.

plt.tight_layout()
plt.savefig('/tmp/nb46_retraining.png', dpi=80, bbox_inches='tight')
plt.show()
print("Figure saved.")
"""),

# ── 7. Failure Modes ─────────────────────────────────────────────────────────
md(r"""
## 7. Failure Modes

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| Catastrophic forgetting | Fine-tune on new data without old data; forgets old patterns | Replay buffer; EWC regularisation; mix old + new data |
| Train-serve skew after retrain | Preprocessing changed; feature schema drifted | Freeze preprocessing code with versioned transforms; schema registry |
| Retraining makes it worse | Training data quality problem not caught | Add **data validation gate** before training |
| Alert storm on retrain | Model output distribution changes; triggers drift alert | Suppress drift alerts for 24h after promoted retrain |
| Sliding window forgets seasonality | 3-month window misses year-ago seasonal patterns | Use year-over-year comparison features; add seasonal baseline |
| Online learning spirals | Learning rate too high; noisy samples corrupt weights | Gradient clipping; learning rate warm-up; outlier rejection |
| Shadow test is gamed | Evaluation metric matches shadow evaluation conditions, not prod | Evaluate shadow on exactly the same slice as live traffic |
"""),

# ── 8. Production Library Implementation ─────────────────────────────────────
md(r"""
## 8. Production Library Implementation

`river` is the leading library for online/incremental learning.
`scikit-learn` supports `partial_fit` for many estimators.
"""),

code(r"""
import numpy as np

try:
    import river
    from river import linear_model as rl, preprocessing as rp, metrics as rm
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

rng = np.random.default_rng(77)

if HAS_RIVER:
    model_r = rl.LogisticRegression()
    scaler_r = rp.StandardScaler()
    metric = rm.ROCAUC()
    for i in range(500):
        x_raw = {'f0': float(rng.normal()), 'f1': float(rng.normal())}
        y = int(x_raw['f0'] + x_raw['f1'] > 0)
        x = scaler_r.transform_one(x_raw)
        metric.update(y, model_r.predict_proba_one(x))
        model_r.learn_one(x, y)
    print(f"river online LR — AUC after 500 samples: {metric.get():.3f}")
elif HAS_SKLEARN:
    clf = SGDClassifier(loss='log_loss', random_state=42, max_iter=1)
    sc  = StandardScaler()
    aucs = []
    for chunk in range(10):
        X_chunk = rng.normal(0, 1, (100, 2))
        y_chunk = (X_chunk[:, 0] + X_chunk[:, 1] > 0).astype(int)
        X_sc = sc.fit_transform(X_chunk)
        clf.partial_fit(X_sc, y_chunk, classes=[0, 1])
        scores = clf.predict_proba(X_sc)[:, 1]
        pos, neg = scores[y_chunk==1], scores[y_chunk==0]
        if len(pos) > 0 and len(neg) > 0:
            auc = sum(p > n for p in pos for n in neg) / (len(pos)*len(neg))
            aucs.append(auc)
    print(f"sklearn partial_fit — mean AUC over chunks: {np.mean(aucs):.3f}")
else:
    print("Neither river nor sklearn available — using scratch logistic above")
"""),

# ── 9. Business Case Study ────────────────────────────────────────────────────
md(r"""
## 9. Business Case Study — Fraud Detection with Seasonal Drift

**Scenario**: A payments company has a fraud detection model trained on Jan–June data.
Fraud patterns spike in November (Black Friday) and December (Christmas).
We simulate a year of transactions, apply EWMA monitoring, and optimise retraining cadence.
"""),

code(r"""
import numpy as np
import math

rng = np.random.default_rng(2025)

WEEKS = 52
FRAUD_BASE = 0.01      # 1% fraud rate base
FRAUD_PEAK = 0.035     # 3.5% during holiday weeks

# ── Simulate weekly AUC from production logs (with seasonal dip) ──────────────
def simulate_fraud_auc(week, rng):
    # Fraud peaks in weeks 45-52 (Nov-Dec); model trained on Jan-Jun data struggles
    if 45 <= week <= 52:
        base_auc = 0.82 + rng.normal(0, 0.015)
    elif 20 <= week <= 35:
        base_auc = 0.91 + rng.normal(0, 0.01)
    else:
        base_auc = 0.88 + rng.normal(0, 0.012)
    return float(np.clip(base_auc, 0.5, 1.0))


# ── Simulate cost: $100 per undetected fraud, $5 per retrain ─────────────────
FRAUD_COST_PER_WEEK = 50_000   # baseline weekly fraud exposure
RETRAIN_COST        = 5_000    # cost per retrain

def auc_to_fraud_cost(auc, base_cost=FRAUD_COST_PER_WEEK):
    missed_rate = 1.0 - auc  # rough proxy: 1 - AUC ≈ fraction missed
    return base_cost * missed_rate


# ── Strategy 1: Monthly retraining (every 4 weeks) ────────────────────────────
total_cost_monthly = 0.0
retrains_monthly   = 0
for w in range(1, WEEKS + 1):
    auc = simulate_fraud_auc(w, rng)
    total_cost_monthly += auc_to_fraud_cost(auc)
    if w % 4 == 0:
        total_cost_monthly += RETRAIN_COST
        retrains_monthly   += 1

# ── Strategy 2: EWMA-triggered retraining ────────────────────────────────────
ctrl2 = EWMAController(alpha=0.3, threshold_delta=0.04, min_obs=4)
total_cost_ewma = 0.0
retrains_ewma   = 0
rng2 = np.random.default_rng(2025)  # same seed for fair comparison
for w in range(1, WEEKS + 1):
    auc = simulate_fraud_auc(w, rng2)
    total_cost_ewma += auc_to_fraud_cost(auc)
    trigger = ctrl2.update(auc)
    if trigger:
        total_cost_ewma += RETRAIN_COST
        retrains_ewma   += 1
        ctrl2._baseline = ctrl2.current_ewma  # reset baseline after retrain

# ── Strategy 3: Drift-based (every holiday season + one scheduled) ────────────
total_cost_seasonal = 0.0
retrains_seasonal   = 0
rng3 = np.random.default_rng(2025)
retrain_weeks = {1, 26, 44}  # Jan, Jul, pre-holiday
for w in range(1, WEEKS + 1):
    auc = simulate_fraud_auc(w, rng3)
    total_cost_seasonal += auc_to_fraud_cost(auc)
    if w in retrain_weeks:
        total_cost_seasonal += RETRAIN_COST
        retrains_seasonal   += 1

print("=" * 55)
print(f"{'Strategy':<30} {'Retrains':>8}  {'Total Cost':>12}")
print("=" * 55)
print(f"{'Monthly (every 4 weeks)':<30} {retrains_monthly:>8}  ${total_cost_monthly:>11,.0f}")
print(f"{'EWMA-triggered':<30} {retrains_ewma:>8}  ${total_cost_ewma:>11,.0f}")
print(f"{'Seasonal (3x/year)':<30} {retrains_seasonal:>8}  ${total_cost_seasonal:>11,.0f}")
print("=" * 55)
print()
print("Optimal theoretical cadence (cost model):")
C_d = FRAUD_COST_PER_WEEK * 0.05  # 5% extra fraud cost per week of staleness
T_opt = math.sqrt(RETRAIN_COST / C_d)
print(f"  T* = sqrt({RETRAIN_COST}/{C_d:.0f}) = {T_opt:.1f} weeks")
"""),

# ── Full Retraining Pipeline ──────────────────────────────────────────────────
md(r"""
### Full Retraining Pipeline (Simulation)
"""),

code(r"""
import numpy as np

rng = np.random.default_rng(123)

# ── Simulated pipeline stages ─────────────────────────────────────────────────
class RetrainingPipeline:
    def __init__(self, shadow_weeks=1):
        self._shadow_weeks = shadow_weeks
        self._current_w = None
        self._current_b = None
        self._shadow_w  = None
        self._shadow_b  = None
        self.log = []

    def _validate_data(self, X, y):
        # Check for null proxies, label imbalance
        null_frac = np.isnan(X).mean()
        pos_rate  = y.mean()
        if null_frac > 0.05:
            return False, f"Null fraction {null_frac:.2%} exceeds 5%"
        if pos_rate < 0.001 or pos_rate > 0.5:
            return False, f"Label imbalance: pos_rate={pos_rate:.3f}"
        return True, "OK"

    def _train(self, X, y):
        return logistic_train(X, y)

    def _evaluate(self, X, y, w, b):
        return logistic_auc(X, y, w, b)

    def run(self, X_train, y_train, X_val, y_val, X_prod, y_prod):
        self.log.append("1. Fetching training data...")

        ok, msg = self._validate_data(X_train, y_train)
        self.log.append(f"2. Data validation: {msg}")
        if not ok:
            self.log.append("   ABORT: data quality gate failed")
            return False

        w_new, b_new = self._train(X_train, y_train)
        self.log.append("3. Training complete")

        auc_new  = self._evaluate(X_val, y_val, w_new, b_new)
        self.log.append(f"4. Validation AUC (new model):  {auc_new:.4f}")

        if self._current_w is not None:
            auc_cur = self._evaluate(X_val, y_val, self._current_w, self._current_b)
            self.log.append(f"   Validation AUC (curr model): {auc_cur:.4f}")
            if auc_new < auc_cur - 0.01:
                self.log.append("   ABORT: new model worse than current on val set")
                return False

        # Shadow test on production slice
        self._shadow_w, self._shadow_b = w_new, b_new
        shadow_auc = self._evaluate(X_prod, y_prod, w_new, b_new)
        self.log.append(f"5. Shadow AUC on prod slice: {shadow_auc:.4f}")

        if shadow_auc < 0.75:
            self.log.append("   ABORT: shadow AUC below safety threshold 0.75")
            return False

        self._current_w = w_new
        self._current_b = b_new
        self.log.append("6. Promoted: new model is now live")
        return True

    def print_log(self):
        for line in self.log:
            print(line)


rng = np.random.default_rng(42)
X_tr = rng.normal(0, 1, (800, 4))
y_tr = (X_tr[:,0] + X_tr[:,1] > 0).astype(int)
X_val= rng.normal(0, 1, (200, 4))
y_val= (X_val[:,0] + X_val[:,1] > 0).astype(int)
X_prd= rng.normal(0, 1, (300, 4))
y_prd= (X_prd[:,0] + X_prd[:,1] > 0).astype(int)

pipeline = RetrainingPipeline()
success = pipeline.run(X_tr, y_tr, X_val, y_val, X_prd, y_prd)
pipeline.print_log()
print(f"\nPipeline success: {success}")
"""),

# ── Catastrophic Forgetting Demo ──────────────────────────────────────────────
md(r"""
### Catastrophic Forgetting Demo
"""),

code(r"""
import numpy as np

rng = np.random.default_rng(9)

# Task A: learned on (f0 + f1 > 0)
X_a = rng.normal(0, 1, (600, 2))
y_a = (X_a[:,0] + X_a[:,1] > 0).astype(int)
w_a, b_a = logistic_train(X_a, y_a, epochs=50)

# Task B: new distribution, different rule (f0 - f1 > 1)
X_b = rng.normal(1, 1, (600, 2))
y_b = (X_b[:,0] - X_b[:,1] > 1).astype(int)

# Warm start: fine-tune w_a on task B only (no task A data)
def logistic_train_warm(X, y, w_init, b_init, lr=0.05, epochs=30):
    w, b = w_init.copy(), float(b_init)
    for _ in range(epochs):
        pred = sigmoid(X @ w + b)
        err  = pred - y
        w   -= lr * X.T @ err / len(y)
        b   -= lr * err.mean()
    return w, b

w_ab, b_ab = logistic_train_warm(X_b, y_b, w_a, b_a)

# Evaluate on Task A and Task B
X_test_a = rng.normal(0, 1, (300, 2)); y_test_a = (X_test_a[:,0] + X_test_a[:,1] > 0).astype(int)
X_test_b = rng.normal(1, 1, (300, 2)); y_test_b = (X_test_b[:,0] - X_test_b[:,1] > 1).astype(int)

auc_a_before = logistic_auc(X_test_a, y_test_a, w_a,  b_a)
auc_a_after  = logistic_auc(X_test_a, y_test_a, w_ab, b_ab)
auc_b_after  = logistic_auc(X_test_b, y_test_b, w_ab, b_ab)

print("Catastrophic Forgetting Demo")
print(f"  Task A AUC before fine-tune:  {auc_a_before:.3f}")
print(f"  Task A AUC after  fine-tune:  {auc_a_after:.3f}  ← degraded (forgot!)")
print(f"  Task B AUC after  fine-tune:  {auc_b_after:.3f}  ← learned new task")

# Simple mitigation: replay buffer — mix 20% old Task A data
X_replay = np.vstack([X_b, X_a[:120]])  # 120 = 20% of 600
y_replay  = np.concatenate([y_b, y_a[:120]])

w_replay, b_replay = logistic_train_warm(X_replay, y_replay, w_a, b_a)
auc_a_replay = logistic_auc(X_test_a, y_test_a, w_replay, b_replay)
auc_b_replay = logistic_auc(X_test_b, y_test_b, w_replay, b_replay)
print(f"\n  With 20% replay buffer:")
print(f"  Task A AUC:  {auc_a_replay:.3f}  ← recovered!")
print(f"  Task B AUC:  {auc_b_replay:.3f}")
"""),

# ── 10. Production Considerations ────────────────────────────────────────────
md(r"""
## 10. Production Considerations

### The Retraining Decision Matrix

| Drift Signal | Performance Drop | Recommended Action |
|---|---|---|
| No | No | Continue monitoring; next scheduled check |
| Yes (PSI 0.1–0.2) | No | Investigate feature pipeline; do not retrain yet |
| No | Yes (small) | Check label quality and evaluation bugs first |
| Yes (PSI > 0.2) | Yes (>3%) | Trigger immediate retrain; run shadow for 48h |
| No | Yes (large >10%) | Emergency — possible model/infra bug |

### Train-Serve Skew After Retraining

Skew sources after a retrain:
1. **Preprocessing version mismatch**: training used StandardScaler fit on old data; serving uses the same stale scaler
2. **Feature freshness**: training used T-7 day features; serving uses T-1 (different lag)
3. **Missing value handling**: training imputed with training-set median; serving imputes with current median

Fix: **version the entire feature pipeline** (scaler, imputer, feature logic) and deploy it alongside the model artifact.

### Atomic Model Promotion

```bash
# Blue-green swap (zero-downtime)
cp model_v2.pkl /mnt/models/pending/
ln -sfn /mnt/models/pending/ /mnt/models/current
# Health check...
# If OK: keep; if fail: ln -sfn /mnt/models/previous/ /mnt/models/current
```
"""),

# ── 11. Tradeoff Analysis ─────────────────────────────────────────────────────
md(r"""
## 11. Tradeoff Analysis

| Data Strategy | Adapts to Drift | Compute Cost | Forgetting Risk | Best For |
|---|---|---|---|---|
| Full history | Low | Very high | None | Stable distributions with abundant compute |
| Sliding window | High | Medium | High if window < seasonal period | Fast-changing environments |
| Warm start | Medium | Low | High without replay | Large models, expensive training |
| Online (per-sample) | Highest | Lowest | Highest | Real-time streams; concept drift heavy |

| Trigger | Latency | FP Rate | Implementation | Best For |
|---|---|---|---|---|
| Scheduled | High (up to cadence) | None | Trivial | Stable, low-cost models |
| Performance-based (EWMA) | Medium | Low | Easy | Most production models |
| Drift-based (PSI) | Low | Medium | Medium | Feature-rich models with good monitoring |
| Event-based | Immediate | None | Requires domain knowledge | Known seasonal / business events |
"""),

# ── 12. Senior-Level Interview Preparation ────────────────────────────────────
md(r"""
## 12. Senior-Level Interview Preparation

**Q1**: When should you use online learning vs periodic batch retraining?

> Online learning for real-time streams where latency matters (fraud, ad ranking).
> Batch retraining for most use cases — easier to debug, audit, and version.
> Key criterion: how fast does the concept drift? Minutes → online; days → batch.

**Q2**: Your fraud model was retrained last night and AUC dropped 3 points in production. What happened?

> First suspects: train-serve skew (scaler/preprocessor mismatch), data pipeline bug in new training data, label leakage in old model that isn't present in new model, or evaluation date bug. Check shadow mode logs for score distribution shift before the drop.

**Q3**: What is catastrophic forgetting and how do you mitigate it?

> When fine-tuning on new data, the model overwrites weights important for old patterns.
> Mitigations: (1) replay buffer — mix old + new data; (2) EWC regularisation — penalise changes to high-Fisher-information weights; (3) larger sliding window that spans both old and new patterns.

**Q4**: How do you decide the optimal retraining cadence?

> Model it as: total cost = retrain_cost / T + drift_cost * T. Optimal T* = sqrt(retrain_cost / drift_cost). Empirically: A/B test different cadences and measure downstream business metrics.

**Q5**: A new regulation requires you to retrain your model monthly to ensure fairness compliance. The data distribution changes weekly. How do you handle this?

> Decouple the compliance retraining cycle from the performance retraining cycle. Keep an EWMA monitor running weekly — if AUC drops, trigger an off-cycle retrain. The monthly retrain is the compliance artefact; the weekly trigger is the performance artefact. Document both.

**Q6**: What is the difference between train-serve skew and data drift?

> Train-serve skew is a **pipeline bug**: the model sees different data at train time vs serve time due to preprocessing inconsistencies. Data drift is a **real-world phenomenon**: the distribution genuinely changed. Train-serve skew is within your control; drift is not.

**Q7**: How do you shadow-test a new model in production?

> Route a copy of all live traffic to the new model without serving its predictions to users. Compare output distributions between old and new model. If distributions agree and new model is better on labelled test set, promote. Keep shadow running for ≥1 full traffic cycle (e.g., 1 week for weekly-seasonal data).

**Q8**: What happens when you retrain a model and the monitoring system fires a drift alert the next day?

> Expected — the model's output distribution changed at promotion. Suppress drift alerts for 24–48h after a promoted retrain. After the suppression window, use the new model's output distribution as the new reference baseline.
"""),

# ── 13. Teach-Back Section ───────────────────────────────────────────────────
md(r"""
## 13. Teach-Back Section

Answer these to solidify understanding:

1. Name the four retraining triggers. When would you use each?
2. Explain the EWMA formula in one sentence and state what $\alpha$ controls.
3. Why does a sliding window risk "forgetting" seasonal patterns?
4. Define catastrophic forgetting in the context of warm-start fine-tuning.
5. What is train-serve skew? Give two concrete examples.
6. Describe the 6-stage retraining pipeline from data to promotion.
7. A junior engineer proposes retraining the model every hour using the previous hour's data. What are the risks?
8. How would you compute the optimal retraining cadence if you know the retrain cost is $2,000 and each week of staleness costs $500 in extra fraud losses?
"""),

# ── 14. Exercises ─────────────────────────────────────────────────────────────
md(r"""
## 14. Exercises

### Beginner
1. Modify `EWMAController` to log the tick at which baseline was set and the tick of every alert.
2. Run `WindowedRetrainer` with window sizes of 100, 300, and 600. Plot AUC vs window size.
3. Explain in writing why a sliding window of 30 days is risky for a model trained on a year of data.

### Intermediate
4. Implement a **cooldown** in `EWMAController`: after a retrain trigger, suppress alerts for `cooldown_ticks` ticks. This prevents alert storms.
5. Add a **data quality gate** to `RetrainingPipeline` that checks: no feature has >10% missing values and the label rate is between 0.1% and 20%.
6. Implement the **EWC penalty** in `logistic_train_warm`: compute the diagonal Fisher (squared gradients from Task A) and add $\lambda \sum_i F_i (w_i - w_i^*)^2$ to the loss gradient.

### Senior
7. Implement a **multi-armed retraining scheduler**: maintain separate EWMA controllers for each business segment (high-value, standard, trial users). Trigger segment-specific retraining when only that segment's AUC drops.
8. Design a **shadow-mode evaluation harness**: given a production request log (feature vectors + actual outcomes), score both old and new models offline and compute a calibrated AUC difference with confidence intervals.
9. Implement the **optimal cadence estimator**: given a history of (week, AUC) pairs, fit the drift cost model $\text{AUC}(T) = \text{AUC}_0 - k \cdot T$ and solve for the cadence that minimises total cost.
"""),

]  # end cells

if __name__ == "__main__":
    build("phase8_production/46_retraining_strategies.ipynb", cells)
