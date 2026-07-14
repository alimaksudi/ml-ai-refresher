"""PROD-05 — Monitoring and Drift Detection builder."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nbbuild import md, code, build

cells = [

# ── 1. Learning Objectives ────────────────────────────────────────────────────
md(r"""
# PROD-05 — Monitoring and Drift Detection

## 1. Learning Objectives

By the end of this notebook you will be able to:

- Distinguish **data drift**, **concept drift**, and **label drift** and explain when each degrades a model
- Implement **Population Stability Index (PSI)** from scratch and interpret alert thresholds
- Implement a **Kolmogorov-Smirnov test** from scratch for continuous feature drift
- Build an **ADWIN** (Adaptive Windowing) algorithm that detects online concept drift
- Monitor **prediction distribution shift** and **confidence score drift**
- Estimate business metrics (revenue, AUC) from production logs without ground-truth labels
- Build a production-ready **DriftMonitor** class that raises structured alerts
- Simulate a 12-month e-commerce recommendation scenario and detect drift at month 6
"""),

# ── 2. Historical Motivation ───────────────────────────────────────────────────
md(r"""
## 2. Historical Motivation

### The Problem with "Train-Once, Deploy-Forever"

In 2012 a major US bank deployed a credit-risk model trained on 2008–2011 data.
By 2014 charge-off rates had climbed 40% above predictions. The model had no idea
the economy had recovered: incomes, spending, and default patterns all shifted.

Similar stories repeat across domains:

| Year | Company | Failure |
|------|---------|---------|
| 2014 | US bank | Credit model misses economic recovery; $200M in bad predictions |
| 2017 | Uber | Surge-pricing model drifts after city-rezoning changes traffic patterns |
| 2020 | Nearly everyone | COVID-19 obliterates every demand-forecasting model overnight |
| 2022 | Twitter ad ranking | User behaviour shift post-acquisition degrades CTR; revenue falls |

### The Vocabulary

- **Data drift** (covariate shift): P(X) changes but P(Y|X) stays the same.
  *Example*: a new phone model floods logs with unusual screen-size features.
- **Concept drift**: P(Y|X) changes; the relationship itself evolves.
  *Example*: "premium" used to predict high spend, but inflation has inverted that.
- **Label drift** (prior probability shift): P(Y) changes.
  *Example*: fraud rate spikes from 0.5% to 2% after a data breach.

Early detection is the difference between a 1-week retrain and a 3-month fire-drill.
"""),

# ── 3. Intuition & Visual Understanding ──────────────────────────────────────
md(r"""
## 3. Intuition & Visual Understanding

### PSI: "How different are two histograms?"

PSI compares a **reference** distribution (training data) to a **current** distribution
(recent production data) bucket-by-bucket:

$$\text{PSI} = \sum_{i=1}^{k} \left( P_i^{\text{cur}} - P_i^{\text{ref}} \right) \ln \frac{P_i^{\text{cur}}}{P_i^{\text{ref}}}$$

This is the KL-divergence of current from reference **plus** the reverse — a symmetric
measure of distribution distance.

**Thresholds (industry standard)**:

| PSI value | Interpretation |
|-----------|---------------|
| < 0.1 | No significant change |
| 0.1 – 0.2 | Moderate shift — investigate |
| > 0.2 | Significant shift — retrain |

### KS Test: "Do these two CDFs diverge?"

The KS statistic is the maximum absolute difference between two empirical CDFs.
Under the null hypothesis of identical distributions its sampling distribution is
known, giving an exact p-value without assuming normality.

### ADWIN: Sliding-window concept drift

ADWIN keeps a growing window of recent error rates.  It continuously checks all
possible splits of that window — if any sub-window has a statistically different
mean, ADWIN drops the older portion and raises a drift signal.  This is adaptive:
the window grows when stable, shrinks when drift is detected.
"""),

# ── 4. Mathematical Foundations ───────────────────────────────────────────────
md(r"""
## 4. Mathematical Foundations

### 4.1 Population Stability Index

$$\text{PSI} = \sum_{i=1}^{k} (A_i - E_i) \ln \left(\frac{A_i}{E_i}\right)$$

where $A_i$ = actual (current) fraction in bin $i$, $E_i$ = expected (reference) fraction.
To avoid $\ln(0)$, bins with zero count are floored to a small $\epsilon$.

### 4.2 Kolmogorov-Smirnov Statistic

$$D = \sup_x |F_n(x) - G_m(x)|$$

Critical value at significance $\alpha$ for samples of size $n$ and $m$:

$$D_\alpha = c(\alpha) \sqrt{\frac{n + m}{n m}}$$

where $c(0.05) \approx 1.358$ (Kolmogorov distribution).

### 4.3 ADWIN Mean-shift Bound

For a window split into sub-windows $W_0$ (old) and $W_1$ (new) of sizes $n_0, n_1$:

$$|\hat{\mu}_0 - \hat{\mu}_1| > \epsilon_{\text{cut}} = \sqrt{\frac{1}{2m}\ln\frac{4n}{\delta}}$$

where $m = \frac{n_0 n_1}{n_0 + n_1}$ (harmonic-mean-like term), $n = n_0 + n_1$,
and $\delta$ is an allowed false-alarm probability (smaller $\delta$ means higher
confidence and a wider threshold). This is a teaching bound; production ADWIN
implementations use the library's full adaptive-window bound and correction logic.

### 4.4 AUC Requires Labels, Often Delayed

The Wilcoxon–Mann–Whitney interpretation of AUC is:

$$\text{AUC} = \frac{|\{(i,j): s_i > s_j, y_i=1, y_j=0\}|}{n_+ \cdot n_-}$$

**Read and symbols:** $s_i$ is prediction score $i$; $y_i$ is its observed binary
label; $n_+$ and $n_-$ are positive and negative counts; the set counts correctly
ordered positive-negative pairs. AUC cannot be computed from scores alone.

In production, compute AUC when **delayed labels** arrive. Clicks or add-to-cart
events may be monitored as explicitly named proxy metrics, but proxy AUC is not
the target-label AUC and must not be reported as though it were.
"""),

# ── 5. Manual Implementation from Scratch ─────────────────────────────────────
md(r"""
## 5. Manual Implementation from Scratch

We implement PSI, KS, and ADWIN entirely with NumPy.
"""),

code(r"""
import numpy as np
import math

# ── PSI ──────────────────────────────────────────────────────────────────────
def compute_psi(reference, current, n_bins=10, eps=1e-6):
    # Bin edges from reference distribution
    _, bin_edges = np.histogram(reference, bins=n_bins)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    ref_frac = ref_counts / ref_counts.sum()
    cur_frac = cur_counts / cur_counts.sum()

    # Floor zeros
    ref_frac = np.maximum(ref_frac, eps)
    cur_frac = np.maximum(cur_frac, eps)

    psi_bins = (cur_frac - ref_frac) * np.log(cur_frac / ref_frac)
    return float(psi_bins.sum()), psi_bins, bin_edges


def psi_alert(psi_value):
    if psi_value < 0.1:
        return "OK"
    elif psi_value < 0.2:
        return "MODERATE_DRIFT"
    else:
        return "SIGNIFICANT_DRIFT"


# ── KS Test ───────────────────────────────────────────────────────────────────
def ks_test(x, y):
    n, m = len(x), len(y)
    combined = np.sort(np.concatenate([x, y]))

    # Empirical CDFs evaluated at every point in combined
    cdf_x = np.searchsorted(np.sort(x), combined, side='right') / n
    cdf_y = np.searchsorted(np.sort(y), combined, side='right') / m

    D = float(np.max(np.abs(cdf_x - cdf_y)))

    # Kolmogorov critical value at alpha=0.05: c(0.05)=1.3581
    c_alpha = 1.3581
    critical = c_alpha * math.sqrt((n + m) / (n * m))
    p_approx = math.exp(-2 * (D * math.sqrt(n * m / (n + m))) ** 2)
    p_approx = min(1.0, p_approx)

    return {
        "D": D,
        "critical_value_05": critical,
        "reject_null": D > critical,
        "p_approx": p_approx,
    }


# ── ADWIN ────────────────────────────────────────────────────────────────────
class ADWIN:
    def __init__(self, delta=0.002):
        self._delta = delta
        self._window = []
        self.drift_detected = False
        self.n_detections = 0

    def add_element(self, value):
        self._window.append(float(value))
        self.drift_detected = self._detect()

    def _detect(self):
        n = len(self._window)
        if n < 5:
            return False
        W = np.array(self._window)
        total_mean = W.mean()
        for split in range(2, n - 2):
            W0 = W[:split]
            W1 = W[split:]
            n0, n1 = len(W0), len(W1)
            m_harm = (n0 * n1) / (n0 + n1)
            eps_cut = math.sqrt(0.5 / m_harm * math.log(4 * n / self._delta))
            if abs(W0.mean() - W1.mean()) > eps_cut:
                self._window = list(W1)  # drop old portion
                self.n_detections += 1
                return True
        return False

    @property
    def window_size(self):
        return len(self._window)


# ── Quick smoke test ─────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
ref = rng.normal(0, 1, 5000)
cur_ok  = rng.normal(0, 1, 2000)         # same distribution
cur_bad = rng.normal(2, 1.5, 2000)       # shifted

psi_ok,  _, _ = compute_psi(ref, cur_ok)
psi_bad, _, _ = compute_psi(ref, cur_bad)
ks_ok  = ks_test(ref, cur_ok)
ks_bad = ks_test(ref, cur_bad)

print(f"PSI (same dist) = {psi_ok:.4f}  → {psi_alert(psi_ok)}")
print(f"PSI (shifted)   = {psi_bad:.4f}  → {psi_alert(psi_bad)}")
print(f"KS (same dist):  D={ks_ok['D']:.4f}, reject={ks_ok['reject_null']}")
print(f"KS (shifted):    D={ks_bad['D']:.4f}, reject={ks_bad['reject_null']}")

adwin = ADWIN(delta=0.002)
stable_stream = rng.normal(0, 1, 200)
for v in stable_stream:
    adwin.add_element(v)
print(f"\nADWIN after 200 stable obs: detections={adwin.n_detections}, window={adwin.window_size}")

drift_stream = np.concatenate([rng.normal(0,1,100), rng.normal(3,1,100)])
adwin2 = ADWIN(delta=0.002)
for v in drift_stream:
    adwin2.add_element(v)
print(f"ADWIN after 100 stable + 100 drifted: detections={adwin2.n_detections}, window={adwin2.window_size}")
"""),

# ── 6. Visualization ─────────────────────────────────────────────────────────
md(r"""
## 6. Visualization

We plot (a) PSI bar charts, (b) KS CDF comparison, and (c) ADWIN window over time.
"""),

code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

rng = np.random.default_rng(42)
ref     = rng.normal(0, 1, 5000)
cur_ok  = rng.normal(0, 1, 2000)
cur_bad = rng.normal(2, 1.5, 2000)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Drift Detection Toolkit", fontsize=15, fontweight='bold')

# ── (a) PSI histogram comparison ─────────────────────────────────────────────
ax = axes[0]
bins = np.linspace(-4, 7, 15)
ax.hist(ref,     bins=bins, alpha=0.5, density=True, color='steelblue', label='Reference (train)')
ax.hist(cur_bad, bins=bins, alpha=0.5, density=True, color='tomato',    label='Current (drifted)')
psi_val, _, _ = compute_psi(ref, cur_bad)
ax.set_title(f"Data Drift — PSI = {psi_val:.3f}\nAlert: {psi_alert(psi_val)}", fontsize=11)
ax.set_xlabel("Feature value")
ax.set_ylabel("Density")
ax.legend(fontsize=9)
# Annotations: each bar pair shows where the distributions diverge most.

# ── (b) KS — Empirical CDFs ──────────────────────────────────────────────────
ax = axes[1]
combined = np.sort(np.concatenate([ref, cur_bad]))
cdf_ref = np.searchsorted(np.sort(ref),     combined, side='right') / len(ref)
cdf_bad = np.searchsorted(np.sort(cur_bad), combined, side='right') / len(cur_bad)
ax.plot(combined, cdf_ref, color='steelblue', label='Reference CDF', linewidth=1.5)
ax.plot(combined, cdf_bad, color='tomato',    label='Current CDF',   linewidth=1.5)
# Mark max-gap
gap_idx = np.argmax(np.abs(cdf_ref - cdf_bad))
ax.axvline(combined[gap_idx], color='black', linestyle='--', linewidth=1.2, label=f'KS-D={np.abs(cdf_ref-cdf_bad).max():.3f}')
ax.set_title("KS Test — CDF Comparison", fontsize=11)
ax.set_xlabel("Feature value")
ax.set_ylabel("Cumulative probability")
ax.legend(fontsize=9)
# The vertical dashed line is the maximum vertical gap between CDFs — the KS statistic.

# ── (c) ADWIN window size over a drifting stream ─────────────────────────────
ax = axes[2]
stream = np.concatenate([rng.normal(0,1,150), rng.normal(3,1,150)])
adwin3 = ADWIN(delta=0.002)
ws, detected_at = [], []
for i, v in enumerate(stream):
    adwin3.add_element(v)
    ws.append(adwin3.window_size)
    if adwin3.drift_detected:
        detected_at.append(i)

ax.plot(ws, color='steelblue', linewidth=1.5, label='ADWIN window size')
ax.axvline(150, color='grey', linestyle=':', linewidth=1.2, label='True drift point')
for d in detected_at[:3]:
    ax.axvline(d, color='tomato', linestyle='--', linewidth=1.0)
ax.set_title("ADWIN — Window Shrinks at Drift", fontsize=11)
ax.set_xlabel("Observations seen")
ax.set_ylabel("Window size")
ax.legend(fontsize=9)
# Window grows when the stream is stable; collapses when ADWIN detects a mean shift.

plt.tight_layout()
plt.savefig('/tmp/nb45_drift_toolkit.png', dpi=80, bbox_inches='tight')
plt.show()
print("Figure saved.")
"""),

# ── 7. Failure Modes ─────────────────────────────────────────────────────────
md(r"""
## 7. Failure Modes

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| PSI misses feature correlation drift | PSI is marginal — it ignores feature interactions | Add multivariate drift (MMD, learned discriminator) |
| ADWIN too slow to detect gradual drift | $\delta$ too permissive; drift is slow | Lower $\delta$; add trend test in parallel |
| Alert fatigue from noisy features | Unimportant feature triggers SIGNIFICANT_DRIFT | Gate alerts on feature importance; only monitor top-K features |
| Proxy AUC is biased | Click proxy ≠ purchase signal | Calibrate proxy on labelled holdout; use AUUC instead |
| PSI bins chosen poorly | All mass in one bin if data is heavy-tailed | Use quantile-based bins instead of equal-width |
| Monitoring lag | Alert fires 7 days after drift began | Use early-warning signals (input drift detects before output drift) |
"""),

# ── 8. Production Library Implementation ─────────────────────────────────────
md(r"""
## 8. Production Library Implementation

`evidently` and `alibi-detect` are the two dominant open-source monitoring libraries.
We show both with guarded imports.
"""),

code(r"""
try:
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

try:
    from alibi_detect.cd import KSDrift, MMDDrift
    HAS_ALIBI = True
except ImportError:
    HAS_ALIBI = False

import numpy as np
rng = np.random.default_rng(99)
X_ref = rng.normal(0, 1, (1000, 4))
X_cur = rng.normal(0.5, 1.2, (500, 4))

if HAS_ALIBI:
    detector = KSDrift(X_ref, p_val=0.05)
    result = detector.predict(X_cur)
    print("alibi-detect KSDrift:", result['data']['is_drift'])
else:
    print("alibi-detect not installed — using scratch implementation")
    ks_res = [ks_test(X_ref[:, j], X_cur[:, j]) for j in range(4)]
    for j, r in enumerate(ks_res):
        print(f"  Feature {j}: D={r['D']:.3f}, reject={r['reject_null']}")

if HAS_EVIDENTLY:
    import pandas as pd
    cols = [f'f{i}' for i in range(4)]
    ref_df = pd.DataFrame(X_ref, columns=cols)
    cur_df = pd.DataFrame(X_cur, columns=cols)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    print("Evidently report generated (call report.show() in Jupyter)")
else:
    print("evidently not installed — using scratch PSI")
    for j in range(4):
        psi_v, _, _ = compute_psi(X_ref[:, j], X_cur[:, j])
        print(f"  Feature {j}: PSI={psi_v:.4f}  → {psi_alert(psi_v)}")
"""),

# ── 9. Realistic Business Case Study ─────────────────────────────────────────
md(r"""
## 9. Business Case Study — E-Commerce Recommendation Model

**Scenario**: An e-commerce platform deployed a recommendation model in January.
In June, a major competitor launched, shifting user browsing behaviour.
We simulate 12 months of data and detect the drift.
"""),

code(r"""
import numpy as np
import math

rng = np.random.default_rng(2024)

# ── Simulate 12 months of feature batches and model predictions ───────────────
MONTHS = 12
DRIFT_MONTH = 6

# Feature: "avg_session_pages" (pages viewed per session)
# Pre-drift: Normal(8, 2); post-drift: Normal(5, 3) — users browse less

monthly_batches = []
for m in range(1, MONTHS + 1):
    if m < DRIFT_MONTH:
        feat = rng.normal(8, 2, 2000)
        preds = 1 / (1 + np.exp(-(feat - 6) / 2)) + rng.normal(0, 0.05, 2000)
    else:
        feat = rng.normal(5, 3, 2000)
        preds = 1 / (1 + np.exp(-(feat - 6) / 2)) + rng.normal(0, 0.08, 2000)

    preds = np.clip(preds, 0.01, 0.99)
    monthly_batches.append({"month": m, "feature": feat, "pred": preds})

# Reference is month 1
ref_feat = monthly_batches[0]["feature"]
ref_pred = monthly_batches[0]["pred"]

results = []
for batch in monthly_batches[1:]:
    feat_psi, _, _ = compute_psi(ref_feat, batch["feature"])
    pred_psi, _, _ = compute_psi(ref_pred, batch["pred"])
    ks_feat = ks_test(ref_feat, batch["feature"])
    results.append({
        "month":     batch["month"],
        "feat_psi":  feat_psi,
        "pred_psi":  pred_psi,
        "ks_D":      ks_feat["D"],
        "ks_reject": ks_feat["reject_null"],
        "feat_alert": psi_alert(feat_psi),
    })

print(f"{'Month':>5}  {'Feat PSI':>9}  {'Pred PSI':>9}  {'KS D':>6}  {'Alert'}")
print("-" * 52)
for r in results:
    print(f"{r['month']:>5}  {r['feat_psi']:>9.4f}  {r['pred_psi']:>9.4f}  "
          f"{r['ks_D']:>6.3f}  {r['feat_alert']}")
"""),

# ── DriftMonitor class ───────────────────────────────────────────────────────
md(r"""
### DriftMonitor — Production Class
"""),

code(r"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class DriftAlert:
    timestamp_idx: int
    feature_name:  str
    metric:        str
    value:         float
    severity:      str
    message:       str

class DriftMonitor:
    def __init__(self, reference_data: Dict[str, np.ndarray], psi_bins=10):
        self._ref = reference_data
        self._psi_bins = psi_bins
        self._adwins = {k: ADWIN() for k in reference_data}
        self.alerts: List[DriftAlert] = []
        self._tick = 0

    def process_batch(self, current_data: Dict[str, np.ndarray]) -> List[DriftAlert]:
        batch_alerts = []
        for feat, cur_vals in current_data.items():
            if feat not in self._ref:
                continue
            ref_vals = self._ref[feat]

            # PSI
            psi_val, _, _ = compute_psi(ref_vals, cur_vals, n_bins=self._psi_bins)
            severity = psi_alert(psi_val)
            if severity != "OK":
                a = DriftAlert(self._tick, feat, "PSI", psi_val, severity,
                               f"PSI={psi_val:.4f} ({severity})")
                batch_alerts.append(a)

            # KS
            ks_res = ks_test(ref_vals, cur_vals)
            if ks_res["reject_null"]:
                a = DriftAlert(self._tick, feat, "KS", ks_res["D"], "KS_REJECT",
                               f"KS D={ks_res['D']:.4f} p≈{ks_res['p_approx']:.4f}")
                batch_alerts.append(a)

            # ADWIN on batch mean
            self._adwins[feat].add_element(float(cur_vals.mean()))
            if self._adwins[feat].drift_detected:
                a = DriftAlert(self._tick, feat, "ADWIN", float(cur_vals.mean()),
                               "CONCEPT_DRIFT", f"ADWIN detected window shift at tick {self._tick}")
                batch_alerts.append(a)

        self.alerts.extend(batch_alerts)
        self._tick += 1
        return batch_alerts

    def summary(self):
        print(f"Total alerts: {len(self.alerts)}")
        by_feat = {}
        for a in self.alerts:
            by_feat.setdefault(a.feature_name, []).append(a.severity)
        for feat, sevs in by_feat.items():
            print(f"  {feat}: {sevs}")


# ── Run DriftMonitor on e-commerce simulation ────────────────────────────────
ref_data = {"session_pages": monthly_batches[0]["feature"],
            "pred_score":    monthly_batches[0]["pred"]}
monitor = DriftMonitor(ref_data)

for batch in monthly_batches[1:]:
    cur_data = {"session_pages": batch["feature"], "pred_score": batch["pred"]}
    alerts = monitor.process_batch(cur_data)
    if alerts:
        print(f"Month {batch['month']:02d}: {len(alerts)} alert(s) — "
              f"{[a.severity for a in alerts]}")
    else:
        print(f"Month {batch['month']:02d}: no alerts")

print()
monitor.summary()
"""),

# ── 10. Drift Visualization ───────────────────────────────────────────────────
code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

months  = [r["month"]    for r in results]
f_psi   = [r["feat_psi"] for r in results]
p_psi   = [r["pred_psi"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("E-Commerce Recommendation Model — 12-Month Drift Monitoring", fontsize=13)

# ── PSI over time ─────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(months, f_psi, 'o-', color='steelblue', label='Feature PSI (session_pages)')
ax.plot(months, p_psi, 's--', color='orange',   label='Prediction score PSI')
ax.axhline(0.10, color='goldenrod', linestyle=':', linewidth=1.5, label='PSI=0.10 (moderate)')
ax.axhline(0.20, color='tomato',    linestyle=':', linewidth=1.5, label='PSI=0.20 (significant)')
ax.axvspan(DRIFT_MONTH, MONTHS, alpha=0.08, color='tomato', label='Post-drift period')
ax.set_xlabel("Month")
ax.set_ylabel("PSI")
ax.set_title("PSI Rises Sharply After Month 6")
ax.legend(fontsize=8)
# The two horizontal dashed lines are alert thresholds. PSI crossing 0.20 should trigger retrain.

# ── Monthly mean feature value ────────────────────────────────────────────────
ax = axes[1]
means = [b["feature"].mean() for b in monthly_batches]
stds  = [b["feature"].std()  for b in monthly_batches]
xs = list(range(1, MONTHS + 1))
ax.errorbar(xs, means, yerr=stds, fmt='o-', color='steelblue', capsize=4, label='Mean ± std')
ax.axvline(DRIFT_MONTH - 0.5, color='tomato', linestyle='--', linewidth=1.5, label='Drift starts (month 6)')
ax.set_xlabel("Month")
ax.set_ylabel("Avg session pages")
ax.set_title("Feature Mean Drops at Month 6")
ax.legend(fontsize=9)
# Error bars show ±1 std; mean drops and spread widens after competitor launch.

plt.tight_layout()
plt.savefig('/tmp/nb45_ecommerce_drift.png', dpi=80, bbox_inches='tight')
plt.show()
print("Drift timeline figure saved.")
"""),

# ── 10. Production Considerations ────────────────────────────────────────────
md(r"""
## 10. Production Considerations

### What to Monitor

**Layer 1 — Input features**: catch data drift before it propagates.
**Layer 2 — Model outputs**: prediction distribution, confidence histogram.
**Layer 3 — Business metrics**: CTR, conversion, revenue — the ultimate truth.
**Layer 4 — System health**: latency, error rate, feature staleness.

### Operational Design

```
Kafka/Kinesis stream
  → Feature log (Parquet, daily partitions)
  → DriftMonitor (hourly batch job)
  → Alert store (PagerDuty / Slack)
  → Retraining trigger (if PSI > 0.2 for 3 consecutive days)
```

### Practical Rules of Thumb

- Monitor **top-20 features by SHAP importance** — monitoring all 200 features creates alert fatigue
- Use **PSI on reference rolling-window** (last 90 days of training), not static training set
- Require **3 consecutive above-threshold batches** before triggering retrain (prevents false alarms)
- **Shadow mode**: run new model in parallel for 1 week before promoting
- Keep **feature distribution snapshots** in a time-series store (e.g. Delta Lake) for root-cause analysis
"""),

# ── 11. Tradeoff Analysis ─────────────────────────────────────────────────────
md(r"""
## 11. Tradeoff Analysis

| Approach | Sensitivity | FP Rate | Latency | Cost |
|----------|------------|---------|---------|------|
| PSI (equal-width bins) | Medium | Medium | Low | Very low |
| PSI (quantile bins) | High | Low | Low | Low |
| KS test | High | Low | Low | Low |
| ADWIN | High (online) | Medium | Real-time | Low |
| MMD (kernel) | Very high | Low | High | High |
| Learned discriminator | Highest | Lowest | Very high | Very high |

**Recommended default**: PSI (quantile bins) for batch jobs + ADWIN for real-time streams.
Add MMD for high-stakes models (medical, financial) where false negatives are expensive.

### Monitoring Cadence Trade-offs

| Cadence | Catches drift early | Alert fatigue | Cost |
|---------|-------------------|--------------|------|
| Real-time | Yes | High | High |
| Hourly | Yes (mostly) | Medium | Medium |
| Daily | For most cases | Low | Low |
| Weekly | Too slow | Very low | Very low |
"""),

# ── 12. Senior-Level Interview Preparation ────────────────────────────────────
md(r"""
## 12. Senior-Level Interview Preparation

**Q1**: You deployed a model 6 months ago and conversion rate is down 15%. Walk me through your debugging process.

> Start with business metric → model output drift → feature drift → data pipeline.
> Isolate by segment (geo, device, user cohort). Check if a specific feature has drifted using PSI.
> Compare current feature distributions against training data. Check if labels have drifted (label drift).

**Q2**: What's the difference between PSI and KS for drift detection?

> PSI is binned, symmetric, and interpretable (which bins shifted?). KS is non-parametric, exact, and better for small samples but doesn't tell you *where* the distribution changed.

**Q3**: How do you monitor a model when you don't have ground-truth labels in production?

> Use **proxy metrics** (clicks, engagement), **delayed labels** (purchases arrive 7 days later), **model confidence** (calibrated model's uncertainty), and **input feature drift** as leading indicators.

**Q4**: What is ADWIN and when would you use it over PSI?

> ADWIN is for **online/streaming** environments where data arrives one sample at a time and you can't afford to wait for a batch. PSI is better for batch monitoring with historical reference data.

**Q5**: A new feature was added to the model inputs last week. Your PSI alert just fired. Is this drift?

> Not necessarily — it could be a distribution change from the new feature or a deployment artifact. Check PSI per-feature to isolate which feature(s) are drifting. Compare feature importance to see if the drifting feature matters.

**Q6**: How do you handle seasonal drift (Black Friday, Christmas)?

> Use a **seasonal reference baseline**: compare current week against the *same week last year*, not against training data. Alternatively, maintain rolling baselines with calendar-aware windowing.

**Q7**: Describe the monitoring stack for a large-scale production ML system.

> Feature logging → offline store (daily) → PSI/KS batch jobs (hourly) → ADWIN on real-time stream → alert routing → Slack/PagerDuty → auto-trigger retraining pipeline.

**Q8**: What is the difference between data drift and concept drift and why does it matter operationally?

> Data drift (P(X) changes) can sometimes be absorbed by the model if P(Y|X) hasn't changed. Concept drift (P(Y|X) changes) *always* requires retraining — the model's learned relationship is now wrong. Operationally: data drift is a **warning**, concept drift is an **emergency**.
"""),

# ── 13. Teach-Back Section ───────────────────────────────────────────────────
md(r"""
## 13. Teach-Back Section

Answer these to solidify understanding:

1. Define PSI in one sentence and state the two alert thresholds used in industry.
2. What does the KS statistic measure geometrically?
3. Explain in plain English how ADWIN decides to shrink its window.
4. What's the difference between **monitoring model outputs** vs **monitoring business metrics**?
5. If PSI > 0.2 but business metrics are unchanged, should you retrain? Argue both sides.
6. Why is it dangerous to use only input-feature drift as your sole alert signal?
7. How would you adapt a monitoring system for a model that makes predictions in real-time vs a batch model that scores overnight?
8. Describe one scenario where ADWIN would produce many false positives and how you'd mitigate it.
"""),

# ── 14. Exercises ─────────────────────────────────────────────────────────────
md(r"""
## 14. Exercises

### Beginner
1. Run `compute_psi` on two Normal(0,1) samples of size 100 and 10,000. Do small samples give noisy PSI? Why?
2. Add a `feature_importance` parameter to `DriftMonitor` and skip monitoring features below a threshold.
3. Modify `ks_test` to return the bin where maximum CDF divergence occurs.

### Intermediate
4. Implement **quantile-based PSI** (use reference quantiles as bin edges instead of equal-width). Compare results on a heavy-tailed distribution.
5. Add **label drift monitoring** to `DriftMonitor`: track the fraction of positive predictions over time.
6. Implement a **rolling-reference PSI** that compares the last 30 days against the previous 30 days (sliding baseline instead of static training baseline).

### Senior
7. Implement **Maximum Mean Discrepancy (MMD)** from scratch using an RBF kernel. Compare its sensitivity to PSI on a subtle mean shift of 0.1σ.
8. Design a **multi-armed alerting strategy**: different alert thresholds for different features based on their SHAP importance. High-importance features alert at PSI > 0.05; low-importance at PSI > 0.3.
9. Build a **drift root-cause analyser**: given a list of drifted features, rank them by contribution to prediction distribution shift using Shapley-style attribution.
"""),

]  # end cells

if __name__ == "__main__":
    build("09_production_ml/05_monitoring_and_drift.ipynb", cells)
