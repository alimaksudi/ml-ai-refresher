import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nbbuild import md, code, build

cells = []

cells.append(md(r"""# PROD-03 — Feature Stores
## Section 09: Production AI

*Centralised feature computation and serving: eliminate training-serving skew, enable feature reuse, guarantee point-in-time correctness.*
"""))

cells.append(md(r"""## 1. Learning Objectives

By the end of this notebook you will be able to:
- Explain the 3 core problems feature stores solve: skew, reuse, and point-in-time correctness
- Describe the dual-store architecture: offline store (training) + online store (serving)
- Define feature groups and entities and organise features by entity type
- Implement point-in-time-correct feature joins that prevent data leakage
- Implement backfill: computing historical features from event logs
- Build a mini feature store from scratch with register, materialize, and lookup
- Compare batch, micro-batch, and streaming feature freshness approaches
- Design the feature store for a production recommendation system
"""))

cells.append(md(r"""## 2. Historical Motivation

### The Feature Engineering Bottleneck (2017–2022)

Before feature stores, every ML team at every company independently re-implemented the same
feature transformations — for training, for serving, and often for A/B testing. This created
three inter-related crises:

**1. Training-Serving Skew**
Uber (2017): A demand forecasting model failed in production because the "time since last surge"
feature was computed differently in the training pipeline (offline, batch) vs the serving API
(online, real-time). The model trained on one distribution, served on another. Result: 15%
accuracy degradation in production vs offline validation.

**2. Feature Duplication**
Airbnb (2017): A survey of feature computation code found the same "user listing view count"
feature implemented 11 times across 7 teams — each with slightly different logic, different
null handling, different time windows. When one team improved it, others didn't benefit.

**3. Point-in-Time Leakage**
Common pattern in churn models: "number of support tickets this month" computed as of today
during training — but at prediction time (e.g. day 10 of the month), the month isn't complete.
Model silently used future information. Feature store point-in-time joins prevent this.

**Feature store timeline:**
- 2017: Uber Michelangelo announces first feature store concept
- 2019: Airbnb Zipline paper — formalises offline/online dual-store pattern
- 2020: Feast (open-source) 0.8 released — makes feature stores accessible
- 2021: Tecton (commercial, ex-Uber Michelangelo founders) Series B
- 2022: Every major cloud has a feature store (Vertex Feature Store, SageMaker Feature Store, Databricks Feature Store)
- 2023: Vector feature stores emerge for LLM/RAG use cases
"""))

cells.append(md(r"""## 3. Intuition and Visual Understanding

### The Dual-Store Architecture

```
                     FEATURE STORE
            ┌────────────────────────────────┐
            │         Feature Registry        │
            │  (name, entity, transform, TTL) │
            └────────────┬───────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
   OFFLINE STORE                   ONLINE STORE
   (Parquet / S3)                  (Redis / DynamoDB)
   ─────────────                  ─────────────────
   - Training data                - Serving requests
   - Backfill history             - Sub-10ms lookup
   - Point-in-time joins          - Latest values only
   - Large, cheap storage         - Small, expensive storage
         │                               │
         ▼                               ▼
    Training Pipeline              Model Serving API
    (batch, daily/weekly)          (real-time, per request)
```

### Point-in-Time Correctness

```
Timeline for entity "user_42":
─────────────────────────────────────────────────────────►
    |           |           |           |           |
  t=0         t=2         t=5         t=8         t=10
  purchase    event       event       LABEL       TODAY
              logged      logged      date

Training label: did user churn by t=10?

WRONG: Look up features as of TODAY (t=10) → uses t=8 events → leakage!
RIGHT: Look up features as of LABEL DATE (t=8) → uses t=0,t=2,t=5 only
                                                → point-in-time correct
```

### Feature Freshness Modes

```
BATCH:        Compute once per day. Cheap, simple, stale by up to 24h.
              Use for: slowly changing features (age, account tier, demographics)

MICRO-BATCH:  Compute every 5-60 min. Moderate cost. Hours-fresh.
              Use for: session-level features (browsing history, cart value)

STREAMING:    Compute on every event. Expensive. Seconds-fresh.
              Use for: real-time features (current cart, just-clicked item)
```
"""))

cells.append(md(r"""## 4. Mathematical Foundations

### 4.1 Point-in-Time Join

Given:
- Feature table $F$ with columns (entity_id, timestamp, feature_values)
- Label table $L$ with columns (entity_id, label_timestamp, label)

Point-in-time join for each row $(e, t_{label})$ in $L$:

$$\text{feature}(e, t_{label}) = F[e, \max\{t_F \in F : t_F \leq t_{label}\}]$$

i.e., take the most recent feature value that was available **at or before** the label timestamp.

### 4.2 Feature Freshness Score

$$\text{freshness}(f, t) = e^{-\lambda (t - t_{last\_computed}(f))}$$

where $\lambda$ controls how quickly staleness decays. Alert when freshness < 0.5.

### 4.3 Feature Importance via Mutual Information

For training a recommendation model, rank features by mutual information with the label:

$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Features with $I(X;Y)$ in the bottom quartile are candidates for pruning.
"""))

cells.append(code(r"""
import numpy as np
import math
import json
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Simulated timestamp helpers (no datetime.now() — use fixed base)
BASE_TS = 1_700_000_000  # fixed Unix timestamp base

def ts(offset_hours):
    return BASE_TS + offset_hours * 3600

print("Imports OK. BASE_TS =", BASE_TS)
"""))

cells.append(md(r"""## 5. Manual Implementation from Scratch

### 5.1 Feature Registry and Store
"""))

cells.append(code(r"""
# Mini Feature Store: registry, offline store, online store, point-in-time join

class FeatureDefinition:
    def __init__(self, name, entity, transform_fn, freshness_mode='batch',
                 ttl_hours=24, description=''):
        self.name           = name
        self.entity         = entity        # 'user', 'item', 'session'
        self.transform_fn   = transform_fn  # callable(raw_events) -> value
        self.freshness_mode = freshness_mode
        self.ttl_hours      = ttl_hours
        self.description    = description

class FeatureStore:
    def __init__(self):
        self._registry      = {}    # feature_name -> FeatureDefinition
        self._offline_store = defaultdict(list)
        # offline: feature_name -> [(entity_id, timestamp, value)]
        self._online_store  = {}
        # online: (feature_name, entity_id) -> latest value

    # ── Registry ──────────────────────────────────────────────────────────────

    def register(self, feature_def):
        self._registry[feature_def.name] = feature_def
        print(f"[FeatureStore] Registered '{feature_def.name}' "
              f"(entity={feature_def.entity}, mode={feature_def.freshness_mode})")

    def list_features(self, entity=None):
        return [f for f in self._registry.values()
                if entity is None or f.entity == entity]

    # ── Offline (training) ───────────────────────────────────────────────────

    def materialize(self, feature_name, entity_events):
        fd = self._registry[feature_name]
        for entity_id, events in entity_events.items():
            for event in events:
                raw   = event['raw']
                tstmp = event['timestamp']
                value = fd.transform_fn(raw)
                self._offline_store[feature_name].append(
                    (entity_id, tstmp, value)
                )
        n = sum(1 for t in self._offline_store[feature_name]
                if True)  # all records
        print(f"[FeatureStore] Materialized '{feature_name}': "
              f"{len(entity_events)} entities")

    # ── Online (serving) ─────────────────────────────────────────────────────

    def update_online(self, feature_name, entity_id, value):
        self._online_store[(feature_name, entity_id)] = value

    def lookup(self, feature_names, entity_id):
        result = {}
        for fn in feature_names:
            key = (fn, entity_id)
            result[fn] = self._online_store.get(key, None)
        return result

    # ── Point-in-Time Join (training) ─────────────────────────────────────────

    def point_in_time_join(self, label_df, feature_names):
        results = []
        for row in label_df:
            entity_id   = row['entity_id']
            label_ts    = row['label_timestamp']
            label       = row['label']
            feature_row = {'entity_id': entity_id, 'label': label}

            for fn in feature_names:
                # Find most recent feature value AT OR BEFORE label_timestamp
                records = [(t, v) for (eid, t, v) in self._offline_store[fn]
                           if eid == entity_id and t <= label_ts]
                if records:
                    best_t, best_v = max(records, key=lambda x: x[0])
                    feature_row[fn] = best_v
                    feature_row[f'{fn}_lag_h'] = (label_ts - best_t) / 3600
                else:
                    feature_row[fn] = None
                    feature_row[f'{fn}_lag_h'] = None
            results.append(feature_row)
        return results

# ── Demo ──────────────────────────────────────────────────────────────────────

fs = FeatureStore()

# Register features
fs.register(FeatureDefinition(
    'user_purchase_count_30d',
    entity='user',
    transform_fn=lambda raw: raw.get('purchase_count', 0),
    freshness_mode='batch',
    description='Number of purchases in last 30 days'
))
fs.register(FeatureDefinition(
    'user_avg_session_duration_min',
    entity='user',
    transform_fn=lambda raw: raw.get('avg_session_min', 0.0),
    freshness_mode='micro_batch',
    description='Average session duration in minutes'
))
fs.register(FeatureDefinition(
    'item_view_count_7d',
    entity='item',
    transform_fn=lambda raw: raw.get('view_count', 0),
    freshness_mode='streaming',
    description='Item view count in last 7 days'
))

# Materialize offline (simulate event data for 3 users)
user_events = {
    'u1': [
        {'timestamp': ts(-48), 'raw': {'purchase_count': 3, 'avg_session_min': 12.5}},
        {'timestamp': ts(-24), 'raw': {'purchase_count': 4, 'avg_session_min': 14.2}},
        {'timestamp': ts(-2),  'raw': {'purchase_count': 4, 'avg_session_min': 13.8}},
    ],
    'u2': [
        {'timestamp': ts(-36), 'raw': {'purchase_count': 1, 'avg_session_min': 5.0}},
        {'timestamp': ts(-12), 'raw': {'purchase_count': 2, 'avg_session_min': 6.3}},
    ],
    'u3': [
        {'timestamp': ts(-10), 'raw': {'purchase_count': 10, 'avg_session_min': 25.0}},
    ],
}

fs.materialize('user_purchase_count_30d',     user_events)
fs.materialize('user_avg_session_duration_min', user_events)

# Update online store with latest values
fs.update_online('user_purchase_count_30d',      'u1', 4)
fs.update_online('user_avg_session_duration_min', 'u1', 13.8)
fs.update_online('user_purchase_count_30d',      'u2', 2)
fs.update_online('user_avg_session_duration_min', 'u2', 6.3)

# Online lookup (serving)
print("\nOnline lookup for user u1 (serving):")
print(fs.lookup(['user_purchase_count_30d', 'user_avg_session_duration_min'], 'u1'))
"""))

cells.append(code(r"""
# Point-in-time correct join

# Label table: did user churn within 7 days after label_timestamp?
label_df = [
    {'entity_id': 'u1', 'label_timestamp': ts(-26), 'label': 0},  # look up features as of t=-26h
    {'entity_id': 'u1', 'label_timestamp': ts(-1),  'label': 0},  # look up features as of t=-1h
    {'entity_id': 'u2', 'label_timestamp': ts(-20), 'label': 1},
    {'entity_id': 'u3', 'label_timestamp': ts(-5),  'label': 0},
]

training_data = fs.point_in_time_join(
    label_df,
    ['user_purchase_count_30d', 'user_avg_session_duration_min']
)

print("Point-in-Time-Correct Training Data:")
print(f"{'entity_id':<10} {'label_ts (h offset)':>20} {'label':>6} "
      f"{'purchase_count':>16} {'lag_h':>8}")
print("-" * 65)
for row in training_data:
    eid  = row['entity_id']
    # Reconstruct hour offset
    for r in label_df:
        if r['entity_id'] == eid:
            h_off = (r['label_timestamp'] - BASE_TS) / 3600
    pc   = row.get('user_purchase_count_30d', '?')
    lag  = row.get('user_purchase_count_30d_lag_h', '?')
    print(f"  {eid:<8} {h_off:>20.0f}h  {row['label']:>6}  {str(pc):>16}  {str(lag):>8}")

print()
print("NOTE: u1 at t=-26h gets purchase_count=3 (from t=-48h feature)")
print("      u1 at t=-1h  gets purchase_count=4 (from t=-24h feature, closest before)")
print("This prevents using future feature values in training — no leakage!")
"""))

cells.append(code(r"""
# 5.2 Backfill: compute historical features from event log

class BackfillEngine:
    def __init__(self, feature_store):
        self.fs = feature_store

    def backfill_from_events(self, feature_name, event_log, entity_col,
                              ts_col, window_hours=24*30):
        fd = self.fs._registry[feature_name]
        entity_events = defaultdict(list)

        # Group events by entity
        for event in event_log:
            entity_id = event[entity_col]
            tstamp    = event[ts_col]
            entity_events[entity_id].append({
                'timestamp': tstamp,
                'raw': event
            })

        # For each entity, compute rolling feature at each event's timestamp
        records_written = 0
        for entity_id, events in entity_events.items():
            events.sort(key=lambda e: e['timestamp'])
            for i, ev in enumerate(events):
                # Window: events in the past `window_hours` before this event
                cutoff = ev['timestamp'] - window_hours * 3600
                window_events = [e for e in events[:i+1]
                                 if e['timestamp'] >= cutoff]
                # Aggregate raw values
                agg_raw = {'purchase_count': 0, 'avg_session_min': 0.0}
                for we in window_events:
                    agg_raw['purchase_count'] += we['raw'].get('purchase_count', 0)
                    agg_raw['avg_session_min'] += we['raw'].get('session_min', 0.0)
                if window_events:
                    agg_raw['avg_session_min'] /= len(window_events)

                value = fd.transform_fn(agg_raw)
                self.fs._offline_store[feature_name].append(
                    (entity_id, ev['timestamp'], value)
                )
                records_written += 1

        print(f"[Backfill] '{feature_name}': {records_written} records written "
              f"for {len(entity_events)} entities")

# Simulate purchase event log
purchase_log = [
    {'user_id': 'u4', 'timestamp': ts(-200), 'purchase_count': 1, 'session_min': 8.0},
    {'user_id': 'u4', 'timestamp': ts(-150), 'purchase_count': 2, 'session_min': 10.0},
    {'user_id': 'u4', 'timestamp': ts(-100), 'purchase_count': 1, 'session_min': 12.0},
    {'user_id': 'u4', 'timestamp': ts(-50),  'purchase_count': 3, 'session_min': 15.0},
    {'user_id': 'u5', 'timestamp': ts(-180), 'purchase_count': 0, 'session_min': 3.0},
    {'user_id': 'u5', 'timestamp': ts(-90),  'purchase_count': 1, 'session_min': 5.0},
]

# Re-register the feature in the store
fs.register(FeatureDefinition(
    'user_purchase_count_30d',
    entity='user',
    transform_fn=lambda raw: raw.get('purchase_count', 0),
    freshness_mode='batch',
    description='Number of purchases in last 30 days'
))

backfill = BackfillEngine(fs)
backfill.backfill_from_events(
    'user_purchase_count_30d',
    purchase_log,
    entity_col='user_id',
    ts_col='timestamp',
    window_hours=24*30
)
print("Backfill complete.")
"""))

cells.append(code(r"""
# 5.3 Feature Freshness Monitor

class FreshnessMontior:
    def __init__(self, decay_lambda=0.05):
        self._last_computed = {}   # feature_name -> timestamp
        self._lambda        = decay_lambda

    def record_computation(self, feature_name, timestamp):
        self._last_computed[feature_name] = timestamp

    def freshness_score(self, feature_name, current_ts):
        last = self._last_computed.get(feature_name)
        if last is None:
            return 0.0
        age_hours = (current_ts - last) / 3600
        return math.exp(-self._lambda * age_hours)

    def check_all(self, current_ts, alert_threshold=0.5):
        report = {}
        for fn, last_ts in self._last_computed.items():
            score = self.freshness_score(fn, current_ts)
            status = 'OK' if score >= alert_threshold else 'STALE'
            age_h  = (current_ts - last_ts) / 3600
            report[fn] = {'score': round(score, 3), 'age_hours': round(age_h, 1), 'status': status}
        return report

monitor = FreshnessMontior(decay_lambda=0.1)
monitor.record_computation('user_purchase_count_30d',      ts(-30))   # 30 hours ago
monitor.record_computation('user_avg_session_duration_min', ts(-4))    # 4 hours ago
monitor.record_computation('item_view_count_7d',            ts(-0.5))  # 30 min ago

current_ts = ts(0)
report = monitor.check_all(current_ts)
print("Feature Freshness Report:")
print(f"{'Feature':<35} {'Score':>6} {'Age(h)':>8} {'Status':>8}")
print("-" * 60)
for fn, r in report.items():
    print(f"  {fn:<33} {r['score']:>6.3f} {r['age_hours']:>8.1f} {r['status']:>8}")
"""))

cells.append(md(r"""## 6. Visualization
"""))

cells.append(code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── Plot 1: Dual-store architecture diagram ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
boxes = [
    ('Feature\nRegistry',  0.50, 0.85, '#1565C0', 0.30, 0.12),
    ('Offline Store\n(Parquet/S3)',  0.25, 0.55, '#1976D2', 0.30, 0.14),
    ('Online Store\n(Redis)',        0.75, 0.55, '#0288D1', 0.30, 0.14),
    ('Training\nPipeline',          0.25, 0.20, '#43A047', 0.28, 0.12),
    ('Model\nServing API',          0.75, 0.20, '#F57C00', 0.28, 0.12),
]
for text, x, y, c, w, h in boxes:
    ax1.add_patch(mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                   boxstyle='round,pad=0.01', color=c, alpha=0.85,
                   transform=ax1.transAxes))
    ax1.text(x, y, text, ha='center', va='center', fontsize=8,
             color='white', fontweight='bold', transform=ax1.transAxes)

arrow_specs = [
    (0.35, 0.85, -0.10, 0.0),   # registry -> offline
    (0.65, 0.85,  0.10, 0.0),   # registry -> online
    (0.25, 0.48,  0.0, -0.16),  # offline -> training
    (0.75, 0.48,  0.0, -0.16),  # online -> serving
]
for sx, sy, dx, dy in arrow_specs:
    ax1.annotate('', xy=(sx+dx, sy+dy), xytext=(sx, sy), xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax1.set_title('Feature Store Dual-Store Architecture\n(offline=training, online=serving)')
# Annotation: offline and online stores use the SAME feature definitions from the registry

# ── Plot 2: Point-in-time join illustration ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
# Timeline for user u1
event_ts = [ts(-48), ts(-24), ts(-2)]
event_ts_h = [-48, -24, -2]
feat_vals  = [3, 4, 4]
label_ts_h = [-26, -1]

ax2.step(event_ts_h + [0], feat_vals + [feat_vals[-1]],
         where='post', color='#1976D2', linewidth=2, label='Feature value (step function)')
for lts in label_ts_h:
    ax2.axvline(lts, color='red', linestyle='--', alpha=0.7)
    # Find the correct PIT value
    valid = [(t, v) for t, v in zip(event_ts_h, feat_vals) if t <= lts]
    if valid:
        best_t, best_v = max(valid, key=lambda x: x[0])
        ax2.plot(lts, best_v, 'ro', markersize=10, zorder=5)
        ax2.annotate(f'PIT={best_v}', xy=(lts, best_v),
                     xytext=(lts+2, best_v+0.2), fontsize=8, color='red')

ax2.scatter(event_ts_h, feat_vals, s=80, color='#1976D2', zorder=5, label='Feature computed')
ax2.set_xlabel('Hours relative to prediction')
ax2.set_ylabel('user_purchase_count_30d')
ax2.set_title('Point-in-Time Correct Join\n(red dashed = label timestamps; dots = PIT value used)')
ax2.legend(fontsize=7)
# Annotation: at label t=-26h, only events at t=-48h are visible → PIT value=3

# ── Plot 3: Feature freshness over time ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ages = np.linspace(0, 48, 200)
for lam, label in [(0.05, 'Slow decay (λ=0.05)'),
                   (0.10, 'Medium decay (λ=0.10)'),
                   (0.20, 'Fast decay (λ=0.20)')]:
    ax3.plot(ages, np.exp(-lam * ages), label=label, linewidth=2)
ax3.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Alert threshold (0.5)')
ax3.set_xlabel('Feature Age (hours)')
ax3.set_ylabel('Freshness Score')
ax3.set_title('Feature Freshness Decay\n(different decay rates for batch vs streaming features)')
ax3.legend(fontsize=8)
# Annotation: batch features (daily) need slower decay; streaming features stay fresh

# ── Plot 4: Training-serving skew illustration ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(42)
vals_train   = rng.normal(5.0, 1.5, 500)
vals_serving = rng.normal(5.0, 1.5, 500) + 1.2  # serving skew: different distribution
vals_fs_fixed= rng.normal(5.0, 1.5, 500)         # with feature store: same distribution

ax4.hist(vals_train,   bins=30, alpha=0.5, color='blue',  label='Training features')
ax4.hist(vals_serving, bins=30, alpha=0.5, color='red',   label='Serving features (skewed)')
ax4.hist(vals_fs_fixed,bins=30, alpha=0.3, color='green', label='With feature store (aligned)')
ax4.set_xlabel('Feature Value')
ax4.set_ylabel('Count')
ax4.set_title('Training-Serving Skew\n(red = skewed; green = feature store eliminates skew)')
ax4.legend(fontsize=7)
# Annotation: skew causes silent accuracy degradation; feature store uses same transform code

# ── Plot 5: Feature entity organisation ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.axis('off')
entities = {
    'User Entity': ['age', 'account_tier', 'purchase_count_30d', 'avg_session_min', 'lifetime_value'],
    'Item Entity': ['category', 'price', 'view_count_7d', 'avg_rating', 'days_since_launch'],
    'Session Entity': ['device_type', 'session_duration_min', 'pages_viewed', 'referral_source'],
}
colors_e = ['#1565C0', '#2E7D32', '#E65100']
y_start = 0.95
for (entity, features), color in zip(entities.items(), colors_e):
    ax5.text(0.05, y_start, entity, fontsize=10, fontweight='bold',
             color=color, transform=ax5.transAxes)
    y_start -= 0.06
    for feat in features:
        ax5.text(0.10, y_start, f'• {feat}', fontsize=8,
                 transform=ax5.transAxes, color='black')
        y_start -= 0.05
    y_start -= 0.03
ax5.set_title('Feature Organisation by Entity\n(each entity has its own feature group)')
# Annotation: entity-centric organisation enables reuse across models

# ── Plot 6: Freshness report bar chart ───────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
feat_names  = list(report.keys())
scores      = [report[f]['score'] for f in feat_names]
ages        = [report[f]['age_hours'] for f in feat_names]
bar_colors  = ['#43A047' if s >= 0.5 else '#EF5350' for s in scores]
short_names = [f.replace('user_', 'u_').replace('item_', 'i_') for f in feat_names]
bars = ax6.bar(short_names, scores, color=bar_colors, alpha=0.85)
ax6.axhline(0.5, color='red', linestyle='--', label='Alert threshold')
ax6.set_ylabel('Freshness Score')
ax6.set_title('Feature Freshness Status\n(green=OK, red=STALE — needs recomputation)')
ax6.legend()
ax6.set_ylim(0, 1.1)
for bar, score, age in zip(bars, scores, ages):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{score:.2f}\n({age:.0f}h)', ha='center', fontsize=7)
ax6.tick_params(axis='x', labelsize=7, rotation=15)
# Annotation: stale features hurt model accuracy; monitor and alert proactively

plt.suptitle('Feature Stores: Architecture, Point-in-Time Joins, Freshness Monitoring',
             fontsize=13, fontweight='bold')
plt.savefig('/tmp/03_feature_stores.png', dpi=100, bbox_inches='tight')
plt.close()
print("Figure saved: /tmp/03_feature_stores.png")
"""))

cells.append(md(r"""## 7. Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| **Training-serving skew** | Different code paths compute the same feature | Shared transform function in feature store |
| **Future leakage** | Training joins on wrong timestamp | Point-in-time-correct join with strict AS OF |
| **Offline-online inconsistency** | Online store not updated after backfill | Dual-write pipeline: materialize offline + update online |
| **Feature staleness alert missed** | No freshness monitoring | TTL + freshness score with Pagerduty alert |
| **Schema drift** | Upstream data changes column type | Schema validation at ingest; versioned feature definitions |
| **Cold start** | New entity has no feature history | Default values in feature definition; use item popularity features |
| **Backfill regression** | Historical computation uses wrong logic | Unit-test transform functions; compare backfill vs online results |
| **Feature explosion** | Too many features, most unused | Feature importance ranking; prune bottom quartile quarterly |
"""))

cells.append(md(r"""## 8. Production Library Implementation
"""))

cells.append(code(r"""
# Production feature store tools

try:
    import feast
    print("Feast (open-source feature store) available:")
    print("  from feast import FeatureStore, Entity, FeatureView")
    print("  store = FeatureStore(repo_path='.')")
    print("  training_df = store.get_historical_features(")
    print("      entity_df=label_df, features=['user_stats:purchase_count_30d']")
    print("  ).to_df()")
    print("  online_features = store.get_online_features(")
    print("      features=['user_stats:purchase_count_30d'],")
    print("      entity_rows=[{'user_id': 'u1'}]")
    print("  ).to_dict()")
except ImportError:
    print("feast not installed — using FeatureStore from scratch above")

try:
    import tecton
    print("Tecton (managed feature store) SDK available")
except ImportError:
    print("tecton not installed (commercial, cloud-only)")

# DuckDB for offline store (high-performance local Parquet queries)
try:
    import duckdb
    print("DuckDB available — fast offline feature queries on Parquet:")
    print("  conn = duckdb.connect()")
    print("  conn.execute(\"SELECT * FROM 'features/*.parquet' WHERE entity_id='u1'\")")
except ImportError:
    print("duckdb not installed — using in-memory dict for offline store")
"""))

cells.append(md(r"""## 9. Realistic Business Case Study

### Recommendation System Feature Store
"""))

cells.append(code(r"""
# Business case: recommendation system with 3 entity types

rng = np.random.default_rng(42)
N_USERS = 1000
N_ITEMS = 5000
N_SESSIONS = 50000

# Simulate user features
user_features = {
    f'u{i}': {
        'age':              int(rng.integers(18, 70)),
        'account_tier':     rng.choice(['free', 'premium', 'vip']),
        'purchase_count_30d': int(rng.integers(0, 20)),
        'avg_session_min':  round(float(rng.exponential(15)), 1),
        'lifetime_value':   round(float(rng.exponential(200)), 2),
    }
    for i in range(N_USERS)
}

# Simulate item features
item_features = {
    f'i{i}': {
        'category':         rng.choice(['electronics', 'clothing', 'books', 'food']),
        'price_usd':        round(float(rng.exponential(50)), 2),
        'view_count_7d':    int(rng.integers(0, 10000)),
        'avg_rating':       round(float(rng.uniform(1, 5)), 1),
        'days_since_launch':int(rng.integers(0, 365)),
    }
    for i in range(N_ITEMS)
}

# Training vs serving feature counts
print("Recommendation System Feature Store Summary")
print("=" * 50)
print(f"  Entities: User ({N_USERS}), Item ({N_ITEMS}), Session ({N_SESSIONS})")
print()
print("  USER FEATURES (5 per user):")
for feat, val in list(user_features['u0'].items()):
    print(f"    {feat:<30}: {type(val).__name__}")
print()
print("  ITEM FEATURES (5 per item):")
for feat, val in list(item_features['i0'].items()):
    print(f"    {feat:<30}: {type(val).__name__}")

# Online serving: combine user + item + context features
def serving_features(user_id, item_id, context):
    u = user_features.get(user_id, {})
    it = item_features.get(item_id, {})
    return {
        'user_purchase_count_30d': u.get('purchase_count_30d', 0),
        'user_avg_session_min':    u.get('avg_session_min', 0.0),
        'user_lifetime_value':     u.get('lifetime_value', 0.0),
        'item_view_count_7d':      it.get('view_count_7d', 0),
        'item_avg_rating':         it.get('avg_rating', 0.0),
        'item_price_usd':          it.get('price_usd', 0.0),
        'context_hour_of_day':     context.get('hour', 12),
        'context_device':          context.get('device', 'desktop'),
    }

# Simulate serving a recommendation request
feats = serving_features('u42', 'i123', {'hour': 14, 'device': 'mobile'})
print("\nServing features for user u42, item i123:")
for k, v in feats.items():
    print(f"  {k:<30}: {v}")

# Cost model
print()
print("Feature Store Operations Cost (1M daily recommendations):")
N_RECS_DAY = 1_000_000
online_lookups_per_rec = 10    # 5 user features + 5 item features
redis_cost_per_10k_ops = 0.002  # USD
offline_storage_gb     = 2.5   # Parquet for 1 year of features
s3_cost_per_gb_month   = 0.023

daily_redis  = N_RECS_DAY * online_lookups_per_rec / 10_000 * redis_cost_per_10k_ops
monthly_s3   = offline_storage_gb * s3_cost_per_gb_month
print(f"  Online store (Redis): ${daily_redis:,.2f}/day = ${daily_redis*30:,.0f}/month")
print(f"  Offline store (S3):   ${monthly_s3:.2f}/month")
print(f"  Total feature store:  ${daily_redis*30 + monthly_s3:,.0f}/month")
"""))

cells.append(md(r"""## 10. Production Considerations

### Feature Store Design Decisions

**Entity Key Design:**
- Use stable, globally unique entity IDs (UUID or hash, never auto-increment INT)
- Composite entities: (user_id, item_id) for user-item interaction features
- Session entity: (user_id, session_start_ts) — sessions are ephemeral

**Point-in-Time Join Implementation:**
- Most feature stores implement this as a sorted merge join (O(N log N))
- Alternative: range join on timestamp using columnar formats (DuckDB/Polars is fast)
- Common bug: using ≤ vs < in the join condition — use ≤ label_timestamp

**Online Store Choices:**
- Redis: fast (< 1ms), supports complex data types, expensive at scale
- DynamoDB: cheap at scale, slightly higher latency, eventual consistency by default
- Cassandra: high-write throughput, wide rows, used by Uber Michelangelo

**Serving Latency Budget:**
- Model serving SLA: 100ms end-to-end
- Feature retrieval budget: 10-20ms (bulk fetch all features for one entity)
- Solution: batch feature lookups with pipeline/multi-get; avoid N+1 lookups

**Streaming Features:**
- Kafka → Flink/Spark Streaming → online store (milliseconds-fresh)
- Cost: 5-10× more expensive than batch
- Only use for features that change faster than batch cadence AND matter for prediction

**Governance:**
- Feature catalog: searchable registry of all features + owners + documentation
- Data lineage: which models use which features — critical for upstream change impact analysis
- Access control: PII features require extra approval (GDPR, CCPA compliance)
"""))

cells.append(md(r"""## 11. Tradeoff Analysis

| Aspect | Batch Features | Micro-Batch | Streaming |
|--------|---------------|-------------|-----------|
| Freshness | Up to 24h stale | 5-60 min stale | Seconds |
| Cost | Low | Medium | High |
| Complexity | Low | Medium | High |
| Use cases | Demographics, history | Session, recent activity | Cart, clicks |

**Feature Store vs No Feature Store:**

| Dimension | No Feature Store | Feature Store |
|-----------|-----------------|---------------|
| Training-serving skew | Common (different code paths) | Eliminated (single transform) |
| Feature reuse | Zero (team-by-team reimplementation) | Full (registry enables discovery) |
| Point-in-time joins | Manual, error-prone | Built-in, tested |
| Cold start overhead | Per-model setup | Amortised across all models |
| Debugging production | Hard (can't reproduce feature values) | Easy (query offline store by timestamp) |

**Build vs Buy:**
- Build: worth it if > 5 models in production and strong platform engineering team
- Buy managed (Tecton, Vertex Feature Store): if time-to-market matters and budget allows
- Open-source (Feast): good middle ground; requires operational overhead
"""))

cells.append(md(r"""## 12. Senior-Level Interview Preparation

**Q1: What is training-serving skew and how does a feature store prevent it?**
Training-serving skew = same feature computed differently in training vs serving.
Example: training computes "days since last login" using batch ETL that runs at midnight;
serving computes it at request time using a different code path. Even a small difference
(different null handling, different timezone) causes the serving distribution to differ
from training. Feature store prevents this by ensuring the same Python transform function
is called in both cases — the training pipeline uses the feature store's offline materialization,
and serving uses the same feature store's online lookup.

**Q2: Explain point-in-time-correct joins and why they matter.**
A point-in-time join ensures that when creating a training row with label at timestamp T,
all features are looked up as they existed AT OR BEFORE T — not at data creation time.
Example: training a churn model. Label: "did user churn by month end?" Feature: "number of
support tickets this month." If you look up the ticket count today (after month end), you're
using information that wasn't available when the model would make a prediction. PIT join
selects the most recent feature value with timestamp ≤ label timestamp.

**Q3: When would you use streaming features vs batch features?**
Streaming features: when the feature changes faster than the batch cadence AND the staleness
causes meaningful degradation. Examples: current cart value (changes every click), last-clicked
item (for real-time recommendations), account balance for fraud detection.
Batch features: anything that changes slowly. Examples: demographic info, account age, 30-day
purchase count. Rule of thumb: start with batch, add streaming only if offline A/B test shows
>3% model improvement from the fresher feature. Streaming is 5-10× more expensive.

**Q4: How do you handle cold start (new entity with no feature history)?**
Three strategies: (1) Default values: register per-feature defaults in the registry
(e.g., new user gets median purchase_count); (2) Fallback features: use item popularity features
that don't require user history (item_view_count_7d, item_avg_rating); (3) Content-based
bootstrap: for new items, use text/image embeddings as features before behavioral data accrues.
The feature store should enforce that lookups return defaults, not nulls, to prevent NaN
propagation into model serving.

**Q5: A data scientist changed the "days since last purchase" feature formula. What could go wrong and how do you prevent it?**
(1) Serving/training skew: if they updated offline code but not online code (or vice versa),
the model now sees a different distribution at serving time. (2) Historical feature values change:
if the old backfill used the old formula, the training data is now inconsistent. (3) Model
regression: the downstream models were trained on the old feature — even if the new formula
is "better," the model isn't calibrated for it.
Prevention: versioned feature definitions (changing the formula = new feature version);
automatically trigger model retraining when a feature version changes; deploy both versions
in parallel during transition; A/B test the new formula.

**Q6: How would you design the feature store for a real-time fraud detection system?**
Key requirements: P99 latency < 5ms for online lookup; features change with every transaction.
Design: (1) Online store: Redis Cluster with read replicas for sub-millisecond lookups;
(2) Streaming: Kafka → Flink for transaction velocity features (transactions_last_1h, amount_last_1h);
(3) Features: per-user (historical pattern), per-merchant (fraud rate), per-card (velocity),
cross-entity (user × merchant interaction); (4) Point-in-time joins for offline training with
strict timestamp ordering; (5) Cold start: use global priors for new cards/merchants.

**Q7: What is a feature backfill and when do you need one?**
A backfill recomputes historical feature values from raw event logs — for example, computing
"purchase_count_30d" for all historical timestamps for all users, to create training data.
You need a backfill: (1) when launching a new model that needs historical training data;
(2) when you add a new feature to an existing model; (3) when you fix a bug in a feature
formula and need to recompute correct historical values. Backfills are expensive (replay all
history) — they're run once and stored in the offline store. Key correctness requirement:
the backfill must use the exact same transform function as the online store.

**Q8: How do you debug a feature store pipeline when a model degrades in production?**
Debugging steps: (1) Check feature freshness scores — is any feature stale?
(2) Compare online store values vs offline store values for the same entity + timestamp —
are they consistent? (3) Check if any upstream data source schema changed (new nulls, new categories);
(4) Run a "shadow check": log the online feature values at serving time for a random sample,
then compare to the offline backfill for the same entity × timestamp pairs;
(5) Plot the feature distribution at serving time vs training time for each feature — look for
distribution shift using PSI or KS test; (6) Check if a feature definition was changed without
version bump (which would cause inconsistency between stored offline values and new online compute).
"""))

cells.append(md(r"""## 13. Teach-Back Section

Explain each of these from memory:

1. **Training-serving skew**: Draw the two code paths (training vs serving) that produce
   different feature values. Now explain how a feature store collapses these two paths into one.

2. **Point-in-time join mechanics**: Given a timeline with 4 feature update events and 3
   label timestamps, draw which feature value is used for each label. What is the exact rule?

3. **Offline vs online store**: Compare the two stores on: latency requirements, storage
   format, access pattern (bulk vs single-entity), cost, and typical implementation.

4. **Feature freshness**: For a fraud detection feature "transactions_in_last_10_minutes",
   what freshness mode (batch/micro-batch/streaming) do you use and why?

5. **Backfill**: A data scientist trains a churn model. They need 3 features: age (static),
   purchase_count_30d (batch), pages_viewed_session (streaming). Which features need a backfill,
   and what does the backfill pipeline look like?

6. **Cold start**: A new user signs up. Your recommendation model needs 8 features, 5 of which
   require at least 7 days of user history. How does your feature store handle the serving request?

7. **Feature versioning**: The feature "user_avg_purchase_value" currently excludes refunds.
   You want to add refund subtraction. Walk through the process of updating this feature without
   breaking models that use the current version.

8. **Feature store ROI**: Your company has 3 models, each with 20 features, and 5 engineers
   each spending 20% of time on feature pipelines. Estimate the time savings from a feature store.
"""))

cells.append(md(r"""## 14. Exercises

### Beginner
1. Implement a `FeatureStore.lookup_batch` method that takes a list of entity IDs and returns
   a dict of `{entity_id: {feature_name: value}}` in one call (no N+1 queries).
2. Add TTL enforcement to the online store: `update_online(feature_name, entity_id, value, ttl_hours)`
   — return `None` on lookup if the entry has expired.
3. Implement a `FeatureRegistry.search(query)` method that returns features whose name or
   description contains the query string (case-insensitive).

### Intermediate
4. Implement a proper point-in-time join using `pandas`: given `label_df` (entity_id, label_ts, label)
   and `feature_df` (entity_id, feature_ts, feature_value), perform an as-of merge that returns
   the most recent feature value at or before each label timestamp. Use `pd.merge_asof`.
5. Implement a `FeatureImportanceRanker` that computes mutual information between each feature
   and a binary label using binning (divide continuous features into 10 bins, compute I(X;Y)).
   Rank features and flag the bottom quartile for pruning.
6. Simulate a streaming feature with Kafka-like semantics: implement a `StreamingFeature` class
   where events arrive with timestamps, and `lookup(entity_id, as_of_ts)` returns the value
   from the most recent event before as_of_ts.

### Senior
7. **Point-in-time join at scale**: Implement a `DistributedPITJoin` for a target
   workload of 1M label rows and 10M feature rows using a partitioned sort-merge/as-of
   strategy. Benchmark both methods only on safe samples (for example 1k×10k), verify
   correctness on identical inputs, fit an empirical scaling curve, and extrapolate the
   naive runtime. Do **not** execute the full $10^{13}$-comparison nested loop. Report
   memory, partitioning, skew, and spill-to-disk assumptions for the target workload.
8. **Feature drift detection**: For each feature in the offline store, implement a `DriftDetector`
   that compares the distribution of the feature over two time windows (last 7 days vs previous 30
   days) using the Kolmogorov-Smirnov test. Alert when KS statistic > 0.1. Simulate 5 features
   where 2 have engineered drift.
"""))

build("09_production_ml/03_feature_stores.ipynb", cells)
print("PROD-03 built.")
