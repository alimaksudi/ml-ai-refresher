import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nbbuild import md, code, build

cells = []

cells.append(md(r"""# PROD-01 — MLOps
## Section 09: Production AI

*The operational discipline for reliably deploying, monitoring, and maintaining ML models at scale.*
"""))

cells.append(md(r"""## 1. Learning Objectives

By the end of this notebook you will be able to:
- Describe the 4 MLOps maturity levels and identify where a real organisation sits
- Trace data through the full ML pipeline: ingestion → features → training → evaluation → deployment → monitoring
- Write CI/CD tests for data schemas, model metrics, and serving contracts
- Implement a model registry from scratch: register, promote, rollback with metadata and lineage
- Design a canary deployment: traffic split, metric comparison, rollback criteria
- Calculate sample size for an A/B test on model accuracy
- Explain data versioning (DVC) concepts and why "data as code" matters
- Design the MLOps pipeline for a production churn prediction model
"""))

cells.append(md(r"""## 2. Historical Motivation

### From Notebooks to Production (2015–2023)

The central crisis of applied ML (2015–2020): companies were **great at training models,
terrible at running them**. The "research-to-production gap" was the dominant failure mode.

**Timeline of the MLOps discipline:**
- 2015: Sculley et al. "Hidden Technical Debt in Machine Learning Systems" (NIPS)
  → First systematic description of ML-specific technical debt (data dependencies, feedback loops,
    configuration debt, pipeline jungles)
- 2017: Uber Michelangelo paper — first public description of an industrial ML platform
  (feature store, training pipelines, model serving, monitoring in one system)
- 2019: Google "Practitioners guide to MLOps" — defines maturity levels 0-3
- 2020: MLflow 1.0, Kubeflow 1.0, DVC release — open-source MLOps stack matures
- 2021 onward: industry surveys continued to report a large research-to-production
  gap, but estimates vary by survey population and by what counts as "production"
- 2022: Vertex AI, SageMaker, Azure ML mature — cloud-managed MLOps platforms
- 2023: ML in production at every FAANG company; MLOps engineer is now a distinct job title

**Key insight from Sculley et al.:** ML code is typically 5% of the total codebase.
The remaining 95% — data pipelines, serving infrastructure, monitoring, feature engineering —
is what MLOps addresses.
"""))

cells.append(md(r"""## 3. Intuition and Visual Understanding

### MLOps Maturity Levels

```
LEVEL 0: Manual Process
├── Data scientists train models in notebooks
├── Manual handoff to engineering for deployment
├── No monitoring, no retraining trigger
└── Failure mode: model rots silently

LEVEL 1: ML Pipeline Automation
├── Training pipeline is code (not notebooks)
├── Automated model evaluation before deployment
├── Basic monitoring (latency, error rate)
└── Manual trigger for retraining

LEVEL 2: CI/CD for ML Pipelines
├── Pipeline itself is version-controlled
├── Automated tests for data, model, serving
├── Automated retraining on data drift
└── Feature store for consistent feature serving

LEVEL 3: Full Automation + Governance
├── Continuous training on streaming data
├── Shadow mode testing before promotion
├── Model cards, lineage, audit trail
└── Regulatory compliance (GDPR, FCRA)
```

### The ML Pipeline

```
[Raw Data] → [Ingest] → [Validate] → [Feature Eng.] → [Train] → [Evaluate]
                                                                       ↓
[Monitoring] ← [Serving] ← [Registry] ← [Package] ← [Test] ←────────┘
     ↓
[Drift Alert] → [Retrain Trigger] → (loop back to Train)
```

### Canary Deployment

```
Traffic:  100% → Old Model
Canary:     5% → New Model    ← measure: accuracy, latency, error rate
                               ← compare against old model baseline
If new model wins: 5% → 25% → 50% → 100% (gradual rollout)
If new model loses: 0% → rollback
```
"""))

cells.append(md(r"""## 4. Mathematical Foundations

### 4.1 A/B Test Sample Size

For an equal-sized, two-sided comparison of independent proportions, detect a
minimum effect δ in model accuracy with:
- α = significance level (type I error, e.g. 0.05)
- β = type II error (e.g. 0.20, so power = 80%)
- p₀ = baseline accuracy

$$\bar p=\frac{p_0+p_1}{2},\qquad
n_{\text{per arm}}\approx
\frac{\left[z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}+
z_{1-\beta}\sqrt{p_0(1-p_0)+p_1(1-p_1)}\right]^2}{(p_1-p_0)^2}.$$

**Read and symbols:** $p_1=p_0+\delta$ is the target rate; $\bar p$ is the
midpoint rate; $z_q$ is the standard-normal quantile with fraction $q$ below it;
$z_{1-\alpha/2}=1.96$ and $z_{1-\beta}=0.842$ for α=0.05 and 80% power.
$n_{\text{per arm}}$ is required in each arm, not total. Clustered users,
repeated measurements, sequential peeking, and non-binary metrics need a
different design.

### 4.2 Canary Rollout Decision Rule

At each check interval, compute the lift:
$$\text{lift} = \frac{\text{metric}_{new} - \text{metric}_{old}}{\text{metric}_{old}}$$

Roll out if: lift > min_lift_threshold AND p-value < α AND N > n_min

Roll back if: lift < -rollback_threshold OR error_rate_{new} > error_rate_{old} × 1.5

### 4.3 Model Staleness Score

$$\text{staleness} = 1 - \frac{\text{performance}(t)}{\text{performance}(t_0)}$$

where $t_0$ is the training date and $t$ is current time.
Retrain trigger: staleness > threshold (e.g. 0.05 = 5% performance degradation).
"""))

cells.append(code(r"""
import numpy as np
import math
import json
import uuid
import time
from datetime import datetime, timedelta
from collections import defaultdict
from statistics import NormalDist
import warnings
warnings.filterwarnings('ignore')

print("Imports OK")
"""))

cells.append(md(r"""## 5. Manual Implementation from Scratch

### 5.1 Model Registry
"""))

cells.append(code(r"""
# Mini Model Registry: register, promote, rollback, query lineage

class ModelRegistry:
    STAGES = ['registered', 'staging', 'production', 'archived']

    def __init__(self):
        self._models = {}           # name -> list of version dicts
        self._production = {}       # name -> version string (current prod)

    def register(self, name, version, metrics, artifact_path,
                 training_data_hash, parent_version=None):
        if name not in self._models:
            self._models[name] = []
        entry = {
            'name':               name,
            'version':            version,
            'stage':              'registered',
            'metrics':            metrics,
            'artifact_path':      artifact_path,
            'training_data_hash': training_data_hash,
            'parent_version':     parent_version,
            'registered_at':      datetime.utcnow().isoformat(),
            'run_id':             str(uuid.uuid4())[:8],
        }
        self._models[name].append(entry)
        print(f"[Registry] Registered {name} v{version}  run_id={entry['run_id']}")
        return entry

    def _get_version(self, name, version):
        for v in self._models.get(name, []):
            if v['version'] == version:
                return v
        raise KeyError(f"Model {name} v{version} not found")

    def transition_stage(self, name, version, new_stage):
        assert new_stage in self.STAGES, f"Invalid stage: {new_stage}"
        entry = self._get_version(name, version)
        old_stage = entry['stage']

        # Archive the current production version when promoting a new one
        if new_stage == 'production' and name in self._production:
            old_prod_ver = self._production[name]
            if old_prod_ver != version:
                old_entry = self._get_version(name, old_prod_ver)
                old_entry['stage'] = 'archived'
                print(f"[Registry] Archived {name} v{old_prod_ver}")

        entry['stage'] = new_stage
        if new_stage == 'production':
            self._production[name] = version

        print(f"[Registry] {name} v{version}: {old_stage} → {new_stage}")
        return entry

    def get_production(self, name):
        if name not in self._production:
            raise KeyError(f"No production model for {name}")
        return self._get_version(name, self._production[name])

    def rollback(self, name):
        versions = self._models.get(name, [])
        # Find most recently archived version
        archived = [v for v in reversed(versions) if v['stage'] == 'archived']
        if not archived:
            raise RuntimeError(f"No archived version to rollback to for {name}")
        prev = archived[0]
        print(f"[Registry] Rolling back {name} to v{prev['version']}")
        return self.transition_stage(name, prev['version'], 'production')

    def lineage(self, name, version):
        chain = []
        current = version
        while current:
            entry = self._get_version(name, current)
            chain.append({'version': entry['version'],
                          'metrics': entry['metrics'],
                          'stage':   entry['stage']})
            current = entry.get('parent_version')
        return chain

    def list_versions(self, name):
        return self._models.get(name, [])

# Demo
registry = ModelRegistry()

# Register 3 versions of a churn model
registry.register('churn_model', 'v1.0', metrics={'auc': 0.812, 'f1': 0.74},
                  artifact_path='s3://models/churn/v1.0/',
                  training_data_hash='abc123')

registry.register('churn_model', 'v1.1', metrics={'auc': 0.829, 'f1': 0.76},
                  artifact_path='s3://models/churn/v1.1/',
                  training_data_hash='def456', parent_version='v1.0')

registry.register('churn_model', 'v2.0', metrics={'auc': 0.851, 'f1': 0.79},
                  artifact_path='s3://models/churn/v2.0/',
                  training_data_hash='ghi789', parent_version='v1.1')

# Promote v1.1 to staging, then production
registry.transition_stage('churn_model', 'v1.1', 'staging')
registry.transition_stage('churn_model', 'v1.1', 'production')

# Promote v2.0 (archives v1.1 automatically)
registry.transition_stage('churn_model', 'v2.0', 'staging')
registry.transition_stage('churn_model', 'v2.0', 'production')

# Rollback
registry.rollback('churn_model')
print(f"\nCurrent production: {registry.get_production('churn_model')['version']}")

# Lineage
print("\nModel lineage:")
for entry in registry.lineage('churn_model', 'v2.0'):
    print(f"  {entry['version']}  AUC={entry['metrics']['auc']}  [{entry['stage']}]")
"""))

cells.append(code(r"""
# 5.2 CI/CD Tests for ML Pipelines

class DataValidator:
    def __init__(self, expected_schema, min_rows=100, max_null_frac=0.1):
        self.schema       = expected_schema      # {col: dtype}
        self.min_rows     = min_rows
        self.max_null_frac = max_null_frac

    def validate(self, data_dict):
        # data_dict: {column_name: list_of_values}
        results = {}

        # Schema check
        for col, expected_dtype in self.schema.items():
            if col not in data_dict:
                results[f'schema_missing_{col}'] = 'FAIL'
            else:
                results[f'schema_{col}'] = 'PASS'

        # Row count check
        n_rows = len(next(iter(data_dict.values())))
        results['min_rows'] = 'PASS' if n_rows >= self.min_rows else f'FAIL ({n_rows}<{self.min_rows})'

        # Null fraction check
        for col, values in data_dict.items():
            null_frac = sum(v is None or (isinstance(v, float) and math.isnan(v))
                           for v in values) / max(len(values), 1)
            key = f'null_frac_{col}'
            results[key] = 'PASS' if null_frac <= self.max_null_frac else f'FAIL ({null_frac:.2f})'

        return results

class ModelEvaluator:
    def __init__(self, min_auc=0.75, max_latency_ms=50, min_throughput_qps=100):
        self.min_auc          = min_auc
        self.max_latency_ms   = max_latency_ms
        self.min_throughput   = min_throughput_qps

    def evaluate(self, metrics):
        results = {}
        results['auc_threshold']       = 'PASS' if metrics.get('auc', 0) >= self.min_auc else 'FAIL'
        results['latency_threshold']   = 'PASS' if metrics.get('p99_ms', 999) <= self.max_latency_ms else 'FAIL'
        results['throughput_threshold']= 'PASS' if metrics.get('qps', 0) >= self.min_throughput else 'FAIL'
        return results

# Run CI tests
schema = {'age': 'int', 'tenure_months': 'int', 'monthly_charges': 'float', 'churn': 'int'}
validator = DataValidator(expected_schema=schema, min_rows=50)

rng = np.random.default_rng(42)
n = 200
data = {
    'age':             list(rng.integers(18, 80, n)),
    'tenure_months':   list(rng.integers(0, 120, n)),
    'monthly_charges': list(rng.uniform(20, 120, n).round(2)),
    'churn':           list(rng.integers(0, 2, n)),
}
data_results = validator.validate(data)
print("Data CI tests:")
for k, v in data_results.items():
    icon = 'PASS' if v == 'PASS' else 'FAIL'
    print(f"  {'OK' if icon=='PASS' else 'NG'}  {k}: {v}")

model_metrics = {'auc': 0.851, 'p99_ms': 32, 'qps': 450}
evaluator = ModelEvaluator()
model_results = evaluator.evaluate(model_metrics)
print("\nModel CI tests:")
for k, v in model_results.items():
    print(f"  {'OK' if v=='PASS' else 'NG'}  {k}: {v}")

all_pass = all(v == 'PASS' for v in {**data_results, **model_results}.values())
print(f"\nCI pipeline: {'ALL PASS — deploy proceed' if all_pass else 'FAILED — block deploy'}")
"""))

cells.append(code(r"""
# 5.3 Canary Deployment Simulation

class CanaryDeployment:
    def __init__(self, model_name, old_version, new_version,
                 traffic_schedule=None, min_lift=0.01, rollback_threshold=-0.03):
        self.model_name         = model_name
        self.old_version        = old_version
        self.new_version        = new_version
        self.traffic_schedule   = traffic_schedule or [0.05, 0.25, 0.50, 1.00]
        self.min_lift           = min_lift
        self.rollback_threshold = rollback_threshold
        self.history            = []

    def _simulate_metrics(self, version, n_requests, rng):
        # New model: +3% AUC, slightly higher latency
        if version == self.new_version:
            accuracy = rng.beta(8.55, 1.45, n_requests)   # mean ~0.855
            latency  = rng.exponential(28, n_requests)
        else:
            accuracy = rng.beta(8.12, 1.88, n_requests)   # mean ~0.812
            latency  = rng.exponential(25, n_requests)
        error_rate = rng.binomial(1, 0.002, n_requests).mean()
        return {
            'accuracy':   accuracy.mean(),
            'p99_ms':     float(np.percentile(latency, 99)),
            'error_rate': error_rate,
            'n':          n_requests,
        }

    def run(self, total_requests_per_step=5000, seed=42):
        rng = np.random.default_rng(seed)
        current_traffic = 0.0
        decision = 'unknown'

        for step, new_frac in enumerate(self.traffic_schedule):
            n_new = int(total_requests_per_step * new_frac)
            n_old = total_requests_per_step - n_new

            m_new = self._simulate_metrics(self.new_version, max(n_new, 1), rng)
            m_old = self._simulate_metrics(self.old_version, max(n_old, 1), rng)

            lift = (m_new['accuracy'] - m_old['accuracy']) / m_old['accuracy']
            err_ratio = m_new['error_rate'] / max(m_old['error_rate'], 1e-6)

            step_result = {
                'step':        step + 1,
                'new_traffic': f"{new_frac*100:.0f}%",
                'lift':        lift,
                'err_ratio':   err_ratio,
                'new_accuracy': m_new['accuracy'],
                'old_accuracy': m_old['accuracy'],
            }

            if lift < self.rollback_threshold or err_ratio > 1.5:
                step_result['decision'] = 'ROLLBACK'
                self.history.append(step_result)
                decision = 'rollback'
                break
            elif lift >= self.min_lift:
                step_result['decision'] = 'CONTINUE'
                current_traffic = new_frac
                if new_frac == 1.0:
                    decision = 'full_rollout'
            else:
                step_result['decision'] = 'HOLD'

            self.history.append(step_result)

        return decision

canary = CanaryDeployment('churn_model', 'v1.1', 'v2.0')
result = canary.run()
print(f"Canary result: {result.upper()}\n")
print(f"{'Step':<6} {'Traffic':>8} {'Old Acc':>8} {'New Acc':>8} {'Lift':>8} {'Decision'}")
print("-" * 55)
for h in canary.history:
    print(f"  {h['step']:<4} {h['new_traffic']:>8} {h['old_accuracy']:>8.4f} {h['new_accuracy']:>8.4f} {h['lift']:>+7.3f}  {h['decision']}")
"""))

cells.append(code(r"""
# 5.4 A/B Test Sample Size Calculator

def ab_test_sample_size(p0, delta, alpha=0.05, power=0.80):
    # Approximate per-arm sample size for two independent proportions.
    p1 = p0 + delta
    if not (0 < p0 < 1 and 0 < p1 < 1):
        raise ValueError("p0 and p0 + delta must be between 0 and 1")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    p_bar = (p0 + p1) / 2
    null_variance = 2 * p_bar * (1 - p_bar)
    alternative_variance = p0 * (1 - p0) + p1 * (1 - p1)
    numerator = z_alpha * math.sqrt(null_variance) + z_power * math.sqrt(alternative_variance)
    n = numerator**2 / (delta**2)
    return math.ceil(n)

# Example: churn model, baseline binary accuracy 0.812, detect +2 percentage points
p0    = 0.812
delta = 0.020

n = ab_test_sample_size(p0, delta)
print(f"A/B Test Sample Size Calculator")
print(f"  Baseline accuracy:  {p0}")
print(f"  Min detectable eff: {delta:+.3f}")
print(f"  Alpha (two-tailed): 0.05")
print(f"  Power:              80%")
print(f"  Required per group: {n:,}")
print(f"  Total required:     {n*2:,}")
print()

# Sensitivity table
print(f"{'Delta':>8} {'Sample/group':>14} {'Total':>10}")
for d in [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]:
    n = ab_test_sample_size(p0, d)
    print(f"  {d:>6.3f} {n:>14,} {n*2:>10,}")
"""))

cells.append(md(r"""## 6. Visualization
"""))

cells.append(code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: MLOps maturity level diagram ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
levels = ['Level 0\nManual', 'Level 1\nPipeline\nAutomation',
          'Level 2\nCI/CD for\nML', 'Level 3\nFull Auto\n+Governance']
capabilities = [2, 5, 8, 10]
ax1.barh(levels, capabilities, color=['#EF5350', '#FFA726', '#66BB6A', '#42A5F5'], alpha=0.85)
ax1.set_xlabel('MLOps Capability Score (0–10)')
ax1.set_title('MLOps Maturity Levels\n(most organisations are at Level 0–1)')
ax1.set_xlim(0, 12)
labels = ['Notebook-only', 'Automated pipelines', 'Full CI/CD', 'Autonomous ML']
for i, (val, label) in enumerate(zip(capabilities, labels)):
    ax1.text(val + 0.2, i, label, va='center', fontsize=8)
# Annotation: each level requires explicit investment in tooling and process

# ── Plot 2: Model registry version timeline ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
versions = ['v1.0', 'v1.1', 'v2.0']
aucs     = [0.812, 0.829, 0.851]
stages   = ['archived', 'production', 'staging']  # after rollback
stage_colors = {'registered': '#9E9E9E', 'staging': '#FFA726',
                'production': '#43A047', 'archived': '#78909C'}
colors = [stage_colors[s] for s in stages]
bars = ax2.bar(versions, aucs, color=colors, alpha=0.85)
ax2.set_ylim(0.79, 0.87)
ax2.set_ylabel('AUC Score')
ax2.set_title('Model Registry: Version Progression\n(color = stage after rollback)')
for bar, auc, stage in zip(bars, aucs, stages):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{auc:.3f}\n[{stage}]', ha='center', fontsize=8)
patches = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85) for c in stage_colors.values()]
ax2.legend(patches, stage_colors.keys(), fontsize=7, loc='lower right')
# Annotation: rollback moves v1.1 back to production; v2.0 is demoted

# ── Plot 3: Canary rollout ────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
steps    = [h['step'] for h in canary.history]
new_accs = [h['new_accuracy'] for h in canary.history]
old_accs = [h['old_accuracy'] for h in canary.history]
traffics = [float(h['new_traffic'].replace('%',''))/100 for h in canary.history]
ax3.plot(steps, new_accs, 'g-o', label='New model (v2.0)', linewidth=2)
ax3.plot(steps, old_accs, 'b--s', label='Old model (v1.1)', linewidth=2)
ax3_twin = ax3.twinx()
ax3_twin.bar(steps, traffics, alpha=0.2, color='purple', label='New traffic %')
ax3_twin.set_ylabel('New Model Traffic Fraction', color='purple')
ax3_twin.set_ylim(0, 1.5)
ax3.set_xlabel('Canary Step')
ax3.set_ylabel('Accuracy')
ax3.set_title('Canary Deployment: Traffic Ramp + Metric Comparison\n(green rising = rollout proceeding)')
ax3.legend(loc='lower right', fontsize=8)
# Annotation: each step increases new model traffic; metrics compared at each checkpoint

# ── Plot 4: A/B test sample size vs delta ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
deltas = np.linspace(0.005, 0.05, 50)
ns     = [ab_test_sample_size(0.812, d) for d in deltas]
ax4.plot(deltas * 100, ns, 'b-', linewidth=2)
ax4.fill_between(deltas * 100, ns, alpha=0.1, color='blue')
ax4.axvline(2.0, color='red', linestyle='--', alpha=0.7, label='MDE=2% → requires large N')
ax4.set_xlabel('Minimum Detectable Effect (%)')
ax4.set_ylabel('Required Sample Size per Group')
ax4.set_title('A/B Test Sample Size vs MDE\n(smaller effect = exponentially more data)')
ax4.legend()
ax4.set_yscale('log')
# Annotation: the curve follows 1/delta^2 — halving the MDE quadruples sample size

# ── Plot 5: CI/CD pipeline test results ──────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
all_tests = {**data_results, **model_results}
test_names = [k.replace('_', '\n') for k in all_tests.keys()]
test_vals  = [1 if v == 'PASS' else 0 for v in all_tests.values()]
colors5    = ['#43A047' if v else '#EF5350' for v in test_vals]
ax5.barh(test_names, test_vals, color=colors5, alpha=0.85)
ax5.set_xlim(0, 1.5)
ax5.set_title('CI/CD Test Suite Results\n(green=PASS, red=FAIL — blocks deploy)')
ax5.set_xlabel('Pass (1) / Fail (0)')
ax5.set_xticks([0, 1])
ax5.set_xticklabels(['FAIL', 'PASS'])
# Annotation: all gates must pass before a model advances to the next pipeline stage

# ── Plot 6: ML pipeline stages ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
stages_pipeline = [
    ('Data\nIngest', 0.05, 0.65, '#42A5F5'),
    ('Validate', 0.20, 0.65, '#66BB6A'),
    ('Feature\nEng.', 0.35, 0.65, '#FFA726'),
    ('Train', 0.50, 0.65, '#EF5350'),
    ('Evaluate', 0.65, 0.65, '#AB47BC'),
    ('Registry', 0.80, 0.65, '#26C6DA'),
    ('Serve', 0.50, 0.30, '#8D6E63'),
    ('Monitor', 0.20, 0.30, '#78909C'),
]
for label, x, y, c in stages_pipeline:
    ax6.add_patch(mpatches.FancyBboxPatch((x-0.07, y-0.1), 0.14, 0.18,
                                           boxstyle="round,pad=0.01", color=c, alpha=0.8,
                                           transform=ax6.transAxes))
    ax6.text(x, y, label, ha='center', va='center', fontsize=7,
             color='white', fontweight='bold', transform=ax6.transAxes)

arrows = [(0.12,0.65,0.05),(0.27,0.65,0.05),(0.42,0.65,0.05),(0.57,0.65,0.05),(0.72,0.65,0.05)]
for ax_x, ay, dx in arrows:
    ax6.annotate('', xy=(ax_x+dx, ay), xytext=(ax_x, ay), xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax6.annotate('', xy=(0.20, 0.40), xytext=(0.50, 0.55), xycoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax6.annotate('', xy=(0.50, 0.40), xytext=(0.50, 0.55), xycoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax6.set_title('ML Pipeline Architecture\n(continuous loop: monitor → retrain trigger)', pad=10)
# Annotation: the pipeline is a loop — monitoring feeds back into retraining

plt.suptitle('MLOps: Model Registry, Canary Deployments, CI/CD, A/B Testing', fontsize=13, fontweight='bold')
plt.savefig('/tmp/01_mlops.png', dpi=100, bbox_inches='tight')
plt.close()
print("Figure saved: /tmp/01_mlops.png")
"""))

cells.append(md(r"""## 7. Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| **Training-serving skew** | Features computed differently offline vs online | Shared feature store; log online features |
| **Model staleness** | No retraining trigger; model rots | Staleness score + automated retraining |
| **Data pipeline failure** | Upstream source changes schema | Schema validation at ingestion; circuit breaker |
| **Silent model degradation** | Performance decays but no alert | Effect-size drift alerts plus delayed-label performance monitoring |
| **Reproducibility failure** | Can't recreate a model version | Pin all dependencies; version data (DVC) |
| **Canary with wrong metric** | Optimise latency, miss accuracy drop | Define rollout gates on multiple metrics |
| **Registry without governance** | No audit trail; can't explain past decisions | Immutable registry entries; model cards |
| **Feature leakage in CI** | Test data leaks into validation | Strict temporal splits in all CI/CD tests |
"""))

cells.append(md(r"""## 8. Production Library Implementation
"""))

cells.append(code(r"""
# Production MLOps tools

try:
    import mlflow
    print("MLflow available:")
    print("  mlflow.log_metric('auc', 0.851)")
    print("  mlflow.register_model(model_uri, 'churn_model')")
    print("  client.transition_model_version_stage('churn_model', 1, 'Production')")
except ImportError:
    print("mlflow not installed — using scratch ModelRegistry above")

try:
    import sklearn.pipeline
    print("sklearn Pipeline available for reproducible feature+model packaging")
except ImportError:
    print("sklearn not installed")

try:
    import great_expectations as ge
    print("Great Expectations available for advanced data validation")
except ImportError:
    print("great_expectations not installed — using DataValidator from scratch")

# DVC concepts (data versioning)
print()
print("DVC data versioning workflow (conceptual):")
print("  dvc init                          # init DVC in git repo")
print("  dvc add data/churn_2024_q1.csv   # hash + .dvc pointer file")
print("  git add data/churn_2024_q1.csv.dvc")
print("  git commit -m 'data: Q1 2024 churn dataset'")
print("  dvc push                          # upload to remote (S3/GCS)")
print("  dvc pull                          # reproduce exact dataset version")
"""))

cells.append(md(r"""## 9. Realistic Business Case Study

### Churn Prediction: Notebook → Production CI/CD

**Context**: An e-commerce company wants to deploy a churn prediction model.
Data scientists trained a Random Forest in a Jupyter notebook (Level 0 MLOps).
The task: elevate it to Level 2 (automated CI/CD pipeline).
"""))

cells.append(code(r"""
# Business case: pipeline cost model and ROI

# Current state (Level 0)
level0 = {
    'deploy_time_days':     14,   # manual handoff, IT tickets
    'model_updates_per_yr': 2,    # twice a year, manually
    'incident_response_hrs': 6,   # time to detect + fix silent degradation
    'incidents_per_yr':     4,
    'ds_ops_hours_per_yr':  120,  # data scientist time on ops
    'ds_hourly_rate':       80,
}

# Target state (Level 2)
level2 = {
    'deploy_time_days':     0.5,   # automated, 4-hour pipeline
    'model_updates_per_yr': 52,    # weekly automated retraining
    'incident_response_hrs': 0.5,  # automated rollback
    'incidents_per_yr':     1,
    'ds_ops_hours_per_yr':  20,    # only escalations
    'platform_cost_yr':     24000, # cloud compute + tooling
}

# Business value
AVG_CHURN_VALUE = 250  # USD lifetime value saved per retained customer
N_CUSTOMERS     = 50000
CHURN_RATE      = 0.12
MODEL_LIFT      = 0.18  # 18% better retention vs no model

customers_saved_level0 = N_CUSTOMERS * CHURN_RATE * MODEL_LIFT * 0.80  # 80% uptime
customers_saved_level2 = N_CUSTOMERS * CHURN_RATE * MODEL_LIFT * 0.98  # 98% uptime (auto retrain)

revenue_level0 = customers_saved_level0 * AVG_CHURN_VALUE
revenue_level2 = customers_saved_level2 * AVG_CHURN_VALUE
ops_cost_level0 = (level0['ds_ops_hours_per_yr'] * level0['ds_hourly_rate'] +
                   level0['incidents_per_yr'] * level0['incident_response_hrs'] * 300)
ops_cost_level2 = (level2['ds_ops_hours_per_yr'] * level0['ds_hourly_rate'] +
                   level2['incidents_per_yr'] * level2['incident_response_hrs'] * 300 +
                   level2['platform_cost_yr'])

print("Churn Model: Level 0 vs Level 2 MLOps")
print("=" * 55)
print(f"{'Metric':<35} {'Level 0':>8} {'Level 2':>8}")
print("-" * 55)
print(f"  Deploy time               {level0['deploy_time_days']:>6.0f}d  {level2['deploy_time_days']:>6.1f}d")
print(f"  Model updates/year        {level0['model_updates_per_yr']:>8}  {level2['model_updates_per_yr']:>8}")
print(f"  Incidents/year            {level0['incidents_per_yr']:>8}  {level2['incidents_per_yr']:>8}")
print(f"  Customers retained/year   {customers_saved_level0:>8.0f}  {customers_saved_level2:>8.0f}")
print(f"  Revenue from model        ${revenue_level0:>7,.0f}  ${revenue_level2:>7,.0f}")
print(f"  Operational cost/year     ${ops_cost_level0:>7,.0f}  ${ops_cost_level2:>7,.0f}")
print(f"  Net value/year            ${revenue_level0-ops_cost_level0:>7,.0f}  ${revenue_level2-ops_cost_level2:>7,.0f}")
print()
lift = (revenue_level2 - ops_cost_level2) - (revenue_level0 - ops_cost_level0)
print(f"Level 2 MLOps incremental value: ${lift:,.0f}/year")
"""))

cells.append(md(r"""## 10. Production Considerations

### Key Production Patterns

**Feature Store:**
- Single source of truth for feature computation — eliminates training/serving skew
- Offline store (Parquet/S3) for training; online store (Redis/DynamoDB) for serving
- Key providers: Feast (open-source), Tecton, Vertex AI Feature Store

**Model Monitoring:**
- Input drift: monitor distribution of incoming features vs training distribution
  - KL divergence, PSI (Population Stability Index), Kolmogorov-Smirnov test
- Output drift: monitor prediction distribution (should match expected churn rate)
- Concept drift: monitor actual outcomes vs predictions over time
- Alert thresholds: PSI > 0.2 = "significant shift", > 0.1 = "moderate shift"

**Retraining Triggers:**
- Schedule-based: weekly retraining regardless (simple but wasteful)
- Performance-based: retrain when accuracy drops > 5% on held-out slice
- Drift-based: retrain when PSI > 0.2 on key features
- Event-based: retrain after major product/business change

**Shadow Mode Testing:**
- New model runs in shadow — receives same traffic, makes predictions, but results are NOT served
- Log shadow predictions; compare offline against actuals
- Promotes confidence before canary exposure

**Blue-Green Deployment:**
- Two identical production environments: Blue (current) and Green (new)
- Switch all traffic at once (vs canary's gradual ramp)
- Instant rollback: flip traffic back to Blue
- Requires 2× infrastructure cost during transition
"""))

cells.append(md(r"""## 11. Tradeoff Analysis

| Pattern | Complexity | Risk | Rollback Speed | When to Use |
|---------|-----------|------|----------------|-------------|
| Big Bang | Low | High | Manual, slow | Toy projects |
| Blue-Green | Medium | Medium | Instant (traffic flip) | Stateless services |
| Canary | High | Low | Automatic at threshold | High-stakes models |
| Shadow Mode | High | Zero | N/A (not serving) | Before first prod deploy |

**MLOps Tool Selection:**

| Need | Open Source | Managed |
|------|-------------|---------|
| Experiment tracking | MLflow | Weights & Biases |
| Pipeline orchestration | Airflow, Prefect | Kubeflow, Vertex Pipelines |
| Model registry | MLflow Registry | SageMaker Model Registry |
| Feature store | Feast | Tecton, Vertex Feature Store |
| Monitoring | Evidently AI | Arize, Fiddler |
| Serving | BentoML, Seldon | SageMaker Endpoints, Vertex Endpoints |

**Level 2 vs Level 3:** Level 3 adds continuous training on streaming data.
This is appropriate for: rapidly changing distributions (e-commerce, news),
high-volume real-time systems (fraud detection), and organisations with strong MLOps maturity.
Most organisations should target Level 2 first.
"""))

cells.append(md(r"""## 12. Senior-Level Interview Preparation

**Q1: What is training-serving skew and how do you prevent it?**
Training-serving skew = features computed differently at training time vs serving time.
Example: you compute "days since last purchase" as of today in training (leaks future info),
but compute it at request time in serving (different value). Fix: use a feature store where
the exact same transformation code runs for both training (from historical data) and serving.
Log online features at serving time and compare to training distribution regularly.

**Q2: Walk me through designing a canary deployment for a fraud detection model.**
(1) Define rollout metric: fraud catch rate (recall) AND precision AND latency;
(2) Set rollback criteria: recall drops >3% or precision drops >5% vs baseline;
(3) Traffic schedule: 1% → 5% → 25% → 50% → 100%, each step held for 24h minimum;
(4) Statistical guard: minimum N requests per step before computing metrics;
(5) Automated rollback: on-call alert + automatic traffic flip if rollback criteria met;
(6) Shadow mode first: run new model in shadow for 1 week before canary starts.

**Q3: How do you detect and respond to model drift?**
Three types: (1) feature drift — input distribution shifts (monitor PSI on key features);
(2) prediction drift — output distribution shifts (monitor mean prediction vs historical);
(3) concept drift — relationship between features and labels changes (monitor accuracy on labelled holdout).
Response: alert at PSI>0.1, trigger retraining at PSI>0.2. For concept drift, immediate investigation
since this is rarely benign (could indicate business change, data quality issue, or adversarial shift).

**Q4: What are the 3 tests every ML CI/CD pipeline must have?**
(1) Data validation: schema check, null fraction, range bounds, row count — fails if data is corrupted;
(2) Model evaluation gate: AUC/precision/recall above threshold on held-out test set — fails if model regressed;
(3) Serving contract test: model accepts expected input format, produces expected output schema and latency — fails if serving is broken.
Optional 4th: comparison test — new model must beat current production model by MDE before advancing.

**Q5: Explain the difference between Level 0 and Level 2 MLOps.**
Level 0: data scientists work in notebooks, manually export models, handoff to engineering via email/ticket.
No automated testing, no monitoring, no retraining. Model rots silently.
Level 2: pipeline is code (version-controlled); CI/CD tests run on every commit; model registry tracks all
versions with metadata; monitoring alerts on drift; retraining is triggered automatically.
The key difference is that at Level 2, shipping a model is as reliable as shipping software.

**Q6: What is a feature store and why does it matter for MLOps?**
A feature store is a central repository for feature computation logic and feature values.
It serves two purposes: (1) online store — low-latency feature retrieval for serving (Redis/DynamoDB);
(2) offline store — bulk feature retrieval for training (S3/Parquet).
Why it matters: eliminates training-serving skew (same code computes features for both);
enables feature reuse across models; allows point-in-time-correct feature lookups (preventing leakage).

**Q7: How do you calculate the ROI of investing in MLOps infrastructure?**
Revenue side: (1) faster deployment → earlier revenue (10 days saved × business value/day);
(2) better uptime → more predictions served (98% vs 80% uptime);
(3) more frequent retraining → better model accuracy → better business outcomes.
Cost side: platform costs (compute, tooling licenses), engineering time for platform build.
Typical ROI: Level 0 → Level 2 migration pays back in 6-12 months for teams with 3+ models in production.

**Q8: Design a retraining strategy for a recommendation model.**
Three triggers: (1) schedule — weekly retraining on rolling 90-day window of interaction data;
(2) performance — retrain immediately if NDCG@10 drops >5% on weekly eval set;
(3) data freshness — retrain if new items account for >15% of catalogue (cold start problem).
Safeguards: A/B test new model vs current in production for 48h before promotion;
use shadow mode for first deployment of each major version; maintain 90-day retrain log for audit.
"""))

cells.append(md(r"""## 13. Teach-Back Section

Explain each of these without notes to a peer:

1. **MLOps Level 0 to Level 2**: Draw the transition diagram from scratch. For each level,
   describe: (a) how a model gets deployed, (b) how you know if it's failing, (c) how you
   retrain it. What is the primary failure mode at each level?

2. **Model Registry mechanics**: Walk through registering a model, promoting it through
   staging → production, and rolling back when it fails. What metadata must a registry
   store for a model to be auditable?

3. **Training-serving skew**: Give a concrete example of how it arises in a real pipeline.
   Then describe the feature store pattern that prevents it.

4. **Canary rollout design**: For a churn model serving 100k requests/day, design a canary
   schedule: step sizes, duration per step, rollout criteria, rollback criteria.

5. **A/B test sample size**: Derive (or explain intuitively) why the sample size scales as
   1/δ². If you need to detect a 1% improvement instead of 2%, how much more data do you need?

6. **CI/CD for ML**: Name the 3 mandatory CI tests for an ML pipeline. For each, give an
   example of what "failure" looks like and what action it triggers (block, alert, rollback).

7. **Data drift response**: You receive an alert that PSI on your top 3 features exceeded 0.2.
   Walk through your diagnostic process: what do you check, in what order?

8. **Reproducibility**: A regulator asks you to reproduce a model you deployed 18 months ago
   for a GDPR audit. What must have been logged/versioned to make this possible?
"""))

cells.append(md(r"""## 14. Exercises

### Beginner
1. Implement a `DataValidator` that checks for: (a) no column has > 10% nulls,
   (b) numeric columns stay within [mean - 5σ, mean + 5σ], (c) categorical columns
   have no unseen labels vs training set.
2. Add a `compare` method to `ModelRegistry` that takes two version strings and returns
   a dict of metric deltas: {'auc': +0.022, 'f1': +0.03}.
3. Use the `ab_test_sample_size` function to build a table: for each combination of
   baseline {0.7, 0.8, 0.9} and MDE {0.01, 0.02, 0.05}, what sample size is needed?

### Intermediate
4. Implement a `DriftDetector` class that monitors prediction distribution using Population
   Stability Index (PSI): PSI = Σ (actual% - expected%) × ln(actual%/expected%). Flag
   drift when PSI > 0.2.
5. Extend `CanaryDeployment` to support automatic metric-based rollout acceleration:
   if lift > 3× min_lift, skip the next traffic step (fast-track to higher traffic).
6. Implement shadow mode: a `ShadowDeployment` class that routes 100% of traffic to
   the production model (serves to users) and simultaneously calls the shadow model
   (logs predictions only). Compare logged shadow vs actual outcomes.

### Senior
7. **Feature store from scratch**: Implement a `FeatureStore` with:
   - `register_feature(name, fn)` — register a computation function
   - `materialize(df, timestamp_col)` — compute offline features from a dataframe
   - `lookup(entity_id, features)` — online lookup from an in-memory store
   - Point-in-time-correct feature lookup: `lookup(entity_id, features, as_of=timestamp)`
   Show that using the feature store prevents training-serving skew in a churn pipeline.
8. **Multi-armed bandit model routing**: Instead of a fixed canary schedule, use Thompson
   sampling to dynamically allocate traffic: models that perform better get more traffic.
   Implement `BanditRouter.route(request)` that tracks Beta(α, β) posterior per model
   (α = wins, β = losses) and samples to select the model for each request.
   Simulate 1000 requests with 3 models of different quality and show convergence.
"""))

build("09_production_ml/01_mlops.ipynb", cells)
print("PROD-01 built.")
