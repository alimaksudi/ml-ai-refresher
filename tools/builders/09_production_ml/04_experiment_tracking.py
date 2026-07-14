import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nbbuild import md, code, build

cells = []

cells.append(md(r"""# PROD-04 — Experiment Tracking
## Core practice introduced after validation; advanced production treatment

**Prerequisites:** MLE-01 and MLE-02. Use the basic run-record discipline from
this notebook for every experiment from Lesson MLE-03 onward. Bayesian search,
statistical comparison, and registry integration are advanced sections that may be
revisited in Section 09.

*Systematic comparison of ML experiments: log everything, reproduce anything, promote only what's statistically better.*
"""))

cells.append(md(r"""## 1. Learning Objectives

By the end of this notebook you will be able to:
- Identify the 6 categories of information every experiment must log
- Implement an ExperimentTracker with log_param, log_metric, compare_runs, best_run
- Implement grid search, random search, and Bayesian optimisation from scratch
- Implement early stopping with patience from scratch
- Apply a t-test to determine whether one run statistically beats another
- Explain how to seed everything for reproducibility
- Track 50 hyperparameter search runs and select the statistically best configuration
- Design the experiment tracking workflow for a production churn model
"""))

cells.append(md(r"""## 2. Historical Motivation

### From Notebooks to Systematic Experiments (2016–2022)

Before experiment tracking tools, ML practitioners kept results in:
- Spreadsheets (no code linkage)
- Notebook filenames: `model_v3_final_FINAL2.ipynb`
- Memory ("I think the best LR was 0.01 but I'm not sure")

**The reproducibility crisis:**
- 2016: A Nature survey found 70% of researchers could not reproduce another scientist's results
- In ML specifically: the inability to reproduce experiments meant teams could not:
  1. Audit which hyperparameters produced a production model
  2. Understand why a model that performed well on one dataset failed on another
  3. Return to a 6-month-old "best" model after the production model degraded

**MLflow (2018):** First open-source experiment tracking tool. Databricks released it;
solved logging + comparison. Now used by hundreds of thousands of ML practitioners.

**Weights & Biases (2018):** SaaS experiment tracking; added rich visualisation,
team collaboration, model registry. Became the standard for deep learning research.

**The statistical comparison problem (2021):**
Teams routinely promoted the model with the highest val-AUC from a hyperparameter sweep —
even when the difference was within noise. A 2021 survey of 43 ML papers found that 73% of
reported improvements were not statistically significant at α=0.05.
"""))

cells.append(md(r"""## 3. Intuition and Visual Understanding

### What to Track in Every Run

```
CATEGORY 1: HYPERPARAMETERS  (immutable once training starts)
  learning_rate=0.01, n_estimators=100, max_depth=5, seed=42

CATEGORY 2: METRICS          (logged at each epoch or step)
  epoch 1: train_loss=0.82, val_loss=0.78, val_auc=0.65
  epoch 2: train_loss=0.71, val_loss=0.69, val_auc=0.72
  ...

CATEGORY 3: ARTIFACTS        (files produced by the run)
  model.pkl, confusion_matrix.png, feature_importance.csv

CATEGORY 4: SYSTEM INFO      (for reproducibility)
  python=3.11, sklearn=1.3.2, numpy=1.24.0, CPU=M2, RAM=32GB

CATEGORY 5: CODE VERSION     (what code produced this run)
  git_commit=abc1234, branch=experiment/feature_engineering_v2

CATEGORY 6: DATA VERSION     (which data was used)
  dataset_hash=sha256:def456, n_train=45000, n_val=5000
```

### Hyperparameter Search Strategies

```
GRID SEARCH:   Exhaustive, expensive, guaranteed to find best in grid
  LR: [0.001, 0.01, 0.1]  x  depth: [3, 5, 7]  →  9 runs

RANDOM SEARCH: Randomly sample, often finds good configs with fewer runs
  LR ~ LogUniform(1e-4, 1e-1)  depth ~ Uniform(1, 10)  →  N runs

BAYESIAN OPT:  Build surrogate model of performance vs hyperparams
               → use acquisition function to pick next best config
               Each run informs the next → faster convergence
```

### Early Stopping Intuition

```
Validation loss over epochs:
  Epoch 1: 0.82
  Epoch 2: 0.78  ← new best (patience counter reset to 0)
  Epoch 3: 0.75  ← new best
  Epoch 4: 0.76  ← no improvement (patience = 1)
  Epoch 5: 0.77  ← no improvement (patience = 2)
  Epoch 6: 0.78  ← no improvement (patience = 3 → STOP)

Best model was at epoch 3. Restore weights from epoch 3.
```
"""))

cells.append(md(r"""## 4. Mathematical Foundations

### 4.1 Gaussian Process Surrogate for Bayesian Optimisation

Predict performance $f(\mathbf{x})$ at untried point $\mathbf{x}$:

$$f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

Using RBF kernel: $k(\mathbf{x}, \mathbf{x}') = \sigma^2 \exp\left(-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\ell^2}\right)$

Given observed data $(\mathbf{X}, \mathbf{y})$, posterior:
$$\mu^*(\mathbf{x}) = k(\mathbf{x}, \mathbf{X})[K(\mathbf{X},\mathbf{X}) + \sigma_n^2 I]^{-1}\mathbf{y}$$
$$\sigma^{*2}(\mathbf{x}) = k(\mathbf{x},\mathbf{x}) - k(\mathbf{x},\mathbf{X})[K(\mathbf{X},\mathbf{X}) + \sigma_n^2 I]^{-1}k(\mathbf{X},\mathbf{x})$$

### 4.2 Expected Improvement Acquisition

$$\text{EI}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f^*, 0)]$$

$$= (\mu^*(\mathbf{x}) - f^*)\Phi(Z) + \sigma^*(\mathbf{x})\phi(Z), \quad Z = \frac{\mu^*(\mathbf{x}) - f^*}{\sigma^*(\mathbf{x})}$$

where $f^* = \max_i y_i$ (best observed), $\Phi$ is CDF, $\phi$ is PDF of standard normal.

### 4.3 Paired Bootstrap for Model Comparison

When models A and B score the **same held-out examples**, their results are paired.
Resample examples (or the independent unit such as user) with replacement, then
recompute the metric difference:

$$\Delta^{(b)}=M\!\left(y^{(b)},\hat y_A^{(b)}\right)-
M\!\left(y^{(b)},\hat y_B^{(b)}\right).$$

**Read and symbols:** $b$ is a bootstrap repetition; $M$ is the chosen metric;
$y$ contains labels; $\hat y_A,\hat y_B$ are paired model predictions; $\Delta$
is A minus B. Percentiles of many $\Delta^{(b)}$ values form a confidence interval.
If the interval excludes zero, the data supports a non-zero difference at the
corresponding confidence level. Also require a practically meaningful effect size.

K-fold scores are dependent because training folds overlap, so an ordinary Welch
t-test on five fold scores understates uncertainty. Use paired out-of-fold
predictions, a paired bootstrap/permutation test, or a corrected repeated-CV method.
An equivalence claim requires a predeclared equivalence margin and an equivalence
test; failing to find a difference does not prove equivalence.
"""))

cells.append(code(r"""
import numpy as np
import math
import uuid
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("Imports OK")
"""))

cells.append(md(r"""## 5. Manual Implementation from Scratch

### 5.1 Experiment Tracker
"""))

cells.append(code(r"""
# Mini Experiment Tracker

class Run:
    def __init__(self, run_id, experiment_name):
        self.run_id    = run_id
        self.name      = experiment_name
        self.params    = {}
        self.metrics   = defaultdict(list)   # metric_name -> [(step, value)]
        self.artifacts = []
        self.tags      = {}
        self.status    = 'running'

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value, step=None):
        if step is None:
            step = len(self.metrics[key])
        self.metrics[key].append((step, value))

    def log_artifact(self, path):
        self.artifacts.append(path)

    def get_metric_history(self, key):
        return [(s, v) for s, v in self.metrics.get(key, [])]

    def get_best_metric(self, key, mode='max'):
        vals = [v for _, v in self.metrics.get(key, [])]
        if not vals:
            return None
        return max(vals) if mode == 'max' else min(vals)

    def finish(self, status='completed'):
        self.status = status

    def to_dict(self):
        return {
            'run_id': self.run_id,
            'params': self.params,
            'best_metrics': {k: self.get_best_metric(k) for k in self.metrics},
            'status': self.status,
        }

class ExperimentTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.runs            = {}

    def start_run(self, run_name=None):
        run_id = str(uuid.uuid4())[:8]
        run    = Run(run_id, run_name or self.experiment_name)
        self.runs[run_id] = run
        return run

    def compare_runs(self, metric, mode='max', top_n=5):
        results = []
        for run_id, run in self.runs.items():
            val = run.get_best_metric(metric, mode)
            if val is not None:
                results.append((val, run_id, run.params))
        results.sort(key=lambda x: x[0], reverse=(mode == 'max'))
        return results[:top_n]

    def best_run(self, metric, mode='max'):
        ranked = self.compare_runs(metric, mode, top_n=1)
        if not ranked:
            return None
        _, run_id, _ = ranked[0]
        return self.runs[run_id]

tracker = ExperimentTracker('churn_model_hpsearch')

# Log a few manual runs
for lr, depth in [(0.01, 3), (0.01, 5), (0.05, 5)]:
    run = tracker.start_run(f'lr{lr}_d{depth}')
    run.log_param('learning_rate', lr)
    run.log_param('max_depth', depth)
    run.log_param('seed', 42)
    auc = 0.78 + lr * 2 + depth * 0.01 + np.random.default_rng(int(lr*100+depth)).normal(0, 0.02)
    run.log_metric('val_auc', auc)
    run.finish()

print("Top runs by val_auc:")
for val, run_id, params in tracker.compare_runs('val_auc', top_n=5):
    print(f"  run={run_id}  auc={val:.4f}  params={params}")
"""))

cells.append(code(r"""
# 5.2 Grid Search

def grid_search(param_grid, evaluate_fn):
    import itertools
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    results = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        score  = evaluate_fn(params)
        results.append({'params': params, 'score': score})
    return sorted(results, key=lambda x: x['score'], reverse=True)

def random_search(param_distributions, evaluate_fn, n_iter=20, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_iter):
        params = {}
        for key, dist in param_distributions.items():
            if dist[0] == 'loguniform':
                _, lo, hi = dist
                params[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            elif dist[0] == 'randint':
                _, lo, hi = dist
                params[key] = int(rng.integers(lo, hi + 1))
            elif dist[0] == 'choice':
                _, options = dist
                params[key] = rng.choice(options)
        score = evaluate_fn(params)
        results.append({'params': params, 'score': score})
    return sorted(results, key=lambda x: x['score'], reverse=True)

# Simulate a churn model AUC as a function of hyperparams
def fake_model_auc(params):
    rng = np.random.default_rng(int(abs(hash(str(params))) % 2**31))
    lr      = params.get('learning_rate', 0.1)
    depth   = params.get('max_depth', 3)
    n_est   = params.get('n_estimators', 100)
    # True AUC surface (no leakage — pure simulation)
    true_auc = 0.72 + 0.08 * math.log(1 + lr * 10) + 0.012 * depth + 0.0003 * n_est
    noise    = rng.normal(0, 0.005)
    return min(max(true_auc + noise, 0.60), 0.95)

# Grid search
grid = {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
gs_results = grid_search(grid, fake_model_auc)
print("Grid Search Top 3:")
for r in gs_results[:3]:
    print(f"  AUC={r['score']:.4f}  {r['params']}")

# Random search
distributions = {
    'learning_rate':  ('loguniform', 0.001, 0.3),
    'max_depth':      ('randint', 2, 10),
    'n_estimators':   ('randint', 50, 500),
}
rs_results = random_search(distributions, fake_model_auc, n_iter=30, seed=7)
print("\nRandom Search Top 3 (30 trials):")
for r in rs_results[:3]:
    print(f"  AUC={r['score']:.4f}  {r['params']}")
"""))

cells.append(code(r"""
# 5.3 Bayesian Optimisation from scratch (GP + Expected Improvement)

def rbf_kernel(X1, X2, length_scale=1.0, sigma=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return sigma**2 * np.exp(-0.5 * dists / length_scale**2)

def gp_predict(X_train, y_train, X_test, noise=1e-4):
    K    = rbf_kernel(X_train, X_train) + noise * np.eye(len(X_train))
    K_s  = rbf_kernel(X_train, X_test)
    K_ss = rbf_kernel(X_test, X_test)
    K_inv = np.linalg.solve(K, K_s).T
    mu    = K_inv @ y_train
    sigma = np.sqrt(np.maximum(np.diag(K_ss) - np.einsum('ij,jk,ki->i', K_inv, K, K_inv.T), 1e-9))
    return mu, sigma

def expected_improvement(mu, sigma, best_y, xi=0.01):
    Z   = (mu - best_y - xi) / (sigma + 1e-9)
    phi = (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * Z**2)
    Phi = 0.5 * (1 + np.array([math.erf(z / math.sqrt(2)) for z in Z]))
    return (mu - best_y - xi) * Phi + sigma * phi

def bayesian_optimisation(evaluate_fn, n_init=5, n_iter=20, seed=42):
    rng = np.random.default_rng(seed)
    # Search space: [learning_rate, max_depth_normalised, n_estimators_normalised]
    # Normalise to [0, 1] for GP

    def decode(x):
        lr    = float(np.exp(x[0] * (np.log(0.3) - np.log(0.001)) + np.log(0.001)))
        depth = int(round(x[1] * 8 + 2))  # [2, 10]
        n_est = int(round(x[2] * 450 + 50))  # [50, 500]
        return {'learning_rate': lr, 'max_depth': depth, 'n_estimators': n_est}

    # Random initialisation
    X_obs = rng.uniform(0, 1, (n_init, 3))
    y_obs = np.array([evaluate_fn(decode(x)) for x in X_obs])

    history = []
    for t in range(n_iter):
        # Sample candidate points
        X_cand = rng.uniform(0, 1, (200, 3))
        mu, sigma = gp_predict(X_obs, y_obs, X_cand)
        ei        = expected_improvement(mu, sigma, y_obs.max())
        best_idx  = np.argmax(ei)
        x_next    = X_cand[best_idx]
        y_next    = evaluate_fn(decode(x_next))
        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)
        history.append({'step': t + n_init + 1, 'best_auc': y_obs.max(), 'auc': y_next})

    best_idx    = np.argmax(y_obs)
    best_params = decode(X_obs[best_idx])
    return best_params, y_obs.max(), history

print("Bayesian Optimisation (5 init + 20 iterations)...")
bo_params, bo_auc, bo_history = bayesian_optimisation(fake_model_auc, n_init=5, n_iter=20)
print(f"Best AUC: {bo_auc:.4f}")
print(f"Best params: {bo_params}")
print(f"\nFinal best by method:")
print(f"  Grid Search:     {gs_results[0]['score']:.4f}")
print(f"  Random Search:   {rs_results[0]['score']:.4f}")
print(f"  Bayesian Optim:  {bo_auc:.4f}")
"""))

cells.append(code(r"""
# 5.4 Early Stopping from scratch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='min', restore_best=True):
        self.patience     = patience
        self.min_delta    = min_delta
        self.mode         = mode
        self.restore_best = restore_best
        self._best_value  = float('inf') if mode == 'min' else -float('inf')
        self._best_epoch  = 0
        self._best_weights= None
        self._counter     = 0
        self.stopped_epoch= None

    def step(self, value, epoch, weights=None):
        improved = (self.mode == 'min' and value < self._best_value - self.min_delta) or \
                   (self.mode == 'max' and value > self._best_value + self.min_delta)
        if improved:
            self._best_value   = value
            self._best_epoch   = epoch
            self._best_weights = weights
            self._counter      = 0
            return False  # keep training
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.stopped_epoch = epoch
                return True  # stop
        return False

    def best_epoch(self):
        return self._best_epoch

    def best_value(self):
        return self._best_value

# Simulate training with early stopping
rng = np.random.default_rng(99)
n_epochs = 50
val_losses = []

# Simulate: drops fast, plateaus, then slowly gets worse
for e in range(n_epochs):
    if e < 15:
        base = 1.0 - 0.03 * e
    else:
        base = 0.55 + 0.005 * (e - 15)
    val_losses.append(base + rng.normal(0, 0.015))

es = EarlyStopping(patience=5, min_delta=0.001, mode='min')
stop_epoch = None
for e, loss in enumerate(val_losses):
    should_stop = es.step(loss, e)
    if should_stop:
        stop_epoch = e
        break

print(f"Early Stopping Summary:")
print(f"  Total epochs available: {n_epochs}")
print(f"  Stopped at epoch: {stop_epoch}")
print(f"  Best epoch: {es.best_epoch()}")
print(f"  Best val_loss: {es.best_value():.4f}")
print(f"  Epochs saved: {n_epochs - stop_epoch} ({(n_epochs-stop_epoch)/n_epochs*100:.0f}% compute saved)")
"""))

cells.append(code(r"""
# 5.5 Paired bootstrap comparison on one untouched test set
from sklearn.metrics import roc_auc_score

def paired_bootstrap_auc(y_true, score_a, score_b, n_boot=2000, seed=42):
    # Return observed AUC difference and a paired percentile confidence interval.
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    if not (len(y_true) == len(score_a) == len(score_b)):
        raise ValueError("labels and paired predictions must have equal length")

    observed = roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        # AUC is undefined if a bootstrap sample contains only one class.
        if np.unique(y_true[idx]).size < 2:
            continue
        differences.append(
            roc_auc_score(y_true[idx], score_a[idx])
            - roc_auc_score(y_true[idx], score_b[idx])
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return observed, (low, high), np.asarray(differences)

rng = np.random.default_rng(42)
y_test = rng.binomial(1, 0.25, 1200)
latent_signal = 1.4 * y_test + rng.normal(0, 1, len(y_test))
score_base = latent_signal + rng.normal(0, 0.75, len(y_test))
score_new = latent_signal + rng.normal(0, 0.55, len(y_test))

delta_auc, interval, bootstrap_deltas = paired_bootstrap_auc(
    y_test, score_new, score_base
)
print("Paired bootstrap on an untouched test set")
print(f"AUC difference (new - base): {delta_auc:+.4f}")
print(f"95% confidence interval:     [{interval[0]:+.4f}, {interval[1]:+.4f}]")
print("Promotion also requires a predeclared minimum useful improvement and system gates.")
"""))

cells.append(code(r"""
# 5.6 Full hyperparameter sweep: 50 runs

N_RUNS = 50
tracker2 = ExperimentTracker('churn_hpsearch_50runs')
rng = np.random.default_rng(0)

run_records = []
for i in range(N_RUNS):
    lr     = float(np.exp(rng.uniform(np.log(0.001), np.log(0.3))))
    depth  = int(rng.integers(2, 11))
    n_est  = int(rng.integers(50, 501))
    subsamp= float(rng.uniform(0.5, 1.0))

    run = tracker2.start_run(f'run_{i:03d}')
    run.log_param('learning_rate', round(lr, 5))
    run.log_param('max_depth', depth)
    run.log_param('n_estimators', n_est)
    run.log_param('subsample', round(subsamp, 3))
    run.log_param('seed', i)

    # Simulate 5-fold CV
    cv_scores = []
    for fold in range(5):
        fold_rng = np.random.default_rng(i * 100 + fold)
        score = fake_model_auc({'learning_rate': lr, 'max_depth': depth,
                                'n_estimators': n_est})
        score += fold_rng.normal(0, 0.005)
        run.log_metric('val_auc', score, step=fold)
        cv_scores.append(score)

    mean_auc = sum(cv_scores) / 5
    run.log_metric('mean_cv_auc', mean_auc)
    run.finish()
    run_records.append({'run_id': run.run_id, 'mean_auc': mean_auc,
                        'cv_scores': cv_scores, 'params': run.params})

# Best run
best_run = tracker2.best_run('mean_cv_auc')
print(f"Best run: {best_run.run_id}")
print(f"Best mean_cv_auc: {best_run.get_best_metric('mean_cv_auc'):.4f}")
print(f"Best params: {best_run.params}")

# Top 5
print("\nTop 5 runs:")
for val, rid, params in tracker2.compare_runs('mean_cv_auc', top_n=5):
    print(f"  {rid}  AUC={val:.4f}  lr={params['learning_rate']:.4f}  depth={params['max_depth']}")
"""))

cells.append(md(r"""## 6. Visualization
"""))

cells.append(code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Hyperparameter search comparison ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
all_aucs   = [r['mean_auc'] for r in run_records]
cummax_auc = [max(all_aucs[:i+1]) for i in range(len(all_aucs))]

# Compare to BO curve
bo_best_so_far = [h['best_auc'] for h in bo_history]
x_rs = range(1, N_RUNS + 1)
x_bo = range(1, len(bo_history) + 1)

ax1.plot(x_rs, cummax_auc, 'b-', linewidth=2, label=f'Random search (N={N_RUNS})')
ax1.plot(x_bo, bo_best_so_far, 'r-', linewidth=2,
         label=f'Bayesian optim (N={len(bo_history)})')
ax1.axhline(gs_results[0]['score'], color='green', linestyle='--', alpha=0.7,
            label=f'Grid search best ({gs_results[0]["score"]:.3f})')
ax1.set_xlabel('Number of Evaluations')
ax1.set_ylabel('Best AUC Found So Far')
ax1.set_title('Hyperparameter Search: Convergence Comparison\n(BO converges faster than random)')
ax1.legend(fontsize=8)
# Annotation: BO needs fewer evaluations to find competitive configurations

# ── Plot 2: AUC distribution of 50 runs ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(all_aucs, bins=15, color='#1976D2', alpha=0.75, edgecolor='white')
ax2.axvline(max(all_aucs), color='red', linestyle='--',
            label=f'Best: {max(all_aucs):.4f}')
ax2.axvline(sum(all_aucs)/len(all_aucs), color='orange', linestyle='--',
            label=f'Mean: {sum(all_aucs)/len(all_aucs):.4f}')
ax2.set_xlabel('Mean CV AUC')
ax2.set_ylabel('Number of Runs')
ax2.set_title('AUC Distribution Across 50 Hyperparameter Runs\n(most runs cluster near the mean)')
ax2.legend()
# Annotation: heavy tail on the right shows good configs are rare — motivates smart search

# ── Plot 3: Early stopping validation loss ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(range(len(val_losses)), val_losses, 'b-', linewidth=1.5, label='Val loss')
ax3.axvline(es.best_epoch(), color='green', linestyle='--',
            label=f'Best epoch ({es.best_epoch()})')
if stop_epoch:
    ax3.axvline(stop_epoch, color='red', linestyle='--',
                label=f'Stopped (epoch {stop_epoch})')
ax3.fill_between(range(len(val_losses)),
                 [min(val_losses)] * len(val_losses),
                 val_losses, alpha=0.1, color='blue')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Validation Loss')
ax3.set_title(f'Early Stopping (patience=5)\n(best: epoch {es.best_epoch()}, stopped: epoch {stop_epoch})')
ax3.legend()
# Annotation: stopping early prevents overfitting and saves compute

# ── Plot 4: Bayesian GP surrogate (1D slice) ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
# 1D illustration: vary learning_rate only
lr_range = np.linspace(0, 1, 100)  # normalised
true_y   = np.array([fake_model_auc({'learning_rate': float(np.exp(x*(np.log(0.3)-np.log(0.001))+np.log(0.001))),
                                      'max_depth': 5, 'n_estimators': 200})
                     for x in lr_range])

# Sample points for GP
sample_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)
sample_y = np.array([fake_model_auc({'learning_rate': float(np.exp(x[0]*(np.log(0.3)-np.log(0.001))+np.log(0.001))),
                                      'max_depth': 5, 'n_estimators': 200})
                     for x in sample_x])

X_test = lr_range.reshape(-1, 1)
mu, sigma = gp_predict(sample_x, sample_y, X_test, noise=1e-4)
ei        = expected_improvement(mu, sigma, sample_y.max())
ei_norm   = (ei - ei.min()) / (ei.max() - ei.min() + 1e-9)

ax4.plot(lr_range, true_y, 'k-', linewidth=1, alpha=0.4, label='True AUC (hidden)')
ax4.plot(lr_range, mu, 'b-', linewidth=2, label='GP mean')
ax4.fill_between(lr_range, mu - 2*sigma, mu + 2*sigma,
                 alpha=0.2, color='blue', label='95% CI')
ax4.plot(sample_x.ravel(), sample_y, 'ko', markersize=8, label='Observed')
ax4_twin = ax4.twinx()
ax4_twin.plot(lr_range, ei_norm, 'r--', linewidth=1.5, alpha=0.6, label='EI (acquisition)')
ax4_twin.set_ylabel('Expected Improvement (normalised)', color='red')
ax4.set_xlabel('Learning Rate (normalised)')
ax4.set_ylabel('AUC')
ax4.set_title('Bayesian Optimisation: GP Surrogate + EI Acquisition\n(next point = max EI = unexplored promising region)')
ax4.legend(loc='lower left', fontsize=7)
# Annotation: EI peaks where GP is uncertain AND predicts high value → smart exploration

# ── Plot 5: AUC vs learning_rate scatter ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
lrs   = [r['params']['learning_rate'] for r in run_records]
aucs5 = [r['mean_auc'] for r in run_records]
ax5.scatter(lrs, aucs5, c=aucs5, cmap='RdYlGn', alpha=0.7, s=40)
ax5.set_xscale('log')
ax5.set_xlabel('Learning Rate (log scale)')
ax5.set_ylabel('Mean CV AUC')
ax5.set_title('AUC vs Learning Rate (50 runs)\n(colour = AUC; optimal LR region visible)')
# Annotation: log-scale reveals the optimal LR range — motivates log-uniform sampling

# ── Plot 6: Descriptive comparison (not an inferential test) ──────────────────
ax6 = fig.add_subplot(gs[2, 1])
# Top/bottom groups were selected from the same sweep, so error bars are descriptive.
top5_aucs    = sorted(all_aucs, reverse=True)[:5]
bottom5_aucs = sorted(all_aucs)[:5]
mean_top     = sum(top5_aucs) / 5
mean_bot     = sum(bottom5_aucs) / 5
std_top      = (sum((x-mean_top)**2 for x in top5_aucs) / 4) ** 0.5
std_bot      = (sum((x-mean_bot)**2 for x in bottom5_aucs) / 4) ** 0.5

ax6.bar(['Top 5 runs', 'Bottom 5 runs'], [mean_top, mean_bot],
        yerr=[std_top, std_bot], color=['#43A047', '#EF5350'],
        alpha=0.85, capsize=5)
ax6.set_ylabel('Mean CV AUC')
ax6.set_title('Top vs Bottom Runs (Descriptive Only)\nSelection makes this unsuitable for a p-value')
ax6.set_ylim(ax6.get_ylim()[0] - 0.01, ax6.get_ylim()[1] + 0.01)
for i, (val, err) in enumerate([(mean_top, std_top), (mean_bot, std_bot)]):
    ax6.text(i, val + err + 0.001, f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
# Annotation: a model must beat baseline by a statistically significant margin before promotion

plt.suptitle('Experiment Tracking: Hyperparameter Search, Early Stopping, Statistical Comparison',
             fontsize=13, fontweight='bold')
plt.savefig('/tmp/04_experiment_tracking.png', dpi=100, bbox_inches='tight')
plt.close()
print("Figure saved: /tmp/04_experiment_tracking.png")
"""))

cells.append(md(r"""## 7. Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| **Irreproducible runs** | Random seeds not fixed | Log all seeds; fix: data, model, environment |
| **Overfitting to val set** | Too many HP trials, all evaluated on same val set | Nested CV: inner loop for HP, outer for generalisation |
| **Invalid uncertainty test** | Treating overlapping CV folds as independent replicates | Compare paired out-of-fold/test predictions with bootstrap or a corrected CV method |
| **Metric mismatch** | Optimising AUC when business needs precision at k | Define the metric before the sweep, not after |
| **Experiment data loss** | Local tracker (no backup) | Use MLflow/W&B; log to durable storage |
| **Grid search missed global optimum** | Grid too coarse; optimal value between grid points | Use random or Bayesian search instead |
| **Early stopping too aggressive** | Patience too low; noisy validation metric | Smooth val metric (EMA); use patience ≥ 5 |
| **Data leakage in CV** | Preprocessing fit on all data before CV split | Fit preprocessor inside each fold (use Pipeline) |
"""))

cells.append(md(r"""## 8. Production Library Implementation
"""))

cells.append(code(r"""
# Production experiment tracking tools

try:
    import mlflow
    print("MLflow available:")
    print("  with mlflow.start_run():")
    print("      mlflow.log_params({'lr': 0.01, 'depth': 5})")
    print("      mlflow.log_metric('val_auc', 0.851)")
    print("      mlflow.sklearn.log_model(model, 'churn_model')")
except ImportError:
    print("mlflow not installed — using ExperimentTracker from scratch")

try:
    import wandb
    print("Weights & Biases available:")
    print("  wandb.init(project='churn-model', config={'lr': 0.01})")
    print("  wandb.log({'val_auc': 0.851, 'epoch': 5})")
except ImportError:
    print("wandb not installed")

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    print("sklearn HP search available (cross_val_score, GridSearchCV, RandomizedSearchCV)")
except ImportError:
    print("sklearn not installed")

try:
    from optuna import create_study
    print("Optuna available (Bayesian/TPE/CMA-ES optimisation):")
    print("  study = optuna.create_study(direction='maximize')")
    print("  study.optimize(objective, n_trials=100)")
except ImportError:
    print("optuna not installed — using scratch Bayesian optimisation above")
"""))

cells.append(md(r"""## 9. Realistic Business Case Study

### Churn Model: 50-Trial Hyperparameter Search with Statistical Selection
"""))

cells.append(code(r"""
# Business case: HP search ROI

# Time cost
ENGINEER_HOURLY_RATE = 80
CLOUD_GPU_HOURLY     = 2.50   # A10G GPU
TRAINING_HOURS       = 0.5    # per trial for a medium churn model

# Manual HP tuning baseline (typical before systematic search)
manual_runs           = 10
manual_search_hours   = 4     # engineer time to try configs
manual_best_auc       = 0.822

# Automated random search
auto_runs             = 50
auto_compute_hours    = auto_runs * TRAINING_HOURS
auto_engineer_hours   = 0.5   # just to launch and review
auto_best_auc         = max(all_aucs)

# Business impact
N_CUSTOMERS           = 100_000
CHURN_RATE            = 0.12
LIFETIME_VALUE        = 300    # USD

def revenue(auc, n_customers=N_CUSTOMERS, churn_rate=CHURN_RATE, ltv=LIFETIME_VALUE,
            model_recall=0.60):
    # Higher AUC correlates with better recall/precision; simple approximation
    effective_recall = model_recall * (auc - 0.5) / 0.5
    customers_saved  = n_customers * churn_rate * effective_recall
    return customers_saved * ltv

rev_manual = revenue(manual_best_auc)
rev_auto   = revenue(auto_best_auc)

print("Churn Model HP Search: ROI Analysis")
print("=" * 55)
print(f"{'Metric':<35} {'Manual':>9} {'Automated':>10}")
print("-" * 55)
print(f"  Runs                          {manual_runs:>9}  {auto_runs:>10}")
print(f"  Engineer hours                {manual_search_hours:>9.1f}  {auto_engineer_hours:>10.1f}")
print(f"  Compute hours (GPU)           {manual_runs*TRAINING_HOURS:>9.1f}  {auto_compute_hours:>10.1f}")
cost_m = (manual_search_hours * ENGINEER_HOURLY_RATE +
           manual_runs * TRAINING_HOURS * CLOUD_GPU_HOURLY)
cost_a = (auto_engineer_hours * ENGINEER_HOURLY_RATE +
           auto_compute_hours * CLOUD_GPU_HOURLY)
print(f"  Total search cost ($)         ${cost_m:>8,.0f}  ${cost_a:>9,.0f}")
print(f"  Best AUC found                {manual_best_auc:>9.4f}  {auto_best_auc:>10.4f}")
print(f"  Annual revenue from model ($) ${rev_manual:>8,.0f}  ${rev_auto:>9,.0f}")
print()
print(f"  Incremental annual revenue:   ${rev_auto - rev_manual:,.0f}")
print(f"  Extra search cost:            ${cost_a - cost_m:,.0f}")
print(f"  ROI of automated HP search:   {(rev_auto - rev_manual) / max(cost_a - cost_m, 1):.0f}x")
"""))

cells.append(md(r"""## 10. Production Considerations

### Reproducibility Checklist

```python
# Seed everything — in this order:
import random; random.seed(42)
import numpy as np; np.random.seed(42)
import os; os.environ['PYTHONHASHSEED'] = '42'

# sklearn
from sklearn.utils import check_random_state
rng = check_random_state(42)

# Log for reproducibility:
mlflow.log_params({
    'seed': 42,
    'python_version': sys.version,
    'sklearn_version': sklearn.__version__,
    'data_hash': sha256(training_data).hexdigest(),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
})
```

### Nested Cross-Validation (Avoid Optimistic Bias)

```
Outer CV (5 folds): split data into 5 train/test pairs
  For each outer fold:
    Inner CV (3 folds) on training data: find best HP
    Evaluate best model on held-out outer test set
    Report outer test score

Result: unbiased estimate of generalisation performance
vs. simple HP search on single val set: optimistically biased
```

### When to Stop a Hyperparameter Search

- Convergence check: best score hasn't improved in last 20 trials
- Budget check: compute budget exhausted
- Evidence check: paired confidence interval clears the predeclared useful-effect threshold
- Diminishing returns: the improvement curve is flattening (second derivative ≈ 0)
"""))

cells.append(md(r"""## 11. Tradeoff Analysis

| Method | Time to Good Config | Final Quality | Requires Expert | Compute |
|--------|--------------------|--------------|-----------------|---------|
| Grid Search | O(N^k) | Good (if grid covers optimum) | Yes (grid design) | High |
| Random Search | O(N) | Good (with enough trials) | Low | Medium |
| Bayesian Optim | O(N · M) | Best (given same N) | Low | Medium |
| Gradient-based (DARTS) | O(epochs) | Very good | High | Low |

**When to use each:**
- Grid: small search spaces (< 3 HP, < 4 values each); when you have strong priors
- Random: > 4 HP, continuous spaces; first exploration of a new problem
- Bayesian (Optuna): when evaluations are expensive (> 10 min each); production HP tuning
- Early stopping: always enable for any iterative training (GBM, NN, LightGBM)
"""))

cells.append(md(r"""## 12. Senior-Level Interview Preparation

**Q1: Why is random search often better than grid search in practice?**
High-dimensional HP spaces are sparse. Most HP don't matter equally. Grid search wastes evaluations
on unimportant HP because it samples all combinations. Random search samples independently on each
dimension — so even with the same number of evaluations, it explores a wider range of the
important HP. Bergstra & Bengio (2012) showed random search finds equally good or better configs
in 1/10 the evaluations for typical high-dimensional HP spaces.

**Q2: Explain Expected Improvement in Bayesian optimisation.**
EI(x) = E[max(f(x) - f*, 0)], the expected improvement over the current best f*.
The GP surrogate gives us both μ(x) (predicted performance) and σ(x) (uncertainty).
EI is high where μ(x) > f* (we think it's better than current best) OR where σ(x) is high
(we're uncertain — exploration bonus). This naturally balances exploitation (try configs we
think are good) and exploration (try configs we're uncertain about).

**Q3: How do you detect overfitting to the validation set in a HP search?**
Signs: (1) val AUC keeps improving as you run more trials but test AUC doesn't;
(2) the improvement margin on val is larger than on test. Prevention: use nested CV —
inner loop for HP search, outer loop for unbiased generalisation estimate. Or: hold out a
"test" set that's never touched during HP search; only evaluate on it for the final selected model.
A practical rule: for > 20 HP trials on the same val set, expect ~0.005-0.01 optimistic bias.

**Q4: When should you use early stopping?**
Any time you're training iteratively (gradient boosting, neural networks) on a metric that
could peak before training ends. Signs you need it: (1) val loss starts increasing while
train loss keeps decreasing (classic overfitting); (2) val metric plateaus for many epochs.
Key parameters: patience (how many non-improving epochs to tolerate), min_delta (minimum
improvement to count as progress), restore_best_weights (load weights from best epoch at end).
Typical patience for neural networks: 10-20 epochs. For GBMs: 50-100 rounds.

**Q5: Two runs have val-AUC of 0.831 vs 0.829. How do you decide which to promote?**
Do not promote from point estimates. Preserve paired out-of-fold predictions or score both
models on the same untouched test examples, then bootstrap the independent unit (usually
user/entity) and compute a confidence interval for the AUC difference. Predeclare the minimum
useful improvement. If the interval includes zero, evidence is inconclusive—not proof of
equivalence. Prefer the simpler model when the measured benefit is not practically meaningful.

**Q6: What should you log in every experiment run to make it reproducible?**
6 categories: (1) Hyperparameters: all of them, even "defaults" (don't assume they don't change);
(2) Metrics: per step/epoch, not just final value (needed to plot learning curves and detect issues);
(3) Artifacts: model weights, scaler, feature list, confusion matrix;
(4) System info: library versions (sklearn, numpy, Python), hardware specs;
(5) Code version: git commit hash and branch name — ensures you can reproduce the code state;
(6) Data version: dataset hash or DVC commit, n_train/n_val/n_test, any filters applied.

**Q7: Your team ran 200 HP trials and claims the best model is "significantly better" than the baseline. What questions do you ask?**
(1) How was "significantly better" measured? Point estimate or statistical test?
(2) What was the resampling unit, and was uncertainty computed from paired predictions rather
than treating overlapping folds as independent?
(3) Was a test set reserved that was never touched during the 200 trials? If not, the val-AUC is optimistically biased.
(4) What's the effect size (absolute AUC improvement)? A p=0.03 improvement of 0.001 AUC may not be worth the deployment overhead.
(5) Does the improvement hold across different data slices (geographic regions, product lines)?
(6) Was early stopping used consistently? If some runs used ES and others didn't, the comparison is unfair.

**Q8: Design a hyperparameter search pipeline for a production fraud detection model.**
Constraints: 30-min training per run, 24h total budget → 48 trials max.
Design: (1) Define search space: LogUniform(LR, 1e-4, 0.1), randint(depth, 3, 10), randint(n_est, 100, 500), uniform(scale_pos_weight, 1, 20) for class imbalance;
(2) Use Bayesian optimisation (Optuna TPE) — faster convergence than random for expensive evaluations;
(3) Metric: average precision (more business-relevant than AUC for fraud);
(4) Cross-validation: 5-fold, temporal split (no data leakage across time);
(5) Early stopping: patience=50 boosting rounds on val average precision;
(6) Statistical comparison: paired bootstrap on out-of-fold or untouched-test predictions,
with a predeclared minimum useful effect before promoting the winner;
(7) Log: all HP, all fold scores, model artifacts, git commit, data hash.
"""))

cells.append(md(r"""## 13. Teach-Back Section

Explain each of these from memory:

1. **6 logging categories**: Walk through what to log in each category (hyperparameters,
   metrics, artifacts, system info, code version, data version) and why omitting any one
   can break reproducibility.

2. **Grid vs random vs Bayesian**: Draw the sampling pattern of each on a 2D HP space.
   For which scenario is each the best choice?

3. **Expected Improvement**: Derive EI intuitively — what are the two terms, and what
   phenomenon does each term capture (exploitation vs exploration)?

4. **Early stopping implementation**: Code EarlyStopping from scratch: constructor, step(),
   should_stop, restore_best_weights. What is the difference between using the stopped epoch
   vs the best epoch?

5. **Nested CV**: Draw the nested CV structure. Why does regular k-fold with HP search give
   an optimistically biased estimate? What does nesting fix?

6. **Statistical comparison**: Two models have paired out-of-fold predictions. Explain how
   to bootstrap the independent unit, build a confidence interval for the metric difference,
   and why five overlapping fold scores are not five independent experiments.

7. **BO convergence**: Show the BO iteration graphically: GP posterior after 3 observations,
   EI acquisition function, next point to evaluate. How does the posterior change after 10
   observations?

8. **Reproducibility**: You're handed a model trained 6 months ago. What do you need to
   reproduce the exact same model? Which of the 6 logging categories contains the answer?
"""))

cells.append(md(r"""## 14. Exercises

### Beginner
1. Extend `ExperimentTracker.compare_runs` to also report the standard deviation of each
   metric (since each run has multiple k-fold metric values logged at different steps).
2. Add a `delete_run` method to `ExperimentTracker` that removes a run and ensures
   `best_run` still works correctly after deletion.
3. Implement `early_stopping_delta_check`: modify `EarlyStopping.step` to compute a
   centered moving average of the last 3 values before comparing to best — smoother,
   less sensitive to spike noise.

### Intermediate
4. Implement parallel random search: simulate 50 HP evaluations with `n_workers=4` parallel
   slots (round-robin assignment). Compare wall-clock time (use simulated latencies) vs
   sequential.
5. Implement a `MetricDashboard` class that: given a list of runs, produces a pandas-like
   summary table (rows=runs, columns=params+metrics), and highlights the Pareto-optimal runs
   on two metrics (e.g. AUC vs latency).
6. Implement `nested_cv_score`: given a dataset, an outer CV (5 folds) and an inner CV
   (3 folds), run random HP search on the inner fold and evaluate the best config on the
   outer fold. Compare to the non-nested val-AUC estimate for bias.

### Senior
7. **Bayesian optimisation with warm start**: Implement `BayesOptimiser.resume(X_obs, y_obs)`
   that initialises the GP from prior observations (e.g. from a previous sweep on similar
   data). Show that warm-start converges faster than cold-start by simulating 2 tasks where
   task 2's optimum is near task 1's optimum.
8. **Multi-objective optimisation**: Extend the Bayesian optimiser to support two objectives
   (AUC and inference latency). Implement the Pareto front criterion: a point dominates another
   if it's better on both objectives. Visualise the Pareto front of 50 trials in 2D space.
"""))

build("09_production_ml/04_experiment_tracking.ipynb", cells)
print("PROD-04 built.")
