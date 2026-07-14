"""NB49 — AI Reliability Patterns builder."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbbuild import md, code, build

cells = [

md(r"""
# 49 — AI Reliability Patterns

## 1. Learning Objectives

By the end of this notebook you will be able to:

- Classify ML system failures into four categories: model, infrastructure, data, dependency
- Implement **RetryWithBackoff** with exponential backoff and jitter from scratch
- Implement a **CircuitBreaker** with closed/half-open/open state transitions from scratch
- Implement a **FallbackChain** with ordered fallback and per-tier health tracking
- Implement an **ErrorBudgetTracker** that computes SLO compliance and burn rate
- Apply chaos engineering principles to test ML system resilience
- Define SLI, SLO, and SLA and compute error budgets from uptime targets
- Design a fraud detection system with 99.99% uptime and zero-downtime ML failover
"""),

md(r"""
## 2. Historical Motivation

### Why Reliability is an ML-Specific Problem

General software reliability (Netflix, Google SRE) is well-understood.
ML systems add a unique failure dimension: **silent degradation**.
A web server either responds or it doesn't. An ML model always responds —
but the response may be wrong in ways that take days to detect.

**Famous ML reliability failures**:

| Year | System | Failure | Impact |
|------|--------|---------|--------|
| 2016 | Knight Capital (trading) | Stale model executed at 10ms; human couldn't override | $440M in 45 minutes |
| 2018 | Amazon Rekognition | Model mismatch in prod vs test environment | 28 US congresspeople misidentified as criminals |
| 2020 | Zillow iBuying | Drift undetected; model overpriced homes | $500M loss; product shutdown |
| 2022 | Twitter recommendation | Latent feature store became stale post-acquisition | Engagement drop, advertiser exodus |

**The SRE insight applied to ML**:
Google's Site Reliability Engineering book defines the goal as
"keeping services up" — for ML that means: model is served, features are fresh,
predictions are calibrated, and business metrics are not degrading.

### SLO/SLI/SLA Vocabulary

| Term | Definition | Example |
|------|-----------|---------|
| SLI (Service Level Indicator) | The metric you measure | p99 latency, error rate |
| SLO (Service Level Objective) | The internal target | p99 < 200ms, error rate < 0.1% |
| SLA (Service Level Agreement) | The external contract | 99.9% uptime or credit issued |
| Error budget | Allowable downtime before SLA breach | 99.9% = 8.76h/year |
| Burn rate | How fast error budget is consumed | 10x = budget gone in 36 days |
"""),

md(r"""
## 3. Intuition & Visual Understanding

### The Reliability Hierarchy for ML Systems

```
Layer 1: Infrastructure reliability
   → Load balancer health checks, auto-scaling, multi-AZ
   → "Is the service responding?"

Layer 2: Data reliability
   → Feature freshness checks, schema validation, pipeline SLOs
   → "Is the input data correct?"

Layer 3: Model reliability
   → Drift monitoring, AUC tracking, calibration checks
   → "Are predictions trustworthy?"

Layer 4: Business reliability
   → Revenue, conversion, fraud rate
   → "Is the system delivering value?"
```

### Circuit Breaker State Machine

```
  ┌─────────────┐    failure_count >= threshold    ┌──────────────┐
  │   CLOSED    │ ─────────────────────────────→  │    OPEN      │
  │ (pass thru) │ ←───────────────────────────── │ (reject all) │
  └─────────────┘    all probe requests succeed   └──────┬───────┘
         ↑                                               │ after timeout
         │          probe request succeeds               ↓
         └───────────────────────────────── ┌───────────────────┐
                                            │   HALF-OPEN       │
                                            │ (allow 1 probe)   │
                                            └───────────────────┘
```

### Error Budget Burn Rate

At 99.99% uptime SLA: error budget = 52 minutes / year.
If you're consuming it at 10x the normal rate, it's gone in 5.2 minutes.
Burn rate > 1 means you'll breach the SLA at this rate.
"""),

md(r"""
## 4. Mathematical Foundations

### 4.1 Error Budget

$$\text{Error budget} = (1 - \text{SLO}) \times \text{time period}$$

At 99.99% monthly: $0.0001 \times 30 \times 24 \times 60 = 4.32$ minutes.

### 4.2 Burn Rate

$$\text{Burn rate} = \frac{\text{error rate}}{\text{SLO error rate}}$$

e.g., SLO = 99.99% (error rate = 0.01%); current error rate = 0.1%:
burn rate = 0.1% / 0.01% = 10x.

Budget consumed in period $T$ = burn\_rate $\times T / \text{window}$.

### 4.3 Exponential Backoff with Jitter

$$\text{delay}_k = \min(\text{cap}, \text{base} \times 2^k) + \text{Uniform}(0, \text{jitter\_scale})$$

Full jitter: $\text{delay}_k \sim \text{Uniform}(0, \min(\text{cap}, \text{base} \times 2^k))$

Jitter prevents thundering-herd when many clients retry simultaneously.

### 4.4 Availability Arithmetic

$$\text{Availability} = \prod_{i} \text{Avail}_i \quad \text{(series)}$$
$$\text{Availability} = 1 - \prod_i (1 - \text{Avail}_i) \quad \text{(parallel/redundant)}$$

Two 99.9% services in series: $0.999 \times 0.999 = 99.8\%$.
Two 99.9% services in parallel (either can serve): $1 - (0.001)^2 = 99.9999\%$.
"""),

md(r"""
## 5. Manual Implementation from Scratch
"""),

code(r"""
import math
import random
import time
import collections
from typing import Callable, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# ── RetryWithBackoff ──────────────────────────────────────────────────────────
class RetryWithBackoff:
    def __init__(self, max_retries: int = 3, base_delay_s: float = 0.1,
                 cap_s: float = 30.0, jitter: bool = True):
        self._max_retries  = max_retries
        self._base         = base_delay_s
        self._cap          = cap_s
        self._jitter       = jitter
        self.attempt_log: List[Tuple[int, float, bool]] = []  # (attempt, delay, success)

    def _delay(self, attempt: int) -> float:
        d = min(self._cap, self._base * (2 ** attempt))
        if self._jitter:
            d = random.uniform(0, d)  # full jitter
        return d

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                self.attempt_log.append((attempt, 0.0, True))
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._delay(attempt)
                    self.attempt_log.append((attempt, delay, False))
                    # In real code: time.sleep(delay)
        raise last_exc


# ── CircuitBreaker ────────────────────────────────────────────────────────────
class CBState(Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 60.0,
                 probe_success_threshold: int = 2):
        self._failure_threshold   = failure_threshold
        self._recovery_timeout    = recovery_timeout_s
        self._probe_threshold     = probe_success_threshold
        self._state               = CBState.CLOSED
        self._failure_count       = 0
        self._probe_success_count = 0
        self._opened_at: Optional[float] = None
        self.transitions: List[Tuple[CBState, CBState, str]] = []

    def _transition(self, new_state: CBState, reason: str):
        self.transitions.append((self._state, new_state, reason))
        self._state = new_state

    @property
    def state(self) -> CBState:
        return self._state

    def allow_request(self, now: float = None) -> bool:
        now = now or time.monotonic()
        if self._state == CBState.CLOSED:
            return True
        if self._state == CBState.OPEN:
            if (now - self._opened_at) > self._recovery_timeout:
                self._probe_success_count = 0
                self._transition(CBState.HALF_OPEN, "recovery timeout elapsed")
                return True
            return False
        # HALF_OPEN: allow probe
        return True

    def record_success(self):
        if self._state == CBState.HALF_OPEN:
            self._probe_success_count += 1
            if self._probe_success_count >= self._probe_threshold:
                self._failure_count = 0
                self._transition(CBState.CLOSED, f"{self._probe_threshold} probe successes")
        elif self._state == CBState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, now: float = None):
        now = now or time.monotonic()
        if self._state == CBState.HALF_OPEN:
            self._transition(CBState.OPEN, "probe failed in HALF_OPEN")
            self._opened_at = now
        elif self._state == CBState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._transition(CBState.OPEN, f"{self._failure_count} failures")
                self._opened_at = now


# ── FallbackChain ─────────────────────────────────────────────────────────────
@dataclass
class FallbackTier:
    name:   str
    fn:     Callable
    calls:  int = 0
    errors: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / max(self.calls, 1)

class FallbackChain:
    def __init__(self, tiers: List[FallbackTier]):
        self._tiers = tiers

    def call(self, *args, **kwargs) -> Tuple[Any, str]:
        for tier in self._tiers:
            tier.calls += 1
            try:
                result = tier.fn(*args, **kwargs)
                return result, tier.name
            except Exception:
                tier.errors += 1
        raise RuntimeError("All fallback tiers exhausted")

    def report(self):
        print("FallbackChain tier health:")
        for t in self._tiers:
            print(f"  {t.name:<20}: {t.calls:>5} calls, {t.errors:>4} errors "
                  f"({t.error_rate:.1%} error rate)")


# ── ErrorBudgetTracker ────────────────────────────────────────────────────────
class ErrorBudgetTracker:
    def __init__(self, slo_percent: float = 99.99, window_minutes: int = 43200):
        self._slo     = slo_percent / 100.0
        self._window  = window_minutes
        self._events: collections.deque = collections.deque()  # (tick, is_error)
        self._tick    = 0

    def record(self, is_error: bool):
        self._events.append((self._tick, is_error))
        self._tick += 1
        # Keep only last window
        while len(self._events) > self._window:
            self._events.popleft()

    def current_error_rate(self) -> float:
        if not self._events:
            return 0.0
        errors = sum(1 for _, e in self._events if e)
        return errors / len(self._events)

    def burn_rate(self) -> float:
        slo_error_rate = 1.0 - self._slo
        current = self.current_error_rate()
        return current / max(slo_error_rate, 1e-12)

    def budget_remaining_pct(self) -> float:
        n = len(self._events)
        if n == 0:
            return 100.0
        errors = sum(1 for _, e in self._events if e)
        budget_total = int(n * (1.0 - self._slo))
        remaining    = max(0, budget_total - errors)
        return 100.0 * remaining / max(budget_total, 1)

    def report(self):
        er   = self.current_error_rate()
        br   = self.burn_rate()
        budg = self.budget_remaining_pct()
        print(f"Error Budget Report (SLO={self._slo*100:.2f}%):")
        print(f"  Current error rate:    {er:.4%}")
        print(f"  Burn rate:             {br:.2f}x  {'⚠ ALERT' if br > 1 else 'OK'}")
        print(f"  Budget remaining:      {budg:.1f}%")
        return {"error_rate": er, "burn_rate": br, "budget_remaining_pct": budg}


# ── Smoke tests ───────────────────────────────────────────────────────────────
rng = random.Random(42)

# RetryWithBackoff
call_count = [0]
def flaky(fail_prob=0.6):
    call_count[0] += 1
    if rng.random() < fail_prob:
        raise ValueError("transient error")
    return "success"

retry = RetryWithBackoff(max_retries=4, base_delay_s=0.001, jitter=True)
try:
    result = retry.call(flaky, fail_prob=0.5)
    print(f"RetryWithBackoff: succeeded after {len(retry.attempt_log)} attempts")
except Exception as e:
    print(f"RetryWithBackoff: failed after {len(retry.attempt_log)} attempts: {e}")

# CircuitBreaker
cb = CircuitBreaker(failure_threshold=3, recovery_timeout_s=1.0)
tick = 0.0
for i in range(6):
    if cb.allow_request(now=tick):
        cb.record_failure(now=tick)
    tick += 0.1
print(f"\nCircuitBreaker after 6 failures: state={cb.state.value}")
tick = 2.0  # simulate time passing
if cb.allow_request(now=tick):
    cb.record_success()
    cb.record_success()
print(f"CircuitBreaker after recovery probe: state={cb.state.value}")
print(f"Transitions: {[(a.value, b.value, r) for a, b, r in cb.transitions]}")
"""),

code(r"""
# FallbackChain demo
def ml_model_predict(x):
    raise RuntimeError("ML model GPU OOM")

def rule_based_predict(x):
    return float(x > 0.5)  # simple threshold rule

def default_predict(x):
    return 0.1  # conservative safe default

chain = FallbackChain([
    FallbackTier("ml_model",    ml_model_predict),
    FallbackTier("rule_based",  rule_based_predict),
    FallbackTier("safe_default", default_predict),
])

for val in [0.3, 0.7, 0.1, 0.9]:
    result, tier = chain.call(val)
    print(f"  input={val:.1f}  result={result:.1f}  tier={tier}")

chain.report()

# ErrorBudgetTracker
import random
rng2 = random.Random(7)
tracker = ErrorBudgetTracker(slo_percent=99.99, window_minutes=1000)
for i in range(1000):
    # 0.05% error rate — well within budget
    is_error = rng2.random() < 0.0005
    tracker.record(is_error)
print()
tracker.report()

# Simulate budget burn (error spike)
for i in range(100):
    tracker.record(rng2.random() < 0.05)  # 5% error rate spike
print()
tracker.report()
"""),

md(r"""
## 6. Visualization
"""),

code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("AI Reliability Patterns", fontsize=14, fontweight='bold')

rng_np = np.random.default_rng(42)

# ── (a) Exponential backoff delay schedule ────────────────────────────────────
ax = axes[0]
attempts = np.arange(0, 8)
base, cap = 0.1, 30.0
delays_no_jitter = np.minimum(cap, base * (2.0 ** attempts))
delays_jitter    = rng_np.uniform(0, delays_no_jitter)

ax.bar(attempts - 0.2, delays_no_jitter, width=0.35, label='No jitter', color='steelblue', alpha=0.8)
ax.bar(attempts + 0.2, delays_jitter,    width=0.35, label='Full jitter', color='tomato', alpha=0.8)
ax.axhline(cap, color='grey', linestyle='--', linewidth=1.2, label=f'Cap={cap}s')
ax.set_xlabel("Retry attempt")
ax.set_ylabel("Delay (s)")
ax.set_title("Exponential Backoff:\nJitter Prevents Thundering Herd", fontsize=11)
ax.legend(fontsize=8)
ax.set_yscale('log')
# Without jitter all clients retry at the same exponential intervals, creating spikes.

# ── (b) Circuit breaker simulation ───────────────────────────────────────────
ax = axes[1]
N_TICKS = 200
error_rate_sim = np.concatenate([
    rng_np.uniform(0, 0.02, 50),    # normal: <2% errors
    rng_np.uniform(0.3, 0.8, 50),   # outage: 30-80% errors
    rng_np.uniform(0, 0.02, 100),   # recovery
])
cb2 = CircuitBreaker(failure_threshold=5, recovery_timeout_s=15)
states_numeric = []
tick_sim = 0.0
for i, er in enumerate(error_rate_sim):
    is_err = rng_np.random() < er
    tick_sim += 0.5
    if cb2.allow_request(now=tick_sim):
        if is_err:
            cb2.record_failure(now=tick_sim)
        else:
            cb2.record_success()
    state_val = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}[cb2.state.value]
    states_numeric.append(state_val)

xs = np.arange(N_TICKS)
ax.fill_between(xs, error_rate_sim, alpha=0.3, color='steelblue', label='Error rate')
ax.plot(xs, error_rate_sim, color='steelblue', linewidth=0.8)
ax2b = ax.twinx()
ax2b.plot(xs, states_numeric, color='tomato', linewidth=2, label='CB state (0=Closed,2=Open)')
ax.set_xlabel("Time tick")
ax.set_ylabel("Error rate", color='steelblue')
ax2b.set_ylabel("CB state", color='tomato')
ax2b.set_yticks([0, 1, 2])
ax2b.set_yticklabels(['Closed', 'Half', 'Open'], fontsize=8)
ax.set_title("Circuit Breaker Opens During\nError Spike", fontsize=11)
ax.axvspan(50, 100, alpha=0.08, color='tomato', label='Outage window')
ax.legend(fontsize=7, loc='upper left')
# CB state jumps to OPEN during the error spike, preventing cascading failures.

# ── (c) Error budget burn down ────────────────────────────────────────────────
ax = axes[2]
budget_start = 100.0
tracker_vis  = ErrorBudgetTracker(slo_percent=99.99, window_minutes=5000)
rng_v = np.random.default_rng(123)
budgets = [budget_start]
for i in range(500):
    # Spike at i=200
    p = 0.05 if 200 <= i < 260 else 0.0008
    tracker_vis.record(rng_v.random() < p)
    budgets.append(tracker_vis.budget_remaining_pct())

ax.plot(budgets, color='steelblue', linewidth=1.5)
ax.axhline(0,  color='tomato', linestyle='--', linewidth=1.5, label='Budget exhausted')
ax.axhline(25, color='orange', linestyle=':', linewidth=1.2, label='25% remaining — alert')
ax.axvspan(200, 260, alpha=0.12, color='tomato', label='Error spike')
ax.set_xlabel("Events processed")
ax.set_ylabel("Error budget remaining (%)")
ax.set_title("Error Budget Burn Down\nDuring Incident", fontsize=11)
ax.legend(fontsize=8)
# Budget drops steeply during the spike (200–260); recovers as error rate normalises.

plt.tight_layout()
plt.savefig('/tmp/nb49_reliability.png', dpi=80, bbox_inches='tight')
plt.show()
print("Figure saved.")
"""),

md(r"""
## 7. Failure Modes

| Failure | Category | Detection | Mitigation |
|---------|---------|-----------|-----------|
| GPU OOM | Infrastructure | Error rate spike | Memory-efficient batching; OOM error → circuit breaker |
| Feature store timeout | Dependency | Latency p99 spike | Fallback to cached/default features; circuit breaker |
| Training data schema change | Data | Validation gate fails | Schema registry; backward-compatible migration |
| Model prediction drift | Model | PSI alert | Retraining trigger; fallback to previous model version |
| Label pipeline delay | Data | Missing labels in evaluation | Delayed label monitoring; alert if >24h lag |
| Service crash (OOM kill) | Infrastructure | Health check failure | Auto-restart; memory limits; kill switch |
| Dependency API rate limit | Dependency | 429 error rate | Retry with backoff; token bucket pre-throttling |
| Silent accuracy degradation | Model | Slow AUC decline | EWMA monitoring; business metric correlation |
"""),

md(r"""
## 8. Production Library Implementation
"""),

code(r"""
try:
    import tenacity
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

try:
    import pybreaker
    HAS_PYBREAKER = True
except ImportError:
    HAS_PYBREAKER = False

import random as rnd

rnd.seed(99)

def flaky_service(fail_prob=0.4):
    if rnd.random() < fail_prob:
        raise ConnectionError("service unavailable")
    return "OK"

if HAS_TENACITY:
    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=0.01, min=0.01, max=1),
           retry=retry_if_exception_type(ConnectionError))
    def reliable_call():
        return flaky_service(0.5)
    try:
        result = reliable_call()
        print(f"tenacity retry: {result}")
    except Exception as e:
        print(f"tenacity: all retries failed: {e}")
else:
    print("tenacity not installed — using scratch RetryWithBackoff above")
    rb = RetryWithBackoff(max_retries=4, base_delay_s=0.001, jitter=True)
    try:
        result = rb.call(flaky_service, 0.5)
        print(f"scratch retry: {result}")
    except Exception as e:
        print(f"scratch retry: all retries failed: {e}")

if HAS_PYBREAKER:
    breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=10)
    @breaker
    def protected():
        return flaky_service(0.8)
    for i in range(6):
        try:
            r = protected()
            print(f"pybreaker call {i}: {r}")
        except Exception as e:
            print(f"pybreaker call {i}: {type(e).__name__}")
else:
    print("pybreaker not installed — using scratch CircuitBreaker above")
    cb_demo = CircuitBreaker(failure_threshold=3, recovery_timeout_s=999)
    tick_d = 0.0
    for i in range(6):
        tick_d += 0.1
        allowed = cb_demo.allow_request(now=tick_d)
        if allowed:
            try:
                r = flaky_service(0.8)
                cb_demo.record_success()
                print(f"call {i}: success (state={cb_demo.state.value})")
            except Exception:
                cb_demo.record_failure(now=tick_d)
                print(f"call {i}: failure (state={cb_demo.state.value})")
        else:
            print(f"call {i}: REJECTED by circuit breaker (state={cb_demo.state.value})")
"""),

md(r"""
## 9. Business Case Study — Payment Fraud Detection with 99.99% Uptime SLA

**Scenario**: A payment processor must detect fraud on every transaction.
Downtime > 4.32 minutes/month breaches the SLA and incurs financial penalties.
When the ML model is degraded, fall back to rule-based heuristics.
"""),

code(r"""
import random
import math

rng_biz = random.Random(2024)

# ── System components ─────────────────────────────────────────────────────────
def ml_fraud_model(txn):
    if rng_biz.random() < txn.get("_fail_rate", 0.0):
        raise RuntimeError("ML model unavailable")
    score = (txn["amount"] > 500) * 0.4 + (txn["is_foreign"] * 0.3) + rng_biz.gauss(0, 0.1)
    return {"score": min(max(score, 0), 1), "model": "ml_v3"}

def rule_based_fraud(txn):
    score = 0.8 if txn["amount"] > 1000 else (0.4 if txn["is_foreign"] else 0.1)
    return {"score": score, "model": "rules_v1"}

def safe_default_fraud(txn):
    return {"score": 0.3, "model": "safe_default"}

fraud_chain = FallbackChain([
    FallbackTier("ml_model",   ml_fraud_model),
    FallbackTier("rule_based", rule_based_fraud),
    FallbackTier("safe_default", safe_default_fraud),
])

budget = ErrorBudgetTracker(slo_percent=99.99, window_minutes=43200)
cb_fraud = CircuitBreaker(failure_threshold=10, recovery_timeout_s=30)

N_TXN = 2000
FAIL_WINDOW = (800, 900)  # simulate ML outage at txns 800-900

results = {"ml": 0, "rule": 0, "default": 0, "blocked": 0}
tick_f = 0.0

for i in range(N_TXN):
    tick_f += 0.05
    fail_rate = 0.95 if FAIL_WINDOW[0] <= i < FAIL_WINDOW[1] else 0.0
    txn = {"amount": rng_biz.uniform(10, 2000),
           "is_foreign": rng_biz.random() < 0.15,
           "_fail_rate": fail_rate}

    if not cb_fraud.allow_request(now=tick_f):
        results["blocked"] += 1
        budget.record(True)  # blocked = error
        continue

    try:
        result, tier = fraud_chain.call(txn)
        if tier == "ml_model":
            cb_fraud.record_success()
            results["ml"] += 1
        elif tier == "rule_based":
            results["rule"] += 1
        else:
            results["default"] += 1
        budget.record(False)
    except Exception:
        cb_fraud.record_failure(now=tick_f)
        budget.record(True)

print("Fraud Detection Results:")
print(f"  ML model served:    {results['ml']:>5} ({results['ml']/N_TXN:.1%})")
print(f"  Rule-based served:  {results['rule']:>5} ({results['rule']/N_TXN:.1%})")
print(f"  Safe default:       {results['default']:>5}")
print(f"  CB blocked:         {results['blocked']:>5}")
print(f"  Circuit breaker state: {cb_fraud.state.value}")
print()
budget.report()
print()
fraud_chain.report()
"""),

md(r"""
## 10. Production Considerations

### Golden Signals for ML Systems

| Signal | Metric | Alert threshold |
|--------|--------|----------------|
| Latency | p99 inference latency | > 2× baseline |
| Traffic | Requests/second | Drop > 20% or spike > 3× |
| Errors | Error rate | > 1% |
| Saturation | GPU utilisation, queue depth | > 80% for >5 min |
| ML-specific | AUC, PSI, prediction entropy | Thresholds per model |

### Chaos Engineering for ML

```python
# Chaos toolkit / manual chaos: inject failures deliberately
class ChaosInjector:
    def wrap(self, fn, fail_rate, latency_ms=0):
        def wrapped(*args, **kwargs):
            if random.random() < fail_rate:
                raise RuntimeError("chaos injection")
            if latency_ms > 0:
                time.sleep(latency_ms / 1000)
            return fn(*args, **kwargs)
        return wrapped
```

**ML-specific chaos tests**:
- Corrupt 10% of feature values → does pipeline detect it?
- Delay feature store by 5s → does serving degrade gracefully?
- Feed adversarial inputs (out-of-distribution) → does model confidence drop?
- Kill one model replica mid-request → does load balancer reroute?

### Bulkhead Isolation

Separate thread pools / resource pools per model tier.
A slow batch scoring job cannot starve the real-time inference pool.
"""),

md(r"""
## 11. Tradeoff Analysis

| Pattern | Reliability Gain | Latency Cost | Complexity | Use When |
|---------|-----------------|-------------|-----------|---------|
| Retry + backoff | High (transient faults) | Medium (adds latency on retry) | Low | Idempotent calls |
| Circuit breaker | High (cascading prevention) | None (fast fail) | Medium | Downstream dependency |
| Fallback chain | High (graceful degrade) | Low | Medium | Always — ML serving |
| Bulkhead | Medium (isolation) | None | Medium | Multi-tenant or mixed SLOs |
| Timeout | Medium | Negative (caps tail) | Very low | Always |
| Health check | Medium | None | Very low | Always |

| SLO level | Error budget (30-day) | Engineering cost | Use For |
|-----------|----------------------|-----------------|---------|
| 99.0% | 7.2 hours | Low | Internal tools |
| 99.9% | 43.2 min | Medium | Most ML APIs |
| 99.99% | 4.32 min | Very high | Payments, fraud |
| 99.999% | 26 seconds | Extreme | Life-critical |
"""),

md(r"""
## 12. Senior-Level Interview Preparation

**Q1**: How do you achieve 99.99% uptime for an ML serving system?

> Multi-AZ deployment (availability via redundancy). Circuit breaker on ML model with fallback to rule-based. Retry-with-backoff for transient errors. Error budget monitoring with automated alerts at 25% remaining. Chaos testing quarterly. Never deploy a new model without shadow mode + circuit breaker protection.

**Q2**: What is the difference between SLO, SLA, and SLI?

> SLI = the measurement (p99 latency, error rate). SLO = the internal target (p99 < 200ms). SLA = the external contract with penalties (99.9% uptime). Set SLO tighter than SLA (e.g., SLO = 99.95% when SLA = 99.9%) so you have time to fix issues before breaching the contract.

**Q3**: Explain exponential backoff with full jitter and why jitter matters.

> Without jitter: all clients back off for 2^k seconds simultaneously, then hammer the server together. With full jitter: each client samples Uniform(0, 2^k) — retries are spread out, preventing correlated load spikes (thundering herd).

**Q4**: A downstream feature store is timing out intermittently. How do you handle it?

> (1) Retry 2x with 10ms backoff. (2) Circuit breaker: open if >10% requests fail in 30s. (3) Fallback: serve with stale cached features if available. (4) Safe default: serve with zero/mean features and log for investigation. (5) Alert on-call when circuit opens.

**Q5**: What is burn rate and what does a burn rate of 10x mean?

> Burn rate = current error rate / SLO error rate. 10x means you're consuming the error budget 10 times faster than sustainable. At 99.9% SLO (budget = 43.2 min/month), a 10x burn rate exhausts the budget in 4.3 minutes.

**Q6**: How do you apply chaos engineering to an ML system?

> (1) Identify steady-state hypothesis (p99 < 200ms, error rate < 0.1%). (2) Inject failures: kill a model replica, corrupt features, delay the feature store. (3) Observe: does the system recover? Does fallback kick in? Does the circuit breaker open? (4) Fix gaps in resilience. Run in staging first; then in prod with blast radius limited to 1% of traffic.

**Q7**: What is the bulkhead pattern and when would you use it in ML?

> Bulkhead = isolate resource pools so one consumer can't starve another. In ML: separate thread pools for real-time inference vs batch scoring. If the batch job saturates threads, real-time inference keeps its pool and hits its SLO. Use when mixing latency-sensitive (real-time) and throughput-optimised (batch) workloads on shared infra.

**Q8**: Your ML model starts returning scores of 0.0 for all inputs after a deploy. How do you detect and recover?

> Detection: prediction distribution monitoring (entropy collapses to 0; PSI > 0.5). Recovery: (1) circuit breaker opens if error rate spikes. (2) Fallback chain serves rule-based. (3) Rollback: model registry promotes previous version. (4) Root cause: verify model artifact hash, feature schema, preprocessing version. Prevention: shadow mode before promote; prediction sanity check gate.
"""),

md(r"""
## 13. Teach-Back Section

1. Define SLI, SLO, SLA in one sentence each.
2. What is an error budget and how do you compute it for 99.9% monthly SLO?
3. Explain exponential backoff in plain English. Why add jitter?
4. Draw the three states of a circuit breaker and describe the transition rules.
5. When should the fallback chain return a "safe default" vs a rule-based result?
6. What is the bulkhead pattern and how does it differ from a circuit breaker?
7. What is burn rate? Give an example at 10x burn on a 99.99% SLA.
8. You are designing chaos tests for a fraud detection ML system. Name 4 specific failure scenarios you would inject.
"""),

md(r"""
## 14. Exercises

### Beginner
1. Compute the monthly error budget (in minutes) for 99.9%, 99.95%, and 99.99% SLOs.
2. Modify `RetryWithBackoff` to log the delay for each attempt and plot the delay schedule.
3. Add a `timeout_s` parameter to `FallbackTier` that marks a tier as failed if it takes longer than the timeout.

### Intermediate
4. Implement **decorrelated jitter**: $\text{delay}_k = \min(\text{cap}, \text{Uniform}(\text{base}, 3 \times \text{delay}_{k-1}))$. Compare its distribution to full jitter on 1000 samples.
5. Add a **sliding window** to `CircuitBreaker`: instead of a cumulative failure count, use the failure rate in the last N requests (more robust to old errors keeping the breaker open).
6. Implement a **chaos injector** class that wraps any function and randomly injects failures (configurable rate), latency (configurable distribution), and corrupted outputs (return None/NaN with given probability).

### Senior
7. Implement a **multi-level fallback with SLO budgets**: each tier has its own SLO. The chain escalates to the next tier only if the current tier's p99 exceeds its SLO budget (not just on errors).
8. Design a **synthetic monitoring** harness for ML: every minute, send a canary transaction with a known expected output. Alert if the actual output deviates by more than ε. This detects silent degradation before user traffic is affected.
9. Build an **adaptive circuit breaker**: the failure threshold automatically adjusts based on observed baseline error rate. In a noisy environment (baseline 2%), open at 10%; in a clean environment (baseline 0.01%), open at 1%.
"""),

]  # end cells

if __name__ == "__main__":
    build("phase9_system_design/49_ai_reliability.ipynb", cells)
