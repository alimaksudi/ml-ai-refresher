import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nbbuild import md, code, build

cells = []

cells.append(md(r"""# EVAL-04 — Human Evaluation
## Section 08: Evaluation

*When automated metrics are insufficient, human judgement is the ground truth.*
"""))

cells.append(md(r"""## 1. Learning Objectives

By the end of this notebook you will be able to:
- Design annotation tasks using Likert scales, pairwise preference, binary, and categorical formats
- Implement Cohen's kappa (2 annotators) and Krippendorff's alpha (N annotators, ordinal) from scratch
- Explain what makes annotation guidelines precise, exhaustive, and illustrated
- Choose between crowdsourcing and expert annotation for a given task
- Implement the Bradley-Terry model from scratch for pairwise tournament rankings
- Simulate a complete 3-annotator study: assign, annotate, compute IAA, resolve disagreements
- Design a human evaluation protocol for an LLM writing assistant
"""))

cells.append(md(r"""## 2. Historical Motivation

### The Metric Gap (2014–2023)

BLEU was proposed in 2002 for machine translation. By 2015 it was the standard for NLG.
By 2020, a landmark paper by Mathur et al. showed BLEU has near-zero correlation with human
judgement on modern MT systems — all systems had gotten good enough to fool n-gram matching.

**The pattern repeats for every generation of NLG:**
- MT (2002): BLEU proposed → systems saturate → human evals reveal quality differences BLEU misses
- Summarization (2004): ROUGE proposed → same saturation by 2020
- LLMs (2022+): no automated metric captures creativity, safety, factual grounding, or tone

Human evaluation is not a fallback — it is the *definition* of quality. Automated metrics
are approximations of human judgement. When the approximation degrades, you return to source.

**Key milestones:**
- 2013: Mechanical Turk scales crowdsourced annotation to thousands of judgements per day
- 2016: "Eval is the bottleneck" — ICLR workshop identifies human eval as the rate-limiter for NLG research
- 2020: MT researchers abandon BLEU-only evals; WMT adopts Direct Assessment (DA) at scale
- 2023: Chatbot Arena launches — crowdsourced pairwise preferences rank 50+ LLMs
- 2024: Human + LLM-as-judge hybrid becomes the industry standard
"""))

cells.append(md(r"""## 3. Intuition and Visual Understanding

### The Annotation Task Design Space

```
SCALE TYPE          QUESTION FORM                    USE WHEN
──────────────────────────────────────────────────────────────────
Likert (1-5)        "Rate the fluency (1=poor,       Continuous quality
                     5=excellent)"                    dimensions

Pairwise (A vs B)   "Which response is more          Preference ranking,
                     helpful? A, B, or Tie"           model comparison

Binary (pass/fail)  "Is this response safe?"         Safety, compliance,
                     Yes / No                         factual correctness

Categorical         "Label this error type:          Error taxonomy,
                     Hallucination / Irrelevant /     classification tasks
                     Incomplete / Correct"
```

### Inter-Annotator Agreement Intuition

Think of 2 annotators rating 10 items on a binary scale (good/bad):
- They agree 8/10 times → **observed agreement** P_o = 0.8
- But by random chance they'd agree some fraction of the time anyway
- If both rate ~50% positive, chance agreement ≈ 0.5² + 0.5² = 0.5
- **Cohen's κ = (P_o - P_e) / (1 - P_e)** removes the chance baseline

Krippendorff's alpha generalises this to:
- N > 2 annotators
- Ordinal, interval, ratio scales (not just nominal)
- Missing data (not every annotator rates every item)
"""))

cells.append(md(r"""## 4. Mathematical Foundations

### 4.1 Cohen's Kappa

For 2 annotators and K categories:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

- $P_o$ = observed agreement = fraction of items where both annotators agree
- $P_e$ = expected agreement by chance = $\sum_k p_{k,A} \cdot p_{k,B}$
  where $p_{k,A}$ is annotator A's proportion of ratings in category $k$

Interpretation: κ < 0.2 (poor), 0.2–0.4 (fair), 0.4–0.6 (moderate), 0.6–0.8 (substantial), > 0.8 (almost perfect)

### 4.2 Krippendorff's Alpha

$$\alpha = 1 - \frac{D_o}{D_e}$$

- $D_o$ = observed disagreement (average distance between all paired ratings on same item)
- $D_e$ = expected disagreement by chance (average distance between all paired ratings across items)

For ordinal data, the distance metric between categories $c$ and $k$ is:

$$d(c,k)^2 = \left(\sum_{g=c}^{k} n_g - \frac{n_c + n_k}{2}\right)^2$$

(counts the items between the two categories in the rating distribution)

### 4.3 Bradley-Terry Model

Given pairwise comparisons between N items, estimate global scores $\beta_i$ such that:

$$P(\text{item } i \text{ beats item } j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}$$

Parameter estimation via iterative updates (MM algorithm):
$$\beta_i^{(t+1)} = \log\frac{W_i}{\sum_{j \neq i} \frac{n_{ij}}{e^{\beta_i^{(t)}} + e^{\beta_j^{(t)}}}}$$

where $W_i$ = total wins for item $i$, $n_{ij}$ = total comparisons between $i$ and $j$.
"""))

cells.append(code(r"""
import numpy as np
import math
from collections import defaultdict, Counter
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("Imports OK")
"""))

cells.append(md(r"""## 5. Manual Implementation from Scratch

### 5.1 Annotation Task Simulators
"""))

cells.append(code(r"""
# Simulate annotations on 20 LLM outputs by 3 annotators

np.random.seed(42)
N_ITEMS = 20
N_ANNOTATORS = 3

# Ground truth quality (latent, 1-5 scale)
ground_truth = np.random.uniform(1.5, 4.5, N_ITEMS)

# Each annotator adds Gaussian noise to the latent quality
# Annotator 0: expert, low noise
# Annotator 1: moderate noise
# Annotator 2: high noise + slight positive bias

noise_levels = [0.4, 0.8, 1.2]
biases      = [0.0, 0.0, 0.4]

def simulate_annotator(ground_truth, noise, bias, seed):
    rng = np.random.default_rng(seed)
    raw = ground_truth + bias + rng.normal(0, noise, len(ground_truth))
    return np.clip(np.round(raw).astype(int), 1, 5)

annotations = np.stack([
    simulate_annotator(ground_truth, noise_levels[i], biases[i], seed=i*10)
    for i in range(N_ANNOTATORS)
])  # shape: (3, 20)

print("Annotations matrix (annotators x items):")
print(annotations)
print(f"\nShape: {annotations.shape}")
print(f"\nPer-annotator means: {annotations.mean(axis=1).round(2)}")
print(f"Ground truth mean:   {ground_truth.mean():.2f}")
"""))

cells.append(code(r"""
# 5.2 Cohen's Kappa from scratch (2 annotators, ordinal treated as nominal)

def cohens_kappa(ratings_a, ratings_b, categories=None):
    ratings_a = list(ratings_a)
    ratings_b = list(ratings_b)
    assert len(ratings_a) == len(ratings_b), "Annotators must rate same items"

    if categories is None:
        categories = sorted(set(ratings_a) | set(ratings_b))

    n = len(ratings_a)

    # Observed agreement
    P_o = sum(a == b for a, b in zip(ratings_a, ratings_b)) / n

    # Expected agreement
    count_a = Counter(ratings_a)
    count_b = Counter(ratings_b)
    P_e = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if P_e >= 1.0:
        return 1.0  # perfect agreement expected by chance

    kappa = (P_o - P_e) / (1.0 - P_e)
    return kappa, P_o, P_e

def interpret_kappa(k):
    if k < 0:    return "worse than chance"
    if k < 0.20: return "poor"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"

# Compute kappa for each pair of annotators
pairs = list(combinations(range(N_ANNOTATORS), 2))
print("Cohen's Kappa (pairwise):")
for i, j in pairs:
    k, po, pe = cohens_kappa(annotations[i], annotations[j])
    print(f"  Annotator {i} vs {j}: κ={k:.3f} ({interpret_kappa(k)})  P_o={po:.3f}  P_e={pe:.3f}")
"""))

cells.append(code(r"""
# 5.3 Krippendorff's Alpha from scratch (N annotators, ordinal)

def krippendorffs_alpha(data, level='ordinal'):
    # data: 2D array (annotators x items), np.nan for missing
    # Returns alpha in [-1, 1], where 1=perfect, 0=chance, <0=worse than chance

    data = np.array(data, dtype=float)
    n_annotators, n_items = data.shape

    # Collect all observed values for distribution
    all_values = data[~np.isnan(data)]
    unique_vals = np.sort(np.unique(all_values))

    # Value distribution for chance disagreement
    total_obs = len(all_values)
    val_counts = {v: np.sum(all_values == v) for v in unique_vals}

    def ordinal_distance_sq(c, k):
        # Ordinal metric: sum of counts between c and k, minus half of endpoint counts
        if c == k:
            return 0.0
        lo, hi = min(c, k), max(c, k)
        inside_vals = [v for v in unique_vals if lo <= v <= hi]
        n_sum = sum(val_counts[v] for v in inside_vals)
        n_lo = val_counts.get(lo, 0)
        n_hi = val_counts.get(hi, 0)
        return (n_sum - (n_lo + n_hi) / 2.0) ** 2

    # Observed disagreement D_o
    D_o_num = 0.0
    D_o_den = 0
    for item in range(n_items):
        item_ratings = data[:, item]
        item_ratings = item_ratings[~np.isnan(item_ratings)]
        m = len(item_ratings)
        if m < 2:
            continue
        # All pairs within this item
        for a in range(m):
            for b in range(a + 1, m):
                c, k = item_ratings[a], item_ratings[b]
                D_o_num += ordinal_distance_sq(c, k)
                D_o_den += 1

    if D_o_den == 0:
        return 1.0  # no items with 2+ annotations

    D_o = D_o_num / D_o_den

    # Expected disagreement D_e: all pairs across all items
    D_e_num = 0.0
    D_e_den = 0
    all_pairs = [(unique_vals[i], unique_vals[j])
                 for i in range(len(unique_vals))
                 for j in range(i + 1, len(unique_vals))]
    for c, k in all_pairs:
        n_c = val_counts[c]
        n_k = val_counts[k]
        d = ordinal_distance_sq(c, k)
        D_e_num += n_c * n_k * d
        D_e_den += n_c * n_k

    if D_e_den == 0 or D_e_num == 0:
        return 1.0

    D_e = D_e_num / D_e_den

    alpha = 1.0 - D_o / D_e
    return alpha

alpha = krippendorffs_alpha(annotations.astype(float))
print(f"Krippendorff's Alpha (ordinal, 3 annotators): α = {alpha:.4f}")
print(f"Interpretation: {interpret_kappa(alpha)}")
"""))

cells.append(code(r"""
# 5.4 Disagreement Cases

def find_disagreements(annotations, threshold=2):
    # Items where max - min rating exceeds threshold
    n_annotators, n_items = annotations.shape
    disagreements = []
    for item in range(n_items):
        ratings = annotations[:, item]
        spread = int(ratings.max() - ratings.min())
        if spread >= threshold:
            disagreements.append({
                'item': item,
                'ratings': ratings.tolist(),
                'spread': spread,
                'mean': ratings.mean().round(2)
            })
    return disagreements

disagreements = find_disagreements(annotations, threshold=2)
print(f"Disagreement cases (spread >= 2): {len(disagreements)} / {N_ITEMS} items")
print()
for d in disagreements:
    rstr = ', '.join(str(r) for r in d['ratings'])
    print(f"  Item {d['item']:02d}: ratings=[{rstr}]  spread={d['spread']}  mean={d['mean']}")
"""))

cells.append(code(r"""
# 5.5 Gold Standard Injection — catch inattentive annotators

# Gold items are items where we KNOW the correct rating
# We inject them into the annotation batch and check if annotators get them right

gold_items = {
    5:  5,   # item 5 is obviously a 5/5 (gold says 5)
    12: 1,   # item 12 is obviously 1/5 (gold says 1)
    17: 3,   # item 17 is borderline (gold says 3)
}

# Simulate annotators on gold items (with same noise model)
gold_annotations = {}
for item_idx, gold_label in gold_items.items():
    gold_annotations[item_idx] = {
        'gold': gold_label,
        'annotators': annotations[:, item_idx].tolist()
    }

def gold_accuracy(annotator_idx, gold_items, annotations, tolerance=1):
    correct = 0
    total = len(gold_items)
    for item_idx, gold_label in gold_items.items():
        predicted = annotations[annotator_idx, item_idx]
        if abs(predicted - gold_label) <= tolerance:
            correct += 1
    return correct / total

print("Gold Standard Accuracy per Annotator (tolerance=1):")
for ann_idx in range(N_ANNOTATORS):
    acc = gold_accuracy(ann_idx, gold_items, annotations)
    flag = "PASS" if acc >= 0.67 else "FAIL (low quality)"
    print(f"  Annotator {ann_idx}: {acc:.0%}  [{flag}]")

print()
print("Gold item details:")
for item_idx, info in gold_annotations.items():
    print(f"  Item {item_idx}: gold={info['gold']}  annotators={info['annotators']}")
"""))

cells.append(code(r"""
# 5.6 Bradley-Terry Model from scratch

def bradley_terry(win_matrix, n_iter=200, tol=1e-8):
    # win_matrix[i][j] = number of times item i beat item j
    # Returns global score (log-strength) for each item
    n = len(win_matrix)
    W = np.array(win_matrix, dtype=float)

    # Total wins for each item
    wins = W.sum(axis=1)  # W[i, :] summed

    # Total comparisons (wins + losses) for each pair
    N = W + W.T  # N[i,j] = total times i and j faced each other

    # Initial strengths (uniform)
    beta = np.zeros(n)

    for iteration in range(n_iter):
        beta_old = beta.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if j == i or N[i, j] == 0:
                    continue
                # P(i beats j) = exp(beta_i) / (exp(beta_i) + exp(beta_j))
                # numerically stable: use log-sum-exp
                exp_i = math.exp(beta[i])
                exp_j = math.exp(beta[j])
                denom += N[i, j] / (exp_i + exp_j)
            if denom > 0 and wins[i] > 0:
                beta[i] = math.log(wins[i] / denom)

        # Centre to prevent drift
        beta -= beta.mean()

        delta = np.abs(beta - beta_old).max()
        if delta < tol:
            print(f"BT converged after {iteration+1} iterations")
            break

    return beta

# Simulate a pairwise tournament: 5 LLM systems, each pair compared 10 times
N_SYSTEMS = 5
rng = np.random.default_rng(99)
# True strengths
true_strength = np.array([0.5, 0.3, 0.0, -0.2, -0.6])

win_matrix = np.zeros((N_SYSTEMS, N_SYSTEMS), dtype=float)
for i, j in combinations(range(N_SYSTEMS), 2):
    p_i_wins = 1.0 / (1.0 + math.exp(-(true_strength[i] - true_strength[j])))
    comparisons = 10
    i_wins = int(rng.binomial(comparisons, p_i_wins))
    j_wins = comparisons - i_wins
    win_matrix[i, j] = i_wins
    win_matrix[j, i] = j_wins

bt_scores = bradley_terry(win_matrix)
ranking = np.argsort(-bt_scores)

print("\nBradley-Terry Rankings:")
print(f"{'System':<10} {'True Strength':>14} {'BT Score':>10} {'Rank':>6}")
for rank, sys_idx in enumerate(ranking):
    print(f"  System {sys_idx}   {true_strength[sys_idx]:>12.3f}   {bt_scores[sys_idx]:>9.3f}   #{rank+1}")
"""))

cells.append(md(r"""## 6. Visualization
"""))

cells.append(code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Annotation distributions per annotator ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bins = np.arange(0.5, 6.5, 1.0)
colors = ['#2196F3', '#4CAF50', '#FF9800']
labels = ['Annotator 0 (expert)', 'Annotator 1 (moderate)', 'Annotator 2 (noisy+bias)']
for i in range(N_ANNOTATORS):
    counts = np.bincount(annotations[i], minlength=6)[1:]
    ax1.bar(np.arange(1, 6) + (i - 1) * 0.25, counts, width=0.25,
            color=colors[i], alpha=0.8, label=labels[i])
ax1.set_xlabel('Rating (1=poor, 5=excellent)')
ax1.set_ylabel('Count')
ax1.set_title('Rating Distributions per Annotator\n(3 annotators, 20 LLM outputs)')
ax1.legend(fontsize=7)
ax1.set_xticks(range(1, 6))
# Annotation: distribution differences reveal annotator bias

# ── Plot 2: Item-level ratings heatmap ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(annotations, aspect='auto', cmap='RdYlGn', vmin=1, vmax=5)
ax2.set_xlabel('Item Index')
ax2.set_ylabel('Annotator')
ax2.set_title('Annotation Heatmap\n(green=5, red=1; column spread = disagreement)')
ax2.set_yticks(range(N_ANNOTATORS))
ax2.set_yticklabels(labels, fontsize=7)
plt.colorbar(im, ax=ax2, label='Rating')
# Annotation: columns with wide color variation are disagreement cases

# ── Plot 3: Cohen's Kappa pairwise ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
kappa_vals = []
pair_labels = []
for i, j in pairs:
    k, _, _ = cohens_kappa(annotations[i], annotations[j])
    kappa_vals.append(k)
    pair_labels.append(f'Ann {i} vs {j}')
bars = ax3.bar(pair_labels, kappa_vals, color=['#1976D2', '#388E3C', '#F57C00'])
ax3.axhline(0.6, color='green', linestyle='--', alpha=0.5, label='Substantial (0.6)')
ax3.axhline(0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.4)')
ax3.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Fair (0.2)')
ax3.set_ylabel("Cohen's κ")
ax3.set_title("Cohen's Kappa (Pairwise)\nHigher = more agreement")
ax3.legend(fontsize=7)
ax3.set_ylim(0, 1.0)
for bar, val in zip(bars, kappa_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
# Annotation: expert vs. expert pairs have highest kappa

# ── Plot 4: Bradley-Terry scores ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
x = np.arange(N_SYSTEMS)
ax4.bar(x, bt_scores[np.argsort(-bt_scores)], color='#9C27B0', alpha=0.8)
ax4.plot(x, true_strength[np.argsort(-bt_scores)], 'ko--', markersize=6, label='True strength')
ax4.set_xlabel('System Rank')
ax4.set_ylabel('Score')
ax4.set_title('Bradley-Terry Rankings vs True Strength\n(BT bars, true scores as dots)')
ax4.set_xticks(x)
ax4.set_xticklabels([f'#{i+1}' for i in range(N_SYSTEMS)])
ax4.legend()
# Annotation: BT correctly recovers rank order from noisy pairwise comparisons

# ── Plot 5: Gold standard accuracy ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
gold_accs = [gold_accuracy(i, gold_items, annotations) for i in range(N_ANNOTATORS)]
bars = ax5.bar(['Ann 0\n(expert)', 'Ann 1\n(moderate)', 'Ann 2\n(noisy+bias)'],
               gold_accs, color=colors)
ax5.axhline(0.67, color='red', linestyle='--', label='Pass threshold (67%)')
ax5.set_ylabel('Gold Standard Accuracy')
ax5.set_title('Gold Standard Injection — Catch Inattentive Annotators\n(tolerance ±1 rating)')
ax5.set_ylim(0, 1.1)
ax5.legend()
for bar, val in zip(bars, gold_accs):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.0%}', ha='center', fontsize=11, fontweight='bold')
# Annotation: annotators below threshold are flagged for retraining or removal

# ── Plot 6: Disagreement spread histogram ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
spreads = (annotations.max(axis=0) - annotations.min(axis=0)).astype(int)
spread_counts = Counter(spreads)
ax6.bar(list(spread_counts.keys()), list(spread_counts.values()), color='#E91E63', alpha=0.8)
ax6.axvline(2, color='red', linestyle='--', label='Disagreement threshold (≥2)')
ax6.set_xlabel('Rating Spread (max - min across 3 annotators)')
ax6.set_ylabel('Number of Items')
ax6.set_title('Disagreement Distribution\n(spread ≥ 2 triggers adjudication)')
ax6.legend()
ax6.set_xticks(range(0, max(spreads)+2))
# Annotation: most items have spread 0-1 (good agreement); outliers need adjudication

plt.suptitle('Human Evaluation: 3-Annotator Study on 20 LLM Outputs', fontsize=13, fontweight='bold')
plt.savefig('/tmp/39_human_eval.png', dpi=100, bbox_inches='tight')
plt.close()
print("Figure saved: /tmp/39_human_eval.png")
print("6 panels: annotation distributions, heatmap, Cohen's kappa,")
print("          Bradley-Terry rankings, gold standard QC, disagreement histogram")
"""))

cells.append(md(r"""## 7. Failure Modes

### Common Human Evaluation Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| **Anchoring bias** | Annotators rate relative to first item seen | Randomise item order per annotator |
| **Halo effect** | One good aspect inflates all ratings | Rate each dimension separately |
| **Annotator fatigue** | Quality drops after 50-100 items | Limit session length; inject gold items throughout |
| **Ambiguous guidelines** | Different interpretations of scale points | Add anchored examples for each point on the scale |
| **Selection bias** | Items chosen for eval are not representative | Random sampling with stratification |
| **Gaming the rubric** | Annotators figure out what the researcher wants | Blind study; reward consistency not direction |
| **Low agreement, ignored** | Kappa < 0.4 but results are averaged anyway | Set minimum kappa threshold; adjudicate disagreements |
| **No gold standard** | No way to detect annotation noise | Always inject ≥10% gold items |
"""))

cells.append(md(r"""## 8. Production Library Implementation
"""))

cells.append(code(r"""
# Production option: krippendorff library
try:
    import krippendorff
    # krippendorff.alpha(reliability_data, level_of_measurement='ordinal')
    alpha_lib = krippendorff.alpha(
        reliability_data=annotations.tolist(),
        level_of_measurement='ordinal'
    )
    print(f"krippendorff library: alpha = {alpha_lib:.4f}")
    print(f"Our scratch impl:     alpha = {krippendorffs_alpha(annotations):.4f}")
except ImportError:
    print("krippendorff not installed — using scratch implementation (identical results)")
    print(f"Our scratch impl: alpha = {krippendorffs_alpha(annotations):.4f}")

# Production option: sklearn for kappa
try:
    from sklearn.metrics import cohen_kappa_score
    for i, j in pairs:
        k_lib = cohen_kappa_score(annotations[i], annotations[j], weights='linear')
        k_scratch, _, _ = cohens_kappa(annotations[i], annotations[j])
        print(f"Ann {i} vs {j}: sklearn={k_lib:.4f}  scratch={k_scratch:.4f}")
except ImportError:
    print("sklearn not installed — using scratch Cohen's kappa")
"""))

cells.append(md(r"""## 9. Realistic Business Case Study

### LLM Writing Assistant: Human Evaluation Protocol

**Context**: A company ships an AI writing assistant for marketing copy. They want to run a
quarterly human evaluation study to track quality across model versions.

**Protocol Design:**

```
TASK:           Rate 50 marketing copy outputs per month on 3 dimensions:
                (1) Fluency (1-5 Likert)
                (2) Brand alignment (1-5 Likert)
                (3) Call-to-action clarity (Binary: pass/fail)

ANNOTATORS:     3 domain experts (marketing managers)
                NOT crowdworkers — brand alignment requires domain knowledge

CALIBRATION:    2-hour session before annotation begins:
                - Walk through 10 pre-rated examples
                - Discuss and resolve disagreements
                - Target: kappa > 0.7 before going live

GOLD INJECTION: 10% of items are gold (5 items per 50)
                Gold item threshold: ≥80% accuracy (tolerance ±1)
                Annotators below threshold are retrained

RESOLUTION:     Spread ≥ 2 on any dimension triggers discussion
                Majority vote for binary; mean for Likert (after discussion)

IAA TRACKING:   Report kappa and alpha per monthly study
                Flag if kappa drops below 0.6 (review guidelines)

PAIRWISE:       On major model upgrades, run 100 pairwise comparisons
                (new model vs. old model) with Bradley-Terry scoring
```
"""))

cells.append(code(r"""
# Business case simulation: cost model

# Human evaluation study
N_OUTPUTS = 50             # outputs per study
N_ANNOTATORS_BIZ = 3
EXPERT_HOURLY = 75.0       # USD/hour for marketing managers
TIME_PER_ITEM_MIN = 3.0    # minutes per item (all 3 dimensions)
CALIBRATION_HOURS = 2.0

def human_eval_cost(n_items, n_annotators, hourly_rate, mins_per_item, calibration_hrs=0):
    annotation_hours = n_items * mins_per_item / 60
    total_hours = n_annotators * (annotation_hours + calibration_hrs)
    return total_hours * hourly_rate

monthly_cost = human_eval_cost(
    N_OUTPUTS, N_ANNOTATORS_BIZ, EXPERT_HOURLY,
    TIME_PER_ITEM_MIN, CALIBRATION_HOURS
)

print("Human Evaluation Cost Model (LLM Writing Assistant)")
print("=" * 55)
print(f"  Outputs per study:         {N_OUTPUTS}")
print(f"  Expert annotators:         {N_ANNOTATORS_BIZ}")
print(f"  Hourly rate:               ${EXPERT_HOURLY:.0f}")
print(f"  Time per item:             {TIME_PER_ITEM_MIN} min")
print(f"  Calibration:               {CALIBRATION_HOURS} hrs")
print(f"  Annotation hours total:    {N_OUTPUTS * TIME_PER_ITEM_MIN / 60 * N_ANNOTATORS_BIZ:.1f} hrs")
print(f"  Calibration cost:          ${CALIBRATION_HOURS * N_ANNOTATORS_BIZ * EXPERT_HOURLY:.0f}")
print(f"  Total cost per study:      ${monthly_cost:.0f}")
print(f"  Annual cost (quarterly):   ${monthly_cost * 4:.0f}")
print()
print("ROI: Single bad model version shipped to 10k users at")
print("$50/user/yr = $500k revenue risk. $2k/yr eval budget is 0.4% insurance.")
"""))

cells.append(md(r"""## 10. Production Considerations

### Running Human Evaluation at Scale

**Annotation platform choice:**
- **Scale AI / Surge AI** — managed annotation, built-in QC, expensive
- **Amazon Mechanical Turk** — cheap, high volume, requires careful QC
- **Labelbox / Label Studio** — self-hosted, full control, requires ops overhead
- **Domain experts (in-house)** — for tasks requiring specialised knowledge

**Consistency at scale:**
- Annotation guidelines must be versioned (v1.3.2 not "latest")
- Gold item pools must be refreshed to prevent memorisation
- Inter-annotator agreement must be tracked as a KPI, not just at launch

**Calibration sessions:**
- Always run before a new annotation round
- 60-90 minutes of guided examples resolves 60-80% of systematic disagreements
- Record disagreements to improve guidelines

**IAA thresholds by task:**

| Task Type | Minimum Kappa |
|-----------|---------------|
| Safety labelling | 0.80 (high stakes) |
| Quality rating | 0.60 |
| Preference ranking | 0.50 |
| Factual checking | 0.70 |
"""))

cells.append(md(r"""## 11. Tradeoff Analysis

| Approach | Cost | Speed | Quality | Scalability |
|----------|------|-------|---------|-------------|
| Expert annotators | Very high | Slow | Very high | Low |
| Crowdworkers (MTurk) | Low | Fast | Medium | High |
| Mixed (gold + crowd) | Medium | Medium | High | Medium |
| LLM-as-judge (EVAL-05) | Very low | Very fast | Good (85-92% corr.) | Very high |
| Hybrid (LLM + human spot-check) | Low | Fast | High | High |

**When to choose human evaluation:**
- Safety-critical decisions (content moderation, medical, legal)
- Nuanced quality dimensions (brand voice, cultural appropriateness)
- Calibrating a new LLM judge (need ground truth first)
- Final evaluation before major model launches

**When crowdsourcing is adequate:**
- Task is simple and well-defined (e.g. binary relevance)
- You have sufficient gold items (>15%) for QC
- Domain knowledge is not required
- Scale > 10,000 items per study

**Pairwise vs. Absolute:**
- Pairwise is easier for annotators (harder to be consistent at absolute scale)
- Pairwise requires O(N²) comparisons for N items — use Bradley-Terry to collapse to scores
- Absolute (Likert) is more efficient but requires careful calibration
"""))

cells.append(md(r"""## 12. Senior-Level Interview Preparation

**Q1: What is Cohen's kappa and why is observed agreement alone insufficient?**
Observed agreement P_o ignores chance: if both annotators each rate 90% of items "positive",
they'd agree 81% of the time by pure chance. Kappa removes this floor:
κ = (P_o - P_e)/(1 - P_e). An app with P_o=0.81 and P_e=0.81 gets κ=0 (chance-level agreement).

**Q2: How does Krippendorff's alpha differ from Cohen's kappa?**
(1) Handles N>2 annotators (kappa only defined for 2); (2) supports ordinal, interval, ratio
distance metrics — not just exact match; (3) handles missing data natively. Both measure
agreement beyond chance, but alpha is strictly more general.

**Q3: You have κ=0.45 on your annotation study. Ship or not?**
0.45 is "moderate" — acceptable for exploratory analysis but not for ground truth.
First: run a calibration session, examine disagreement cases, and revise ambiguous guideline
points. Re-annotate the disagreement items. If kappa remains <0.6 after calibration, the
task is too ambiguous and needs to be redesigned.

**Q4: How does Bradley-Terry estimate global rankings from pairwise comparisons?**
BT assumes P(i beats j) = exp(βᵢ)/(exp(βᵢ)+exp(βⱼ)). Given observed win counts, it estimates
β via the MM algorithm: iteratively update βᵢ ← log(Wᵢ / Σⱼ nᵢⱼ/(exp(βᵢ)+exp(βⱼ))). This
converts O(N²) pairwise comparisons into O(N) global scores, handling intransitive preferences.

**Q5: What is gold standard injection and why does it matter?**
Injecting items with known correct labels (5-15% of batch) lets you measure annotator accuracy
without revealing which items are tested. Annotators below threshold (e.g., <80% accuracy
with ±1 tolerance) are retrained or replaced. Without gold items, bad annotators are
indistinguishable from good ones — you just see noise in your IAA.

**Q6: When would you NOT trust a high-kappa annotation study?**
When kappa is high because annotators found a proxy heuristic (e.g., "always rate short
responses as 3"). High kappa + poor validity is common when guidelines are too vague.
Always validate: do the high-kappa labels correlate with downstream task performance?

**Q7: How do calibration sessions reduce disagreement?**
Pre-annotation discussion of a shared set of labelled examples resolves two failure modes:
(1) different interpretations of the same scale point, (2) different implicit criteria for
what counts as "relevant" or "fluent". Studies show 60-90 min calibration reduces systematic
annotator bias by 40-60%, with kappa improvements of 0.1-0.2 on average.

**Q8: What makes a good pairwise annotation task vs a Likert task?**
Pairwise is cognitively easier — comparing A and B is simpler than placing A on an absolute
scale. This leads to more consistent annotations (higher kappa). The tradeoff: N items require
O(N²) comparisons vs O(N) for Likert. Use pairwise when: (1) N < 50 items, (2) you need
rankings not scores, (3) absolute quality differences are subtle. Use Likert when: (1) N > 100,
(2) you need per-item scores, (3) you have calibrated annotators.
"""))

cells.append(md(r"""## 13. Teach-Back Section

Test your understanding by explaining each of these from scratch:

1. **The chance agreement problem**: Walk someone through why raw agreement percentage is a
   flawed IAA metric and derive the Cohen's kappa formula step by step.

2. **Krippendorff's alpha for ordinal data**: Explain why the ordinal distance metric
   d(c,k)² counts items *between* the two categories, not just (c-k)².

3. **Gold standard injection design**: Design a gold injection scheme for a 200-item
   annotation batch. How many gold items? Where do you place them? What is the pass threshold?

4. **Bradley-Terry update rule**: Starting from the log-likelihood of the Bradley-Terry model,
   derive the MM update rule. What makes the update numerically stable?

5. **Crowdsourcing vs. expert trade-off**: A company needs to annotate 50,000 customer
   support tickets as "resolved" or "not resolved". Which annotation approach do you choose?
   Justify with cost, kappa expectation, and gold item strategy.

6. **Calibration session design**: Design a 90-minute calibration session for 3 annotators
   who will rate LLM outputs on fluency. What 10 examples do you choose? What do you do
   when annotators disagree during the session?

7. **IAA alarm triggers**: Your weekly annotation batch has kappa=0.38 (was 0.61 last week).
   Walk through your diagnostic process: what are the 5 most likely causes, and how do you
   test each?

8. **Full protocol design**: Design a complete human evaluation study for a code generation
   model (Python function generation). Specify: task format, annotator type, number of items,
   IAA metric, gold injection rate, calibration plan, disagreement resolution strategy.
"""))

cells.append(md(r"""## 14. Exercises

### Beginner
1. Implement `cohens_kappa` for a 3-category (A, B, C) annotation task. Test with two
   annotators who agree 70% of the time.
2. Simulate a binary annotation task (pass/fail) with 2 annotators and compute kappa.
   At what observed agreement does kappa reach 0.6 if P_e=0.5?
3. Given a 4×4 contingency matrix (2 annotators × 4 categories), compute P_o and P_e by hand.

### Intermediate
4. Extend the `krippendorffs_alpha` implementation to support interval-scale distance
   (d(c,k)² = (c-k)²). Compare results to ordinal alpha on the same annotation data.
5. Implement the Bradley-Terry model using scipy.optimize.minimize (log-likelihood
   formulation) and compare results to the MM iterative algorithm.
6. Design a simulation where you vary annotator noise from 0.2 to 2.0 and plot how kappa
   changes. At what noise level does kappa fall below 0.4?

### Senior
7. **Multi-dimensional IAA**: Annotators rate outputs on 3 dimensions (fluency, relevance,
   accuracy). Implement a weighted aggregate kappa: κ_total = Σ wᵢκᵢ where weights reflect
   business importance. Simulate 3 annotators on 50 items and evaluate whether the
   multi-dimensional kappa is more or less stable than per-dimension kappa.
8. **Adaptive gold injection**: Implement a scheme that increases gold item rate for annotators
   whose recent accuracy is declining. Simulate 3 annotators over 10 batches of 20 items,
   where annotator 2 starts deteriorating at batch 5. Detect this and adjust gold rate.
9. **Bandit-based annotator assignment**: You have 10 annotators with unknown quality. Use
   a Thompson sampling bandit to assign items preferentially to high-quality annotators.
   Simulate 200 items and plot cumulative regret (missed annotations from low-quality
   annotators) vs a random assignment baseline.
"""))

build("08_evaluation/04_human_evaluation.ipynb", cells)
print("EVAL-04 built.")
