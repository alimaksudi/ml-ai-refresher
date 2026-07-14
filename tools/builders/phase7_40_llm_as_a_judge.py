import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nbbuild import md, code, build

cells = []

cells.append(md(r"""# Notebook 40 — LLM-as-a-Judge
## Phase 7: Evaluation

*Using LLMs to evaluate LLM outputs: 85–92% correlation with human judgement at 1/100th the cost.*
"""))

cells.append(md(r"""## 1. Learning Objectives

By the end of this notebook you will be able to:
- Explain why LLM-as-judge achieves 85-92% correlation with human judgement
- Design a judge prompt with the 3 required elements: rubric, examples, output format
- Implement single-answer grading (score 1-10 with reasoning) from scratch
- Implement pairwise comparison with position-bias mitigation (swap and average)
- Identify and measure the 3 main LLM judge biases: position, verbosity, self-preference
- Compute Spearman correlation between LLM judge scores and human scores
- Explain MT-Bench and Chatbot Arena's use of LLM-as-judge
- Design a cost-effective evaluation pipeline for 1000 responses/day
"""))

cells.append(md(r"""## 2. Historical Motivation

### The Evaluation Scaling Problem (2022–2024)

By late 2022, LLMs could produce outputs so fluent that automated metrics (BLEU, ROUGE)
completely failed to differentiate quality. Human evaluation remained the gold standard
but was expensive, slow, and didn't scale.

**Key cost comparison:**
- Human evaluation: $0.50–$2.00 per response (expert) or $0.05–$0.20 (crowd)
- LLM-as-judge (GPT-4 API): ~$0.01 per response
- LLM-as-judge (Llama-3 local): ~$0.001 per response

**Key accuracy result:**
- Zheng et al. (2023), *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*:
  GPT-4 as judge achieves 80-85% agreement with expert human evaluators
  (human-human agreement baseline: ~80-85% for same task)
  → GPT-4 judge is statistically indistinguishable from a human expert

**Industry adoption:**
- MT-Bench (2023): 80-question benchmark judged by GPT-4, 50+ LLMs ranked
- Chatbot Arena (2023): Crowdsourced pairwise + LLM judge hybrid
- AlpacaEval (2023): LLM judge at scale; Claude Sonnet 3.7 achieves 91% win-rate
- Internal evals at Anthropic, OpenAI, Google: LLM judge is primary signal for RLHF

**The judge bias problem (2023-2024):**
Wang et al. (2023): GPT-4 as judge shows position bias (prefers first answer in pairwise),
verbosity bias (longer = better), and self-preference (GPT-4 prefers GPT-4 outputs).
The fix: swap + aggregate pairwise comparisons.
"""))

cells.append(md(r"""## 3. Intuition and Visual Understanding

### The Judge Prompt Design Space

```
ELEMENT 1: RUBRIC — what dimensions matter and at what weight
┌─────────────────────────────────────────────────────────┐
│ Score 1-10 based on:                                    │
│   - Accuracy: Does the answer correctly answer the Q?  │
│     (40% weight)                                        │
│   - Completeness: All aspects addressed? (30%)          │
│   - Clarity: Clear, well-structured? (20%)              │
│   - Conciseness: No unnecessary padding? (10%)          │
└─────────────────────────────────────────────────────────┘

ELEMENT 2: EXAMPLES — anchored reference points
┌─────────────────────────────────────────────────────────┐
│ Score 1: Completely wrong or refuses to answer          │
│ Score 4: Partially correct but missing key elements     │
│ Score 7: Mostly correct, minor gaps                     │
│ Score 10: Perfect, comprehensive, clear, concise        │
└─────────────────────────────────────────────────────────┘

ELEMENT 3: OUTPUT FORMAT — structured, parseable
┌─────────────────────────────────────────────────────────┐
│ Respond ONLY with JSON:                                 │
│ {"score": <int 1-10>, "reasoning": "<2-3 sentences>"}  │
└─────────────────────────────────────────────────────────┘
```

### Position Bias: Why Swapping Works

```
Round 1:  [A, B] → judge prefers A (score: A=7, B=5) — position bias inflates A
Round 2:  [B, A] → same judge, same outputs — judge prefers B... ← same real B, now first
          → If judge is consistent about quality: A still wins (7>5, 7>5)
          → If judge has position bias: B wins when first → averaged out
Aggregate: winner = response with higher score in BOTH rounds, or avg if mixed
```

### The 3 Biases

```
1. POSITION BIAS:  P(choose first) > 0.5 even when quality is equal
   Fix: always run both orderings; take the response that wins in both, or average.

2. VERBOSITY BIAS: longer_response.score > shorter_response.score when quality is equal
   Fix: instruct judge to ignore length; penalise padding explicitly in rubric.

3. SELF-PREFERENCE: model_A.judge.score(model_A_output) > model_B.judge.score(model_A_output)
   Fix: use a different model as judge; use ensemble of judges.
```
"""))

cells.append(md(r"""## 4. Mathematical Foundations

### 4.1 Spearman Rank Correlation

Measures monotonic correlation (order agreement, not linear):

$$r_s = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)}$$

where $d_i$ is the rank difference for the $i$-th item between two orderings.
Range: [-1, 1]. LLM judge target: $r_s > 0.85$ vs human scores.

### 4.2 Position Bias Mitigation

For pairwise comparison of responses A and B:
1. Judge(A, B) → score_A_first, score_B_first
2. Judge(B, A) → score_B_second, score_A_second (swapped)
3. Debiased score for A: $(score\_A\_first + score\_A\_second) / 2$
4. Debiased score for B: $(score\_B\_first + score\_B\_second) / 2$

### 4.3 Verbosity Bias Correction

Residual regression: fit a linear model to predict score from token count on gold
examples with known quality. The residuals (score - predicted-from-length) give
length-adjusted scores.

### 4.4 Win Rate

$$\text{win\_rate}(A) = \frac{\text{wins}(A)}{\text{wins}(A) + \text{losses}(A) + 0.5 \cdot \text{ties}(A)}$$

Used in AlpacaEval: win rate of model vs GPT-4 on 805 questions from AlpacaFarm.
"""))

cells.append(code(r"""
import numpy as np
import math
import json
import re
from collections import defaultdict

np.random.seed(42)

# Utility: Spearman rank correlation from scratch
def spearman_correlation(x, y):
    n = len(x)
    assert n == len(y), "Must be same length"
    rank_x = _ranks(x)
    rank_y = _ranks(y)
    d_sq = sum((rx - ry)**2 for rx, ry in zip(rank_x, rank_y))
    if n < 2:
        return 0.0
    rs = 1.0 - 6.0 * d_sq / (n * (n**2 - 1))
    return rs

def _ranks(arr):
    n = len(arr)
    sorted_idx = sorted(range(n), key=lambda i: arr[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[sorted_idx[j]] == arr[sorted_idx[j+1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-indexed average
        for k in range(i, j+1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks

# Quick test
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
print(f"Spearman([1..5], [5..1]) = {spearman_correlation(x, y):.4f}  (expected: -1.0)")
x2 = [1, 2, 3, 4, 5]
y2 = [1, 2, 3, 4, 5]
print(f"Spearman([1..5], [1..5]) = {spearman_correlation(x2, y2):.4f}  (expected: +1.0)")
"""))

cells.append(md(r"""## 5. Manual Implementation from Scratch

### 5.1 Simulated LLM Responses and Human Scores
"""))

cells.append(code(r"""
# Simulate 50 question-answer pairs with human scores and LLM judge scores

N_EXAMPLES = 50

# Ground truth human scores (1-10)
human_scores = np.clip(
    np.random.normal(6.0, 2.0, N_EXAMPLES), 1.0, 10.0
).round(1)

# Response lengths (tokens) — varies across examples
response_lengths = (50 + human_scores * 20 + np.random.normal(0, 30, N_EXAMPLES)).astype(int)
response_lengths = np.clip(response_lengths, 20, 400)

# System A responses: quality tracks human scores + noise
def llm_judge_scores(human_scores, response_lengths, bias_params):
    # Simulate LLM judge with configurable biases
    pos_bias     = bias_params.get('position', 0.0)   # adds to score if response is "first"
    verb_bias    = bias_params.get('verbosity', 0.0)  # correlation with length
    noise_std    = bias_params.get('noise', 0.5)
    rng          = np.random.default_rng(bias_params.get('seed', 0))

    base = human_scores + rng.normal(0, noise_std, N_EXAMPLES)
    # Verbosity bias: longer responses score higher regardless of quality
    length_effect = verb_bias * (response_lengths - response_lengths.mean()) / response_lengths.std()
    raw = base + length_effect + pos_bias
    return np.clip(np.round(raw, 1), 1.0, 10.0)

# Judge 1: good judge (high correlation, low bias)
judge_good = llm_judge_scores(human_scores, response_lengths,
    {'position': 0.0, 'verbosity': 0.1, 'noise': 0.4, 'seed': 1})

# Judge 2: biased judge (position + verbosity bias)
judge_biased = llm_judge_scores(human_scores, response_lengths,
    {'position': 0.5, 'verbosity': 0.8, 'noise': 0.8, 'seed': 2})

rs_good   = spearman_correlation(list(human_scores), list(judge_good))
rs_biased = spearman_correlation(list(human_scores), list(judge_biased))

print(f"Spearman correlation (good judge vs. human):   {rs_good:.4f}")
print(f"Spearman correlation (biased judge vs. human): {rs_biased:.4f}")
print(f"\nTarget for production judge: r_s > 0.85")
print(f"Good judge: {'PASS' if rs_good > 0.85 else 'FAIL'}")
print(f"Biased judge: {'PASS' if rs_biased > 0.85 else 'FAIL'}")
"""))

cells.append(code(r"""
# 5.2 Judge Prompt Template (the 3 elements)

SINGLE_ANSWER_PROMPT = r'''You are an expert evaluator for AI-generated responses.

RUBRIC:
Score the response on a scale of 1 to 10 based on:
- Accuracy (40%): Does the response correctly answer the question?
- Completeness (30%): Are all relevant aspects addressed?
- Clarity (20%): Is the response clear and well-structured?
- Conciseness (10%): No unnecessary filler or repetition?

EXAMPLES:
- Score 1: Completely wrong, refuses to answer, or off-topic
- Score 4: Partially correct but missing major components
- Score 7: Mostly correct with minor gaps or clarity issues
- Score 10: Perfect, comprehensive, clear, and concise

OUTPUT FORMAT (respond ONLY with valid JSON):
{"score": <integer 1-10>, "reasoning": "<2-3 sentence justification>"}

Question: {question}
Response: {response}
'''

PAIRWISE_PROMPT = r'''You are an expert evaluator comparing two AI-generated responses.

RUBRIC: Evaluate which response is better based on accuracy, completeness, and clarity.
Ignore response length — longer does not mean better.
If both are equal quality, output "tie".

OUTPUT FORMAT (respond ONLY with valid JSON):
{"winner": "A" or "B" or "tie", "reasoning": "<2-3 sentences>"}

Question: {question}
Response A: {response_a}
Response B: {response_b}
'''

REFERENCE_GUIDED_PROMPT = r'''You are an expert evaluator scoring a response against a gold reference.

RUBRIC: Compare the response to the reference answer.
Score 1 if the response contradicts or ignores the reference.
Score 10 if the response matches or improves on the reference.

OUTPUT FORMAT (respond ONLY with valid JSON):
{"score": <integer 1-10>, "alignment": "low/medium/high", "reasoning": "<2-3 sentences>"}

Question: {question}
Reference answer: {reference}
Response to evaluate: {response}
'''

print("Judge prompts defined.")
print(f"Single-answer prompt length: {len(SINGLE_ANSWER_PROMPT)} chars")
print(f"Pairwise prompt length:      {len(PAIRWISE_PROMPT)} chars")
print(f"Reference-guided prompt length: {len(REFERENCE_GUIDED_PROMPT)} chars")
"""))

cells.append(code(r"""
# 5.3 Mock LLM Judge (simulates an API response for demonstration)
# In production: replace with actual API call to OpenAI / Claude / Gemini

class MockLLMJudge:
    def __init__(self, quality_corr=0.9, verbosity_bias=0.0, position_bias=0.0,
                 seed=42):
        # quality_corr: how well judge tracks true quality
        self._quality_corr = quality_corr
        self._verb_bias    = verbosity_bias
        self._pos_bias     = position_bias
        self._rng          = np.random.default_rng(seed)
        self._call_count   = 0

    def _score_from_quality(self, true_quality, length=100, position=0):
        noise = self._rng.normal(0, 1.0 - self._quality_corr + 0.1)
        score = true_quality + noise
        score += self._verb_bias * (length - 150) / 100.0
        score += self._pos_bias * (1 if position == 0 else -0.5)
        return float(np.clip(round(score, 1), 1.0, 10.0))

    def grade_single(self, question, response, true_quality=7.0, length=100):
        self._call_count += 1
        score = self._score_from_quality(true_quality, length)
        reasoning = f"Response scored {score}/10 based on accuracy and clarity."
        return {"score": score, "reasoning": reasoning}

    def grade_pairwise(self, question, response_a, response_b,
                       quality_a=7.0, quality_b=5.0,
                       length_a=100, length_b=150,
                       a_is_first=True):
        self._call_count += 1
        pos = 0 if a_is_first else 1
        score_a = self._score_from_quality(quality_a, length_a, position=(0 if a_is_first else 1))
        score_b = self._score_from_quality(quality_b, length_b, position=(1 if a_is_first else 0))
        if abs(score_a - score_b) < 0.5:
            winner, reasoning = "tie", "Both responses are of comparable quality."
        elif score_a > score_b:
            winner = "A"
            reasoning = f"Response A is clearer and more accurate (A:{score_a:.1f} vs B:{score_b:.1f})."
        else:
            winner = "B"
            reasoning = f"Response B is more complete (B:{score_b:.1f} vs A:{score_a:.1f})."
        return {"winner": winner, "reasoning": reasoning,
                "_score_a": score_a, "_score_b": score_b}

judge = MockLLMJudge(quality_corr=0.9, verbosity_bias=0.1, position_bias=0.3, seed=7)
result = judge.grade_single("What is gradient descent?", "response text", true_quality=8.0, length=120)
print("Single-answer grading result:")
print(json.dumps(result, indent=2))
"""))

cells.append(code(r"""
# 5.4 Pairwise comparison with position-bias mitigation

def debiased_pairwise(judge, question, response_a, response_b,
                      quality_a, quality_b, length_a, length_b):
    # Round 1: [A, B]
    r1 = judge.grade_pairwise(question, response_a, response_b,
                               quality_a, quality_b, length_a, length_b,
                               a_is_first=True)
    # Round 2: [B, A]  (swapped)
    r2 = judge.grade_pairwise(question, response_b, response_a,
                               quality_b, quality_a, length_b, length_a,
                               a_is_first=False)

    # Average debiased scores
    score_a_avg = (r1['_score_a'] + r2['_score_a']) / 2
    score_b_avg = (r1['_score_b'] + r2['_score_b']) / 2

    if score_a_avg > score_b_avg + 0.3:
        winner = 'A'
    elif score_b_avg > score_a_avg + 0.3:
        winner = 'B'
    else:
        winner = 'tie'

    return {
        'winner': winner,
        'score_a_avg': round(score_a_avg, 2),
        'score_b_avg': round(score_b_avg, 2),
        'round1': r1['winner'],
        'round2_effective': 'A' if r2['winner']=='B' else ('B' if r2['winner']=='A' else 'tie'),
    }

# Demonstrate: A is better quality (8 vs 5) but B is longer (250 vs 80 tokens)
result = debiased_pairwise(
    judge,
    question="Explain backpropagation.",
    response_a="Short but accurate answer...",
    response_b="Very long, detailed but partially wrong answer...",
    quality_a=8.0, quality_b=5.0,
    length_a=80, length_b=250
)
print("Debiased pairwise result:")
print(json.dumps(result, indent=2))
print()
print(f"A wins: {result['winner']=='A'}  (correct: A is higher quality)")
"""))

cells.append(code(r"""
# 5.5 Measure position bias

def measure_position_bias(judge, n_trials=100):
    # When quality_a == quality_b, a fair judge should pick 'tie' or split 50/50
    # A biased judge picks 'A' more often (first-position preference)
    wins_a_first = 0  # A in position 1, A wins
    wins_b_first = 0  # B in position 1, B wins

    rng = np.random.default_rng(123)
    for _ in range(n_trials):
        q = 6.0  # equal quality
        l = 100

        # A is first
        r1 = judge.grade_pairwise("Q", "Resp A", "Resp B", q, q, l, l, a_is_first=True)
        if r1['winner'] in ('A', 'tie'):
            wins_a_first += 1

        # B is first (so B occupies position 1)
        r2 = judge.grade_pairwise("Q", "Resp B", "Resp A", q, q, l, l, a_is_first=False)
        if r2['winner'] in ('B', 'tie'):
            wins_b_first += 1

    position_bias = (wins_a_first - wins_b_first) / n_trials
    return {
        'win_rate_when_first_a': wins_a_first / n_trials,
        'win_rate_when_first_b': wins_b_first / n_trials,
        'position_bias_score': position_bias,
        'interpretation': 'biased toward first position' if abs(position_bias) > 0.1 else 'unbiased'
    }

bias_result = measure_position_bias(judge, n_trials=200)
print("Position Bias Measurement (equal-quality responses):")
print(json.dumps(bias_result, indent=2))

# Compare to a fair judge
fair_judge = MockLLMJudge(quality_corr=0.9, verbosity_bias=0.0, position_bias=0.0, seed=8)
fair_bias  = measure_position_bias(fair_judge, n_trials=200)
print("\nFair judge position bias:")
print(json.dumps(fair_bias, indent=2))
"""))

cells.append(code(r"""
# 5.6 Verbosity bias measurement

def measure_verbosity_bias(judge, n_trials=100):
    # Compare short high-quality vs long low-quality responses
    # A fair judge should prefer the high-quality one
    correct_choices = 0

    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        quality_short = 8.0   # short but good
        quality_long  = 5.0   # long but mediocre
        length_short  = 50
        length_long   = 400

        r = judge.grade_pairwise(
            "Q", "Short but good response", "Long but mediocre response",
            quality_short, quality_long, length_short, length_long,
            a_is_first=True
        )
        if r['winner'] == 'A':  # A is the short, high-quality one
            correct_choices += 1

    accuracy = correct_choices / n_trials
    return {
        'accuracy_short_quality_wins': accuracy,
        'verbosity_bias_severity': 'high' if accuracy < 0.6 else 'medium' if accuracy < 0.8 else 'low'
    }

verb_result = measure_verbosity_bias(judge, n_trials=200)
print("Verbosity Bias (short-quality vs long-mediocre):")
print(json.dumps(verb_result, indent=2))

fair_verb = measure_verbosity_bias(fair_judge, n_trials=200)
print("\nFair judge verbosity bias:")
print(json.dumps(fair_verb, indent=2))
"""))

cells.append(code(r"""
# 5.7 Calibration: LLM judge vs human scores (Spearman correlation)

# Simulate 50 examples: human scores + judge scores
human_scores_list = list(human_scores)

# Grade each example with our mock judge
judge_calibration = MockLLMJudge(quality_corr=0.92, verbosity_bias=0.05, position_bias=0.0, seed=5)

llm_judge_scores_list = []
for i in range(N_EXAMPLES):
    result = judge_calibration.grade_single(
        f"Question {i}",
        f"Response {i}",
        true_quality=human_scores[i],
        length=int(response_lengths[i])
    )
    llm_judge_scores_list.append(result['score'])

rs = spearman_correlation(human_scores_list, llm_judge_scores_list)
print(f"Calibration Study (N={N_EXAMPLES} examples)")
print(f"Spearman correlation (LLM judge vs human): {rs:.4f}")
print(f"Target: > 0.85   Result: {'PASS' if rs > 0.85 else 'FAIL'}")
print()

# Score distribution comparison
print("Human score distribution (quartiles):")
hs = sorted(human_scores_list)
print(f"  Q1={hs[N_EXAMPLES//4]:.1f}  median={hs[N_EXAMPLES//2]:.1f}  Q3={hs[3*N_EXAMPLES//4]:.1f}")

ls = sorted(llm_judge_scores_list)
print("LLM judge score distribution (quartiles):")
print(f"  Q1={ls[N_EXAMPLES//4]:.1f}  median={ls[N_EXAMPLES//2]:.1f}  Q3={ls[3*N_EXAMPLES//4]:.1f}")
"""))

cells.append(md(r"""## 6. Visualization
"""))

cells.append(code(r"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: LLM judge vs human scores scatter ────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(human_scores_list, llm_judge_scores_list, alpha=0.6, color='#1976D2', s=40)
ax1.plot([1, 10], [1, 10], 'r--', alpha=0.5, label='Perfect agreement')
ax1.set_xlabel('Human Score')
ax1.set_ylabel('LLM Judge Score')
ax1.set_title(f'LLM Judge vs Human Score Calibration\nSpearman r = {rs:.3f}')
ax1.legend()
ax1.set_xlim(1, 10); ax1.set_ylim(1, 10)
# Annotation: points near the diagonal show strong agreement; outliers reveal judge biases

# ── Plot 2: Good vs biased judge calibration ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(human_scores_list, list(judge_good), alpha=0.6, color='green',
            s=40, label=f'Good judge (rs={rs_good:.2f})')
ax2.scatter(human_scores_list, list(judge_biased), alpha=0.4, color='red',
            s=40, label=f'Biased judge (rs={rs_biased:.2f})', marker='^')
ax2.plot([1, 10], [1, 10], 'k--', alpha=0.3)
ax2.set_xlabel('Human Score')
ax2.set_ylabel('LLM Judge Score')
ax2.set_title('Good vs Biased Judge\n(calibration against human ground truth)')
ax2.legend(fontsize=8)
# Annotation: red triangles scatter further from the diagonal — bias adds noise

# ── Plot 3: Position bias measurement ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
categories = ['Biased Judge\n(A first)', 'Biased Judge\n(B first)',
              'Fair Judge\n(A first)', 'Fair Judge\n(B first)']
win_rates = [
    bias_result['win_rate_when_first_a'],
    bias_result['win_rate_when_first_b'],
    fair_bias['win_rate_when_first_a'],
    fair_bias['win_rate_when_first_b'],
]
colors_pb = ['#E53935', '#EF9A9A', '#43A047', '#A5D6A7']
bars = ax3.bar(categories, win_rates, color=colors_pb, alpha=0.8)
ax3.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='No-bias baseline (0.5)')
ax3.set_ylabel('Win Rate of First Response')
ax3.set_title('Position Bias Measurement\n(equal quality responses, N=200 trials)')
ax3.set_ylim(0, 1.0)
ax3.legend()
for bar, val in zip(bars, win_rates):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', fontsize=9)
# Annotation: biased judge picks first response >50% even when quality is equal

# ── Plot 4: Verbosity bias — score vs response length ────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(response_lengths, llm_judge_scores_list, alpha=0.5, color='#7B1FA2', s=40,
            label='Judge score')
ax4.scatter(response_lengths, human_scores_list, alpha=0.3, color='gray', s=30,
            label='Human score', marker='x')
# Fit lines
m_judge, b_judge = np.polyfit(response_lengths, llm_judge_scores_list, 1)
m_human, b_human = np.polyfit(response_lengths, human_scores_list, 1)
x_range = np.linspace(response_lengths.min(), response_lengths.max(), 50)
ax4.plot(x_range, m_judge * x_range + b_judge, color='#7B1FA2',
         label=f'Judge trend (slope={m_judge:.3f})')
ax4.plot(x_range, m_human * x_range + b_human, color='gray', linestyle='--',
         label=f'Human trend (slope={m_human:.3f})')
ax4.set_xlabel('Response Length (tokens)')
ax4.set_ylabel('Score')
ax4.set_title('Verbosity Bias Analysis\n(steeper judge slope = more verbosity bias)')
ax4.legend(fontsize=7)
# Annotation: if judge slope >> human slope, judge rewards length regardless of quality

# ── Plot 5: Debiased vs single-round pairwise ────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
# Simulate 50 pairwise comparisons
N_PAIRS = 50
quality_a_arr = np.random.uniform(4, 10, N_PAIRS)
quality_b_arr = np.random.uniform(4, 10, N_PAIRS)
true_winner   = ['A' if qa > qb else ('tie' if abs(qa-qb)<0.5 else 'B')
                  for qa, qb in zip(quality_a_arr, quality_b_arr)]

biased_pjudge = MockLLMJudge(quality_corr=0.85, position_bias=0.6, seed=99)

single_correct = 0
debiased_correct = 0

for i in range(N_PAIRS):
    qa, qb = quality_a_arr[i], quality_b_arr[i]
    tw = true_winner[i]

    r_single = biased_pjudge.grade_pairwise("Q", "A", "B", qa, qb, 100, 100, a_is_first=True)
    if r_single['winner'] == tw or tw == 'tie':
        single_correct += 1

    r_db = debiased_pairwise(biased_pjudge, "Q", "A", "B", qa, qb, 100, 100)
    if r_db['winner'] == tw or tw == 'tie':
        debiased_correct += 1

acc_single   = single_correct / N_PAIRS
acc_debiased = debiased_correct / N_PAIRS
ax5.bar(['Single-round\npairwise', 'Debiased\n(swap + avg)'],
        [acc_single, acc_debiased], color=['#E53935', '#43A047'], alpha=0.85)
ax5.set_ylabel('Accuracy vs True Winner')
ax5.set_title(f'Debiasing Improves Pairwise Accuracy\n(biased judge, N={N_PAIRS} pairs)')
ax5.set_ylim(0, 1.0)
for i, val in enumerate([acc_single, acc_debiased]):
    ax5.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
# Annotation: swap + average consistently improves accuracy for biased judges

# ── Plot 6: MT-Bench style model ranking ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
mt_models = ['Model A\n(GPT-4 class)', 'Model B\n(Claude class)',
             'Model C\n(Llama3 class)', 'Model D\n(Small model)']
mt_scores = [9.1, 8.7, 7.2, 5.4]
mt_colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#888888']
bars = ax6.barh(mt_models, mt_scores, color=mt_colors, alpha=0.85)
ax6.axvline(7.0, color='red', linestyle='--', alpha=0.5, label='Passing threshold (7.0)')
ax6.set_xlabel('Average Judge Score (1-10)')
ax6.set_title('MT-Bench Style LLM Ranking\n(GPT-4 judge, 80 questions)')
ax6.set_xlim(0, 10)
ax6.legend()
for bar, val in zip(bars, mt_scores):
    ax6.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val}', va='center', fontsize=10, fontweight='bold')
# Annotation: MT-Bench uses 2-turn conversations across 8 categories; GPT-4 as judge

plt.suptitle('LLM-as-a-Judge: Biases, Calibration, and Debiasing', fontsize=13, fontweight='bold')
plt.savefig('/tmp/40_llm_judge.png', dpi=100, bbox_inches='tight')
plt.close()
print("Figure saved: /tmp/40_llm_judge.png")
print("6 panels: calibration scatter, good vs biased, position bias,")
print("          verbosity bias, debiasing accuracy, MT-Bench ranking")
"""))

cells.append(md(r"""## 7. Failure Modes

| Failure | Description | Fix |
|---------|-------------|-----|
| **Position bias** | Judge prefers first response in pairwise | Always swap; take consensus winner |
| **Verbosity bias** | Longer = better regardless of quality | Add explicit rubric note; penalise padding |
| **Self-preference** | GPT-4 prefers GPT-4 outputs | Use a different model as judge |
| **Rubric collapse** | Judge ignores rubric, uses gut feel | Include rubric examples anchored to specific scores |
| **Overconfident scoring** | All scores cluster at 7-8 | Add forced differentiation requirement |
| **Format brittleness** | Judge produces invalid JSON → parse error | Parse with fallback regex; retry on failure |
| **Cost explosion** | 2 API calls per pairwise × 1000 pairs/day | Cache identical prompts; batch API calls |
| **Calibration drift** | Judge accuracy drops after model update | Re-run calibration study quarterly |
"""))

cells.append(md(r"""## 8. Production Library Implementation
"""))

cells.append(code(r"""
# Production option: use OpenAI / Anthropic API as judge

try:
    import anthropic
    print("anthropic SDK available — in production, call claude-sonnet-4-6 as judge")
    print("Example call:")
    print('  client = anthropic.Anthropic(api_key=...)')
    print('  msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=256, ...)')
except ImportError:
    print("anthropic not installed — using MockLLMJudge for all demonstrations")

try:
    import openai
    print("openai SDK available — can use gpt-4o as judge")
except ImportError:
    print("openai not installed")

# Production judge wrapper (drop-in for MockLLMJudge)
class ProductionJudge:
    def __init__(self, provider='anthropic', model='claude-sonnet-4-6'):
        self.provider = provider
        self.model    = model
        self._cost_per_call = 0.01  # rough estimate USD

    def grade_single(self, question, response, **kwargs):
        prompt = SINGLE_ANSWER_PROMPT.replace('{question}', question).replace('{response}', response)
        # In production: call API and parse JSON
        # result = self._call_api(prompt)
        # return json.loads(result)
        print(f"[WOULD CALL {self.model}] prompt length: {len(prompt)} chars")
        return {"score": 7, "reasoning": "placeholder — real judge not called"}

prod_judge = ProductionJudge()
result = prod_judge.grade_single("What is a transformer?", "A transformer is a neural architecture...")
print(f"\nProductionJudge result: {result}")
"""))

cells.append(md(r"""## 9. Realistic Business Case Study

### Evaluating 1000 Customer Support Responses per Day

**Context**: A telecom company uses an LLM to draft responses to customer support tickets.
Quality control team must evaluate output quality before full deployment.
"""))

cells.append(code(r"""
# Cost comparison: human vs LLM judge for 1000 responses/day

RESPONSES_PER_DAY = 1000
DAYS_PER_MONTH = 22  # working days

# Human evaluation costs
HUMAN_EXPERT_RATE     = 30.0   # USD/hour (support QA specialist)
HUMAN_MINS_PER_ITEM   = 2.5    # minutes to read + rate one response
HUMAN_AGREEMENT_LEVEL = 0.88   # Spearman correlation (expert human vs gold)

# Crowdworker costs
CROWD_RATE      = 12.0   # USD/hour (MTurk HIT)
CROWD_MINS      = 1.5
CROWD_AGREEMENT = 0.72   # lower quality

# LLM judge costs
LLM_COST_PER_CALL   = 0.008   # USD (Claude Haiku or GPT-4o-mini)
LLM_CALLS_PER_ITEM  = 2       # single-answer + pairwise (debiased = 2 calls)
LLM_AGREEMENT       = 0.89    # Spearman correlation vs human gold

def daily_eval_cost(n_items, rate_per_hour, mins_per_item, n_rounds=1):
    hours = n_items * mins_per_item / 60 * n_rounds
    return hours * rate_per_hour

expert_daily   = daily_eval_cost(RESPONSES_PER_DAY, HUMAN_EXPERT_RATE, HUMAN_MINS_PER_ITEM)
crowd_daily    = daily_eval_cost(RESPONSES_PER_DAY, CROWD_RATE, CROWD_MINS)
llm_daily      = RESPONSES_PER_DAY * LLM_COST_PER_CALL * LLM_CALLS_PER_ITEM

print("Daily Evaluation Cost Comparison (1,000 responses/day)")
print("=" * 60)
print(f"{'Method':<25} {'Daily Cost':>10} {'Monthly':>10} {'Corr. w/ Gold':>14}")
print("-" * 60)
methods = [
    ('Expert human',   expert_daily,   HUMAN_AGREEMENT_LEVEL),
    ('Crowdworker',    crowd_daily,    CROWD_AGREEMENT),
    ('LLM judge',      llm_daily,      LLM_AGREEMENT),
]
for name, daily, corr in methods:
    monthly = daily * DAYS_PER_MONTH
    print(f"  {name:<23} ${daily:>8,.0f}   ${monthly:>8,.0f}     {corr:.2f}")

print()
expert_monthly = expert_daily * DAYS_PER_MONTH
llm_monthly    = llm_daily * DAYS_PER_MONTH
savings_pct    = (expert_monthly - llm_monthly) / expert_monthly * 100
print(f"LLM judge savings vs. expert human: {savings_pct:.0f}%")
print(f"LLM judge savings vs. crowdworker:  {(crowd_daily*DAYS_PER_MONTH - llm_monthly)/(crowd_daily*DAYS_PER_MONTH)*100:.0f}%")
print()
print("Recommended hybrid: LLM judge for all 1000/day + human spot-check of 50/day")
hybrid_monthly = llm_monthly + daily_eval_cost(50, HUMAN_EXPERT_RATE, HUMAN_MINS_PER_ITEM) * DAYS_PER_MONTH
print(f"Hybrid cost: ${hybrid_monthly:,.0f}/month  (LLM + 5% human QC)")
print(f"vs. full human: ${expert_monthly:,.0f}/month")
print(f"Savings: {(expert_monthly - hybrid_monthly)/expert_monthly:.0%}")
"""))

cells.append(md(r"""## 10. Production Considerations

### MT-Bench and Chatbot Arena

**MT-Bench (2023, Zheng et al.):**
- 80 multi-turn questions across 8 domains (writing, roleplay, extraction, reasoning, math, coding, STEM, humanities)
- GPT-4 as judge with few-shot examples and chain-of-thought rubric
- Key finding: GPT-4 judge agrees with expert human 80-85% of the time
- Key bias: judges inflate scores on math/coding when answers look confident (even if wrong)

**Chatbot Arena (LMSYS, 2023-present):**
- Crowdsourced pairwise preferences from 500,000+ human votes
- Elo-based ranking; added LLM-as-judge for rapid turnaround
- Key finding: LLM judge rankings correlate r=0.93 with crowdsourced Elo
- Used to rank GPT-4, Claude, Llama, Gemini, and 50+ other models

### Production Checklist for LLM-as-Judge Deployment

```
[ ] Calibration study: run judge vs. human on 100+ gold examples → r_s > 0.85
[ ] Position bias test: equal-quality pairs → win rate < 0.55 for first response
[ ] Verbosity bias test: short-quality vs long-mediocre → correct 80%+ of time
[ ] Format robustness: inject malformed responses → parse success rate > 99%
[ ] Cost monitoring: daily API spend tracked with alert thresholds
[ ] Judge model version pinned (model updates can shift scoring distributions)
[ ] Quarterly recalibration: refresh gold examples, rerun bias tests
[ ] Human spot-check: 5% of items reviewed by human weekly
```
"""))

cells.append(md(r"""## 11. Tradeoff Analysis

| Approach | Agreement with Human | Cost/item | Latency | Bias Risk |
|----------|---------------------|-----------|---------|-----------|
| Expert human | ~85% (baseline) | $1.25 | Days | Low |
| Crowdworker | ~72% | $0.30 | Hours | Medium |
| LLM judge (GPT-4o) | ~89% | $0.02 | Seconds | Position + Verbosity |
| LLM judge (Llama-3 local) | ~82% | $0.002 | Seconds | Higher bias risk |
| Debiased LLM judge | ~91% | $0.04 | Seconds | Mostly mitigated |
| Human + LLM hybrid | ~93% | $0.10 | Hours | Very low |

**Key tradeoffs:**
- **Throughput vs accuracy**: LLM judge enables 100× throughput but needs bias mitigation
- **Model capability vs cost**: GPT-4 judge is best but 10× more expensive than Haiku
- **Single-answer vs pairwise**: pairwise is more reliable but 2× cost (4× with swap)
- **Self-judge vs independent**: always use a different model family as judge

**When LLM-as-judge is NOT appropriate:**
- Safety evaluation (legal liability requires human sign-off)
- Novel task types not in judge's training distribution
- When the judge has obvious self-preference for the model under test
"""))

cells.append(md(r"""## 12. Senior-Level Interview Preparation

**Q1: What are the 3 main biases in LLM-as-judge, and how do you mitigate each?**
(1) Position bias: always swap A and B; report winner only if consistent across both orderings.
(2) Verbosity bias: add explicit rubric instruction "ignore response length"; penalise padding.
(3) Self-preference: never use model X to judge model X's outputs; use a different model family.
Meta-mitigation: calibrate judge against human scores before deployment; set r_s > 0.85 threshold.

**Q2: Why does swapping the pairwise order work to remove position bias?**
If judge has position bias β (boosts first response's score), then:
- Order [A,B]: score_A += β, score_B -= β → A wins
- Order [B,A]: score_B += β, score_A -= β → B wins
Averaging the two rounds: net bias contribution = 0 for both responses.
The actual quality difference remains; the positional bias cancels.

**Q3: How would you design a judge prompt for evaluating customer support responses?**
3 elements: (1) Rubric specifying dimensions with weights (accuracy 40%, tone 30%, resolution 20%, conciseness 10%); (2) Anchored examples — show a score-2 response (rude, incorrect), score-5 (polite but incomplete), score-8 (resolves issue, friendly), score-10 (perfect + empathetic); (3) Output format: JSON with integer score 1-10, one-sentence per dimension, overall reasoning.

**Q4: Your LLM judge shows r_s = 0.78 vs human on calibration. What do you do?**
78% is below 0.85 threshold. Diagnose: (a) look at items where judge and human disagree most — is there a pattern? (b) check if judge is miscalibrated for a specific category; (c) add more examples to rubric for that category; (d) consider upgrading judge model; (e) run debiasing (swap for pairwise) — position bias alone can drop r_s by 0.05-0.10.

**Q5: When should you NOT use LLM-as-judge?**
(1) Safety/policy compliance — legal liability; (2) tasks requiring real-world grounding (verifying factual claims about recent events); (3) code execution correctness — run the code instead; (4) when judge was trained on data generated by the model under evaluation (circular); (5) when ground truth is perfectly measurable (exact match, unit tests, math).

**Q6: Explain Chatbot Arena's evaluation methodology.**
Users visit a website, chat with two anonymous LLMs, and choose which one they prefer (or tie). Wins and losses feed an Elo rating system. With 500k+ votes, the Elo ranking is stable and representative. Key advantage over MT-Bench: open-ended user prompts, not a fixed benchmark. Key challenge: expensive (requires many users), slow (weeks to accumulate votes), prompt distribution may not match production.

**Q7: How do you compute win rate and what is AlpacaEval?**
Win rate = wins / (wins + losses + 0.5 × ties). AlpacaEval evaluates 805 instruction-following prompts, using GPT-4 as judge to compare each model against text-davinci-003 baseline. A model's AlpacaEval score = fraction of times GPT-4 prefers it over baseline. Claude Sonnet 3.7 scores ~91%, meaning GPT-4 prefers it on 91% of prompts vs the baseline.

**Q8: Design a hybrid evaluation pipeline for a production LLM product.**
(1) Daily: LLM judge evaluates 100% of outputs (Haiku-class model, fast + cheap) — flags p90 failures; (2) Weekly: human spot-check 50 flagged items — compute human-judge agreement, update calibration; (3) Monthly: full calibration study: 100 gold items, compute r_s, recalibrate rubric; (4) On release: pre-ship debiased pairwise study (new vs old model, 200 comparisons, swap + BT ranking); (5) Alert: if daily LLM judge mean drops >0.5 points week-over-week, escalate to human review.
"""))

cells.append(md(r"""## 13. Teach-Back Section

Explain each of these from memory to a peer who has not read this notebook:

1. **The LLM-as-judge value proposition**: Why does it achieve 85-92% correlation with
   human judgement? What properties of modern LLMs make them suitable as judges?
   Where does the remaining 8-15% disagreement come from?

2. **Judge prompt anatomy**: Walk through the 3 elements of a good judge prompt and
   explain why removing any one element degrades judge reliability.

3. **Position bias mechanics**: Prove mathematically that averaging scores across
   both orderings cancels additive position bias. When does this NOT work?

4. **Verbosity bias vs legitimate length preference**: A response scores 8/10 and is
   400 tokens. Another scores 6/10 and is 80 tokens. Is the longer response rated
   higher due to verbosity bias or legitimate quality? How would you determine which?

5. **MT-Bench methodology**: Describe the MT-Bench setup from memory: number of questions,
   number of domains, type of questions, judge model, and what the score means.

6. **Calibration study design**: Walk through how you'd design a 100-example calibration
   study to validate a new LLM judge before deploying it for production evaluation.

7. **Self-preference bias identification**: You're evaluating outputs from Model A and
   using Model A as judge. Describe 2 experiments to detect and quantify self-preference.

8. **Hybrid evaluation architecture**: Design a cost-effective evaluation pipeline for
   1 million responses per month. Specify: LLM judge model, human review fraction,
   escalation criteria, calibration cadence, and estimated monthly cost.
"""))

cells.append(md(r"""## 14. Exercises

### Beginner
1. Implement `spearman_correlation` from scratch using only Python lists (no numpy).
   Verify: spearman([1,2,3], [1,2,3]) = 1.0 and spearman([1,2,3], [3,2,1]) = -1.0.
2. Write a judge prompt template for evaluating Python code generation. Include all 3
   required elements: rubric (correctness, efficiency, style), anchored examples,
   and JSON output format.
3. Compute the position bias score for a judge that picks the first response 65% of
   the time when both responses are equal quality. Is this considered high bias?

### Intermediate
4. Implement `measure_verbosity_bias` with residual correction: fit a linear model
   predicting score from length on gold examples, then use residuals as bias-corrected scores.
5. Implement a reference-guided judge that computes semantic similarity between the
   response and a gold reference using simple token overlap (Jaccard similarity). Add a
   factual penalty if the response contradicts named entities in the reference.
6. Build a mini MT-Bench evaluation: 10 questions × 3 models. Use the MockLLMJudge
   to score each (model, question) pair. Produce a ranking table with confidence intervals
   (bootstrap 95% CI on mean score, 200 resamples).

### Senior
7. **Multi-judge ensemble**: Implement a 3-judge ensemble (different mock bias profiles).
   Aggregate via majority vote for pairwise, mean for single-answer. Show that ensemble
   r_s vs human is higher than any individual judge. Prove this holds when judge errors
   are uncorrelated; derive when it fails.
8. **Adaptive judge routing**: You have a fast cheap judge (r_s=0.80) and a slow expensive
   judge (r_s=0.92). Route easy items (where fast judge is confident, score >= 8 or <= 3)
   to cheap judge, ambiguous items (4-7) to expensive judge. Simulate 500 items and show
   cost vs accuracy tradeoff curve for different routing thresholds.
9. **Judge calibration with Platt scaling**: The LLM judge outputs scores on 1-10 but
   is systematically over-confident (clustering around 7-8). Implement Platt scaling
   (logistic regression on judge score → calibrated probability of human rating >= 7).
   Show reliability diagram before and after calibration on a held-out validation set.
"""))

build("phase7_evaluation/40_llm_as_a_judge.ipynb", cells)
print("NB40 built.")
