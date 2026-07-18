"""Builder for NLP-03 — LLM Pretraining and Data Pipeline."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells = [
md(r"""# NLP-03 · LLM Pretraining and Data Pipeline
### From a versioned corpus to a domain-adapted checkpoint

The tiny-language-model gate proved next-token training. This lesson asks what happens
before and after that loop: curate a corpus, prevent evaluation contamination, estimate
training work, continue a real checkpoint on domain text, and measure both improvement
and forgetting. **Prerequisites:** NLP-06, FND-04, MLE-02, and PROD-04."""),
md(r"""## 1 · Learning Objectives

- Distinguish base pretraining from continued pretraining.
- Apply explicit provenance, quality, deduplication, and contamination rules.
- Keep tokenizer/checkpoint/data versions compatible.
- calculate next-token loss and a first-order compute budget.
- Measure domain improvement and original-domain retention separately.
- Choose continued pretraining only when data and evidence justify it."""),
md(r"""## 2 · Historical Motivation

Decoder pretraining made one objective reusable across tasks; scaling-law research then
showed that parameters, tokens, and compute must be budgeted together. Domain adaptation
reuses that checkpoint instead of restarting. The practical lesson is not “more text is
always better”: duplicated, unlicensed, poisoned, or evaluation-contaminated data can
make a larger run less trustworthy."""),
md(r"""## 3 · Intuition and Practical Problem

A general model reads ordinary prose but performs poorly on astronomy language. We can
continue next-token training on licensed astronomy text. This is like a trained musician
studying a new genre: prior skill helps, but exclusive practice can weaken the old
repertoire. The analogy stops at mechanism—the model only changes parameters to reduce
loss.

Use continued pretraining for domain language, style, or vocabulary at scale. Avoid it
when facts change often (prefer RAG), when a prompt solves the task, or when you cannot
build domain and retention evaluation sets."""),
md(r"""## 4 · Mathematical Foundations

After reading tokens $x_1,ldots,x_t$, the model predicts $x_{t+1}$. Read the loss aloud:
“average the negative log probability assigned to every true next token.”

$$J=-\frac1{T-1}\sum_{t=1}^{T-1}\log p_\theta(x_{t+1}\mid x_{1:t}).$$

$T$ is sequence length; $t$ is position; $x_{1:t}$ is visible context; $p_\theta$ is
the probability from parameters $\theta$; $J$ is mean loss in nats. If true-token
probabilities are `0.5` and `0.25`, $J=(-\log .5-\log .25)/2=1.040$ nats.

A rough dense-Transformer training estimate is $C\approx6ND$ FLOPs: $N$ parameters,
$D$ training tokens, and factor 6 approximating forward plus backward work. For 1M
parameters and 10M tokens, $C\approx6\times10^{13}$ FLOPs. It is planning arithmetic,
not a hardware-runtime guarantee; architecture and utilization matter."""),
md(r"""## 5 · Manual Implementation from Scratch

The local lab normalizes documents, rejects a short item, removes an exact duplicate,
and blocks an exact evaluation substring. It then trains the existing tiny decoder on
base text, copies its checkpoint, and continues training only the copy on curated domain
text."""),
code(r"""import sys
from pathlib import Path
roots=[Path.cwd(), *Path.cwd().parents]
repo=next(p for p in roots if (p/'projects/language_model_adaptation').exists())
sys.path[:0]=[str(repo/'projects/language_model_adaptation/src'), str(repo/'projects/tiny_language_model/src')]
from language_model_adaptation.lab import run_adaptation_lab
report=run_adaptation_lab(seed=42)
print(report['data_pipeline'])
print(report['continued_pretraining'])"""),
md(r"""## 6 · Visualization

The domain bar should fall; the retention bar may rise. Both are evidence—the second is
not an inconvenient metric to hide."""),
code(r"""import matplotlib.pyplot as plt
m=report['continued_pretraining']
labels=['domain','base retention']
before=[m['domain_loss_before'],m['base_retention_loss_before']]
after=[m['domain_loss_after'],m['base_retention_loss_after']]
x=range(2)
plt.bar([i-.18 for i in x],before,.36,label='before')
plt.bar([i+.18 for i in x],after,.36,label='after')
plt.xticks(list(x),labels); plt.ylabel('held-out token loss'); plt.legend(); plt.show()"""),
md(r"""## 7 · Failure Modes and Common Mistakes

| Symptom | Cause | Evidence | Repair |
|---|---|---|---|
| suspicious validation gain | contamination | overlap report | rebuild split before training |
| domain improves, base worsens | forgetting | retention loss | mix replay data or reduce updates |
| checkpoint will not load | tokenizer/config mismatch | version manifest | restore exact paired artifacts |
| loss falls only on duplicates | memorization | duplicate clusters | deduplicate before split |

Beginners often split after windowing, fit a new tokenizer that changes IDs, select on
the final test set, or report domain loss without retention."""),
md(r"""## 8 · Library or Tool Implementation

Production pipelines may use dataset manifests, MinHash, distributed trainers, bf16,
FSDP, and checkpoint sharding. Those tools scale the same contract; they do not replace
provenance, split integrity, or a local baseline. Pin revisions and keep the core lesson
offline."""),
md(r"""## 9 · Realistic Case Study

A legal team considers adaptation on licensed contracts. It first blocks benchmark and
client-test overlap, compares exact and near-duplicate clusters, holds out contract types,
and measures general-language retention. If the need is fresh case retrieval rather than
language adaptation, it chooses RAG instead."""),
md(r"""## 10 · Learning and Production Considerations

Record corpus hashes, licenses, filters, mixture weights, tokenizer, model config,
optimizer, precision, seeds, tokens, FLOPs, and checkpoints. Exact overlap detection is
only a floor; paraphrase and benchmark contamination need stronger audits."""),
md(r"""## 11 · Tradeoff Analysis

| Method | Solves | Strength | Risk |
|---|---|---|---|
| prompt | task framing | cheapest | limited behavior change |
| RAG | changing knowledge | source updates | retrieval dependency |
| continued pretraining | domain language | broad domain exposure | forgetting and compute |
| train from scratch | new base model | full control | enormous data/compute need |"""),
md(r"""## 12 · Readiness Check

Explain why the experiment's domain loss `4.043→1.758` is good evidence and its base
loss `1.853→3.189` is a warning. A strong answer proposes a replay-mixture experiment,
not deletion of the retention metric."""),
md(r"""## 13 · Teach-Back

1. Why curate before splitting and training?
2. What does exact deduplication miss?
3. Why keep the tokenizer fixed?
4. What does $6ND$ estimate and omit?
5. When should RAG replace continued pretraining?"""),
md(r"""## 14 · Exercises, Self-Check, and Solutions

**Worked (10 min):** calculate $6ND$ for `N=2M`, `D=5M`: `6×10^13` FLOPs.
**Guided (20 min):** add one duplicate; hint: normalize before hashing. Expected:
duplicate count rises by one. **Independent (45 min):** mix 20% base replay and report
both losses. Pass if domain still improves and retention damage shrinks. **Challenge
(60 min):** add paraphrase-contamination detection and document false positives.

Summary: continued pretraining adapts language distribution, but every domain gain must
be paired with split integrity and retention evidence. **Memory aid:** *Curate, version,
continue, then measure both the new skill and the old one.*"""),
]
build('05_nlp_and_llms/03_llm_pretraining_and_data_pipeline.ipynb', cells)
