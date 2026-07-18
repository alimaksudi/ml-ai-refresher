"""Builder for NLP-07 — Instruction Tuning and LoRA."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells=[
md(r"""# NLP-07 · Instruction Tuning and LoRA
### From next-token completion to response-supervised behavior

NLP-03 produced a domain-adapted checkpoint. Now we format instruction-response pairs,
mask prompt labels, perform real SFT, and compare full parameter updates with rank-4
LoRA adapters. **Prerequisites:** NLP-03, DL-03, and DL-04."""),
md(r"""## 1 · Learning Objectives
- Construct shifted inputs and response-only labels.
- Explain how prompts condition responses without direct prompt loss.
- Train full-SFT and LoRA candidates from the same checkpoint.
- Trace LoRA shapes, initialization, trainable parameters, merging, and limitations.
- Compare training, held-out, and retention evidence before selecting an adapter."""),
md(r"""## 2 · Historical Motivation
Raw pretrained models continue text; supervised instruction tuning demonstrates the
desired interaction. Full tuning updates every parameter and optimizer state. LoRA
instead learns low-rank corrections to selected linear projections, making adaptation
state smaller—but not guaranteeing equal quality or preserved behavior."""),
md(r"""## 3 · Intuition and Practical Problem
A prompt is the question sheet; the response is the portion graded directly. The model
must read the question, but we need not reward it for reproducing the question. LoRA is
a removable correction layer over a frozen base. This analogy does not imply adapters
store isolated human-readable skills.

Use SFT for stable formats and demonstrated behavior. Try prompting first for small
changes and RAG for changing facts. Avoid tuning without representative held-out and
retention sets."""),
md(r"""## 4 · Mathematical Foundations
Read aloud: “average negative log probability only over response positions.”

$$J_{SFT}=-\frac{1}{\sum_t m_t}\sum_t m_t\log p_\theta(x_{t+1}\mid x_{1:t}).$$

$m_t\in\{0,1\}$ is the response-label mask; all other symbols match NLP-03. With losses
`[2.0,1.0,0.4,0.2]` and mask `[0,0,1,1]`, $J=(0.4+0.2)/2=0.3$.

For PyTorch's column-vector convention, a frozen linear layer becomes
$h=W_0x+(\alpha/r)BAx$. $W_0\in\mathbb R^{d_{out}\times d_{in}}$ is frozen,
$A\in\mathbb R^{r\times d_{in}}$, $B\in\mathbb R^{d_{out}\times r}$, rank $r$ is
small, and $\alpha/r$ scales the update. Initializing $B=0$ makes the initial correction
exactly zero. LoRA stores $r(d_{in}+d_{out})$ trainable weights instead of
$d_{in}d_{out}$ for one projection."""),
md(r"""## 5 · Manual Implementation from Scratch
The project builds each `user/assistant` sequence, shifts targets one character, assigns
`-100` to prompt and padding labels, and uses PyTorch cross-entropy's ignore index.
Full and LoRA models start from identical continued-pretraining weights."""),
code(r"""import sys
from pathlib import Path
repo=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'projects/language_model_adaptation').exists())
sys.path[:0]=[str(repo/'projects/language_model_adaptation/src'),str(repo/'projects/tiny_language_model/src')]
from language_model_adaptation.lab import run_adaptation_lab
report=run_adaptation_lab(seed=42)
print(report['instruction_tuning'])"""),
md(r"""## 6 · Visualization"""),
code(r"""import matplotlib.pyplot as plt
t=report['instruction_tuning']; names=['full','lora']
plt.bar([0,1],[t[n]['train_loss_after'] for n in names],label='train')
plt.bar([.3,1.3],[t[n]['held_out_loss'] for n in names],width=.3,label='held out')
plt.xticks([.15,1.15],names); plt.ylabel('response-token loss'); plt.legend(); plt.show()"""),
md(r"""## 7 · Failure Modes and Common Mistakes
| Symptom | Cause | Check | Repair |
|---|---|---|---|
| prompt dominates loss | labels not masked | supervised-token count | set prompt labels to `-100` |
| adapter changes output at start | nonzero B | zero-delta test | initialize B to zero |
| tiny train loss, weak held-out | memorization | validation loss | more diverse data/regularization |
| base behavior regresses | combined adapter changes outputs | retention suite | lower capacity/LR or replay |

Do not claim LoRA “prevents forgetting,” compare models with different starting
checkpoints, or treat parameter percentage as a quality metric."""),
md(r"""## 8 · Library or Tool Implementation
PEFT libraries automate adapter insertion and saving. Verify target-module names,
fan-in/fan-out convention, bias policy, precision, and merged/unmerged equivalence.
The scratch implementation remains the behavioral oracle."""),
md(r"""## 9 · Realistic Case Study
A support model needs a strict response schema. The team first tests structured prompts,
then SFTs reviewed examples. It compares full and LoRA tuning on format validity, task
correctness, old-task retention, latency, and state size—not training loss alone."""),
md(r"""## 10 · Learning and Production Considerations
Chat templates and special tokens are part of the model contract. Pack examples without
allowing one response to attend across unintended boundaries. Version base weights,
adapter, tokenizer, template, split, and evaluation suite together."""),
md(r"""## 11 · Tradeoff Analysis
| Approach | Trainable state | Strength | Weakness |
|---|---:|---|---|
| prompt | none | fastest baseline | limited adaptation |
| LoRA | small | portable and cheaper | rank/targets need evaluation |
| full SFT | all | maximum update capacity | memory and regression risk |
| RAG | no weight update | fresh sourced facts | retrieval complexity |"""),
md(r"""## 12 · Readiness Check
The full model reaches lower train loss, while LoRA uses `768` versus `16,928` trainable
parameters and has better held-out loss in this tiny run. That does not prove LoRA is
universally better; it exposes full-SFT overfitting on four examples."""),
md(r"""## 13 · Teach-Back
1. Which target positions receive loss?
2. Why do response gradients still depend on prompt states?
3. Why is B initialized to zero?
4. What does rank control?
5. What evidence would justify full SFT over LoRA?"""),
md(r"""## 14 · Exercises, Self-Check, and Solutions
**Worked (10 min):** apply mask `[0,0,1,1]` to four losses; answer `0.3`.
**Guided (20 min):** print supervised character counts; hint: count labels not equal to
`-100`. **Independent (45 min):** compare ranks 1, 4, 8 on held-out and retention loss.
**Challenge (60 min):** verify merged and unmerged LoRA logits match within `1e-6`.

Summary: SFT teaches from response labels; LoRA changes how many parameters move, not
what evidence is required. **Memory aid:** *Mask the prompt, train the response, and
measure the adapter like any other model.*"""),]
build('05_nlp_and_llms/07_instruction_tuning_and_lora.ipynb',cells)
