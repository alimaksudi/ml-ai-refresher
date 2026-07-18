"""Builder for NLP-08 — Preference Learning and Alignment."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells=[
md(r"""# NLP-08 · Preference Learning and Alignment
### From demonstrations to chosen-versus-rejected behavior

SFT shows one desired response. Preference data compares two responses to the same
prompt. This lesson trains real DPO on the tiny SFT checkpoint, explains reward models,
and keeps PPO conceptual and correctly labeled. **Prerequisites:** NLP-07 and FND-02."""),
md(r"""## 1 · Learning Objectives
- Build and audit `(prompt, chosen, rejected)` triples.
- Calculate Bradley–Terry and DPO losses.
- Sum response-token log probabilities conditioned on the prompt.
- Explain the frozen reference policy and beta.
- Distinguish KL-regularized RL from PPO's clipped surrogate.
- Measure preference gain and SFT retention without claiming safety."""),
md(r"""## 2 · Historical Motivation
RLHF commonly trains a preference reward model and optimizes a policy while limiting
drift. PPO is one possible optimizer. DPO derives an offline classification loss for a
specific KL-regularized preference model, removing online rollouts and a separately
deployed reward model but retaining strong data and evaluation assumptions."""),
md(r"""## 3 · Intuition and Practical Problem
Two candidate answers receive a comparison, like a reviewer choosing the better draft.
The reference policy is an anchor: DPO asks whether the new policy improves the chosen
response *relative to how the reference scored both responses*. The analogy stops at
truth—reviewer preference can be inconsistent, biased, or wrong.

Use preference optimization only with meaningful comparisons after a strong SFT
baseline. Avoid it when supervised corrections suffice or preference labels lack
agreement and coverage."""),
md(r"""## 4 · Mathematical Foundations
For reward scores $r_w,r_l$, Bradley–Terry loss is
$J_{RM}=-\log\sigma(r_w-r_l)$. $\sigma(z)=1/(1+e^{-z})$. If the margin is `1`, loss is
`-log(0.731)=0.313`; if it is `-1`, loss is `1.313`.

DPO uses policy $\pi_\theta$, frozen reference $\pi_{ref}$, chosen $y_w$, rejected
$y_l$, prompt $x$, and scale $\beta>0$:

$$J_{DPO}=-\log\sigma\{\beta[(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x))
-(\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))]\}.$$

Sequence log probability sums response-token log probabilities only. At zero relative
margin, loss is `log(2)=0.693`. For margin $m$, gradient magnitude is
$\sigma(-m)$: it approaches `1` for a very negative margin and `0` for a very positive
one. Length can confound summed log probabilities and must be audited.

The high-level RL objective $E[r]-\beta KL(\pi||\pi_{ref})$ is **not itself PPO**.
PPO additionally uses sampled rollouts, advantages, policy ratios, and clipping."""),
md(r"""## 5 · Manual Implementation from Scratch
The project starts policy and reference from identical full-SFT weights, freezes the
reference, computes chosen/rejected response sequence log probabilities, and updates
only the policy with DPO."""),
code(r"""import sys
from pathlib import Path
repo=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'projects/language_model_adaptation').exists())
sys.path[:0]=[str(repo/'projects/language_model_adaptation/src'),str(repo/'projects/tiny_language_model/src')]
from language_model_adaptation.lab import run_adaptation_lab
report=run_adaptation_lab(seed=42)
print(report['preference_alignment'])"""),
md(r"""## 6 · Visualization"""),
code(r"""import numpy as np, matplotlib.pyplot as plt
m=np.linspace(-5,5,200); loss=np.logaddexp(0,-m); grad=1/(1+np.exp(m))
plt.plot(m,loss,label='DPO logistic loss'); plt.plot(m,grad,label='gradient magnitude')
plt.xlabel('relative preference margin'); plt.legend(); plt.show()"""),
md(r"""## 7 · Failure Modes and Common Mistakes
| Symptom | Cause | Evidence | Repair |
|---|---|---|---|
| longer answers always lose | summed-length confound | length slice | balance/audit lengths |
| labels contradict | low agreement | annotator matrix | adjudicate or model uncertainty |
| preference rises, task falls | alignment tax | retention suite | tune beta/data/early stop |
| reward rises, behavior exploits gaps | reward hacking | adversarial human review | improve data/model and constrain rollout |

Never call preference accuracy truth, call the KL objective PPO, train the reference,
or include prompt tokens in response log probability."""),
md(r"""## 8 · Library or Tool Implementation
DPO trainers can manage reference models, masking, padding, and distributed execution.
Verify chat-template boundaries, chosen/rejected lengths, reference identity, beta,
gradient accumulation, and evaluation code against the scratch loss before scaling."""),
md(r"""## 9 · Realistic Case Study
A medical assistant prefers responses that acknowledge uncertainty and request source
verification. Clinicians label pairs with an abstain option. The team measures agreement,
factual correctness, safety, length, specialties, and SFT retention; preference alone
cannot authorize deployment."""),
md(r"""## 10 · Learning and Production Considerations
Preference collection needs a rubric, randomized response order, annotator metadata,
privacy controls, disagreement handling, and held-out prompts. Monitor both the optimized
signal and independent outcomes that the signal might fail to represent."""),
md(r"""## 11 · Tradeoff Analysis
| Method | Data | Strength | Weakness |
|---|---|---|---|
| SFT | target responses | simple baseline | no pairwise tradeoff signal |
| reward model | preference pairs | reusable score | exploitable proxy |
| PPO-style RLHF | reward + rollouts | online optimization | many moving parts |
| DPO | offline pairs | simpler executed loop | reference/data sensitivity |"""),
md(r"""## 12 · Readiness Check
The local DPO loss falls `0.693→near 0`, held-out preference accuracy changes `0→1`,
and SFT retention loss worsens. This proves the objective changed a narrow preference,
not that the model became aligned, safe, or truthful."""),
md(r"""## 13 · Teach-Back
1. What is one preference example?
2. Why freeze the reference?
3. Which tokens form sequence log probability?
4. Why is negative-margin gradient large?
5. Why does preference accuracy not prove safety?"""),
md(r"""## 14 · Exercises, Self-Check, and Solutions
**Worked (10 min):** calculate logistic loss at margins `1` and `-1`: about `0.313`
and `1.313`. **Guided (20 min):** reverse one label; inspect its margin. **Independent
(45 min):** sweep beta and report preference plus retention. **Challenge (60 min):**
add annotator disagreement and a rule for ties/abstentions.

Summary: preference optimization changes relative response likelihood under a labeled
proxy and a reference constraint. **Memory aid:** *Optimize the comparison, anchor to
the reference, and audit everything the preference label leaves out.*"""),]
build('05_nlp_and_llms/08_preference_learning_and_alignment.ipynb',cells)
