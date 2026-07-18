# Student Mastery Path and Remediation Guide

This is the learner-facing route. Semantic lesson IDs are stable identifiers; follow
the canonical route card generated near the top of each notebook.

## How to study one module

Use two passes rather than attempting every senior detail at once.

### Core pass — required

1. Read objectives and intuition.
2. Work through the mathematical foundation with a tiny numeric example.
3. Predict code output before running it.
4. Run and modify the implementation.
5. Reproduce one failure mode.
6. Complete the Required Core Mastery Gate.
7. Continue only at 8/10 with successful teach-back.

### Extension pass — revisit after the section gate

- detailed history;
- complete derivations beyond the core formula;
- production architecture;
- senior interview material;
- long challenges in Section 14.

Trying to master both passes simultaneously is optional and will substantially
increase the stated workload.

## Gate A — Mathematical language and coding readiness

After PRE-01 through PRE-05, without notes:

- evaluate an expression while respecting operation order;
- rearrange a one-variable equation;
- read a graph as an input-output relationship;
- explain derivative and probability intuition;
- trace NumPy matrix shapes;
- write and test a small function;
- load, validate, filter, group, and safely join a DataFrame;
- use a traceback to locate an error.

If weak in notation or algebra, repeat PRE-01 and PRE-02. If code is the blocker,
repeat PRE-03 and PRE-05 using a new five-row dataset.

## Gate B — Mathematical ML readiness

After FND-01 and FND-02:

- distinguish scalar, vector, matrix, and their shapes;
- compute a dot product and explain its meaning;
- distinguish probability, likelihood, expectation, and sample statistic;
- explain sampling variation and why a point estimate is incomplete.

Do not proceed if matrix multiplication is still memorized as a rule with no shape
meaning, or if conditional probability is confused with intersection.

## Gate C — First valid ML experiment

After FND-03, CML-01, FND-04, CML-02, MLE-01, and MLE-02:

- define prediction unit, target, prediction time, and naive baseline;
- split before fitting learned transformations;
- distinguish loss from metric;
- explain a gradient update using squared loss;
- fit regression and classification baselines;
- choose a metric and threshold from the decision cost;
- explain why the test set is not a tuning tool.

Remediate data-contract problems in FND-03, optimization problems in CML-01/FND-04,
and evaluation problems in MLE-01/MLE-02. Do not compensate with a more complex model.

## Gate D — Classical ML mastery

Complete the wine-classifier checkpoint. A passing score requires valid evidence,
not merely accuracy. Then revisit optional derivations and senior extensions from
CML-01 through MLE-05 and complete MLE-06.

## Gate E — Deep Learning mastery

Complete DL-01 through DL-05 and the digit checkpoint. You must be able to trace
tensor shapes, write train/eval loops, gradient-check, diagnose learning curves, and
justify why a neural model does or does not beat a simpler baseline.

## Gate F — Language Model foundations

Complete NLP-01 and DL-06 through DL-08, then complete the
`projects/tiny_language_model` checkpoint. You must be able to trace the full path
from raw text to token IDs, shifted targets, `B × T × C` hidden states, causal
attention, next-token logits, cross-entropy, backpropagation, validation, checkpoint,
and generation without using a hosted API.

Passing requires a one-batch overfit diagnostic, a controlled architecture ablation,
comparison with a bigram baseline, a character-versus-BPE experiment using bits per
character, cached-versus-naive inference equivalence, a measured KV-cache benchmark,
and teach-back of why next-token likelihood is not the same as factual truth. Complete
this gate before NLP-06, NLP-02, NLP-03, prompting,
hallucination mitigation, or RAG. If it does not pass, remediate the smallest failed
dependency: tensor shapes in DL-01, optimization in DL-04, sequence targets in DL-06,
attention and masking in DL-07, or the complete decoder in DL-08.

## Gate G — Transformer model-family mastery

Complete NLP-06 and `projects/transformer_families`. You must prove the behavioral
difference between causal, bidirectional, padding, and cross-attention masks; train the
GPT, BERT, and T5 diagnostic objectives; and trace every attention score as
`B × H × T_query × T_key`.

Passing requires all mask-invariant tests, real loss reduction, an intentionally broken
mask and repair, and at least 17/20 on the teach-back. Complete this gate before sentence
embeddings, the LLM adaptation pipeline, prompting, hallucination mitigation, or RAG.

## Gate H — Sentence embedding mastery

Complete NLP-02 and `projects/sentence_embeddings`. You must trace raw text through
tokenization, bidirectional encoding, padding-aware pooling, L2 normalization, the
`B × B` similarity matrix, MNR loss, and held-out retrieval without a hosted API.

Passing requires padding and normalization invariants, zero exact split overlap, real
loss reduction, better held-out MRR than the untrained encoder, comparison with TF-IDF,
an inspected hard-negative/false-negative example, and at least 17/20 on the teach-back.
Complete this gate before similarity search, chunking, vector databases, or RAG.

## Gate I — Pretraining and data-pipeline mastery

Complete NLP-03 and the pretraining checkpoint in `projects/language_model_adaptation`.
Passing requires curation counts, zero known exact contamination, real domain-loss
improvement, an honestly reported retention regression, and at least 17/20 teach-back.

## Gate J — Instruction tuning and LoRA mastery

Complete NLP-07 and the SFT/LoRA checkpoint. Passing requires shifted response labels,
prompt/padding masking, zero initial LoRA delta, real loss reduction, parameter-state
comparison, held-out evidence, and at least 17/20 teach-back.

## Gate K — Preference alignment mastery

Complete NLP-08 and the alignment checkpoint. Passing requires response-only sequence
log probabilities, a frozen reference, real DPO loss reduction, held-out preference
improvement, retention evidence, correct PPO/DPO distinction, and at least 17/20.

## Gate L — LLM evaluation mastery

Complete EVAL-02 and `projects/language_model_adaptation/EVALUATION_CHECKPOINT.md`.
Passing requires a predeclared evaluation contract, untouched paired examples,
correct manual lexical metrics, real local model evidence, uncertainty plus coverage
limits, slice inspection, and a regression gate that enforces retention guardrails.
Complete this gate before changing prompts in NLP-04.

## Gate M — Controlled prompt experimentation mastery

Complete NLP-04 and `projects/prompt_evaluation/MASTERY_CHECKPOINT.md`. Passing
requires prompt and data hashes, a frozen local model/tokenizer/decoding contract,
development-only challenger selection, one-time paired final testing, separate schema
and correctness metrics, robustness evidence, and an honest release decision. Complete
this gate before NLP-05 threat modelling and guardrails.

## Gate N — Guardrail and trust-boundary mastery

Complete NLP-05 and `projects/guardrail_evaluation/MASTERY_CHECKPOINT.md`. Passing
requires an explicit four-action policy, PII redaction trace, action confusion matrix,
critical-escape and over-restriction analysis, independent schema/citation/support/truth
checks, authorization outside the model, and a residual-risk decision. Complete this
gate before treating retrieved evidence as model context.

## Retention schedule

- Next day: repeat teach-back without notes.
- One week: redo the independent task with different data.
- End of section: complete the cumulative gate before reviewing solutions.
- One month: diagnose one intentionally broken experiment from scratch.

Record attempts, scores, misconception, remediation module, and retry date. Page
completion is not mastery evidence.

Use [`INTEGRATION_PROJECTS.md`](INTEGRATION_PROJECTS.md) after each corresponding
gate. Lesson exercises build one skill; integration projects test whether skills work
together.
