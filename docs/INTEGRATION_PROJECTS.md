# Curriculum Integration Projects

Projects sit at curriculum checkpoints. Individual lessons use small independent
applications; these projects require several concepts to work together without
creating repetitive toy projects.

## A. Data readiness audit — after PRE-01 through FND-03

- **Goal:** turn a messy measurement table into documented, trustworthy data.
- **Dataset columns:** entity ID, timestamp, group, two numeric measurements, one
  category, target, unit, and one intentionally leaked future field.
- **Workflow:** validate schema/units, inspect missingness and duplicates, define the
  prediction unit/time, remove leakage, choose a split, and produce a data card.
- **Expected output:** cleaned table, validation report, split manifest, three plots,
  and an explanation of every excluded column.
- **Evaluation:** no entity/time leakage, reproducible counts, valid joins, and a
  correct teach-back of EDA versus model evaluation.

## B. First valid prediction experiment — after CML-01 through MLE-02

- **Goal:** compare naive, linear-regression, and logistic-regression baselines.
- **Dataset columns:** five numeric features, one category, timestamp, continuous
  target, and binary decision target.
- **Workflow:** frame decisions, split first, fit pipelines, calculate task-aligned
  metrics, choose a validation threshold from costs, and test once.
- **Expected output:** experiment table, confusion matrix, residual plot, uncertainty
  statement, and decision memo.
- **Evaluation:** baseline comparison, no test selection, correct loss/metric
  distinction, and limitations in plain language.

## C. Classical ML vertical slice — after CML-03 through MLE-06 and EVAL-01

- **Goal:** determine whether ensembles earn complexity over linear models and explore
  unlabeled structure without presenting clusters as truth.
- **Dataset:** existing wine-classifier data and documented feature schema.
- **Workflow:** compare linear/tree/forest/boosting pipelines, tune within CV, evaluate
  imbalance/explanations, and run PCA/clustering separately.
- **Expected output:** tested artifact, model card, experiment report, explanation
  stability caveat, and cluster/anomaly limitations.
- **Evaluation:** wine mastery checkpoint plus reproducible unsupervised diagnostics.

## D. Neural representation study — after DL-01 through NLP-01

- **Goal:** compare majority, logistic, MLP, and CNN models fairly.
- **Dataset columns:** sklearn digits' 64 pixels and one digit label.
- **Workflow:** fixed stratified split, stable training, validation checkpoint
  selection, dropout ablation, multi-seed comparison, and final test evaluation.
- **Expected output:** learning curves, saved weights, metrics, shape trace, and
  complexity recommendation.
- **Evaluation:** digit mastery checkpoint; the neural model is not required to win.

## E1. Tiny language model — after NLP-01 and DL-06 through DL-08

- **Goal:** train and evaluate a decoder-only character language model without an API
  key, then explain why every component exists.
- **Dataset columns:** the bundled data is a raw character stream; generated training
  examples contain input token IDs, next-token targets, split, and window position.
- **Workflow:** contiguous text split, character and BPE tokenizers fit on training text,
  shifted windows, bigram baseline, one-batch overfit test, controlled tokenizer
  comparison using bits per character, causal Transformer training, validation
  checkpoint selection, decoding comparison, cached-versus-naive inference equivalence,
  KV-cache latency measurement, and one architecture ablation.
- **Expected output:** learning curves, dataset hash, model configuration, parameter
  count, token compression, context coverage, baseline and model validation loss,
  within-tokenizer perplexity, cross-tokenizer bits per character, saved weights,
  tokenizer merges, fixed-seed samples, and a limitation statement.
- **Inference evidence:** maximum cached-logit difference, identical greedy output,
  per-layer cache shapes, estimated cache bytes, median latency, tokens per second, and
  speedup across several prompt lengths.
- **Evaluation:** no window may cross the split; future tokens must not affect earlier
  logits; the student must report a losing comparison honestly. Use
  `projects/tiny_language_model/MASTERY_CHECKPOINT.md`.

## E2. Transformer model families — after DL-08 and NLP-06

- **Goal:** prove how attention visibility and training objectives distinguish a
  causal decoder, bidirectional encoder, and encoder-decoder.
- **Dataset columns:** sequence ID, source tokens, corrupted/masked tokens, target
  tokens, padding mask, objective, and split or diagnostic role.
- **Workflow:** shared components, mask-invariant tests, next-token training,
  masked-token training, classification, source-to-target training, parameter
  comparison, one broken-mask diagnosis, and architecture selection.
- **Expected output:** initial/final losses, diagnostic accuracies, attention shapes,
  parameter counts, saved weights, limitation statement, and task-choice rationale.
- **Evaluation:** synthetic accuracy is learnability evidence only; passing requires
  behavioral invariants and the human mastery checkpoint.

## E3. Text representation study — after DL-06 through NLP-03

- **Goal:** compare TF-IDF, static embeddings, recurrence, and a small Transformer
  representation on one classification or similarity task.
- **Dataset columns:** text ID, raw text, label or pair relevance, and split.
- **Workflow:** tokenize, establish lexical baseline, train/evaluate under equal
  splits, inspect OOV/length slices, and document compute.
- **Expected output:** baseline table, slice analysis, embedding visualization, and
  recommendation tied to quality and latency.
- **Evaluation:** no vocabulary leakage, correct padding/masking, measured benefit.

## F. Prompt evaluation lab — after EVAL-02 and NLP-04 through NLP-05

- **Goal:** treat prompt changes as versioned experiments.
- **Dataset columns:** case ID, input, expected schema, rubric/reference, safety slice,
  and prompt version.
- **Workflow:** fix decoding, validate schema, compare prompt versions, inspect
  stochastic variation, and classify failures.
- **Expected output:** versioned prompt, eval set, validity/task/safety report, and
  regression examples.
- **Evaluation:** use a declared local model or fixed versioned outputs; synthetic
  accuracy may test the harness but cannot rank prompt methods. No selection on the
  final holdout and no unsupported “better” claim. Report correctness, grounding,
  citation validity, refusal, and safety separately when they apply.

## G. Measured RAG foundation — after RAG-01 through RAG-03

- **Goal:** determine which chunking and retriever baseline is justified.
- **Dataset columns:** document, section, query, slice, and gold evidence IDs in
  `projects/rag_foundations/data`.
- **Workflow:** run lexical, LSA, and hybrid retrieval over three chunkers; calculate
  component metrics; diagnose failed and unanswerable queries.
- **Expected output:** hashed report, evidence traces, slice comparison, and the
  simplest-supported-system recommendation.
- **Evaluation:** RAG foundation mastery checkpoint.

**RAG-06 extension:** use `hybrid_corpus.json` and `hybrid_queries.json` to prove one
exact-identifier ranking win for BM25, one paraphrase ranking win for dense LSA, one
mixed-intent recall improvement from RRF, one no-gain control, and one correct
abstention. Preserve identical hashes, chunks, labels, candidate depth, and `top_k`
across every retriever comparison.

## H. Evidence-grounded assistant — after RAG-05 through RAG-08 and EVAL-03

- **Goal:** add vector indexing, reranking, and generation only when each component
  improves a declared failure.
- **Dataset columns:** query ID, evidence IDs, answer, citations, abstention, latency,
  token/cost estimate, and evaluator labels.
- **Workflow:** preserve baseline, add one component at a time, separate retrieval and
  generation errors, test injection/unanswerable cases, and enforce citations.
- **Expected output:** ablation report, grounded assistant, failure taxonomy, cost budget.
- **Evaluation:** recall, support/correctness, abstention, latency, security, and no
  component without measurable value.

**EVAL-03 gate:** compare lexical, dense, and hybrid systems under identical corpus,
query, answer, chunking, metric, and threshold versions. Required-term correctness and
extractive support remain named proxies. Reranking is justified only when relevant
evidence already appears in the candidate set but ranking or answer selection fails.

## I. Workflow versus agent challenge — after AGT-01 through AGT-05 and EVAL-04/EVAL-05

- **Goal:** solve one task with deterministic code, one agent, and optionally multiple agents.
- **Dataset columns:** task ID, expected result, tool trace, permissions, steps,
  retries, outcome, latency, and cost.
- **Workflow:** establish deterministic baseline, add typed tools, evaluate traces,
  add memory/reflection only for demonstrated failures, and compare complexity.
- **Expected output:** system comparison, permission model, failure traces, and a
  simplest-adequate-design recommendation.
- **Evaluation:** success, unsafe action rate, cost, latency, and reproducibility.

## J. Production reliability drill — after PROD-01 through SYS-04

- **Goal:** operate one classical, neural, or RAG project under controlled failures.
- **Dataset columns:** trace ID, model/data version, prediction, confidence, latency,
  slice, delayed label, alert, and response action.
- **Workflow:** package/deploy, define SLOs, inject schema drift/dependency failure/
  stale index/bad release, investigate, degrade safely, and roll back.
- **Expected output:** tested service, dashboards, runbook, incident timeline,
  postmortem, cost model, threat model, and architecture decision record.
- **Evaluation:** detection time, false alarms, rollback, quality SLOs, security/privacy,
  and evidence-backed retraining decisions.
