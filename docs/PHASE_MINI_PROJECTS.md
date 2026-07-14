# Phase Integration Mini-Projects

Mini-projects sit at phase boundaries. Individual lessons use small independent
applications; these projects require several concepts to work together without
creating 61 repetitive toy projects.

## A. Data readiness audit — after 00A–03A

- **Goal:** turn a messy measurement table into documented, trustworthy data.
- **Dataset columns:** entity ID, timestamp, group, two numeric measurements, one
  category, target, unit, and one intentionally leaked future field.
- **Workflow:** validate schema/units, inspect missingness and duplicates, define the
  prediction unit/time, remove leakage, choose a split, and produce a data card.
- **Expected output:** cleaned table, validation report, split manifest, three plots,
  and an explanation of every excluded column.
- **Evaluation:** no entity/time leakage, reproducible counts, valid joins, and a
  correct teach-back of EDA versus model evaluation.

## B. First valid prediction experiment — after 04–10

- **Goal:** compare naive, linear-regression, and logistic-regression baselines.
- **Dataset columns:** five numeric features, one category, timestamp, continuous
  target, and binary decision target.
- **Workflow:** frame decisions, split first, fit pipelines, calculate task-aligned
  metrics, choose a validation threshold from costs, and test once.
- **Expected output:** experiment table, confusion matrix, residual plot, uncertainty
  statement, and decision memo.
- **Evaluation:** baseline comparison, no test selection, correct loss/metric
  distinction, and limitations in plain language.

## C. Classical ML vertical slice — after 06–13B and 36

- **Goal:** determine whether ensembles earn complexity over linear models and explore
  unlabeled structure without presenting clusters as truth.
- **Dataset:** existing wine-classifier data and documented feature schema.
- **Workflow:** compare linear/tree/forest/boosting pipelines, tune within CV, evaluate
  imbalance/explanations, and run PCA/clustering separately.
- **Expected output:** tested artifact, model card, experiment report, explanation
  stability caveat, and cluster/anomaly limitations.
- **Evaluation:** wine mastery checkpoint plus reproducible unsupervised diagnostics.

## D. Neural representation study — after 13A–20

- **Goal:** compare majority, logistic, MLP, and CNN models fairly.
- **Dataset columns:** sklearn digits' 64 pixels and one digit label.
- **Workflow:** fixed stratified split, stable training, validation checkpoint
  selection, dropout ablation, multi-seed comparison, and final test evaluation.
- **Expected output:** learning curves, saved weights, metrics, shape trace, and
  complexity recommendation.
- **Evaluation:** digit mastery checkpoint; the neural model is not required to win.

## E. Text representation study — after 17–22

- **Goal:** compare TF-IDF, static embeddings, recurrence, and a small Transformer
  representation on one classification or similarity task.
- **Dataset columns:** text ID, raw text, label or pair relevance, and split.
- **Workflow:** tokenize, establish lexical baseline, train/evaluate under equal
  splits, inspect OOV/length slices, and document compute.
- **Expected output:** baseline table, slice analysis, embedding visualization, and
  recommendation tied to quality and latency.
- **Evaluation:** no vocabulary leakage, correct padding/masking, measured benefit.

## F. Prompt evaluation lab — after 38 and 23–24

- **Goal:** treat prompt changes as versioned experiments.
- **Dataset columns:** case ID, input, expected schema, rubric/reference, safety slice,
  and prompt version.
- **Workflow:** fix decoding, validate schema, compare prompt versions, inspect
  stochastic variation, and classify failures.
- **Expected output:** versioned prompt, eval set, validity/task/safety report, and
  regression examples.
- **Evaluation:** no selection on final holdout and no unsupported “better” claim.

## G. Measured RAG foundation — after 25, 29, and 25A

- **Goal:** determine which chunking and retriever baseline is justified.
- **Dataset columns:** document, section, query, slice, and gold evidence IDs in
  `projects/rag_foundations/data`.
- **Workflow:** run lexical, LSA, and hybrid retrieval over three chunkers; calculate
  component metrics; diagnose failed and unanswerable queries.
- **Expected output:** hashed report, evidence traces, slice comparison, and the
  simplest-supported-system recommendation.
- **Evaluation:** RAG foundation mastery checkpoint.

## H. Evidence-grounded assistant — after 26–30 and 37

- **Goal:** add vector indexing, reranking, and generation only when each component
  improves a declared failure.
- **Dataset columns:** query ID, evidence IDs, answer, citations, abstention, latency,
  token/cost estimate, and evaluator labels.
- **Workflow:** preserve baseline, add one component at a time, separate retrieval and
  generation errors, test injection/unanswerable cases, and enforce citations.
- **Expected output:** ablation report, grounded assistant, failure taxonomy, cost budget.
- **Evaluation:** recall, support/correctness, abstention, latency, security, and no
  component without measurable value.

## I. Workflow versus agent challenge — after 31–40

- **Goal:** solve one task with deterministic code, one agent, and optionally multiple agents.
- **Dataset columns:** task ID, expected result, tool trace, permissions, steps,
  retries, outcome, latency, and cost.
- **Workflow:** establish deterministic baseline, add typed tools, evaluate traces,
  add memory/reflection only for demonstrated failures, and compare complexity.
- **Expected output:** system comparison, permission model, failure traces, and a
  simplest-adequate-design recommendation.
- **Evaluation:** success, unsafe action rate, cost, latency, and reproducibility.

## J. Production reliability drill — after 41–50

- **Goal:** operate one classical, neural, or RAG project under controlled failures.
- **Dataset columns:** trace ID, model/data version, prediction, confidence, latency,
  slice, delayed label, alert, and response action.
- **Workflow:** package/deploy, define SLOs, inject schema drift/dependency failure/
  stale index/bad release, investigate, degrade safely, and roll back.
- **Expected output:** tested service, dashboards, runbook, incident timeline,
  postmortem, cost model, threat model, and architecture decision record.
- **Evaluation:** detection time, false alarms, rollback, quality SLOs, security/privacy,
  and evidence-backed retraining decisions.
