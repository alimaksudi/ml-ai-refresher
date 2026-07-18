# Curriculum Sequence Review: Zero to Deep ML Mastery

> **Historical audit:** this document records the analysis that produced the current
> route. Some tables retain the lesson numbers that existed when the audit was written.
> Use [`LEGACY_ID_MAP.md`](LEGACY_ID_MAP.md) to translate them. All active dependencies,
> paths, route cards, and new content use the semantic IDs in `CURRICULUM_PATH.json`.

## Executive verdict

The curriculum has strong breadth and a mostly recognizable progression, but it is not yet a clean zero-to-hero path for **deep ML mastery**. The main spine—foundations → classical ML → deep learning → NLP/LLMs → RAG → production—is sound. The largest weakness is that evaluation, experimental method, data workflow, and project practice are treated as later topics instead of habits introduced before the first model and reinforced throughout.

**Readiness score: 74/100.**

- Scope and coverage: 18/20
- Prerequisite correctness: 14/20
- Beginner cognitive load: 13/20
- Deliberate practice and checkpoints: 11/20
- Depth toward ML mastery: 14/20
- Production integration: 14/20

The course is a strong broad refresher. It becomes a strong mastery curriculum after reordering the experimental-method spine, adding missing fundamentals, and inserting assessed projects between phases.

## How to read the module audit

Each row answers the requested ten questions in compact form:

- **Needs / taught?**: what the learner must know and whether the current course has already taught it.
- **Problem / why here**: what the module enables and the reason for its location.
- **Placement / neighbors**: keep, move earlier, or move later; then the ideal immediate predecessor and successor.
- **Overload / repetition / missing**: likely beginner overload, avoidable duplication, and the most important gap.

“Partial” means an idea appeared earlier but was not taught or assessed deeply enough to be a safe prerequisite.

## Module-by-module audit

### Prerequisite phase and mathematical foundations

| Module | Needs / taught earlier? | Problem solved / why here | Placement / ideal neighbors | Overload / repetition / missing |
|---|---|---|---|---|
| 00A Math language and arithmetic | None / yes | Makes notation, ratios, powers, sums, and units readable; correct entry point | Keep first; before: diagnostic; after: 00B | Overload: sigma notation and logs together. Repeat arithmetic only as retrieval practice. Missing: dimensional analysis and estimation. |
| 00B Algebra, functions, graphs | Arithmetic / yes, 00A | Turns relationships into equations and graphs; required before models and loss curves | Keep; 00A → 00B → 00C | Overload: function notation plus transformations. Missing: inequalities, piecewise functions, and inverse functions. |
| 00C Calculus and probability intuition | Algebra and graphs / yes | Introduces change, slopes, uncertainty; prepares optimization and statistics | Keep, but separate calculus intuition from probability intuition if novices struggle; 00B → 00C → 00D | Overload: two major subjects in one notebook. Probability repeats in 02 appropriately but needs explicit spiral labels. Missing: conditional probability tree practice. |
| 00D Python, NumPy, Jupyter | Arithmetic, functions / yes | Enables all runnable work; must precede data and matrix code | Move before or interleave with 00C; 00B → 00D → 00C/01 | Overload: Python, arrays, broadcasting, debugging, and notebooks together. Missing: functions, tests, plotting, file I/O, environments, and reproducibility basics. |
| 01 Linear algebra | Algebra, NumPy arrays / yes, 00B/00D | Represents tabular data, transformations, projections, similarity, and neural layers | Keep after Python; 00D → 01 → 02 | Overload: matrix multiplication, span, basis, eigen ideas in one pass. Missing: shapes as a first-class skill, rank, positive semidefinite matrices, SVD/PCA bridge. |
| 02 Probability and statistics | Algebra, probability intuition, vectors / yes | Models uncertainty and supports inference, loss, evaluation, and data reasoning | Keep; 01 → 02 → data/experimental method | Overload: distributions, estimation, Bayes, and hypothesis tests. Missing: sampling distributions, confidence intervals, effect size, multiple testing, causal caveat. |
| 03 Optimization and gradient descent | Functions, derivatives, vectors, loss functions / **partial**: math is taught, but supervised loss is not grounded in a model yet | Finds parameters minimizing an objective; essential before neural training | **Move after linear regression/loss**, or split into optimization intuition now and full gradient descent after 04; 04 → 03 → 05 | Critical overload: abstract objectives before a meaningful modeling problem. Repeats gradient descent in 04/14/15. Missing: convexity intuition, learning-rate schedules, regularization as objective term. |
| 03A Data workflow, EDA, cleaning, Pandas, SQL | Python, statistics / yes | Turns raw data into defensible modeling data; this is required before the first model | **Move before 03 and all models**; 02 → 03A → experimental design/split | Overload: EDA, cleaning, Pandas, and SQL in one module. Separate SQL appendix or lab. Missing: data provenance, target definition, sampling bias, schema/data contracts, reproducible splits. |

### Classical ML and core model-development practice

| Module | Needs / taught earlier? | Problem solved / why here | Placement / ideal neighbors | Overload / repetition / missing |
|---|---|---|---|---|
| 04 Linear regression | Data workflow, vectors, statistics, loss / partial; data workflow currently immediately before, loss only abstractly in 03 | Predicts continuous outcomes and gives the clearest first full learning algorithm | Keep as first model, but teach squared loss here before full gradient descent; train-test split → 04 → 03 optimization | Overload: derivation, closed form, GD, inference, regularization. Split basic regression from diagnostics/regularization. Missing: baseline model, residual assumptions, uncertainty, categorical encoding. |
| 05 Logistic regression | Linear model, probability, log loss, binary classification framing / partial; classification task and log loss arrive inside module | Predicts probabilities/classes with an interpretable boundary | Keep after regression and optimization; 03 → 05 → classification metrics | Overload: sigmoid, odds, log-odds, likelihood, cross-entropy, thresholds. Missing: calibration and explicit threshold/business-cost exercise. |
| 06 Decision trees | Classification/regression, impurity, validation / **partial**: tasks known, validation and metrics not yet taught | Learns nonlinear rules and interactions without scaling | Move after basic metrics and holdout validation; metrics/split → 06 → 07 | Overload: entropy, Gini, recursive partitioning, pruning. Feature importance introduced here should be named as prerequisite to SHAP. Missing: instability and honest pruning selection. |
| 07 Random forest | Trees, bootstrap sampling, variance / partial; trees yes, resampling not deeply taught | Reduces tree variance through bagging and feature randomness | Keep after trees; 06 → 07 → boosting | Overload: bootstrap, OOB, decorrelation at once. Repeats tree importance. Missing: uncertainty across trees and OOB-vs-CV comparison. |
| 08 Gradient boosting/XGBoost | Trees, residuals/gradients, loss, regularization, validation / partial; validation comes later | Builds strong tabular models by sequential error correction | Move after validation and feature engineering basics; 07 → validation/tuning → 08 | Overload: functional gradients plus XGBoost regularized objective. Separate generic boosting from XGBoost internals. Missing: early stopping, categorical alternatives, calibration, compute tradeoffs. |
| 09 Evaluation metrics | Classification/regression tasks and predictions / yes by 04–08, but taught too late | Converts predictions into task-aligned evidence | **Split and move earlier**: baseline + regression metrics before 04 evaluation; classification metrics immediately after 05; advanced metric selection later | Overload: all regression, classification, ranking, and calibration metrics together. Repeated substantially by 36. Missing: uncertainty intervals and metric variance. |
| 10 Validation and leakage | Data workflow, metrics, model fitting / yes, but five models were already evaluated informally | Estimates generalization and prevents invalid experiments | **Move before trees/ensembles**, with holdout before 04 and CV after first models; metrics → validation → trees | Overload: split, CV, temporal/group splits, leakage simultaneously. Missing: nested CV and model-selection bias. |
| 11 Feature engineering | Data types, preprocessing, split/leakage, model behavior / yes if 03A and 10 move earlier | Makes raw variables learnable while preserving evaluation integrity | Move before ensembles or pair with a pipelines module; validation → preprocessing/feature engineering → boosting | Overload: encoding, scaling, transformations, interactions. Missing: `ColumnTransformer`, fit/transform contract, feature selection stability, schema handling. |
| 12 Imbalanced learning | Classification metrics, thresholds, validation, sampling / yes under revised order | Handles rare outcomes without accuracy traps or leakage | Keep after metrics/validation/feature engineering; boosting → 12 → interpretability | Overload: resampling, class weights, PR curves, thresholds, cost. Missing: prevalence shift, calibration after resampling, decision analysis. |
| 13 Explainability/SHAP | Model behavior, coefficients, tree feature importance, conditional expectation / partial; feature importance appears in trees, conditional expectation may be weak | Explains local/global model behavior and audits reliance | Keep after a dedicated interpretability ladder; permutation importance → PDP/ICE → SHAP | Overload: Shapley axioms and correlated-feature caveats. Missing: first teach coefficients, impurity importance limitations, permutation importance, PDP/ICE, then SHAP; add explanation stability and causal warning. |

### Deep learning, NLP, and LLMs

| Module | Needs / taught earlier? | Problem solved / why here | Placement / ideal neighbors | Overload / repetition / missing |
|---|---|---|---|---|
| 14 Neural networks from scratch | Linear algebra, optimization, regression/classification, activations, loss / yes under revised core | Learns nonlinear representations and composes differentiable layers | Keep; classical capstone → 14 → 15 | Overload: architecture, forward pass, loss, training at once. Missing: tensor shapes, initialization, normalization, regularization, PyTorch fundamentals. |
| 15 Backpropagation | Computation graph, chain rule, forward pass, loss / partial: chain rule/NN taught | Computes gradients efficiently; correct immediately after a forward-only network | Keep, but avoid re-teaching the entire NN; 14 → 15 → training practice | Overload: indices/Jacobians. Missing: autodiff, gradient checking, vanishing/exploding gradients, optimizer comparison. |
| 16 CNN | Backprop, tensors, dense nets, image structure / yes | Exploits locality and weight sharing for images | Keep after a neural-training lab; training practice → 16 → sequence models | Overload: convolution shapes, padding, stride, pooling. Missing: augmentation, transfer learning, modern residual networks, vision evaluation. |
| 17 RNN/LSTM | Backprop, sequences, recurrence / yes | Models ordered data and motivates long-range dependency failures | Keep as historical bridge, potentially shorten; CNN → 17 → attention | Overload: LSTM gate equations. Some content is historical rather than mastery-critical. Missing: teacher forcing, masking, sequence batching. |
| 18 Attention | Vectors/matrices, similarity, softmax, sequences / yes | Removes recurrence bottleneck and learns content-based dependencies | Keep; 17 → 18 → 19 | Overload: Q/K/V shapes and scaling. Missing: masking, cross-attention, multi-head shape drills, computational complexity. |
| 19 Transformers | Attention, residuals, normalization, embeddings, optimization / **partial**: attention yes; residuals/layer norm/modern training weak | Builds the dominant sequence architecture | Keep, but add missing neural components before it; attention → transformer block → NLP | Overload: full encoder/decoder and training mechanics. Missing: layer norm, residual pathways, positional methods, decoder masking, KV cache. |
| NLP-06 Transformer model families | Trained causal decoder, attention masks, tensor shapes / yes through DL-08 and the tiny-LM gate | Separates GPT generation, BERT representation learning, and T5 source-to-target modeling before later modules assume them | Keep immediately after DL-08 in the canonical route; DL-08 → NLP-06 → sentence embeddings / LLM lifecycle | Added shared scratch components, behavioral mask tests, and real controlled objectives. Extension gaps: pretrained local checkpoint mapping, relative positions, and modern attention variants. |
| 20 TF-IDF and word embeddings | Text preprocessing, sparse vectors, similarity / partial; linear algebra yes, NLP task framing new | Establishes lexical and distributional text representations | **Move before RNN/attention/transformers** as NLP foundations; classical ML → 20 → sequence models | Overload: TF-IDF plus multiple embedding families. Missing: tokenization, vocabulary/OOV, sparse text classification baseline, bias in embeddings. |
| 21 Sentence embeddings | Word embeddings, BERT-style encoding, padding masks, cosine, cross-entropy, split discipline / yes through NLP-01, DL-07/08, and NLP-06 | Compresses contextual token states into retrieval-ready sentence geometry | Keep after NLP-06 and before similarity search; NLP-06 → NLP-02 → RAG-01 | Rebuilt around real local training: masked pooling, normalization, MNR, per-anchor hard negatives, false-negative warning, leakage test, TF-IDF/untrained baselines, Recall@k/MRR/margin, and behavioral tests. Extension gaps: pretrained checkpoint comparison, multilingual/domain-scale evaluation, and multi-seed uncertainty. |
| NLP-03 LLM pretraining and data pipeline | Causal LM, validation/leakage, tracking, tokenizer/checkpoint compatibility / yes through tiny-LM, MLE-02, PROD-04, NLP-06 | Curates and versions data, estimates work, continues a real checkpoint, and measures domain gain plus forgetting | Keep after NLP-06; NLP-06 → NLP-03 → NLP-07 | Split from the overloaded lifecycle. Added executed curation, contamination rejection, continued pretraining, domain validation, and retention evidence. Extension gaps: near-dedup at scale and distributed systems. |
| NLP-07 Instruction tuning and LoRA | Continued checkpoint, shifted targets, backprop, stable training / yes through NLP-03 and DL-03/04 | Teaches response behavior with real masked SFT and compares full updates with low-rank adapters | Keep after NLP-03; NLP-03 → NLP-07 → NLP-08 | Added real full/LoRA optimization, zero-update invariant, parameter count, held-out loss, and overfit diagnosis. Extension gaps: packing, multi-turn masks, and QLoRA. |
| NLP-08 Preference learning and alignment | SFT policy, sigmoid/BCE, sequence log probabilities / yes through NLP-07 and FND-02 | Optimizes chosen over rejected responses while anchoring to a reference | Keep after SFT and before LLM evaluation; NLP-07 → NLP-08 → EVAL-02 | Corrects PPO/DPO conflation and negative-margin gradient error. Adds real DPO, held-out preference margin/accuracy, and retention cost. PPO remains correctly labeled conceptual scope. |
| 23 Prompt engineering | Transformer/LLM behavior, decoding, evaluation / partial; evaluation is deferred to 38 | Controls model behavior without weight updates | Move after basic LLM evaluation; LLM behavior/decoding → evaluation basics → prompting | Overload: many patterns without an experiment framework. Repeated later in agents. Missing: decoding parameters, structured outputs, prompt versioning, ablation. |
| 24 Hallucination and guardrails | LLM failure modes, calibration limits, evaluation, retrieval / **partial**: RAG not yet taught and LLM eval later | Detects/mitigates unsafe or unsupported outputs | Split: teach failure taxonomy here; move RAG-grounding and operational guardrails after RAG/evaluation | Lumps hallucination, safety, policy, validation. Missing: threat modeling, prompt injection, privacy, red teaming, refusal evaluation. |

### Retrieval and agents

| Module | Needs / taught earlier? | Problem solved / why here | Placement / ideal neighbors | Overload / repetition / missing |
|---|---|---|---|---|
| 25 Similarity search | Vectors, cosine/dot product, sentence embeddings / yes | Retrieves nearest semantic items; correct RAG foundation | Keep; 21 → 25 → chunking | Overload: exact/approximate distinctions. Missing: normalization, distance concentration, recall-latency evaluation. |
| 26 Vector databases | Similarity search, indexing, metadata, persistence / yes | Operates retrieval at scale | Move after chunking and retrieval evaluation basics; chunking → vector DB → hybrid | Overload: ANN algorithms plus database concerns. Missing: index build/update lifecycle, filtering interactions, tenancy/security. |
| 27 Hybrid search | Dense retrieval, lexical retrieval, rank fusion / yes if TF-IDF moved earlier | Recovers exact terms while retaining semantic recall | Keep after vector DB; vector DB → hybrid → reranking | Overload: score normalization and fusion. Missing: query taxonomy and per-slice comparison. |
| 28 Reranking | Candidate retrieval, passage labels, ranking metrics, development/evaluation separation / yes through RAG-06 and EVAL-03; cross-encoders are introduced here | Reorders a fixed candidate set when relevant evidence is present but poorly ranked | Keep after retrieval evaluation basics; hybrid → component evaluation → reranking | Rebuilt around a manual reorder and held-out measured benchmark. Extension risks remain: neural architecture, hard-negative training, calibration, and realistic service latency. |
| 29 Chunking strategies | Documents, tokenization, embeddings, retrieval metrics / partial; metrics deferred to 37 | Creates retrievable units and controls context coherence | **Move to immediately after similarity search and before vector DB ingestion**; 25 → 29 → 26 | Current order teaches indexing/ranking before defining the indexed unit. Missing: parsing/layout, metadata, parent-child chunks, empirical chunk experiments. |
| 30 Advanced RAG | All retrieval components, prompting, evaluation / partial; RAG evaluation comes later | Composes query rewriting, routing, retrieval, generation, and fallback | Move after 37 RAG evaluation basics; RAG baseline → evaluation → advanced RAG | Overload: many architectures without a stable baseline. Missing: end-to-end error taxonomy, ablations, cost/latency budgets, security. |
| 31 Agent fundamentals | LLM prompting, structured outputs, tools, state, evaluation / partial; structured output and evaluation weak | Frames agents as controlled loops rather than magic autonomy | Keep only after LLM evaluation and a simple RAG app; LLM eval → 31 → 32 | Overload: autonomy before reliability. Missing: deterministic workflow baseline and explicit “when not to use an agent.” |
| 32 Planning and tool use | Agent loop, schemas/APIs, state machines, error handling / partial; software/API basics not explicit | Executes multi-step tasks against external capabilities | Keep; 31 → 32 → agent evaluation | Overload: planning algorithms plus function calling. Missing: idempotency, retries, permissions, sandboxing, typed tool contracts. |
| 33 Memory systems | Agent state, embeddings/retrieval, persistence, privacy / yes except privacy | Preserves useful context across steps/sessions | Keep after tool use; 32 → agent evaluation → 33 | Overload: working, episodic, semantic memory together. Repeats vector DB/RAG. Missing: retention, deletion, poisoning, memory evaluation. |
| 34 Reflection/self-correction | Agent loop, evaluation signal, tool traces / **partial**: agent evaluation is deferred | Uses feedback to revise outputs/actions | Move after evaluation methods; agent eval → 34 → multi-agent | Overload: reflection can imply reliability without evidence. Missing: stopping rules, verifier independence, regression risks, cost accounting. |
| 35 Multi-agent systems | Single-agent control/evaluation, distributed coordination / partial | Coordinates specialized roles when decomposition truly helps | Keep last and mark optional/advanced; 34 → 35 → production reliability | Overload: orchestration and emergent failure. Repeats planning/memory. Missing: single-agent baseline, communication protocols, deadlock, security, measured benefit. |

### Evaluation, production, system design, and capstone

| Module | Needs / taught earlier? | Problem solved / why here | Placement / ideal neighbors | Overload / repetition / missing |
|---|---|---|---|---|
| 36 Classical ML evaluation | Metrics, validation, experimental design / yes, largely repeats 09/10 | Deepens uncertainty, slicing, comparison, and decision quality | **Combine with 09/10** or redefine strictly as advanced evaluation after projects | Major unnecessary repetition unless it adds confidence intervals, statistical tests, calibration, robustness, and slice analysis. |
| 37 RAG evaluation | Retrieval pipeline, ranking metrics, grounded generation / yes after baseline RAG | Separates retrieval from generation failures | **Move before advanced RAG**; baseline RAG → 37 → reranking/advanced RAG | Overload: many proxy metrics. Missing: labelled dataset creation, retrieval recall@k first, confidence intervals, failure taxonomy. |
| 38 LLM evaluation | LLM behavior, prompting, sampling, test design / partial | Measures quality when exact labels are insufficient | **Move before prompt engineering/agents** at introductory depth; LLM basics → 38 → prompting | Overload: reference, model-based, and task-based evaluation. Missing: contamination, stochastic variance, adversarial tests, reproducible decoding. |
| 39 Human evaluation | Clear rubric, sampling, statistics, LLM task / yes | Produces defensible judgments for subjective outputs | Keep after basic LLM evaluation; 38 → 39 → judge models | Overload: study design and annotation operations. Missing: power/sample size, accessibility, worker welfare, adjudication. |
| 40 LLM-as-a-judge | Human rubrics, LLM evaluation, bias/reliability / yes | Scales rubric-based comparison with known limitations | Keep after 39; human eval → 40 → agent eval | Overload: judge calibration and bias. Missing: position/verbosity bias controls, agreement thresholds, judge drift. |
| 41 MLOps | Complete model lifecycle, validation, software testing, deployment / partial; deployment and software engineering are not a dedicated prerequisite | Makes training reproducible and deployable | Move core lifecycle earlier as a running project spine; advanced MLOps remains here | Overload: CI/CD, registries, orchestration, serving. Missing: packaging, config/secrets, unit/integration/data tests, lineage. |
| 42 LLMOps | LLM apps, evaluation, prompting, RAG, monitoring / yes under revised order | Operates prompts/models/retrieval with quality and cost controls | Keep after LLM/RAG evaluation; 41 → 42 → monitoring | Repeats MLOps and evaluation. Missing: prompt registry, token/cost accounting, caching, provider fallback, privacy. |
| 43 Feature stores | Feature engineering, leakage, online serving, data systems / partial; online serving/data architecture weak | Aligns reusable offline and online features | Move after scalable ML systems or make optional; serving/data pipelines → 43 → monitoring | Too specialized before experiment tracking. Missing: point-in-time correctness lab, ownership, backfills, build-vs-buy. |
| 44 Experiment tracking | Experimental method, training runs, metrics / yes—and needed much earlier | Preserves parameters, artifacts, metrics, and comparisons | **Move to immediately after validation and use thereafter**; validation → 44 → ensemble experiments | Current late placement is a critical practice error. Missing: run naming, lineage, reproducibility exercise, hypothesis log. |
| 45 Monitoring/drift | Offline evaluation, deployed inference, distributions, logging / yes except deployment practice | Detects data, concept, performance, and service degradation | Keep after deployment; deployment → 45 → 46 | Overload: drift tests plus observability. Missing: labels delayed/missing, alert thresholds, segment monitoring, incident response. |
| 46 Retraining strategies | Monitoring signals, pipelines, validation, deployment safety / yes | Decides when/how to update models safely | Keep; 45 → 46 → reliability | Overload: triggers, automation, online learning. Missing: champion/challenger, rollback, backtesting, feedback loops. |
| 47 Scalable ML systems | Serving, data systems, distributed systems, SLOs / partial | Meets throughput, latency, cost, and reliability constraints | Move before feature stores; MLOps/deployment → 47 → 43 | Overload: batching, caching, sharding, queues. Missing: capacity estimation lab and consistency tradeoffs. |
| 48 Production RAG | RAG evaluation, indexing lifecycle, security, serving / partial; security weak | Makes RAG reliable under changing data and load | Keep after scalable systems; 47 → 48 → reliability | Repeats advanced RAG/LLMOps. Missing: ACL-aware retrieval, deletion, freshness SLOs, prompt injection defense. |
| 49 AI reliability | Evaluation, monitoring, distributed failure, safety / yes under revised order | Designs graceful degradation and incident controls | Keep near end; production systems → 49 → architecture | Overload: many patterns. Missing: hazard analysis, runbooks/game days, error budgets, human escalation design. |
| 50 End-to-end architecture | All prior lifecycle and system topics / yes | Integrates requirements, data, model, serving, evaluation, and operations | Keep as synthesis before final capstone design review; 49 → 50 → capstone | Risks becoming diagram vocabulary. Missing: explicit requirements/capacity exercise, architecture decision records, threat model, cost model. |
| 51 Deployable wine classifier | Data workflow, classical ML, evaluation, MLOps / yes, but arrives far too late | Demonstrates a complete vertical slice | **Move a smaller version to after Phase 2**, then revisit/extend in production; classical core → capstone v1 → deep learning | Current capstone does not exercise DL, NLP, RAG, agents, or much of Phases 3–9, so it cannot validate whole-course mastery. Add separate track capstones. |
| Wine classifier project | Python packaging, training, tests, API, monitoring, Docker / partial before Phase 8 | Converts notebook knowledge into software | Use incrementally from Phase 2 onward; v1 model → API/tests → monitoring/retraining | Overload if first encountered at the end. Missing: CI, data validation, model registry/deployment target, load test, rollback drill. |

## Dependency map

```text
Arithmetic and notation
→ required before algebra and functions
→ because the student must parse equations, ratios, exponents, and summations.

Algebra and functions
→ required before vectors, probability models, and loss functions
→ because the student must manipulate expressions and understand input-output mappings.

Python and NumPy
→ required before linear algebra labs and data workflow
→ because the student must inspect shapes and run experiments, not only read theory.

Vectors
→ required before matrices
→ because a matrix is easiest to understand as rows/columns of vectors and as a transformation between vector spaces.

Probability intuition
→ required before statistics and logistic regression
→ because predictions and uncertainty must be interpreted as probabilities rather than arbitrary scores.

Statistics and sampling
→ required before EDA, validation, and metric uncertainty
→ because observed data is a sample and every reported score has variability.

Data workflow and target definition
→ required before train-test split and modeling
→ because the prediction unit, target, population, and timestamp determine whether an experiment is valid.

Supervised-learning framing and a naive baseline
→ required before linear regression
→ because students need to know what is learned, what is held fixed, and what improvement means.

Loss functions
→ required before gradient descent
→ because a gradient only has meaning relative to an objective being minimized.

Train-test split
→ required before cross-validation
→ because cross-validation generalizes the idea of estimating out-of-sample performance across repeated splits.

Linear regression and squared loss
→ required before full gradient-descent derivation
→ because a concrete, visual objective makes slope and parameter updates meaningful.

Binary classification and logistic regression
→ required before classification metrics
→ because confusion matrices, thresholds, precision, recall, ROC, and calibration all require class predictions or probabilities.

Basic task metrics
→ required before model comparison
→ because “better” is undefined until the task and cost determine a metric.

Holdout validation and leakage
→ required before tuning trees and ensembles
→ because flexible models otherwise reward accidental overfitting and invalid preprocessing.

Preprocessing fit/transform semantics
→ required before pipelines
→ because a pipeline exists to learn transformations only on training data and apply them consistently later.

Decision trees
→ required before random forests and gradient boosting
→ because both ensembles change how many trees are fit and how their errors are combined.

Coefficients and tree feature importance
→ required before permutation importance, PDP/ICE, and SHAP
→ because SHAP should be learned as one advanced explanation method, not as the learner's first idea of importance.

Neural forward pass and loss
→ required before backpropagation
→ because backprop computes how that specific loss changes through the computation graph.

Backpropagation and stable neural training
→ required before CNNs, RNNs, and transformers
→ because architecture equations are not useful without the ability to train and debug them.

Tokenization, TF-IDF, and word embeddings
→ required before neural sequence models and sentence embeddings
→ because students need the lexical baseline and the text-to-number problem before advanced representations.

Sequence models
→ required before attention
→ because attention's motivation is clearest as a response to recurrence and fixed-state bottlenecks.

Attention, residual connections, and layer normalization
→ required before transformers
→ because a transformer is composed from these mechanisms.

Sentence embeddings and cosine similarity
→ required before semantic search
→ because dense retrieval ranks vector representations with similarity functions.

Similarity search
→ required before chunking experiments
→ because learners need a measurable retrieval task to compare chunk boundaries.

Chunking and document metadata
→ required before vector database ingestion
→ because the indexed unit and its provenance must be defined before storage and retrieval.

Lexical and dense retrieval
→ required before hybrid search
→ because hybrid retrieval combines their complementary candidates or scores.

Retrieval metrics and a labelled query set
→ required before reranking and advanced RAG
→ because improvements cannot be distinguished from added complexity without recall/ranking evidence.

Basic LLM evaluation
→ required before prompt engineering experiments
→ because prompt changes are model changes and need repeatable test cases and metrics.

RAG evaluation
→ required before production RAG
→ because production monitoring must extend known offline quality measures.

Single-agent loop and deterministic workflow baseline
→ required before planning, reflection, and multi-agent systems
→ because added autonomy must demonstrate value over a simpler controlled workflow.

Agent evaluation
→ required before reflection and self-correction
→ because a system cannot reliably correct itself without an independent signal of correctness.

Model evaluation and a deployed service
→ required before monitoring
→ because monitoring detects changes relative to defined offline quality and runtime expectations.

Monitoring and safe deployment
→ required before automated retraining
→ because a trigger is useful only if the signal is meaningful and a bad replacement can be rolled back.
```

## Critical sequence problems

1. **Gradient descent precedes a concrete supervised loss.** Lesson FND-04 can teach optimization intuition, but full parameter-gradient mechanics should follow linear regression and squared loss.
2. **Data workflow is too late within foundations.** EDA, target definition, data quality, and splitting should frame every model experiment.
3. **Metrics and validation arrive after five models.** Students can fit models before they can make a valid claim about them. Introduce holdout evaluation with the first model, classification metrics after logistic regression, and CV before tuning ensembles.
4. **Experiment tracking is Lesson PROD-04.** It should become a habit immediately after validation, not a late production tool.
5. **NLP foundations follow transformers.** TF-IDF, tokenization, sparse baselines, and word embeddings should precede neural sequence models.
6. **Chunking follows vector DB, hybrid search, and reranking.** The indexed unit must be designed before ingestion and ranking.
7. **Evaluation is isolated in Phase 7.** LLM, RAG, and agent evaluation must sit beside their systems, with Phase 7 reserved for advanced evaluation design.
8. **SHAP is effectively the first dedicated interpretability module.** Build an interpretability ladder from coefficients and tree importance through permutation importance and PDP/ICE to SHAP.
9. **The only applied capstone is at the end and covers mainly classical ML.** It neither scaffolds early project skills nor assesses mastery of DL/NLP/RAG/system design.

## Missing prerequisites and important missing concepts

### Required for a true ML mastery claim

- Problem framing: prediction unit, target construction, decision/action, cost of errors, baseline, constraints.
- Experimental method: hypotheses, baselines, ablations, uncertainty intervals, statistical comparison, reproducibility.
- Generalization theory at an intuitive level: bias-variance, under/overfitting, regularization, learning curves, capacity.
- Preprocessing pipelines: train-only fitting, column-wise transforms, feature names/schema, inference consistency.
- Hyperparameter optimization: random/Bayesian search, nested validation, early stopping, selection bias.
- Unsupervised learning: PCA/SVD, clustering, anomaly detection, representation evaluation.
- Data-centric ML: label quality, annotation, sampling bias, dataset shift, slices, data versioning.
- Calibration and decision theory: probability quality, expected cost/utility, threshold selection.
- Software engineering for ML: packaging, configuration, testing pyramid, CI, artifacts, interfaces, reproducible environments.
- Deep-learning practice: PyTorch, tensors/autodiff, initialization, normalization, regularization, optimizers, schedulers, debugging.
- Responsible ML: fairness definitions/tradeoffs, privacy, security, causal-vs-predictive explanations, governance/model cards.
- Causal inference basics: confounding, interventions, selection bias, A/B experiments; enough to prevent causal claims from predictive models.
- Distributed/data systems basics before feature stores and scalable serving.

### Useful electives rather than core prerequisites

- Time-series forecasting and temporal validation.
- Recommender systems and learning-to-rank.
- Reinforcement learning.
- Graph ML.
- Generative vision/diffusion.

These broaden “AI mastery,” but should not displace the missing experimental and modeling fundamentals above.

## Modules to combine

- **09 Evaluation Metrics + 10 Validation + 36 Classical ML Evaluation** → a three-stage evaluation thread: foundations with the first models, model selection after several models, advanced inference/robustness after the classical capstone.
- **23 Prompt Engineering + introductory part of 38 LLM Evaluation** → prompt experimentation with a fixed eval set, then retain advanced evaluation as a later module.
- **24 Hallucination/Guardrails + 49 Reliability** → keep a short LLM failure introduction in 24; move operational mitigations, fallback, escalation, and incident patterns into reliability.
- **30 Advanced RAG + 48 Production RAG** → keep architectural improvement and production operation distinct, but share one evolving project and one evaluation suite to prevent repeated architecture surveys.
- **41 MLOps + 44 Experiment Tracking** → tracking is the first concrete MLOps practice and should be introduced together early, then extended in Phase 8.

## Modules to separate

- **00C** → calculus intuition; probability intuition.
- **03A** → data/EDA/cleaning with Pandas; SQL for analysis as a lab or prerequisite appendix.
- **04** → basic regression and loss; regression diagnostics/regularization/inference.
- **08** → gradient boosting concepts; XGBoost engineering and regularization.
- **13** → model-specific/global importance; local explanations and SHAP.
- **20** → TF-IDF/sparse NLP baseline; learned word embeddings.
- **22** → pretraining/data/scaling; instruction tuning and preference alignment; inference optimization.
- **24** → factual support/hallucination; safety/security guardrails.
- **41** → experiment/reproducibility lifecycle; deployment/orchestration lifecycle.

## Revised course order

### Phase A — Readiness and computational foundations

1. Diagnostic and learning-how-to-learn guide
2. 00A Mathematical language and arithmetic
3. 00B Algebra, functions, graphs
4. 00D Python/Jupyter foundations (core)
5. 00C1 Calculus intuition
6. 00C2 Probability intuition
7. 01 Linear algebra, shapes, transformations
8. 02 Probability, statistics, sampling, uncertainty
9. 03A1 Data workflow, EDA, cleaning, Pandas

**Major reason:** learners can manipulate data and run small experiments before abstract optimization. Python is available when mathematical ideas become computational.

### Phase B — Learning from data and experimental method

10. Problem framing, target definition, baselines, supervised-learning loop **(new)**
11. Train/validation/test split and leakage basics **(from 10)**
12. 04A Linear regression and squared loss
13. 03 Optimization and gradient descent, now grounded in regression
14. 04B Regression diagnostics, regularization, uncertainty
15. 05 Logistic regression and classification
16. 09A Classification metrics, calibration, thresholds, decision costs
17. 10B Cross-validation, group/time splits, model selection
18. 44A Experiment tracking and reproducibility
19. 11 Preprocessing, pipelines, and feature engineering

**Major reason:** a learner should understand what constitutes valid evidence before fitting increasingly flexible models.

### Phase C — Classical ML depth

20. 06 Decision trees and pruning
21. 07 Random forests and bagging
22. 08A Gradient boosting
23. 08B XGBoost, early stopping, tuning
24. 12 Imbalanced learning and decision analysis
25. Interpretability ladder: coefficients → permutation → PDP/ICE → 13 SHAP
26. Unsupervised learning: PCA/SVD, clustering, anomaly detection **(new)**
27. 36 Advanced evaluation: confidence intervals, comparison, robustness, slices
28. Classical ML vertical-slice capstone v1 (wine project)

**Major reason:** ensembles now build on valid validation and preprocessing; the phase ends with evidence-backed software, not just more algorithms.

### Phase D — Deep learning and representation learning

29. PyTorch, tensors, autodiff, training-loop lab **(new)**
30. 14 Neural networks from scratch
31. 15 Backpropagation and gradient checking
32. Stable training: initialization, normalization, optimizers, regularization **(new)**
33. 16 CNN and transfer learning
34. 20A Tokenization and TF-IDF text baseline
35. 20B Word embeddings
36. 17 RNN/LSTM as sequence-model bridge
37. 18 Attention
38. Residual connections, layer normalization, positional representation **(new)**
39. 19 Transformers
40. Deep-learning mini capstone

**Major reason:** NLP representation basics precede advanced sequence architecture, and practical neural training fills the gap between scratch derivations and usable mastery.

### Phase E — LLMs and retrieval systems

41. 21 Sentence embeddings
42. 22A LLM pretraining/data/scaling
43. 22B instruction tuning/alignment/inference
44. 38A Basic LLM evaluation and reproducible decoding
45. 23 Prompt experimentation with an eval set
46. 24A Hallucination/failure taxonomy and structured outputs
47. 25 Similarity search
48. 29 Chunking, parsing, metadata
49. Baseline RAG and retrieval evaluation dataset **(new)**
50. 26 Vector databases
51. 27 Hybrid search
52. 37 Retrieval/RAG evaluation
53. 28 Reranking
54. 30 Advanced RAG
55. 24B Safety, privacy, prompt injection, guardrails
56. RAG capstone with ablations

**Major reason:** every added RAG component can be measured against a simple baseline; chunking precedes indexing.

### Phase F — Agents and rigorous evaluation

57. 31 Agent fundamentals versus deterministic workflows
58. 32 Planning, typed tools, permissions, error handling
59. Agent evaluation and trace taxonomy **(new)**
60. 33 Memory systems
61. 34 Reflection/self-correction with stopping rules
62. 39 Human evaluation
63. 40 LLM-as-a-judge
64. 35 Multi-agent systems (advanced elective)
65. Agent capstone comparing workflow, single-agent, and multi-agent baselines

**Major reason:** autonomy follows evaluation, and every agentic technique must justify its complexity against a simpler baseline.

### Phase G — Production and system mastery

66. Software packaging, tests, CI, configuration, deployment **(new, or capstone extraction)**
67. 41 MLOps lifecycle and registries
68. 42 LLMOps
69. 47 Scalable ML systems and capacity estimation
70. 43 Feature stores (optional specialization)
71. 45 Monitoring and drift
72. 46 Retraining, champion/challenger, rollback
73. 48 Production RAG
74. 49 Reliability, security, incident response
75. 50 End-to-end architecture and cost/threat models
76. Final capstone: choose classical ML, DL, or RAG/agent track; meet a common production rubric

**Major reason:** production topics now follow deployed vertical slices and culminate in demonstrated reliability, rather than remaining architecture vocabulary.

## Recommended mastery checkpoints

Each checkpoint should require a readiness score of at least 80%, plus successful teach-back. A learner who fails should receive a targeted remediation path rather than simply continue.

| Gate | Evidence required |
|---|---|
| A: Mathematical/computational readiness | Trace vector/matrix shapes; manipulate a function; explain derivative and probability in words; implement a NumPy function; debug a notebook. |
| B: Valid experiment | Define unit/target/baseline; make leakage-safe splits; fit linear/logistic regression; select a justified metric; report uncertainty and limitations. |
| C: Classical ML mastery | Compare linear, tree, bagging, and boosting models with one reproducible pipeline; diagnose overfit; tune without test leakage; explain predictions with caveats. |
| D: Deep-learning mastery | Derive and gradient-check a small network; train/debug in PyTorch; reason about tensor shapes; compare dense/CNN/sequence inductive biases. |
| E: LLM/RAG mastery | Build a lexical baseline and dense retrieval; create a labelled eval set; run chunking/reranking ablations; report retrieval and generation failures separately. |
| F: Agent mastery | Compare deterministic workflow vs agent; enforce typed tools/permissions; evaluate task success, cost, latency, and failure; justify or reject multi-agent design. |
| G: Production mastery | Package/test/deploy; define SLOs; monitor slices and drift; perform champion/challenger and rollback; defend architecture, cost, privacy, and threat decisions. |

## Recommended mini-projects between phases

1. **After foundations — Data audit notebook:** clean a messy dataset, define its population and target, visualize sampling problems, and produce a data card.
2. **After first linear/classification models — Decision memo:** compare naive, regression, and logistic baselines; choose a threshold from business costs; communicate uncertainty.
3. **After classical ML — Tabular vertical slice:** leakage-safe pipeline, tracking, model comparison, calibration, explanations, tests, and model card. Evolve the existing wine project here.
4. **After deep learning — Representation study:** compare a linear baseline, MLP, and task-appropriate CNN/sequence model; include learning curves and error slices.
5. **After NLP/LLM basics — Prompt evaluation lab:** fixed dataset, deterministic settings where possible, prompt variants, bootstrap uncertainty, qualitative failure taxonomy.
6. **After RAG — Evidence-grounded assistant:** lexical vs dense vs hybrid; chunk/rerank ablations; retrieval recall and answer support; latency/cost budget; injection tests.
7. **After agents — Workflow complexity challenge:** solve one task with deterministic code, one agent, and multiple agents; select the simplest system meeting the quality target.
8. **After production — Reliability drill:** introduce schema drift, delayed labels, dependency failure, stale index, and bad model deployment; detect, degrade safely, and roll back.
9. **Final capstone — Production defense:** an end-to-end system plus design review, live failure demo, evaluation report, model/system card, runbook, cost model, and postmortem.

## Final assessment

The present course follows a logical **topic survey** from basic math to advanced AI systems, and its consistent notebook structure is a meaningful strength. It does not yet fully support “master ML deeply” because breadth outruns repeated applied practice, several prerequisites arrive after their consumers, and evaluation is organized as a phase instead of the spine of the curriculum.

The highest-impact changes are:

1. Move data workflow, split/leakage basics, metrics, and experiment tracking ahead of flexible models.
2. Ground loss before gradient descent.
3. Move NLP foundations before advanced sequence models.
4. Move chunking before indexing and RAG evaluation before advanced RAG.
5. Introduce LLM/agent evaluation before prompting and self-correction.
6. Add unsupervised learning, experimental rigor, preprocessing pipelines, practical deep-learning training, responsible ML, and software engineering.
7. Replace the single late capstone with an evolving vertical slice and phase gates.

With those changes implemented and assessed, the design would plausibly reach **90–94/100** readiness for its stated target. The remaining gap would depend on execution quality: exercise depth, feedback, real datasets, cumulative retention checks, and whether learners must defend decisions rather than merely run notebooks.
