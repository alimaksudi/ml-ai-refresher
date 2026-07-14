# Curriculum Build Progress

This file is the **build tracker**. Each `/loop` iteration: read this file,
build the next unchecked notebook(s), tick them off, update "Next up".

- **Status:** COMPLETE
- **Built:** 62 / 62
- **Next up:** —

Legend: `[x]` built & verified · `[~]` in progress · `[ ]` not started

## Canonical mastery path

The numbered phase list below remains the artifact/build inventory. It is **not**
the learning order. The machine-validated order is defined in
[`docs/CURRICULUM_PATH.json`](docs/CURRICULUM_PATH.json).

The implemented Foundations → Classical ML path is:

1. 00A → 00B → 00D → 00C → 01 → 02
2. 03A Data workflow, problem framing, baseline, and first holdout
3. 04 Linear regression and squared loss
4. 03 Optimization and gradient descent grounded in that loss
5. 05 Logistic regression
6. 09 Task-aligned metrics, calibration, thresholds, and decision cost
7. 10 Cross-validation and leakage
8. 44 Experiment tracking and reproducibility discipline
9. 11 Feature engineering and leak-safe pipelines
10. 06 → 07 → 08 Trees and ensembles
11. 12 Imbalanced learning → 13 Explainability → 36 advanced evaluation
12. Wine classifier mastery checkpoint

The implemented Deep Learning path is:

1. 13A PyTorch tensors, data loaders, autograd, train/eval loops, and checkpoints
2. 14 Neural networks from scratch
3. 15 Backpropagation and gradient checking
4. 15A Initialization, normalization, regularization, optimization, and diagnosis
5. 16 CNN and spatial inductive bias
6. 20 TF-IDF and word embeddings before sequence architectures
7. 17 RNN/LSTM → 18 Attention → 19 Transformers
8. Digit-classifier mastery checkpoint with baseline, ablation, and teach-back

---

## Prerequisite Phase — Zero-Math and Programming Onboarding
- [x] 00A — Mathematical Language and Arithmetic → `notebooks/phase_minus1_onboarding/00a_math_language_and_arithmetic.ipynb`
- [x] 00B — Algebra, Functions, and Graphs → `notebooks/phase_minus1_onboarding/00b_algebra_functions_graphs.ipynb`
- [x] 00C — Calculus and Probability Intuition → `notebooks/phase_minus1_onboarding/00c_calculus_probability.ipynb`
- [x] 00D — Python, NumPy, and Jupyter Foundations → `notebooks/phase_minus1_onboarding/00d_python_numpy_jupyter.ipynb`
- [x] 00E — Practical Python, Pandas, Debugging, and Tests → `notebooks/phase_minus1_onboarding/00e_python_pandas_debugging.ipynb`

## Phase 0 — Mathematical Foundations
- [x] 01 — Linear Algebra Essentials → `notebooks/phase0_foundations/01_linear_algebra_essentials.ipynb`
- [x] 02 — Probability and Statistics → `notebooks/phase0_foundations/02_probability_and_statistics.ipynb`
- [x] 03 — Optimization and Gradient Descent → `notebooks/phase0_foundations/03_optimization_and_gradient_descent.ipynb`
- [x] 03A — Data Workflow, EDA, Cleaning, Pandas, and SQL → `notebooks/phase0_foundations/03a_data_workflow_eda_cleaning.ipynb`

## Phase 1 — Classical Machine Learning
- [x] 04 — Linear Regression → `notebooks/phase1_classical_ml/04_linear_regression.ipynb`
- [x] 05 — Logistic Regression → `notebooks/phase1_classical_ml/05_logistic_regression.ipynb`
- [x] 06 — Decision Trees → `notebooks/phase1_classical_ml/06_decision_trees.ipynb`
- [x] 07 — Random Forest → `notebooks/phase1_classical_ml/07_random_forest.ipynb`
- [x] 08 — Gradient Boosting and XGBoost → `notebooks/phase1_classical_ml/08_gradient_boosting_xgboost.ipynb`

## Phase 2 — ML Engineering Foundations
- [x] 09 — Evaluation Metrics → `notebooks/phase2_ml_engineering/09_evaluation_metrics.ipynb`
- [x] 10 — Validation and Data Leakage → `notebooks/phase2_ml_engineering/10_validation_and_data_leakage.ipynb`
- [x] 11 — Feature Engineering → `notebooks/phase2_ml_engineering/11_feature_engineering.ipynb`
- [x] 12 — Imbalanced Learning → `notebooks/phase2_ml_engineering/12_imbalanced_learning.ipynb`
- [x] 13 — Explainability (SHAP) → `notebooks/phase2_ml_engineering/13_explainability_shap.ipynb`
- [x] 13B — Unsupervised Learning Foundations → `notebooks/phase2_ml_engineering/13b_unsupervised_learning.ipynb`

## Phase 3 — Deep Learning Foundations
- [x] 13A — PyTorch Foundations and Training Loops → `notebooks/phase3_deep_learning/13a_pytorch_foundations.ipynb`
- [x] 14 — Neural Networks from Scratch → `notebooks/phase3_deep_learning/14_neural_networks_from_scratch.ipynb`
- [x] 15 — Backpropagation → `notebooks/phase3_deep_learning/15_backpropagation.ipynb`
- [x] 15A — Stable Neural Training → `notebooks/phase3_deep_learning/15a_stable_neural_training.ipynb`
- [x] 16 — CNN → `notebooks/phase3_deep_learning/16_cnn.ipynb`
- [x] 17 — RNN and LSTM → `notebooks/phase3_deep_learning/17_rnn_and_lstm.ipynb`
- [x] 18 — Attention Mechanism → `notebooks/phase3_deep_learning/18_attention_mechanism.ipynb`
- [x] 19 — Transformers → `notebooks/phase3_deep_learning/19_transformers.ipynb`

## Phase 4 — Modern NLP and LLMs
- [x] 20 — TF-IDF and Word Embeddings → `notebooks/phase4_nlp_llms/20_tfidf_word_embeddings.ipynb`
- [x] 21 — Sentence Embeddings → `notebooks/phase4_nlp_llms/21_sentence_embeddings.ipynb`
- [x] 22 — LLM Training Pipeline → `notebooks/phase4_nlp_llms/22_llm_training_pipeline.ipynb`
- [x] 23 — Prompt Engineering → `notebooks/phase4_nlp_llms/23_prompt_engineering.ipynb`
- [x] 24 — Hallucination and Guardrails → `notebooks/phase4_nlp_llms/24_hallucination_guardrails.ipynb`

## Phase 5 — Retrieval-Augmented Generation
- [x] 25 — Similarity Search → `notebooks/phase5_rag/25_similarity_search.ipynb`
- [x] 25A — Measured Baseline RAG Retrieval → `notebooks/phase5_rag/25a_measured_rag_baseline.ipynb`
- [x] 25B — Grounded Answer and Citation Evaluation → `notebooks/phase5_rag/25b_grounded_answer_evaluation.ipynb`
- [x] 26 — Vector Databases → `notebooks/phase5_rag/26_vector_databases.ipynb`
- [x] 27 — Hybrid Search → `notebooks/phase5_rag/27_hybrid_search.ipynb`
- [x] 28 — Reranking → `notebooks/phase5_rag/28_reranking.ipynb`
- [x] 29 — Chunking Strategies → `notebooks/phase5_rag/29_chunking_strategies.ipynb`
- [x] 30 — Advanced RAG Architectures → `notebooks/phase5_rag/30_advanced_rag.ipynb`

## Phase 6 — Agentic AI
- [x] 31 — Agent Fundamentals → `notebooks/phase6_agents/31_agent_fundamentals.ipynb`
- [x] 32 — Planning and Tool Use → `notebooks/phase6_agents/32_planning_tool_use.ipynb`
- [x] 33 — Memory Systems → `notebooks/phase6_agents/33_memory_systems.ipynb`
- [x] 34 — Reflection and Self-Correction → `notebooks/phase6_agents/34_reflection_self_correction.ipynb`
- [x] 35 — Multi-Agent Systems → `notebooks/phase6_agents/35_multi_agent_systems.ipynb`

## Phase 7 — Evaluation
- [x] 36 — Classical ML Evaluation → `notebooks/phase7_evaluation/36_classical_ml_evaluation.ipynb`
- [x] 37 — RAG Evaluation → `notebooks/phase7_evaluation/37_rag_evaluation.ipynb`
- [x] 38 — LLM Evaluation → `notebooks/phase7_evaluation/38_llm_evaluation.ipynb`
- [x] 39 — Human Evaluation → `notebooks/phase7_evaluation/39_human_evaluation.ipynb`
- [x] 40 — LLM-as-a-Judge → `notebooks/phase7_evaluation/40_llm_as_a_judge.ipynb`

## Phase 8 — Production AI
- [x] 41 — MLOps → `notebooks/phase8_production/41_mlops.ipynb`
- [x] 42 — LLMOps → `notebooks/phase8_production/42_llmops.ipynb`
- [x] 43 — Feature Stores → `notebooks/phase8_production/43_feature_stores.ipynb`
- [x] 44 — Experiment Tracking → `notebooks/phase8_production/44_experiment_tracking.ipynb`
- [x] 45 — Monitoring and Drift Detection → `notebooks/phase8_production/45_monitoring_drift.ipynb`
- [x] 46 — Retraining Strategies → `notebooks/phase8_production/46_retraining_strategies.ipynb`

## Phase 9 — AI System Design
- [x] 47 — Scalable ML Systems → `notebooks/phase9_system_design/47_scalable_ml_systems.ipynb`
- [x] 48 — Production RAG Systems → `notebooks/phase9_system_design/48_production_rag.ipynb`
- [x] 49 — AI Reliability Patterns → `notebooks/phase9_system_design/49_ai_reliability.ipynb`
- [x] 50 — End-to-End AI Architecture → `notebooks/phase9_system_design/50_e2e_architecture.ipynb`

## Phase 10 — Applied Capstone
- [x] 51 — Deployable Real-Data ML Vertical Slice → `notebooks/phase10_applied_capstone/51_deployable_wine_classifier.ipynb`
- [x] Project — Wine classifier training, API, monitoring, tests, and Docker → `projects/wine_classifier/`
