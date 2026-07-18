# Curriculum Build Progress

This file is the **build tracker**. Each `/loop` iteration: read this file,
build the next unchecked notebook(s), tick them off, update "Next up".

- **Status:** COMPLETE
- **Built:** 62 / 62
- **Next up:** —

Legend: `[x]` built & verified · `[~]` in progress · `[ ]` not started

## Canonical mastery path

The section list below is the artifact/build inventory. Semantic IDs remain stable;
the machine-validated teaching order is defined in
[`docs/CURRICULUM_PATH.json`](docs/CURRICULUM_PATH.json).

The implemented Foundations → Classical ML path is:

1. PRE-01 → PRE-02 → PRE-03 → PRE-04 → FND-01 → FND-02
2. FND-03 Data workflow, problem framing, baseline, and first holdout
3. CML-01 Linear regression and squared loss
4. FND-04 Optimization and gradient descent grounded in that loss
5. CML-02 Logistic regression
6. MLE-01 Task-aligned metrics, calibration, thresholds, and decision cost
7. MLE-02 Cross-validation and leakage
8. PROD-04 Experiment tracking and reproducibility discipline
9. MLE-03 Feature engineering and leak-safe pipelines
10. CML-03 → CML-04 → CML-05 Trees and ensembles
11. MLE-04 Imbalanced learning → MLE-05 Explainability → EVAL-01 advanced evaluation
12. Wine classifier mastery checkpoint

The implemented Deep Learning path is:

1. DL-01 PyTorch tensors, data loaders, autograd, train/eval loops, and checkpoints
2. DL-02 Neural networks from scratch
3. DL-03 Backpropagation and gradient checking
4. DL-04 Initialization, normalization, regularization, optimization, and diagnosis
5. DL-05 convolutional networks and spatial inductive bias
6. NLP-01 TF-IDF and word embeddings before sequence architectures
7. DL-06 RNN/LSTM → DL-07 Attention → DL-08 Transformers
8. Pre-RAG language-model gate → `projects/tiny_language_model`
9. Digit-classifier mastery checkpoint with baseline, ablation, and teach-back
10. NLP-06 GPT/BERT/T5 model-family gate → `projects/transformer_families`

---

## Section 00 — Zero-Math and Programming Onboarding
- [x] PRE-01 — Mathematical Language and Arithmetic → `notebooks/00_prerequisites/01_math_language_and_arithmetic.ipynb`
- [x] PRE-02 — Algebra, Functions, and Graphs → `notebooks/00_prerequisites/02_algebra_functions_and_graphs.ipynb`
- [x] PRE-03 — Python, NumPy, and Jupyter Foundations → `notebooks/00_prerequisites/03_python_numpy_and_jupyter.ipynb`
- [x] PRE-04 — Calculus and Probability Intuition → `notebooks/00_prerequisites/04_calculus_and_probability.ipynb`
- [x] PRE-05 — Practical Python, Pandas, Debugging, and Tests → `notebooks/00_prerequisites/05_python_pandas_and_debugging.ipynb`

## Section 01 — Mathematical Foundations
- [x] FND-01 — Linear Algebra Essentials → `notebooks/01_ml_foundations/01_linear_algebra_essentials.ipynb`
- [x] FND-02 — Probability and Statistics → `notebooks/01_ml_foundations/02_probability_and_statistics.ipynb`
- [x] FND-03 — Data Workflow, EDA, Cleaning, Pandas, and SQL → `notebooks/01_ml_foundations/03_data_workflow_eda_and_cleaning.ipynb`
- [x] FND-04 — Optimization and Gradient Descent → `notebooks/01_ml_foundations/04_optimization_and_gradient_descent.ipynb`

## Section 02 — Classical Machine Learning
- [x] CML-01 — Linear Regression → `notebooks/02_classical_ml/01_linear_regression.ipynb`
- [x] CML-02 — Logistic Regression → `notebooks/02_classical_ml/02_logistic_regression.ipynb`
- [x] CML-03 — Decision Trees → `notebooks/02_classical_ml/03_decision_trees.ipynb`
- [x] CML-04 — Random Forest → `notebooks/02_classical_ml/04_random_forest.ipynb`
- [x] CML-05 — Gradient Boosting and XGBoost → `notebooks/02_classical_ml/05_gradient_boosting_and_xgboost.ipynb`

## Section 03 — ML Engineering Foundations
- [x] MLE-01 — Evaluation Metrics → `notebooks/03_ml_engineering/01_evaluation_metrics.ipynb`
- [x] MLE-02 — Validation and Data Leakage → `notebooks/03_ml_engineering/02_validation_and_data_leakage.ipynb`
- [x] MLE-03 — Feature Engineering → `notebooks/03_ml_engineering/03_feature_engineering.ipynb`
- [x] MLE-04 — Imbalanced Learning → `notebooks/03_ml_engineering/04_imbalanced_learning.ipynb`
- [x] MLE-05 — Explainability (SHAP) → `notebooks/03_ml_engineering/05_explainability_with_shap.ipynb`
- [x] MLE-06 — Unsupervised Learning Foundations → `notebooks/03_ml_engineering/06_unsupervised_learning_foundations.ipynb`

## Section 04 — Deep Learning Foundations
- [x] DL-01 — PyTorch Foundations and Training Loops → `notebooks/04_deep_learning/01_pytorch_foundations.ipynb`
- [x] DL-02 — Neural Networks from Scratch → `notebooks/04_deep_learning/02_neural_networks_from_scratch.ipynb`
- [x] DL-03 — Backpropagation → `notebooks/04_deep_learning/03_backpropagation.ipynb`
- [x] DL-04 — Stable Neural Training → `notebooks/04_deep_learning/04_stable_neural_training.ipynb`
- [x] DL-05 — CNN → `notebooks/04_deep_learning/05_convolutional_neural_networks.ipynb`
- [x] DL-06 — RNN and LSTM → `notebooks/04_deep_learning/06_rnn_and_lstm.ipynb`
- [x] DL-07 — Attention Mechanism → `notebooks/04_deep_learning/07_attention_mechanism.ipynb`
- [x] DL-08 — Transformers → `notebooks/04_deep_learning/08_transformers.ipynb`
- [x] Offline tiny language model — true causal training, validation, checkpointing,
  decoding comparisons, character-versus-BPE evaluation, automated tests, and human mastery assessment →
  `projects/tiny_language_model/`
  Includes correct per-layer KV caching, context-limit reset behavior, and measured
  cached-versus-naive inference.

## Section 05 — Modern NLP and LLMs
- [x] NLP-01 — TF-IDF and Word Embeddings → `notebooks/05_nlp_and_llms/01_tfidf_and_word_embeddings.ipynb`
- [x] NLP-06 — Transformer Model Families → `notebooks/05_nlp_and_llms/06_transformer_model_families.ipynb`
- [x] NLP-02 — Sentence Embeddings from Scratch → `notebooks/05_nlp_and_llms/02_sentence_embeddings.ipynb`
  Includes a local BERT-style bi-encoder, masked pooling, MNR loss, hard negatives,
  TF-IDF/untrained baselines, held-out retrieval metrics, behavioral tests, and a
  human mastery checkpoint → `projects/sentence_embeddings/`
- [x] NLP-03 — LLM Training Pipeline → `notebooks/05_nlp_and_llms/03_llm_training_pipeline.ipynb`
- [x] NLP-04 — Prompt Engineering → `notebooks/05_nlp_and_llms/04_prompt_engineering.ipynb`
- [x] NLP-05 — Hallucination and Guardrails → `notebooks/05_nlp_and_llms/05_hallucination_and_guardrails.ipynb`

## Section 06 — Retrieval-Augmented Generation
- [x] RAG-01 — Similarity Search → `notebooks/06_rag/01_similarity_search.ipynb`
- [x] RAG-02 — Chunking Strategies → `notebooks/06_rag/02_chunking_strategies.ipynb`
- [x] RAG-03 — Measured Baseline RAG Retrieval → `notebooks/06_rag/03_measured_retrieval_baseline.ipynb`
- [x] RAG-04 — Grounded Answer and Citation Evaluation → `notebooks/06_rag/04_grounded_answer_evaluation.ipynb`
- [x] RAG-05 — Vector Databases → `notebooks/06_rag/05_vector_databases.ipynb`
- [x] RAG-06 — Hybrid Search → `notebooks/06_rag/06_hybrid_search.ipynb`
- [x] RAG-07 — Reranking → `notebooks/06_rag/07_reranking.ipynb`
- [x] RAG-08 — Advanced RAG Architectures → `notebooks/06_rag/08_advanced_rag.ipynb`

## Section 07 — Agentic AI
- [x] AGT-01 — Agent Fundamentals → `notebooks/07_ai_agents/01_agent_fundamentals.ipynb`
- [x] AGT-02 — Planning and Tool Use → `notebooks/07_ai_agents/02_planning_and_tool_use.ipynb`
- [x] AGT-03 — Memory Systems → `notebooks/07_ai_agents/03_memory_systems.ipynb`
- [x] AGT-04 — Reflection and Self-Correction → `notebooks/07_ai_agents/04_reflection_and_self_correction.ipynb`
- [x] AGT-05 — Multi-Agent Systems → `notebooks/07_ai_agents/05_multi_agent_systems.ipynb`

## Section 08 — Evaluation
- [x] EVAL-01 — Classical ML Evaluation → `notebooks/08_evaluation/01_classical_ml_evaluation.ipynb`
- [x] EVAL-02 — LLM Evaluation → `notebooks/08_evaluation/02_llm_evaluation.ipynb`
- [x] EVAL-03 — RAG Evaluation → `notebooks/08_evaluation/03_rag_evaluation.ipynb`
- [x] EVAL-04 — Human Evaluation → `notebooks/08_evaluation/04_human_evaluation.ipynb`
- [x] EVAL-05 — LLM-as-a-Judge → `notebooks/08_evaluation/05_llm_as_a_judge.ipynb`

## Section 09 — Production AI
- [x] PROD-01 — MLOps → `notebooks/09_production_ml/01_mlops.ipynb`
- [x] PROD-02 — LLMOps → `notebooks/09_production_ml/02_llmops.ipynb`
- [x] PROD-03 — Feature Stores → `notebooks/09_production_ml/03_feature_stores.ipynb`
- [x] PROD-04 — Experiment Tracking → `notebooks/09_production_ml/04_experiment_tracking.ipynb`
- [x] PROD-05 — Monitoring and Drift Detection → `notebooks/09_production_ml/05_monitoring_and_drift.ipynb`
- [x] PROD-06 — Retraining Strategies → `notebooks/09_production_ml/06_retraining_strategies.ipynb`

## Section 10 — AI System Design
- [x] SYS-01 — Scalable ML Systems → `notebooks/10_system_design/01_scalable_ml_systems.ipynb`
- [x] SYS-02 — Production RAG Systems → `notebooks/10_system_design/02_production_rag.ipynb`
- [x] SYS-03 — AI Reliability Patterns → `notebooks/10_system_design/03_ai_reliability.ipynb`
- [x] SYS-04 — End-to-End AI Architecture → `notebooks/10_system_design/04_end_to_end_architecture.ipynb`

## Section 11 — Applied Capstone
- [x] CAP-01 — Deployable Real-Data ML Vertical Slice → `notebooks/11_capstone/01_deployable_wine_classifier.ipynb`
- [x] Project — Wine classifier training, API, monitoring, tests, and Docker → `projects/wine_classifier/`
