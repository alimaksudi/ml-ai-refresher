"""Builder for Notebook 37 — RAG Evaluation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 37 · RAG Evaluation
    ### Phase 7 — Evaluation · *ML/AI Senior Mastery Curriculum*

    > A RAG system can fail in two independent places: the **retriever** (wrong chunks
    > returned) or the **generator** (hallucination, irrelevance). Evaluating only the
    > final answer tells you *that* something failed but not *where*. This notebook
    > teaches component-level RAG evaluation — the RAGAS metric family — implemented
    > from scratch so you understand every formula, plus the failure taxonomy and
    > production quality-gate patterns.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **RAG failure taxonomy**: retrieval failure vs. generation failure — how to distinguish them.
    - **Context precision** (from scratch): of retrieved chunks, what fraction are relevant?
    - **Context recall** (from scratch): of all relevant chunks, what fraction were retrieved?
    - **Faithfulness** (from scratch): is every claim in the answer supported by the context?
    - **Answer relevance** (from scratch): does the answer actually address the question?
    - **Answer correctness**: against a gold-standard ground truth answer.
    - **RAGAS framework**: the complete metric suite — understand what each measures.
    - **Tracing failures** to source: build a diagnostic matrix (retrieval-ok × generation-ok).
    - **Production quality gate**: alert when any metric falls below SLA.

    **Why it matters**
    - Shipping a RAG system without component-level evaluation is flying blind. A system
      with great overall accuracy may have terrible faithfulness (the generator is inventing
      facts) — a liability risk. A system with poor context recall is systematically missing
      relevant information. Senior engineers can diagnose and fix RAG quality issues only
      if they measure the right things at the right level.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **RAGAS (Es et al., 2023).** "Ragas: Automated Evaluation of Retrieval Augmented
    Generation." Introduced a framework with four core metrics: Faithfulness, Answer
    Relevance, Context Precision, Context Recall. Each metric uses an LLM as the judge —
    but the formulas are deterministic given the judge's binary outputs, making them
    reliable and interpretable.

    **TruLens (TruEra, 2023).** Similar framework, adds "triad" evaluation: context
    relevance, groundedness, answer relevance. Strong emphasis on explainability.

    **ARES (Saad-Falcon et al., 2023).** Uses smaller fine-tuned classifiers instead of
    LLM judges for efficiency. Introduced the idea of domain-specific evaluation.

    **LangSmith, Langfuse (2023–2024).** Production tracing platforms that capture every
    RAG call (query, retrieved chunks, generated answer) for offline evaluation. Essential
    infrastructure for continuous monitoring.

    **Why automated evaluation?** Human evaluation is gold standard but expensive ($0.5–$5
    per example). LLM-as-judge is cheaper ($0.01–$0.05) and correlates 85–92% with human
    judgement on faithfulness and relevance metrics.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The RAG evaluation pipeline:**
    ```
    Query → [Retriever] → Context chunks → [Generator] → Answer
                ↓                               ↓
        Context Precision              Faithfulness
        Context Recall                 Answer Relevance
                    ↓___________________________↓
                         Answer Correctness (vs. ground truth)
    ```

    **Failure taxonomy — 2×2 diagnostic matrix:**
    ```
                       Generator
                   Good    |   Bad
    Retriever  ┌──────────────────────┐
    Good       │ ✓ Both work          │ Hallucination failure
               │ High precision +     │ (model ignores context or
               │ faithfulness         │ invents beyond it)
               ├──────────────────────┤
    Bad        │ Lucky guess          │ ✗ Complete failure
               │ (model knows from   │ (wrong chunks + generation
               │ parametric memory)   │ error)
               └──────────────────────┘
    ```

    **RAGAS metric intuition:**
    - **Context Precision@k**: of the k retrieved chunks, how many are actually useful? (Quality of retrieval)
    - **Context Recall**: of all facts in the ground truth, how many are covered by retrieved context? (Coverage)
    - **Faithfulness**: of all statements in the answer, how many are supported by context? (No hallucination)
    - **Answer Relevance**: does the answer address the actual question asked? (Not just tangentially related)
    """),

    code(r"""
    import re
    import math
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Context Precision@k

    Given retrieved chunks $C = \{c_1, \ldots, c_k\}$ with relevance labels $rel_i \in \{0,1\}$:

    $\text{Context Precision@k} = \frac{1}{k} \sum_{i=1}^k rel_i$

    RAGAS version (position-weighted, rewards relevant chunks ranked higher):

    $\text{CP@k} = \frac{\sum_{i=1}^k (rel_i \cdot \text{Prec@i})}{\text{number of relevant chunks in top-k}}$

    where $\text{Prec@i} = \frac{\sum_{j=1}^i rel_j}{i}$ (precision at position $i$).

    ### 4.2 Context Recall

    Given ground truth answer statements $S = \{s_1, \ldots, s_m\}$:

    $\text{Context Recall} = \frac{|\{s_i : s_i \text{ is attributable to some chunk } c_j\}|}{|S|}$

    "Attributable" = the statement can be inferred from the chunk.

    ### 4.3 Faithfulness

    Given generated answer with statements $A = \{a_1, \ldots, a_n\}$ and context $C$:

    $\text{Faithfulness} = \frac{|\{a_i : a_i \text{ is supported by } C\}|}{|A|}$

    A statement is "supported" if the context contains information that entails it
    (not necessarily verbatim — semantic entailment).

    ### 4.4 Answer Relevance

    Uses a reverse question generation approach: given answer $A$, generate $N$ questions
    $\{q_1, \ldots, q_N\}$ that the answer addresses, then measure their similarity to
    the original question $q$:

    $\text{Answer Relevance} = \frac{1}{N} \sum_{i=1}^N \cos(\mathbf{e}_{q_i}, \mathbf{e}_q)$

    High relevance: generated questions resemble the original question.

    ### 4.5 Answer Correctness

    Combines semantic similarity with factual overlap:

    $\text{Answer Correctness} = \alpha \cdot F1_{\text{factual}} + (1-\alpha) \cdot \cos(\mathbf{e}_{\text{answer}}, \mathbf{e}_{\text{ground truth}})$

    where $F1_{\text{factual}}$ is computed over extracted factual statements.
    """),

    md(r"""
    ## 5 · Mini RAG Pipeline + Evaluation from Scratch

    ### 5a — Synthetic corpus and mini RAG
    """),

    code(r"""
    # 5a. Synthetic knowledge base and minimal RAG pipeline for evaluation.

    CORPUS = [
        {'id': 0,  'text': 'NDCG@k measures ranking quality. DCG = sum(rel_i / log2(i+1)). IDCG is the ideal DCG.'},
        {'id': 1,  'text': 'BM25 uses TF saturation (k1=1.5) and length normalisation (b=0.75). Robertson IDF.'},
        {'id': 2,  'text': 'Cosine similarity: dot(a,b)/(|a||b|). Range -1 to 1. Scale-invariant.'},
        {'id': 3,  'text': 'HNSW builds a hierarchical navigable small world graph. Layers: sparse at top, dense at bottom.'},
        {'id': 4,  'text': 'RRF (Reciprocal Rank Fusion): score(d) = sum(1/(k+rank_r(d))). k=60 standard. Combines multiple ranked lists.'},
        {'id': 5,  'text': 'Attention mechanism: softmax(QK^T/sqrt(d_k))V. Multi-head: H parallel heads, concatenate, project.'},
        {'id': 6,  'text': 'Transformer positional encoding uses sine and cosine functions of different frequencies.'},
        {'id': 7,  'text': 'Adam optimizer: m_t = beta1*m_{t-1} + (1-beta1)*g_t. v_t = beta2*v_{t-1} + (1-beta2)*g_t^2.'},
        {'id': 8,  'text': 'Gradient descent update: theta = theta - lr * gradient(loss, theta).'},
        {'id': 9,  'text': 'Precision@k: of top-k retrieved docs, fraction that are relevant. Recall@k: of all relevant, fraction in top-k.'},
        {'id': 10, 'text': 'Cross-encoder: encodes query+doc jointly. Cannot pre-compute. High quality for reranking.'},
        {'id': 11, 'text': 'Bi-encoder: encodes query and doc independently. Can pre-compute doc embeddings. Fast retrieval.'},
        {'id': 12, 'text': 'HyDE: generate a hypothetical document that answers the query, then embed and search.'},
        {'id': 13, 'text': 'Chunking strategies: fixed-size with overlap, sentence-boundary, recursive, semantic (cosine threshold).'},
        {'id': 14, 'text': 'Vector database adds persistence, CRUD, metadata filtering, and multi-tenancy over FAISS.'},
        {'id': 15, 'text': 'Calibration: a model is calibrated if P(correct | score=0.8) = 0.8. Brier score and ECE measure this.'},
        {'id': 16, 'text': 'ROC curve plots TPR vs FPR at all thresholds. AUC = probability model ranks positive above negative.'},
        {'id': 17, 'text': 'F1 score: harmonic mean of precision and recall. F1 = 2*P*R/(P+R).'},
        {'id': 18, 'text': 'Parent-child chunking: retrieve small children for precision, return large parent for generation context.'},
        {'id': 19, 'text': 'ReAct agents interleave Thought / Action / Observation steps in a loop until task complete.'},
    ]

    # Simple embedding: deterministic hash-based (stand-in for sentence transformer).
    def embed(text, dim=16):
        rng_e = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng_e.standard_normal(dim)
        return v / (np.linalg.norm(v) + 1e-9)

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # Pre-embed corpus.
    for doc in CORPUS:
        doc['embedding'] = embed(doc['text'])

    def retrieve(query, k=3):
        q_emb = embed(query)
        scored = [(cosine_sim(q_emb, d['embedding']), d) for d in CORPUS]
        scored.sort(key=lambda x: -x[0])
        return [(d, round(s, 3)) for s, d in scored[:k]]

    def generate_answer(query, context_chunks):
        # Simulated answer generation: extract key phrases from top chunk.
        if not context_chunks:
            return 'I do not have information about that.'
        top_text = context_chunks[0][0]['text']
        return f'Based on the retrieved context: {top_text[:120]}...'

    print(f'Corpus: {len(CORPUS)} documents.')
    print('Mini RAG demo:')
    q = 'How does NDCG work?'
    chunks = retrieve(q, k=3)
    answer = generate_answer(q, chunks)
    print(f'  Query: {q}')
    print(f'  Retrieved chunks:')
    for doc, score in chunks:
        print(f'    [{score:.3f}] {doc["text"][:65]}...')
    print(f'  Answer: {answer[:80]}...')
    """),

    md(r"""
    ### 5b — Context Precision from scratch
    """),

    code(r"""
    # 5b. Context Precision: of retrieved chunks, fraction that are relevant.

    def context_precision_at_k(retrieved_chunks, relevant_ids):
        # Simple: fraction of retrieved chunks whose id is in relevant_ids.
        relevant_ids_set = set(relevant_ids)
        n_relevant = sum(1 for doc, _ in retrieved_chunks if doc['id'] in relevant_ids_set)
        return n_relevant / len(retrieved_chunks) if retrieved_chunks else 0.0

    def context_precision_ragas(retrieved_chunks, relevant_ids):
        # RAGAS version: position-weighted (average precision style).
        relevant_ids_set = set(relevant_ids)
        n_relevant_found = 0
        precision_sum = 0.0
        for i, (doc, _) in enumerate(retrieved_chunks):
            if doc['id'] in relevant_ids_set:
                n_relevant_found += 1
                precision_sum += n_relevant_found / (i + 1)
        if n_relevant_found == 0:
            return 0.0
        return precision_sum / n_relevant_found

    # Test questions with ground-truth relevant doc IDs.
    test_cases_cp = [
        {'query': 'How does NDCG work?',           'relevant_ids': [0, 9]},
        {'query': 'What is cosine similarity?',     'relevant_ids': [2, 11]},
        {'query': 'How do transformers use attention?', 'relevant_ids': [5, 6]},
        {'query': 'Explain BM25 scoring',           'relevant_ids': [1, 4]},
        {'query': 'What is chunking in RAG?',       'relevant_ids': [13, 18]},
    ]

    print('Context Precision evaluation:')
    print(f'  {"Query":40s} {"CP@3 simple":12s} {"CP@3 RAGAS":12s}')
    cp_simple_scores, cp_ragas_scores = [], []
    for tc in test_cases_cp:
        chunks = retrieve(tc['query'], k=3)
        cp_s = context_precision_at_k(chunks, tc['relevant_ids'])
        cp_r = context_precision_ragas(chunks, tc['relevant_ids'])
        cp_simple_scores.append(cp_s)
        cp_ragas_scores.append(cp_r)
        print(f'  {tc["query"][:40]:40s} {cp_s:.3f}        {cp_r:.3f}')
    print(f'\n  Mean CP (simple): {np.mean(cp_simple_scores):.3f}')
    print(f'  Mean CP (RAGAS):  {np.mean(cp_ragas_scores):.3f}')
    """),

    md(r"""
    ### 5c — Context Recall from scratch
    """),

    code(r"""
    # 5c. Context Recall: of ground-truth statements, fraction covered by retrieved context.

    def statement_in_context(statement, context_texts, sim_threshold=0.35):
        # Check if a statement is semantically covered by any context chunk.
        s_emb = embed(statement)
        for ctx_text in context_texts:
            if cosine_sim(s_emb, embed(ctx_text)) > sim_threshold:
                return True
        return False

    def context_recall(ground_truth_statements, retrieved_chunks, threshold=0.35):
        context_texts = [doc['text'] for doc, _ in retrieved_chunks]
        covered = sum(1 for s in ground_truth_statements
                      if statement_in_context(s, context_texts, threshold))
        return covered / len(ground_truth_statements) if ground_truth_statements else 0.0

    # Test cases with ground truth statements.
    test_cases_cr = [
        {
            'query': 'How does NDCG work?',
            'ground_truth_statements': [
                'NDCG stands for Normalised Discounted Cumulative Gain.',
                'DCG sums relevance divided by log of position.',
                'NDCG normalises DCG by ideal DCG.',
            ],
            'relevant_ids': [0, 9],
        },
        {
            'query': 'Explain Adam optimizer',
            'ground_truth_statements': [
                'Adam combines momentum (first moment) and RMSProp (second moment).',
                'Adam uses exponential moving averages of gradients.',
                'beta1 controls first moment, beta2 controls second moment.',
            ],
            'relevant_ids': [7],
        },
    ]

    print('Context Recall evaluation:')
    cr_scores = []
    for tc in test_cases_cr:
        chunks = retrieve(tc['query'], k=3)
        cr = context_recall(tc['ground_truth_statements'], chunks)
        cr_scores.append(cr)
        print(f'\n  Query: "{tc["query"]}"')
        print(f'  Ground truth statements: {len(tc["ground_truth_statements"])}')
        for s in tc['ground_truth_statements']:
            ctx_texts = [d['text'] for d, _ in chunks]
            covered = statement_in_context(s, ctx_texts)
            print(f'    {"[COVERED]" if covered else "[MISSING]":10s} {s[:60]}...')
        print(f'  Context Recall: {cr:.3f}')
    """),

    md(r"""
    ### 5d — Faithfulness from scratch
    """),

    code(r"""
    # 5d. Faithfulness: of claims in the answer, fraction supported by context.

    def extract_statements(text):
        # Simulate NLP statement extraction: split on sentences.
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def statement_supported_by_context(statement, context_texts, threshold=0.30):
        s_emb = embed(statement)
        return any(cosine_sim(s_emb, embed(c)) > threshold for c in context_texts)

    def faithfulness(answer, retrieved_chunks):
        context_texts = [doc['text'] for doc, _ in retrieved_chunks]
        statements = extract_statements(answer)
        if not statements:
            return 1.0, []   # No statements to check — vacuously faithful.
        supported = []
        for stmt in statements:
            is_supported = statement_supported_by_context(stmt, context_texts)
            supported.append((stmt, is_supported))
        score = sum(1 for _, s in supported if s) / len(supported)
        return score, supported

    # Test answers: one faithful, one hallucinating.
    faithful_answer = (
        'NDCG measures ranking quality. DCG is the sum of relevance divided by log2 of position. '
        'NDCG normalises DCG by the ideal DCG. Cosine similarity measures angle between vectors.'
    )
    hallucinating_answer = (
        'NDCG was invented in 1998 by researchers at Stanford. '
        'It uses a complex neural network to compute relevance scores. '
        'The formula involves a special transformer attention mechanism.'
    )

    print('Faithfulness evaluation:')
    for label, ans in [('Faithful answer', faithful_answer), ('Hallucinating answer', hallucinating_answer)]:
        chunks = retrieve('How does NDCG work?', k=3)
        score, details = faithfulness(ans, chunks)
        print(f'\n  {label}: faithfulness={score:.3f}')
        for stmt, supported in details:
            print(f'    {"[OK]  " if supported else "[FAIL]"} {stmt[:65]}')
    """),

    md(r"""
    ### 5e — Answer Relevance from scratch
    """),

    code(r"""
    # 5e. Answer Relevance: do the questions the answer implies match the original query?

    def generate_questions_from_answer(answer, n=3):
        # Simulate LLM generating questions that the answer addresses.
        # In production: call LLM with: "Generate N questions that this answer addresses."
        sentences = extract_statements(answer)
        questions = []
        for s in sentences[:n]:
            # Simple heuristic: turn statement into question form.
            q = s.replace('is', 'What is').replace('uses', 'How does it use')
            questions.append(q + '?' if not q.endswith('?') else q)
        # Pad if not enough.
        while len(questions) < n:
            questions.append(f'What additional context is provided?')
        return questions[:n]

    def answer_relevance(query, answer, n_questions=3):
        q_emb = embed(query)
        generated_qs = generate_questions_from_answer(answer, n=n_questions)
        similarities = [cosine_sim(q_emb, embed(gq)) for gq in generated_qs]
        score = float(np.mean(similarities))
        return score, generated_qs, similarities

    test_qa_pairs = [
        ('What is BM25?',
         'BM25 is a bag-of-words retrieval function. It uses TF saturation and length normalisation.'),
        ('How do I deploy a model to production?',
         'The cosine similarity between two vectors is the dot product divided by the product of their norms.'),
    ]

    print('Answer Relevance evaluation:')
    for query, answer in test_qa_pairs:
        score, gen_qs, sims = answer_relevance(query, answer)
        print(f'\n  Query:  "{query}"')
        print(f'  Answer: "{answer[:60]}..."')
        print(f'  Generated questions vs. original:')
        for gq, sim in zip(gen_qs, sims):
            print(f'    [{sim:.3f}] {gq[:60]}')
        print(f'  Answer Relevance: {score:.3f}')
    """),

    md(r"""
    ### 5f — Full RAGAS suite evaluation on test set
    """),

    code(r"""
    # 5f. Evaluate full RAGAS suite on a mini test set.

    TEST_SET = [
        {
            'query': 'How does NDCG measure ranking quality?',
            'ground_truth': 'NDCG normalises DCG by ideal DCG. DCG sums relevance divided by log2 of position.',
            'gt_statements': ['DCG sums relevance by log position.', 'NDCG normalises by ideal ranking.'],
            'relevant_ids': [0, 9],
        },
        {
            'query': 'What is cosine similarity?',
            'ground_truth': 'Cosine similarity is the dot product divided by the product of norms. Range -1 to 1.',
            'gt_statements': ['Cosine similarity uses dot product and norms.', 'Range is -1 to 1.'],
            'relevant_ids': [2],
        },
        {
            'query': 'How does the attention mechanism work in transformers?',
            'ground_truth': 'Attention computes softmax(QK^T/sqrt(d_k))V. Multi-head attention uses H parallel heads.',
            'gt_statements': ['Attention uses Q, K, V matrices.', 'Multi-head attention uses parallel heads.'],
            'relevant_ids': [5, 6],
        },
        {
            'query': 'What is HyDE?',
            'ground_truth': 'HyDE generates a hypothetical document that answers the query then embeds it for search.',
            'gt_statements': ['HyDE generates a hypothetical document.', 'It then embeds the document for retrieval.'],
            'relevant_ids': [12],
        },
        {
            'query': 'Explain chunking strategies for RAG',
            'ground_truth': 'Chunking strategies include fixed-size, sentence-boundary, recursive, and semantic approaches.',
            'gt_statements': ['Fixed-size chunking uses overlap.', 'Semantic chunking uses cosine distance threshold.'],
            'relevant_ids': [13, 18],
        },
    ]

    results_ragas = []
    for tc in TEST_SET:
        chunks = retrieve(tc['query'], k=3)
        answer = generate_answer(tc['query'], chunks)

        cp = context_precision_ragas(chunks, tc['relevant_ids'])
        cr = context_recall(tc['gt_statements'], chunks)
        faith, _ = faithfulness(answer, chunks)
        ar, _, _ = answer_relevance(tc['query'], answer)

        results_ragas.append({'query': tc['query'], 'cp': cp, 'cr': cr, 'faithfulness': faith, 'ar': ar})

    print('RAGAS Suite Results:')
    print(f'  {"Query":42s} {"CP":6s} {"CR":6s} {"Faith":6s} {"AR":6s}')
    print('  ' + '-'*70)
    for r in results_ragas:
        print(f'  {r["query"][:42]:42s} {r["cp"]:.3f}  {r["cr"]:.3f}  {r["faithfulness"]:.3f}  {r["ar"]:.3f}')
    print('  ' + '-'*70)
    means = {k: np.mean([r[k] for r in results_ragas]) for k in ['cp', 'cr', 'faithfulness', 'ar']}
    print(f'  {"Mean":42s} {means["cp"]:.3f}  {means["cr"]:.3f}  {means["faithfulness"]:.3f}  {means["ar"]:.3f}')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — RAGAS metric radar chart.
    metrics_names = ['Context\nPrecision', 'Context\nRecall', 'Faithfulness', 'Answer\nRelevance']
    metric_keys = ['cp', 'cr', 'faithfulness', 'ar']
    mean_vals = [means[k] for k in metric_keys]

    # Simulate two systems for comparison.
    system_a = mean_vals
    system_b = [max(0, v - 0.12 + rng.normal(0, 0.04)) for v in mean_vals]

    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})
    for vals, label, color in [(system_a + [system_a[0]], 'System A (hybrid)', 'steelblue'),
                               (system_b + [system_b[0]], 'System B (BM25-only)', 'coral')]:
        ax.plot(angles, vals, 'o-', lw=2, label=label, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Figure 1 — RAGAS radar chart: System A vs. System B', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 1.** RAGAS radar chart comparing two RAG systems. The outer ring = perfect
    score (1.0). System A (hybrid retrieval, blue) outperforms System B (BM25-only, red)
    across all four dimensions. The most informative comparison: **Context Recall** — if
    System B has low CR but high Faithfulness, the retriever is missing relevant chunks but
    the generator is not hallucinating; fix: improve retrieval. If high CR but low Faithfulness,
    the generator is hallucinating despite good retrieval; fix: constrain the generator.
    """),

    code(r"""
    # Figure 2 — Failure taxonomy heatmap.
    # Simulate 20 RAG evaluations with different failure modes.
    n_evals = 20
    retrieval_ok = rng.random(n_evals) > 0.35   # ~65% retrieval OK
    generation_ok = np.where(retrieval_ok, rng.random(n_evals) > 0.2, rng.random(n_evals) > 0.6)

    categories = {
        'Both OK': int(np.sum(retrieval_ok & generation_ok)),
        'Retrieval fail': int(np.sum(~retrieval_ok & generation_ok)),
        'Generation fail (hallucination)': int(np.sum(retrieval_ok & ~generation_ok)),
        'Both fail': int(np.sum(~retrieval_ok & ~generation_ok)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_tax = ['seagreen', 'orange', 'coral', 'red']
    axes[0].bar(range(len(categories)), list(categories.values()), color=colors_tax, alpha=0.8)
    axes[0].set_xticks(range(len(categories)))
    axes[0].set_xticklabels(list(categories.keys()), rotation=20, ha='right', fontsize=9)
    axes[0].set_ylabel('Count (out of 20 queries)')
    axes[0].set_title('Figure 2a — RAG failure taxonomy')
    for i, (k, v) in enumerate(categories.items()):
        axes[0].text(i, v + 0.3, str(v), ha='center', fontsize=11, fontweight='bold')

    # CP vs Faithfulness scatter — shows which quadrant each query falls in.
    sim_cp = np.where(retrieval_ok, rng.uniform(0.6, 1.0, n_evals), rng.uniform(0.0, 0.4, n_evals))
    sim_faith = np.where(generation_ok, rng.uniform(0.6, 1.0, n_evals), rng.uniform(0.0, 0.4, n_evals))
    scatter_colors = ['seagreen' if r and g else 'orange' if not r and g else 'coral' if r and not g else 'red'
                      for r, g in zip(retrieval_ok, generation_ok)]
    axes[1].scatter(sim_cp, sim_faith, c=scatter_colors, s=80, alpha=0.8)
    axes[1].axvline(0.5, color='gray', ls='--', alpha=0.5)
    axes[1].axhline(0.5, color='gray', ls='--', alpha=0.5)
    axes[1].set_xlabel('Context Precision'); axes[1].set_ylabel('Faithfulness')
    axes[1].set_title('Figure 2b — Diagnostic scatter: retrieval vs. generation quality')
    axes[1].text(0.75, 0.75, 'Both OK', ha='center', fontsize=9, color='seagreen')
    axes[1].text(0.15, 0.75, 'Retrieval fail', ha='center', fontsize=9, color='orange')
    axes[1].text(0.75, 0.20, 'Generation fail', ha='center', fontsize=9, color='coral')
    axes[1].text(0.15, 0.20, 'Both fail', ha='center', fontsize=9, color='red')
    plt.suptitle('Figure 2 — RAG failure taxonomy and diagnostic scatter')
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 2.** RAG failure taxonomy. Left: count of each failure mode across 20 test queries.
    Right: scatter plot of Context Precision (retrieval quality) vs. Faithfulness (generation
    quality). The four quadrants of the diagnostic scatter are the key diagnostic tool:
    - **Top-right (green)**: both retrieval and generation work → system is healthy.
    - **Top-left (orange)**: retrieval fails but generator makes do → low context precision,
      generator may be relying on parametric knowledge (dangerous).
    - **Bottom-right (red)**: retrieval works but generator hallucinates → check generation
      prompt, temperature, output constraints.
    - **Bottom-left (dark red)**: complete failure → start with retrieval debugging.
    """),

    code(r"""
    # Figure 3 — RAGAS metrics for each test query.
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results_ragas))
    w = 0.2
    labels_q = [r['query'][:25]+'...' for r in results_ragas]
    metric_plot = [('cp', 'Context Precision', 'steelblue'),
                   ('cr', 'Context Recall', 'seagreen'),
                   ('faithfulness', 'Faithfulness', 'coral'),
                   ('ar', 'Answer Relevance', 'purple')]
    for i, (key, label, color) in enumerate(metric_plot):
        vals = [r[key] for r in results_ragas]
        ax.bar(x + (i - 1.5)*w, vals, w, label=label, color=color, alpha=0.8)
    ax.axhline(0.7, color='red', ls='--', alpha=0.5, label='SLA threshold (0.70)')
    ax.set_xticks(x); ax.set_xticklabels(labels_q, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('RAGAS score'); ax.set_ylim(0, 1.15); ax.legend(fontsize=9)
    ax.set_title('Figure 3 — RAGAS scores per query (bar chart)')
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 3.** RAGAS scores broken down by metric and query. Bars below the SLA
    threshold (red dashed line at 0.70) indicate metric failures that should trigger
    investigation. Different queries have different failure patterns:
    - Low CP + low CR → retrieval configuration issue (embedding model, chunk size, k).
    - Low Faithfulness only → generator configuration issue (temperature, system prompt, context window).
    - Low AR only → query-answer mismatch (answer drifted from the question).
    Use this per-query breakdown to diagnose and prioritise improvements.
    """),

    code(r"""
    # Figure 4 — RAGAS metric correlations.
    # Simulate 50 data points across the test suite.
    n_sim = 50
    sim_cp_all   = rng.uniform(0.2, 1.0, n_sim)
    sim_cr_all   = sim_cp_all * 0.85 + rng.normal(0, 0.1, n_sim)   # cp and cr correlated
    sim_faith    = 0.6 + 0.3 * sim_cp_all + rng.normal(0, 0.12, n_sim)   # faithfulness improves with better retrieval
    sim_ar       = rng.uniform(0.4, 0.95, n_sim)   # answer relevance mostly independent

    sim_cr_all   = np.clip(sim_cr_all, 0, 1)
    sim_faith    = np.clip(sim_faith, 0, 1)

    pairs = [('Context Precision', sim_cp_all, 'Context Recall', sim_cr_all),
             ('Context Precision', sim_cp_all, 'Faithfulness',   sim_faith)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xlab, xval, ylab, yval) in zip(axes, pairs):
        ax.scatter(xval, yval, alpha=0.6, color='steelblue', s=40)
        m, b = np.polyfit(xval, yval, 1)
        x_line = np.linspace(0, 1, 50)
        ax.plot(x_line, m*x_line + b, 'r-', alpha=0.7)
        corr = float(np.corrcoef(xval, yval)[0, 1])
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f'r = {corr:.2f}')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle('Figure 4 — RAGAS metric correlations')
    plt.tight_layout(); plt.show()
    print(f'CP-CR correlation:   {np.corrcoef(sim_cp_all, sim_cr_all)[0,1]:.3f}')
    print(f'CP-Faith correlation: {np.corrcoef(sim_cp_all, sim_faith)[0,1]:.3f}')
    """),

    md(r"""
    **Figure 4.** RAGAS metric correlations. **Context Precision and Context Recall** are
    strongly positively correlated — a better retriever tends to both retrieve more relevant
    chunks (precision) and cover more of the relevant chunks (recall). This is expected:
    both measure retrieval quality from complementary angles. **Context Precision and
    Faithfulness** are moderately correlated — better retrieval gives the generator better
    material to work with, reducing hallucination. However, the correlation is not 1.0:
    even with perfect retrieval, the generator can still hallucinate. This justifies
    measuring faithfulness independently of retrieval quality.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **LLM judge inconsistency** | Same answer scored differently across runs | High temperature judge; non-deterministic | Use temperature=0 for judge; cache evaluations |
    | **Evaluation data contamination** | Evaluation queries in training data | Test set not held out | Use strictly held-out queries; date-split for time-series |
    | **Faithfulness false positives** | Hallucinated claim not caught | Statement extraction too coarse; threshold too low | Lower similarity threshold; use dedicated NLI model |
    | **Context Recall underestimates** | Low CR even though answer is correct | Embedding model doesn't align text and statements | Use same embedding model as retriever; phrase statements like chunks |
    | **Answer Relevance saturates** | AR always high regardless of answer quality | Generated questions too generic | Use structured reverse-question generation prompt |
    | **Metric gaming** | System optimises CP by over-filtering chunks | Optimises for eval metric not user value | Pair with human evaluation; use blind held-out set |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 RAGAS library (guarded).
    try:
        import ragas  # noqa: F401
        print('RAGAS available.')
    except ImportError:
        lines = [
            '[ragas not installed — production RAGAS pattern]:',
            '  from ragas import evaluate',
            '  from ragas.metrics import (faithfulness, answer_relevancy,',
            '      context_precision, context_recall)',
            '',
            '  # Build evaluation dataset.',
            '  from datasets import Dataset',
            '  eval_data = Dataset.from_dict({',
            '      "question": [q for q in queries],',
            '      "answer": [a for a in answers],',
            '      "contexts": [[c for c in chunks] for chunks in all_chunks],',
            '      "ground_truth": [gt for gt in ground_truths],',
            '  })',
            '',
            '  # Evaluate.',
            '  result = evaluate(eval_data, metrics=[',
            '      faithfulness, answer_relevancy,',
            '      context_precision, context_recall,',
            '  ])',
            '  print(result)  # Returns DataFrame with per-query scores',
        ]
        print('\n'.join(lines))
    """),

    code(r"""
    # 8.2 Langfuse tracing (guarded).
    try:
        from langfuse import Langfuse  # noqa: F401
        print('Langfuse available.')
    except ImportError:
        lines = [
            '[langfuse not installed — tracing pattern]:',
            '  from langfuse import Langfuse',
            '  langfuse = Langfuse(public_key="pk-...", secret_key="sk-...")',
            '',
            '  # In your RAG handler:',
            '  trace = langfuse.trace(name="rag-query")',
            '  retrieval_span = trace.span(name="retrieval", input={"query": query})',
            '  chunks = retriever.retrieve(query)',
            '  retrieval_span.end(output={"chunks": [c.text for c in chunks]})',
            '',
            '  generation_span = trace.span(name="generation", input={"context": context})',
            '  answer = generator.generate(query, context)',
            '  generation_span.end(output={"answer": answer})',
            '',
            '  # Score via RAGAS offline on logged traces:',
            '  trace.score(name="faithfulness", value=0.92)',
            '  trace.score(name="context_precision", value=0.85)',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Business Case Study — Customer Support Chatbot Quality Gate

    **Scenario.** An e-commerce company deploys a RAG chatbot for customer support
    (order status, returns, policies). 5,000 queries/day. They set RAGAS-based quality
    gates to catch degradation before customers notice.

    **Quality gate SLAs:**
    - Faithfulness ≥ 0.90 (legal requirement: no false claims about policies).
    - Context Precision ≥ 0.70 (retrieval must be relevant).
    - Context Recall ≥ 0.75 (must not miss relevant policy sections).
    - Answer Relevance ≥ 0.80 (must answer the actual question).

    **Alerting rules:**
    - Any metric below SLA for > 50 consecutive queries → alert PagerDuty.
    - Faithfulness below 0.80 (hard floor) on any single query → log + human review.

    **Evaluation cadence:**
    - Continuous: every query evaluated by lightweight RAGAS-style scorer ($0.002/query).
    - Weekly: sample 200 queries for full LLM-as-judge evaluation ($0.05/query).
    - Monthly: 50 queries sent for human evaluation ($2/query).

    **Results after 3 months:**
    - Caught 3 retrieval degradation incidents (chunk size changes broke context recall).
    - Caught 1 generation issue (temperature bump from 0 to 0.7 increased hallucination rate).
    - Mean time to detect: 15 minutes (vs. days for customer complaint-based detection).
    - Monthly evaluation cost: $450 (vs. $15K for full human evaluation at same coverage).
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Evaluation dataset curation.** Build a golden evaluation set of 200–500 queries
      with verified ground truths. Cover: typical queries (60%), edge cases (20%), adversarial
      queries (20% — queries where the system is expected to decline or acknowledge uncertainty).
      Refresh quarterly as product evolves.
    - **Metric latency.** LLM-based RAGAS evaluation adds 1–2 seconds per query. For
      real-time evaluation (every production query): use lightweight classifiers (fine-tuned
      small model for faithfulness) with < 50ms latency. Use full LLM evaluation on samples.
    - **Metric drift.** RAGAS scores can drift if the LLM judge model is updated. Pin the
      judge model version; re-calibrate when changing.
    - **False negative risk.** Faithfulness score of 0.85 doesn't mean 85% of statements
      are correct — it means the scorer found 85% supported. Scorer errors are especially
      dangerous for sensitive domains (medical, legal, financial). Always pair with human
      spot-checks.
    - **Versioned evaluation.** Tie every evaluation run to a system version (embedding
      model version, chunk config, retriever config, LLM version). Without versioning, you
      cannot attribute metric changes to specific system changes.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **RAG evaluation strategy comparison:**

    | Method | Cost/query | Latency | Accuracy vs. human | Scale |
    |---|---|---|---|---|
    | Human evaluation | $1–$5 | 1–2 days | 100% (gold) | 50–200 queries |
    | LLM-as-judge (RAGAS) | $0.02–$0.10 | 2–5s | 85–92% | Thousands |
    | Fine-tuned classifier | $0.001–$0.005 | < 100ms | 78–85% | Millions |
    | Embedding similarity only | $0.0001 | < 10ms | 65–75% | Unlimited |

    **Which RAGAS metric to prioritise:**
    - Safety-critical (medical, legal, financial): Faithfulness is paramount.
    - Information retrieval products: Context Precision + Context Recall (retrieval quality).
    - Conversational assistants: Answer Relevance (user experience).
    - Knowledge-intensive tasks: Answer Correctness vs. ground truth.
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What are the four RAGAS metrics and what does each measure?"* → Context Precision
      (of retrieved chunks, what fraction are relevant?), Context Recall (of all relevant
      chunks, what fraction were retrieved?), Faithfulness (of claims in the answer, what
      fraction are supported by context?), Answer Relevance (does the answer address the
      actual question?). Together they cover both retrieval quality and generation quality.
    - *"How do you distinguish a retrieval failure from a generation failure?"* → Use the
      2×2 diagnostic matrix: measure Context Precision (retrieval quality) and Faithfulness
      (generation quality) independently. Low CP + high Faithfulness = retrieval problem
      (bad chunks, but generator is not hallucinating). High CP + low Faithfulness =
      generation problem (good chunks, generator is hallucinating). Low both = systemic failure.

    **Deep-dive questions**
    - *"What is faithfulness and how is it computed without a reference answer?"* →
      Faithfulness decomposes the generated answer into individual statements, then checks
      each statement against the retrieved context. "Supported" = the context entails the
      statement (semantic entailment, not verbatim match). No reference answer needed —
      only the generated answer and the retrieved context. This is what makes RAGAS practical
      for production monitoring where ground truths are expensive to collect.
    - *"Why is context recall hard to measure accurately?"* → Context Recall requires knowing
      ALL relevant chunks for a query — the "complete ground truth retrieval set". This is
      expensive to annotate (requires domain experts to read all possible chunks). In practice:
      use a small annotated set; measure recall only against annotated relevant chunks; accept
      that recall is a lower bound (un-annotated relevant chunks won't be counted).

    **Common mistakes:** evaluating end-to-end accuracy only (misses component failures);
    using default threshold=0.5 for all metrics (faithfulness SLA should be 0.9 not 0.7 for
    high-stakes domains); not versioning the judge model (metric drift misattributed to system
    changes); no human spot-checks (false negatives in faithfulness scoring).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **RAGAS four metrics.** Define each. Which metric does NOT require a ground-truth answer?
    2. **Failure taxonomy.** Draw the 2×2 matrix. What does "retrieval OK, generation fail" mean in production terms?
    3. **Faithfulness formula.** Write it. Can faithfulness be 1.0 if the answer is factually wrong?
    4. **Context Recall vs. Precision.** Give a scenario where CP is high but CR is low. Is that better or worse than high CR and low CP?
    5. **Answer Relevance method.** Explain the reverse-question-generation approach. Why is this smarter than comparing answer to query directly?
    6. **Production SLA.** Faithfulness SLA = 0.90. In 1,000 queries, faithfulness averages 0.88. Alert or not? What would you investigate?
    7. **Judge model.** What happens if the faithfulness judge model is updated (new version)? Why is this a problem for trend monitoring?
    8. **Cost analysis.** You have 10K queries/day. LLM judge costs $0.05/query. Fine-tuned classifier costs $0.002/query with 80% accuracy. Design an evaluation strategy targeting < $200/day.
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. A RAG system has CP=0.9 but CR=0.3. Interpret this. What does it mean for the user experience?
    2. A chatbot achieves Faithfulness=0.95 but Answer Relevance=0.50. What is the user likely experiencing?

    **Beginner → Intermediate (coding)**
    3. Implement `answer_correctness` from scratch: compare the generated answer to a ground-truth answer using token-level F1 (intersection of tokens / union). Test on 5 query-answer pairs.
    4. Implement a **RAGAS quality gate**: given a dict of metrics and SLA thresholds, return a pass/fail decision and a list of violations. Test it triggers correctly on failing inputs.

    **Intermediate (analysis)**
    5. Vary k (number of retrieved chunks) from 1 to 10. Plot Context Precision and Context Recall vs. k. At what k does precision start to degrade? At what k does recall saturate? What is the optimal k?
    6. Compare two embedding models (simulate by adding different noise levels). Which RAGAS metric changes most? Which is most sensitive to embedding quality?

    **Senior (design)**
    7. *System design:* design a continuous RAG evaluation system for a legal document chatbot. 1,000 queries/day. Faithfulness SLA=0.95 (legal liability). Design: evaluation cadence, metrics, alerting thresholds, human review workflow, evaluation dataset refresh schedule, cost budget.
    8. *Interview:* "Our RAG chatbot has Faithfulness=0.82 which is below our 0.90 SLA. Context Precision is 0.88. What are the top 3 things you'd change?" (Expected: constrain generator temperature → 0; add citation requirement to system prompt; filter out retrieved chunks with score < 0.4; evaluate whether the embedding model aligns with the domain vocabulary.)
    """),

    md(r"""
    ---
    ### Summary
    RAG evaluation requires measuring both retrieval and generation independently.
    **Context Precision** and **Context Recall** measure retrieval quality from complementary
    angles. **Faithfulness** catches generation hallucinations without needing a reference
    answer. **Answer Relevance** catches query drift. The 2×2 diagnostic matrix (retrieval
    OK × generation OK) is the key diagnostic tool for triaging failures. In production:
    continuous lightweight evaluation + weekly LLM-as-judge sampling + monthly human spot-checks.

    **Next:** `38 · LLM Evaluation` — evaluating LLM outputs beyond RAG: BLEU, ROUGE-L,
    BERTScore, perplexity, pass@k, and benchmark-based evaluation.
    """),
]

build("phase7_evaluation/37_rag_evaluation.ipynb", cells)
