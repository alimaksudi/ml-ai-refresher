"""Builder for Notebook 28 — Reranking."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 28 · Reranking
    ### Phase 5 — Retrieval-Augmented Generation · *ML/AI Senior Mastery Curriculum*

    > Notebooks 25–27 built fast retrieval systems that maximise **Recall@100** —
    > retrieving a large candidate set with high probability of including the relevant
    > documents. But returning 100 documents to the user (or the LLM) is wasteful.
    > A **reranker** is a slower, more accurate model that takes the top-100 candidates
    > and re-scores them to produce a precise top-10. This two-stage architecture is
    > the backbone of every production RAG system.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Two-stage retrieval architecture**: why split into recall stage and precision stage.
    - **Bi-encoder vs cross-encoder**: the fundamental difference in how they attend to
      query and document; why cross-encoders are more accurate but slower.
    - **Cross-encoder scoring from scratch**: concatenate query+document, compute a
      relevance score. Understand the computational asymmetry.
    - **Full reranking pipeline from scratch**: BM25 first stage → dense second stage
      (optional) → cross-encoder reranker → precision@k evaluation.
    - **Latency budget analysis**: how to fit reranking within a 200ms SLA.
    - **Production reranking**: Cohere Rerank API and sentence-transformers
      CrossEncoder (guarded imports).
    - **When to rerank and when not to**: cost-benefit for different corpus sizes and
      quality requirements.

    **Why it matters**
    - NDCG@10 (ordering quality) on a typical RAG task improves 15–30% when a
      cross-encoder reranker is added over a bi-encoder first stage. For legal
      document search or medical Q&A, the difference between the correct answer at
      position 1 vs. position 10 is critical. Reranking is the highest-ROI single
      improvement to a deployed RAG pipeline.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Traditional multi-stage IR (1980s–2000s).** Cascade retrieval was standard in
    information retrieval before neural models: fast BM25 → slower feature-based
    learning-to-rank (LambdaMART, RankBoost) → final scoring. The insight that
    "you can afford to be slow on a small candidate set" predates transformers.

    **BERT cross-encoders (Nogueira & Cho, 2019).** The first paper to use BERT as
    a reranker: concatenate query and document with [SEP], use [CLS] embedding for
    binary relevance classification. Dramatically outperformed BM25 + sparse features
    on MS MARCO passage ranking.

    **Mono/DuoBERT pipeline (Nogueira et al., 2019).** Formalised the multi-stage:
    BM25 (1000) → monoBERT reranker (100) → duoBERT pairwise reranker (10).
    Showed that cascade stages can be stacked profitably.

    **Cohere Rerank API (2023).** Productionised cross-encoder reranking as a managed
    API — the de facto production choice when latency is not sub-10ms. Single API
    call, no model hosting required.

    **ColBERT late interaction (Khattab & Zaharia, 2020).** A middle ground between
    bi-encoder (fast but less accurate) and cross-encoder (accurate but slow):
    per-token embeddings scored with MaxSim. Enables ANN-compatible search with
    near-cross-encoder accuracy.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Bi-encoder (retrieval stage):**
    ```
    Query ──► Encoder ──► q_emb [768d]
                                        ──► dot product ──► score
    Document ──► Encoder ──► d_emb [768d]
    ```
    The encoder processes query and document **independently**. This means
    $d_{\text{embs}}$ can be pre-computed and stored — $O(1)$ lookup per document
    at query time. Cost: $O(N)$ for brute-force (ANN: $O(\log N)$).

    **Cross-encoder (reranking stage):**
    ```
    "[CLS] query [SEP] document [SEP]" ──► Encoder ──► [CLS] embedding ──► score
    ```
    The encoder processes query and document **jointly** — every layer of the
    transformer can attend from any query token to any document token. Much richer
    interaction, but cannot be pre-computed: must run inference per (query, doc) pair.
    Cost: $O(k_{\text{rerank}} \times T_{\text{inference}})$ where $k_{\text{rerank}}$
    is the candidate set size.

    **Latency budget intuition:**
    - BM25 search, $N=1M$: ~5ms
    - Dense ANN search, $N=1M$: ~3ms
    - RRF fusion, 200 candidates: ~1ms
    - Cross-encoder inference, 100 × 200 tokens: ~80–200ms (CPU), ~10–30ms (GPU)
    - **Total two-stage**: ~100–210ms (CPU) / ~20–40ms (GPU)

    **Why cross-encoders win on precision.** A bi-encoder maps a document to a fixed
    vector before seeing the query. If the query is "What is the defendant's motive?",
    the document embedding cannot know to emphasise the motive-related sentences —
    it must summarise everything into one vector. The cross-encoder sees the full
    query–document pair and can focus attention specifically on motive-related phrases.
    """),

    code(r"""
    import numpy as np
    import math
    import re
    import time
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (9, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Bi-encoder scoring

    $$s_{\text{bi}}(q, d) = \text{cos}(\mathbf{E}_q(q),\, \mathbf{E}_d(d))$$

    where $\mathbf{E}_q$ and $\mathbf{E}_d$ are encoder functions (often the same
    model with mean pooling). Crucially, $\mathbf{E}_d(d)$ can be pre-computed.

    ### 4.2 Cross-encoder scoring

    $$s_{\text{cross}}(q, d) = \mathbf{w}^\top \mathbf{E}([q; \text{SEP}; d])_{[CLS]}$$

    The encoder runs over the concatenated sequence. No pre-computation possible.
    Complexity: $O(k \cdot (|q| + |d|)^2)$ due to attention.

    ### 4.3 NDCG@k (Normalised Discounted Cumulative Gain)

    The standard ranking quality metric:
    $$\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i+1)}$$
    $$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

    where $r_i \in \{0, 1\}$ is the relevance of the result at rank $i$, and
    $\text{IDCG@k}$ is the DCG of the perfect ranking. NDCG@k rewards placing
    relevant documents higher in the ranking.

    ### 4.4 Precision@k and Recall@k

    $$\text{Precision@k} = \frac{|\text{top-k} \cap \text{relevant}|}{k}$$
    $$\text{Recall@k} = \frac{|\text{top-k} \cap \text{relevant}|}{|\text{relevant}|}$$

    The two-stage architecture explicitly targets:
    - Stage 1 (retrieval): high **Recall@100** — get all relevant docs in candidate set.
    - Stage 2 (reranker): high **Precision@10** and **NDCG@10** — rank them correctly.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    ### 5a BM25 first-stage retriever (from Notebook 27)
    """),

    code(r"""
    # 5a. Reuse BM25 from Notebook 27.
    class BM25:
        def __init__(self, k1=1.5, b=0.75):
            self.k1 = k1; self.b = b
            self.corpus = []; self.tokenised = []
            self.df = Counter(); self.N = 0; self.avgdl = 0.0

        def _tok(self, text):
            return re.findall(r'\w+', text.lower())

        def fit(self, corpus):
            self.corpus = corpus
            self.tokenised = [self._tok(d) for d in corpus]
            self.N = len(corpus)
            self.avgdl = sum(len(t) for t in self.tokenised) / max(1, self.N)
            self.df = Counter()
            for toks in self.tokenised:
                for term in set(toks):
                    self.df[term] += 1
            return self

        def idf(self, term):
            n = self.df.get(term, 0)
            return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

        def score(self, query, di):
            toks = self.tokenised[di]; dl = len(toks); tfc = Counter(toks)
            s = 0.0
            for t in self._tok(query):
                tf = tfc.get(t, 0)
                norm_tf = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl)))
                s += self.idf(t) * norm_tf
            return s

        def search(self, query, k=10):
            scores = [(self.score(query, i), i) for i in range(self.N)]
            scores.sort(reverse=True)
            return [(idx, s) for s, idx in scores[:k]]

    # Legal document corpus (simplified).
    corpus = [
        'The defendant was seen near the crime scene on the night of the incident',
        'Witness testimony confirms the suspect had prior knowledge of the vault location',
        'Financial records show transfers to offshore accounts totalling 2 million dollars',
        'The accused provided a false alibi claiming to be out of town during the robbery',
        'Security footage captures the suspect wearing a red jacket at 11pm',
        'DNA evidence found at the scene does not match the defendant',
        'The defendant has no prior criminal record and cooperated with investigators',
        'Expert witness states the financial fraud required insider access to the system',
        'Phone records indicate the suspect communicated with the co-conspirator',
        'Character witness describes the defendant as peaceful and law-abiding',
        'CCTV footage from a petrol station confirms the alibi at 10:45pm',
        'Bank statements show unusual cash withdrawals in the week before the incident',
    ]

    bm25 = BM25()
    bm25.fit(corpus)
    print(f'Corpus: {len(corpus)} legal documents')
    """),

    md(r"""
    ### 5b Bi-encoder dense retriever (simulated)
    """),

    code(r"""
    # 5b. Simulated dense embeddings (semantic axes: evidence, alibi, financial, identity).
    D = 16

    def doc_embed(text, rng):
        words = text.lower().split()
        v = rng.normal(0, 0.05, D)
        v[0] += sum(1 for w in words if w in ['evidence', 'dna', 'witness', 'footage', 'cctv'])
        v[1] += sum(1 for w in words if w in ['alibi', 'false', 'cooperated', 'confirmed'])
        v[2] += sum(1 for w in words if w in ['financial', 'bank', 'money', 'fraud', 'offshore'])
        v[3] += sum(1 for w in words if w in ['suspect', 'defendant', 'accused', 'criminal'])
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    corpus_embs = np.stack([doc_embed(d, rng) for d in corpus])

    def dense_search(q_emb, k=10):
        sims = corpus_embs @ q_emb
        idx = np.argpartition(sims, -k)[-k:]
        return [(i, float(sims[i])) for i in idx[np.argsort(sims[idx])[::-1]]]

    # Query: "What evidence links the defendant to the financial crime?"
    q_emb = np.zeros(D, dtype=np.float32)
    q_emb[0] = 0.6; q_emb[2] = 0.8   # evidence + financial
    q_emb /= np.linalg.norm(q_emb)

    print('Dense first-stage results (semantic):')
    for rank, (idx, s) in enumerate(dense_search(q_emb, k=6)):
        print(f'  [{rank+1}] sim={s:.3f}  {corpus[idx][:65]}')
    """),

    md(r"""
    ### 5c Cross-encoder scoring from scratch
    """),

    code(r"""
    # 5c. Simplified cross-encoder: concatenate query and doc token-level features,
    # score with a learned linear weight (simulated by feature engineering).
    # This is a pedagogical approximation — production cross-encoders are full
    # transformer models. The key idea: query-aware scoring impossible in bi-encoders.

    def tokenise(text):
        return set(re.findall(r'\w+', text.lower()))

    def cross_encoder_score(query, doc):
        # Simulates the richer query-document interaction of a real cross-encoder.
        q_terms = tokenise(query)
        d_terms = tokenise(doc)
        words = doc.lower().split()

        # Feature 1: query term overlap (bi-encoder can do this via embedding).
        overlap = len(q_terms & d_terms) / max(1, len(q_terms))

        # Feature 2: query-conditioned positional signal — does each query term
        # appear near another query term in the document? (cross-attention surrogate)
        query_positions = defaultdict(list)
        for i, w in enumerate(words):
            if w in q_terms:
                query_positions[w].append(i)
        proximity_score = 0.0
        q_list = list(q_terms)
        for i in range(len(q_list)):
            for j in range(i + 1, len(q_list)):
                pi = query_positions.get(q_list[i], [])
                pj = query_positions.get(q_list[j], [])
                if pi and pj:
                    min_dist = min(abs(a - b) for a in pi for b in pj)
                    proximity_score += 1.0 / (1 + min_dist)

        # Feature 3: query-document length ratio (cross-encoder sees full length).
        len_ratio = min(1.0, len(q_terms) / max(1, len(d_terms)))

        score = 0.5 * overlap + 0.4 * min(1.0, proximity_score / max(1, len(q_terms))) + 0.1 * len_ratio
        return float(score)

    query = 'What evidence links the defendant to the financial crime'
    print(f'Cross-encoder scores for: "{query}"')
    ce_scores = [(i, cross_encoder_score(query, doc)) for i, doc in enumerate(corpus)]
    ce_scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (idx, s) in enumerate(ce_scores[:6]):
        print(f'  [{rank+1}] ce={s:.4f}  {corpus[idx][:65]}')
    """),

    md(r"""
    ### 5d Full two-stage reranking pipeline
    """),

    code(r"""
    # 5d. Complete pipeline: BM25 recall stage → cross-encoder rerank stage.
    def rerank_pipeline(query, q_emb, bm25, corpus_embs, corpus,
                        k_retrieve=10, k_rerank=5, use_hybrid=True):
        # Stage 1: fast retrieval (BM25 or hybrid).
        bm25_results = bm25.search(query, k=k_retrieve)
        if use_hybrid:
            dense_results = dense_search(q_emb, k=k_retrieve)
            # Simple RRF fusion.
            rrf_scores = defaultdict(float)
            for rank, (idx, _) in enumerate(bm25_results):
                rrf_scores[idx] += 1.0 / (60 + rank + 1)
            for rank, (idx, _) in enumerate(dense_results):
                rrf_scores[idx] += 1.0 / (60 + rank + 1)
            candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k_retrieve]
        else:
            candidates = bm25_results

        # Stage 2: cross-encoder reranking (expensive but small k).
        reranked = []
        for idx, _ in candidates:
            ce = cross_encoder_score(query, corpus[idx])
            reranked.append((idx, ce))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k_rerank]

    # Ground truth: which documents are actually relevant?
    # Query: "What evidence links the defendant to the financial crime?"
    # Relevant: docs 2, 7, 11 (financial), 1 (insider knowledge), 8 (phone records).
    relevant_docs = {1, 2, 7, 8, 11}

    print('=== Two-stage pipeline ===')
    results = rerank_pipeline(query, q_emb, bm25, corpus_embs, corpus,
                               k_retrieve=10, k_rerank=5)
    found = set(idx for idx, _ in results)
    precision_5 = len(found & relevant_docs) / 5
    recall_5    = len(found & relevant_docs) / len(relevant_docs)
    print(f'Precision@5: {precision_5:.2f}  Recall@5: {recall_5:.2f}')
    print('Results:')
    for rank, (idx, s) in enumerate(results):
        rel = '[REL]' if idx in relevant_docs else '     '
        print(f'  [{rank+1}] {rel} ce={s:.4f}  {corpus[idx][:60]}')

    # Baseline: BM25 only, no reranking.
    print('\n=== BM25 only (no reranking) ===')
    bm25_top5 = bm25.search(query, k=5)
    bm25_found = set(idx for idx, _ in bm25_top5)
    print(f'Precision@5: {len(bm25_found & relevant_docs)/5:.2f}  '
          f'Recall@5: {len(bm25_found & relevant_docs)/len(relevant_docs):.2f}')
    for rank, (idx, s) in enumerate(bm25_top5):
        rel = '[REL]' if idx in relevant_docs else '     '
        print(f'  [{rank+1}] {rel} bm25={s:.3f}  {corpus[idx][:60]}')
    """),

    md(r"""
    ### 5e NDCG@k evaluation
    """),

    code(r"""
    # 5e. NDCG@k: the standard ranking quality metric.
    def ndcg_at_k(ranked_ids, relevant_set, k):
        dcg = 0.0
        for i, doc_id in enumerate(ranked_ids[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(i + 2)   # log2(rank+1) where rank is 1-indexed
        # Ideal DCG: all relevant at the top.
        n_ideal = min(k, len(relevant_set))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
        return dcg / idcg if idcg > 0 else 0.0

    # Compare ranking quality across pipeline stages.
    bm25_top10   = [idx for idx, _ in bm25.search(query, k=10)]
    dense_top10  = [idx for idx, _ in dense_search(q_emb, k=10)]
    hybrid_top10 = [idx for idx, _ in rerank_pipeline(
        query, q_emb, bm25, corpus_embs, corpus, k_retrieve=10, k_rerank=10)]

    for name, ranked in [('BM25', bm25_top10), ('Dense', dense_top10),
                          ('Hybrid+Rerank', hybrid_top10)]:
        nd = ndcg_at_k(ranked, relevant_docs, k=5)
        p5 = len(set(ranked[:5]) & relevant_docs) / 5
        r5 = len(set(ranked[:5]) & relevant_docs) / len(relevant_docs)
        print(f'{name:15s}: NDCG@5={nd:.3f}  P@5={p5:.2f}  R@5={r5:.2f}')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Bi-encoder vs cross-encoder: attention patterns (schematic).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    q_tokens = ['What', 'evidence', 'financial', 'crime']
    d_tokens = ['Financial', 'records', 'show', 'offshore', 'accounts']

    # Bi-encoder: no cross-attention (compressed to single vector).
    bi_matrix = np.zeros((len(q_tokens), len(d_tokens)))
    ce_matrix = rng.uniform(0.1, 0.9, (len(q_tokens), len(d_tokens)))
    # Simulate strong cross-attention between matching terms.
    ce_matrix[1, 0] = 0.95  # evidence → Financial (weak)
    ce_matrix[2, 0] = 0.95  # financial → Financial
    ce_matrix[2, 3] = 0.85  # financial → offshore
    ce_matrix[3, 4] = 0.80  # crime → accounts

    for ax, mat, title in [
        (axes[0], bi_matrix, 'Bi-encoder: NO cross-attention\n(query & doc encoded separately)'),
        (axes[1], ce_matrix, 'Cross-encoder: full cross-attention\n(query & doc attend to each other)'),
    ]:
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(d_tokens))); ax.set_xticklabels(d_tokens, fontsize=9)
        ax.set_yticks(range(len(q_tokens))); ax.set_yticklabels(q_tokens, fontsize=9)
        ax.set_xlabel('Document tokens'); ax.set_ylabel('Query tokens')
        ax.set_title(title)
    plt.colorbar(im, ax=axes[1], label='Attention weight')
    plt.suptitle('Figure 1 — Bi-encoder vs cross-encoder: attention asymmetry')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** The fundamental difference between bi-encoder and cross-encoder.
    **Left:** the bi-encoder encodes query and document independently — there is no
    cross-attention (zero matrix). The query "financial crime evidence" is compressed
    into a single 768-dim vector that cannot be conditioned on the specific document.
    **Right:** the cross-encoder concatenates query and document and runs a full
    transformer — every query token can attend to every document token. "financial"
    attends strongly to "Financial" and "offshore"; "crime" attends to "accounts".
    This full interaction is why cross-encoders produce much more accurate relevance
    scores — they can identify *why* a document is relevant to *this specific query*.
    """),

    code(r"""
    # Figure 2 — NDCG@k vs k for each retrieval method.
    ks = list(range(1, 11))

    bm25_ranked    = [idx for idx, _ in bm25.search(query, k=12)]
    dense_ranked   = [idx for idx, _ in dense_search(q_emb, k=12)]
    hybrid_ranked  = [idx for idx, _ in rerank_pipeline(
        query, q_emb, bm25, corpus_embs, corpus, k_retrieve=12, k_rerank=12)]

    fig, ax = plt.subplots()
    for name, ranked in [('BM25', bm25_ranked), ('Dense', dense_ranked), ('Hybrid+Rerank', hybrid_ranked)]:
        ndcgs = [ndcg_at_k(ranked, relevant_docs, k) for k in ks]
        ax.plot(ks, ndcgs, 'o-', label=name)
    ax.set_xlabel('k'); ax.set_ylabel('NDCG@k')
    ax.set_title('Figure 2 — NDCG@k: retrieval method comparison')
    ax.set_xticks(ks); ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** NDCG@k across k values for each method. The reranker consistently
    achieves higher NDCG, especially at low k (k=1 to 5) where ranking order matters
    most for RAG (the LLM reads the top-3 to top-5 passages). BM25 peaks later because
    its relevant documents may be ranked lower in the initial list. The key insight:
    **the reranker doesn't improve recall** (it only re-orders the same candidate set)
    but dramatically improves **ordering quality**. If a relevant document wasn't
    retrieved in stage 1, the reranker cannot recover it.
    """),

    code(r"""
    # Figure 3 — Latency breakdown: retrieval vs reranking vs k_rerank.
    # Simulated latency (representative of CPU inference).
    import time

    def simulate_latency(n_docs, k_retrieve, k_rerank, d=768):
        # BM25: O(N * q_len), roughly 0.5ms per 1K docs.
        bm25_latency   = n_docs * 0.5e-6   # 0.5 us per doc
        # Dense ANN (HNSW): O(log N), roughly constant.
        dense_latency  = 3e-3              # 3ms
        # Cross-encoder: O(k_rerank * seq_len^2), ~2ms per doc on CPU.
        rerank_latency = k_rerank * 2e-3   # 2ms per doc
        total = bm25_latency + dense_latency + rerank_latency
        return bm25_latency * 1000, dense_latency * 1000, rerank_latency * 1000, total * 1000

    k_reranks = [5, 10, 20, 50, 100]
    fig, ax = plt.subplots()
    totals = []
    for k_r in k_reranks:
        b, d_l, r, tot = simulate_latency(100_000, 100, k_r)
        totals.append(tot)

    ax.stackplot(k_reranks,
                 [simulate_latency(100_000, 100, k)[0] for k in k_reranks],
                 [simulate_latency(100_000, 100, k)[1] for k in k_reranks],
                 [simulate_latency(100_000, 100, k)[2] for k in k_reranks],
                 labels=['BM25', 'Dense ANN', 'Cross-encoder rerank'])
    ax.axhline(200, color='red', ls='--', label='200ms SLA')
    ax.set_xlabel('k_rerank (candidates sent to reranker)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Figure 3 — Latency breakdown: where does time go?')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print('Latency breakdown (N=100K, k_retrieve=100):')
    for k_r in k_reranks:
        b, d_l, r, tot = simulate_latency(100_000, 100, k_r)
        print(f'  k_rerank={k_r:3d}: BM25={b:.1f}ms  Dense={d_l:.1f}ms  Rerank={r:.1f}ms  Total={tot:.1f}ms')
    """),

    md(r"""
    **Figure 3.** Latency breakdown for the two-stage pipeline. BM25 and dense ANN
    are fast ($O(\text{ms})$) and dominate at low $k_{\text{rerank}}$. The cross-encoder
    reranker cost is **linear in $k_{\text{rerank}}$** — this is why the candidate set
    must be bounded (typically 20–100). At $k_{\text{rerank}} = 100$ on CPU, the
    reranker alone takes 200ms, violating a typical 200ms SLA. The production solution:
    **GPU inference** (cross-encoder ~10× faster), **batched reranking**, or **Cohere
    Rerank API** (managed inference, ~100ms round-trip). Key design rule: set
    $k_{\text{rerank}} \leq 50$ for CPU serving and $k_{\text{rerank}} \leq 200$ for GPU.
    """),

    code(r"""
    # Figure 4 — Precision@k vs k_retrieve: why you need a large first stage.
    N = len(corpus)
    k_retrieves = [2, 4, 6, 8, 10, 12]
    p_bm25, p_dense, p_hybrid = [], [], []

    for k in k_retrieves:
        bm25_k    = set(idx for idx, _ in bm25.search(query, k=k))
        dense_k   = set(idx for idx, _ in dense_search(q_emb, k=k))
        hybrid_k  = bm25_k | dense_k
        p_bm25.append(len(bm25_k & relevant_docs) / len(relevant_docs))
        p_dense.append(len(dense_k & relevant_docs) / len(relevant_docs))
        p_hybrid.append(len(hybrid_k & relevant_docs) / len(relevant_docs))

    fig, ax = plt.subplots()
    ax.plot(k_retrieves, p_bm25,   'o-', label='BM25 recall')
    ax.plot(k_retrieves, p_dense,  's-', label='Dense recall')
    ax.plot(k_retrieves, p_hybrid, 'D-', label='Hybrid recall')
    ax.axhline(0.8, color='gray', ls=':', label='Target Recall@k=0.8')
    ax.set_xlabel('k_retrieve'); ax.set_ylabel('Recall (fraction of relevant retrieved)')
    ax.set_title('Figure 4 — First-stage recall vs k_retrieve (must exceed reranker window)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** First-stage recall vs. retrieval width $k$. The reranker can only
    improve the ordering of retrieved candidates — it **cannot recover documents not
    in the candidate set**. This means stage-1 recall must be very high. If the
    relevant documents are not in the top-100 candidates, the reranker cannot surface
    them. The target is typically Recall@100 $\geq 0.95$ from stage 1. Note that
    **hybrid retrieval** achieves higher recall at every $k$ than either BM25 or dense
    alone — this is the primary motivation for combining both systems in stage 1.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Stage-1 recall ceiling** | Reranker never finds the right doc | Relevant docs not in candidate set | Increase k_retrieve; use hybrid retrieval in stage 1 |
    | **Reranker latency violation** | p99 > SLA | k_rerank too large on CPU | Move to GPU; reduce k_rerank; use Cohere managed API |
    | **Reranker domain mismatch** | Reranker hurts recall vs. BM25 | Reranker trained on different domain | Fine-tune on domain; or use BM25 fallback |
    | **Reranker overfit to length** | Long docs always win | Cross-encoder biased toward verbosity | Add doc length as de-ranking signal; truncate to 512 tokens |
    | **Cold start** | No reranker model for new language | No cross-encoder for language | mDeBERTa or Cohere multilingual reranker |
    | **Score saturation** | All docs get ~0.9 reranker score | Reranker not discriminating | Check model: may be miscalibrated; try different threshold |
    | **Missing query term** | High-relevance doc ranked low | Query expansion not applied | Add query expansion before first stage |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 sentence-transformers CrossEncoder (guarded).
    try:
        from sentence_transformers.cross_encoder import CrossEncoder  # noqa: F401
        lines = [
            'from sentence_transformers.cross_encoder import CrossEncoder',
            '',
            '# Load a cross-encoder (fine-tuned on MS MARCO).',
            'model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")',
            '',
            '# Score a list of (query, doc) pairs.',
            'pairs = [(query, doc) for doc in candidates]',
            'scores = model.predict(pairs)  # (k_retrieve,) float array',
            '',
            '# Re-sort by score.',
            'reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)',
            'top_10 = reranked[:10]',
        ]
        print('\n'.join(lines))
    except ImportError:
        lines = [
            '[sentence-transformers not installed — production pattern]:',
            '  from sentence_transformers.cross_encoder import CrossEncoder',
            '  model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")',
            '  pairs = [(query, doc) for doc in candidates]',
            '  scores = model.predict(pairs)',
            '  reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)',
        ]
        print('\n'.join(lines))
    """),

    code(r"""
    # 8.2 Cohere Rerank API (guarded).
    try:
        import cohere  # noqa: F401
        lines = [
            'import cohere',
            'co = cohere.Client(api_key="...")',
            '',
            'results = co.rerank(',
            '    query=query,',
            '    documents=candidates,      # list of strings',
            '    top_n=10,',
            '    model="rerank-english-v3.0"',
            ')',
            'for hit in results.results:',
            '    print(hit.relevance_score, candidates[hit.index])',
        ]
        print('\n'.join(lines))
    except ImportError:
        lines = [
            '[cohere not installed — production pattern]:',
            '  import cohere',
            '  co = cohere.Client(api_key="...")',
            '  results = co.rerank(',
            '      query=query, documents=candidates,',
            '      top_n=10, model="rerank-english-v3.0")',
            '  for hit in results.results:',
            '      print(hit.relevance_score, candidates[hit.index])',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Legal Document Search

    **Scenario.** A law firm uses RAG to answer lawyer queries over 500K case files,
    contracts, and expert reports. The cost of a wrong answer at position 1 is high:
    a lawyer citing an irrelevant precedent in court wastes hours and damages credibility.

    **Requirements:**
    - Precision@5: $\geq 0.90$ (4.5 of 5 results must be relevant).
    - Recall@100: $\geq 0.97$ from stage 1 (miss rate < 3%).
    - Latency: p99 $\leq 400$ms (legal queries are infrequent; latency is acceptable).
    - Domain: legal English; US case law, UK law, EU regulations.

    **Architecture:**
    - Stage 1: Hybrid BM25 + `text-embedding-3-large` (d=3072, truncated to d=1024),
      k_retrieve=100. Hybrid recall@100 = 0.97 measured on 200 golden queries.
    - Stage 2: Cohere `rerank-english-v3.0`, k_rerank=5. Adds ~150ms round-trip.
    - Total latency: ~220ms (within SLA).
    - Precision@5 improved from 0.61 (BM25 only) to 0.91 (hybrid + reranker).

    **Monitoring:** weekly Precision@5 on 50 sampled golden queries with lawyer
    feedback. Alert if Precision@5 drops below 0.85. Monthly review of failures
    to identify systematic misses (e.g., newer regulatory documents not yet indexed).
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Sequence length limit.** Most cross-encoders cap at 512 tokens. Long documents
      must be truncated or chunked before reranking. Truncate to the most relevant
      passage or use a sliding-window approach (Notebook 29).
    - **Batch reranking.** Cross-encoders support batched inference — batch all
      $k_{\text{rerank}}$ pairs in one forward pass (or a few batches). On GPU,
      batching 50 pairs at once is ~5× faster than scoring one at a time.
    - **Model selection.** MS MARCO-trained models (MiniLM, DeBERTa) are good
      general-purpose rerankers. For domain-specific tasks (legal, medical), fine-tune
      on domain data or use a managed API (Cohere, Jina) with a general model.
    - **Score calibration.** Cross-encoder output is not a probability unless trained
      with sigmoid cross-entropy. Use relative ordering (rank by score), not absolute
      thresholds, unless the model was specifically calibrated.
    - **Streaming results.** For interactive UIs, stream the top-5 from stage 1 while
      reranking runs asynchronously. Replace preliminary results with reranked results
      when available (similar to Google's progressive loading).
    - **ColBERT as an alternative.** ColBERT achieves near-cross-encoder accuracy with
      ANN-compatible indexing by storing per-token embeddings and using MaxSim scoring.
      It's 10–50× faster than a cross-encoder at reranking time but requires 10–30×
      more storage per document than a bi-encoder.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Reranker type comparison:**

    | Model type | Accuracy | Latency | Can precompute | Storage | Use case |
    |---|---|---|---|---|---|
    | BM25 | Moderate | Very fast | Yes (index) | Low | Baseline, exact match |
    | Bi-encoder | Good | Fast (ANN) | Yes (embeddings) | Medium | First-stage retrieval |
    | Cross-encoder | Excellent | Slow | No | None (online) | Reranking top-50 |
    | ColBERT | Very good | Medium | Partial (token embs) | High | If budget allows |
    | Cohere Rerank API | Excellent | ~100ms RTT | No | None | Managed, no infra |

    **k_rerank tradeoffs:**

    | k_rerank | Latency (CPU) | Latency (GPU) | NDCG@10 benefit | Recommended when |
    |---|---|---|---|---|
    | 10 | ~20ms | ~5ms | Low (first stage already ok) | Latency-critical apps |
    | 50 | ~100ms | ~15ms | High | Default production setting |
    | 100 | ~200ms | ~25ms | Moderate (diminishing) | High-stakes precision tasks |
    | 200+ | >400ms | ~50ms | Marginal | Async (batch) reranking only |

    **Two-stage vs. end-to-end:**

    | Approach | Quality | Latency | Cost | Recommended for |
    |---|---|---|---|---|
    | BM25 only | Moderate | 5ms | Minimal | Low-quality tolerance |
    | Hybrid only | Good | ~10ms | Low | Latency < 50ms |
    | Hybrid + reranker | Excellent | ~150ms | Medium | Most RAG applications |
    | End-to-end fine-tuned | Best | 100–500ms | High | Enterprise, domain-critical |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"Why can't you just use a cross-encoder for everything?"* → Cross-encoders
      cannot be pre-computed. For $N=1M$ documents and 1000 QPS, you'd need
      $1M \times 1000 = 1B$ forward passes per second — infeasible. The two-stage
      approach runs cross-encoder only on the top-50–100 candidates, making it tractable.
    - *"What is the difference between bi-encoder and cross-encoder?"* → Bi-encoder:
      encodes query and document independently; supports pre-computation; $O(\log N)$
      search via ANN. Cross-encoder: encodes query+document jointly; full cross-
      attention; cannot pre-compute; runs $O(k_{\text{rerank}})$ inferences at query
      time. Cross-encoders produce more accurate scores because they see both texts
      simultaneously.

    **Deep-dive questions**
    - *"What is NDCG@k and why is it better than Precision@k for evaluating a reranker?"*
      → NDCG@k accounts for rank position — a relevant document at rank 1 is worth more
      than at rank 5. Precision@k treats all top-k positions equally. NDCG matches the
      user experience: the first result is read by almost everyone; the fifth result
      by far fewer. For RAG systems where the LLM reads results in order, NDCG@3 or
      NDCG@5 is the most relevant metric.
    - *"The reranker is improving Precision@5 from 0.7 to 0.85, but latency is 300ms
      on CPU and our SLA is 200ms. What are your options?"* → (1) Move to GPU (10×
      speedup → 30ms). (2) Use Cohere Rerank API (~100ms managed). (3) Reduce
      k_rerank from 100 to 30 (~90ms). (4) Use a smaller cross-encoder model
      (MiniLM-L4 vs L12: 3× faster, ~5% NDCG loss). (5) ColBERT if storage budget allows.

    **Whiteboard question**
    - "A user complaint: 'RAG gives me 5 results but the right answer is always #3 or #4,
      never #1.' What is the root cause and how do you fix it?" → Root cause: poor
      ranking quality in stage 1 (or no reranker). Fix: add cross-encoder reranker.
      Diagnosis: measure NDCG@1 and NDCG@5 on golden queries to confirm ranking issue
      vs. recall issue.

    **Common mistakes:** forgetting that reranker cannot recover docs not in stage-1
    candidate set; running cross-encoder on all N docs (ignoring latency); not
    measuring NDCG@k (only measuring recall); choosing k_rerank=200 on CPU without
    checking latency.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Two-stage goal.** Stage 1 optimises ______; stage 2 optimises ______. Why
       can't you skip stage 1?
    2. **Bi vs cross.** Explain the encoding difference. Why can't cross-encoders
       pre-compute document representations?
    3. **NDCG@k.** Define it. Why is it preferred over Precision@k for ranking tasks?
    4. **Latency budget.** Cross-encoder on 100 candidates at 2ms/doc = ? ms. What
       are 3 ways to reduce this?
    5. **Stage-1 recall.** If stage-1 Recall@100 is 0.85 and you have 10 relevant docs
       total, how many can the reranker surface at most?
    6. **ColBERT.** What makes ColBERT a "middle ground" between bi- and cross-encoder?
       What is the storage cost?
    7. **k_rerank selection.** Your SLA is 100ms, GPU inference is 1ms/doc. What is
       the maximum k_rerank you can afford, leaving 30ms for retrieval?
    8. **Failure mode.** A reranker trained on MS MARCO is deployed for legal document
       search. Precision@5 is worse than BM25 alone. What happened? How do you fix it?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Prove that increasing $k_{\text{retrieve}}$ cannot decrease stage-1 recall.
       Why does this not mean you should always use the largest possible $k_{\text{retrieve}}$?
    2. NDCG@5 for a ranking [REL, NOT, REL, NOT, NOT] with 3 total relevant docs.
       Compute DCG@5 and IDCG@5 by hand. Show your working.

    **Beginner → Intermediate (coding)**
    3. Add **MRR@k (Mean Reciprocal Rank)** to the evaluation suite: for each query,
       find the rank of the first relevant result; MRR = mean of 1/rank. Compare MRR@10
       for BM25, dense, and hybrid+rerank on the toy corpus.
    4. Implement **score normalisation** for the cross-encoder: fit a min-max scaler
       on a calibration set of 20 (query, doc) pairs with known relevance labels.
       Show how calibrated scores enable thresholding (only return docs with score >0.7).

    **Intermediate (analysis)**
    5. Implement a **sliding-window chunker** (Notebook 29 preview): split each document
       into overlapping 128-token windows with 32-token stride. Index the chunks with
       BM25 and rerank with the cross-encoder. Compare Precision@5 on the toy corpus
       vs. full-document reranking.
    6. Simulate the **ColBERT MaxSim** scoring: for each document, store token-level
       embeddings (e.g., word2vec of each token from Notebook 20). At query time,
       for each query token find the maximum cosine similarity across all doc tokens
       (MaxSim), then sum over query tokens. Compare to bi-encoder cosine similarity.

    **Senior (design)**
    7. *System design:* design the complete retrieval stack for a medical literature
       search system (PubMed, 35M papers). Requirements: Precision@5 $\geq 0.88$,
       Latency $\leq 300$ms, must handle medical jargon exact-match ("IL-6 cytokine
       storm") AND semantic ("inflammation signalling"). Specify: stage-1 system,
       stage-2 reranker model, k_retrieve, k_rerank, latency breakdown, evaluation plan.
    8. *Interview:* "Our legal RAG system achieves Recall@100=0.97 (stage 1) but
       Precision@5=0.55 (after reranking). We've tried 3 different cross-encoders
       and none improves Precision@5 above 0.60. Where is the problem?" Diagnose
       systematically: what would you check first, second, third?
    """),

    md(r"""
    ---
    ### Summary
    Reranking is the single highest-ROI improvement to a RAG pipeline. The two-stage
    architecture separates concerns: **fast retrieval** (BM25 + dense hybrid) maximises
    Recall@100; **cross-encoder reranking** maximises Precision@10 and NDCG@10 on the
    candidate set. Cross-encoders are accurate because they jointly encode query and
    document with full cross-attention — but they cannot pre-compute, so $k_{\text{rerank}}$
    must be small (20–100). Production options: sentence-transformers CrossEncoder
    (self-hosted GPU), Cohere Rerank API (managed, ~100ms), or ColBERT (ANN-compatible,
    10× higher storage).

    **Next:** `29 · Chunking Strategies` — how to split documents before indexing:
    fixed-size, sentence-boundary, semantic, recursive, and parent-child chunking.
    Chunk size determines what the cross-encoder sees and strongly impacts both
    retrieval and generation quality.
    """),
]

build("phase5_rag/28_reranking.ipynb", cells)
