"""Builder for Lesson RAG-06 — Hybrid Search."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # RAG-06 · Hybrid Search
    ### Section 06 — Retrieval-Augmented Generation · *ML/AI Senior Mastery Curriculum*

    > Dense vector search (Lesson RAG-01) excels at semantic similarity but fails on
    > **exact-match** queries — product codes, proper nouns, rare terms, serial numbers.
    > BM25 sparse retrieval excels at exact matches but fails on paraphrase and
    > semantic intent. **Hybrid search** fuses both signals to get the best of both
    > worlds. This notebook teaches BM25 from scratch, two fusion strategies (RRF and
    > alpha-weighted), and why every production RAG pipeline should be hybrid.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Why hybrid search**: the failure modes of pure dense and pure sparse retrieval.
    - **BM25 from scratch**: TF saturation, IDF, document length normalisation —
      derive and implement every term.
    - **Dense retrieval from scratch**: cosine similarity (reference Lesson RAG-01).
    - **Reciprocal Rank Fusion (RRF)**: the rank-based fusion formula; why it is
      robust to score-scale mismatch between systems.
    - **Alpha-weighted score fusion**: normalise scores, blend with weight $\alpha$.
    - **When RRF beats alpha-weighted and vice versa.**
    - **Production hybrid search**: LangChain EnsembleRetriever + BM25Retriever
      (guarded import).
    - **Tune $\alpha$**: measuring Recall@k across a grid of $\alpha$ values.

    **Why it matters**
    - BEIR benchmarks (2021) showed that BM25 outperforms many dense models on
      exact-match tasks. Production RAG at Notion, Elastic, Cohere consistently uses
      hybrid. Not using hybrid when your corpus has product codes or named entities
      is a leading cause of "why doesn't RAG find X" support tickets.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **BM25 (Robertson & Zaragoza, 1994–2009).** Okapi BM25 emerged from the Okapi
    IR system at City University London. It improved on classic TF-IDF by adding:
    (1) **TF saturation** (diminishing returns for repeated terms), and
    (2) **document length normalisation** (long documents shouldn't score higher
    just for having more term occurrences). BM25 dominated TREC competitions for a
    decade and remains the default sparse baseline in 2024.

    **Dense retrieval (Karpukhin et al., DPR, 2020).** Dense Passage Retrieval
    showed that fine-tuned bi-encoders outperform BM25 on open-domain QA (Natural
    Questions, TriviaQA). This triggered widespread adoption of vector search.

    **The hybrid insight.** BEIR (Thakur et al., 2021) benchmarked dense retrievers
    on 18 datasets and found BM25 still won on 6 of them (exact-match heavy tasks).
    Hybrid search fuses both to cover all tasks. **SPLADE** (2021) took this further
    by learning a sparse representation that mimics BM25 structure with learned
    expansion — but pure BM25+dense hybrid is simpler and nearly as good.

    **Reciprocal Rank Fusion (Cormack et al., 2009).** RRF was proposed as a simple,
    parameter-free method to combine ranked lists from multiple retrieval systems.
    It works without knowing the score distributions of each system — only ranks.
    Widely adopted in production because it requires no tuning.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The failure modes of each retrieval type.**

    | Query type | Dense search | BM25 sparse |
    |---|---|---|
    | "shoe that's comfortable for running" | Excellent (semantic) | Misses "running shoe" if those words absent |
    | "Nike Air Max 90 size 10" | Misses if no exact match | Excellent (exact tokens) |
    | "product ID SKU-48291" | Fails (no semantic content) | Excellent if in index |
    | "What is the capital of France?" | Excellent | Good (Paris is a frequent term) |

    **BM25 intuition.** A search engine scores document $d$ for query $q$:
    - Each query term $t$ contributes a score based on how often it appears in $d$
      (TF, but capped — finding "python" 10× isn't 10× better than finding it 1×).
    - Terms that appear in fewer documents get higher weight (IDF).
    - Short documents are preferred over long ones containing the same term count.

    **RRF intuition.** System A ranks document X at position 3; System B ranks it
    at position 7. RRF score = 1/(60+3) + 1/(60+7) = 0.0159 + 0.0149 = 0.0308.
    A document ranked 1st by both systems would score 1/(60+1) + 1/(60+1) = 0.033.
    The constant 60 prevents a rank-1 result from dominating when the other system
    ranks it very low. RRF is **scale-agnostic** — it doesn't matter if BM25 scores
    are in [0, 20] and dense scores are in [-1, 1].

    **Alpha-weighted fusion.** Normalise each system's scores to [0,1], then:
    $\text{score}(d) = \alpha \cdot \text{dense\_score}(d) + (1-\alpha) \cdot \text{bm25\_score}(d)$.
    Requires calibrating $\alpha$ per domain. More interpretable than RRF but
    sensitive to score outliers.
    """),

    code(r"""
    import numpy as np
    import math
    import re
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

    ### 4.1 BM25 scoring

    For query $q = \{t_1, \dots, t_n\}$ and document $d$:

    $$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

    where:
    - $f(t,d)$ = term frequency of $t$ in $d$
    - $|d|$ = document length (tokens)
    - $\text{avgdl}$ = average document length in the corpus
    - $k_1 \in [1.2, 2.0]$ = TF saturation parameter (default 1.5)
    - $b \in [0, 1]$ = length normalisation parameter (default 0.75)

    **IDF** (Robertson's variant, avoids division by zero):
    $$\text{IDF}(t) = \log\!\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$

    where $N$ = total docs, $n(t)$ = docs containing $t$.

    ### 4.2 Reciprocal Rank Fusion

    Given $R$ ranked lists $\{L_1, \dots, L_R\}$ and a constant $k$ (default 60):
    $$\text{RRF}(d) = \sum_{r=1}^{R} \frac{1}{k + \text{rank}_r(d)}$$

    Documents not appearing in a list get rank = $N+1$ (worst possible).

    ### 4.3 Alpha-weighted fusion

    Min-max normalise each system's scores independently, then:
    $$s_{\text{hybrid}}(d) = \alpha \cdot \hat{s}_{\text{dense}}(d) + (1 - \alpha) \cdot \hat{s}_{\text{BM25}}(d)$$

    Optimise $\alpha$ by grid search on a labelled development set, maximising Recall@k.

    ### 4.4 Why RRF with k=60?

    The constant $k=60$ was empirically shown by Cormack et al. to perform well across
    diverse retrieval tasks. It dampens the advantage of rank-1 results and makes the
    formula robust to systems with very different quality. Higher $k$ gives more uniform
    weights (approaching average-rank); lower $k$ gives more weight to the top rank.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    ### 5a BM25 from scratch
    """),

    code(r"""
    # 5a. BM25 implementation from scratch.
    class BM25:
        def __init__(self, k1=1.5, b=0.75):
            self.k1 = k1
            self.b = b
            self.corpus = []
            self.tokenised = []
            self.df = Counter()    # doc frequency per term
            self.N = 0
            self.avgdl = 0.0

        def _tokenise(self, text):
            return re.findall(r'\w+', text.lower())

        def fit(self, corpus):
            self.corpus = corpus
            self.tokenised = [self._tokenise(doc) for doc in corpus]
            self.N = len(corpus)
            self.avgdl = sum(len(t) for t in self.tokenised) / max(1, self.N)
            self.df = Counter()
            for tokens in self.tokenised:
                for term in set(tokens):
                    self.df[term] += 1
            return self

        def idf(self, term):
            n = self.df.get(term, 0)
            return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

        def score(self, query, doc_idx):
            tokens = self.tokenised[doc_idx]
            dl = len(tokens)
            tf_counter = Counter(tokens)
            score = 0.0
            for term in self._tokenise(query):
                tf = tf_counter.get(term, 0)
                idf = self.idf(term)
                norm_tf = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
                )
                score += idf * norm_tf
            return score

        def search(self, query, k=5):
            scores = [(self.score(query, i), i) for i in range(self.N)]
            scores.sort(reverse=True)
            return [(idx, s) for s, idx in scores[:k]]

    # Corpus: product catalogue.
    corpus = [
        'Nike Air Max 90 white running shoe size 10',
        'Adidas Ultraboost 22 black running sneaker',
        'comfortable running shoes for long distance training',
        'Nike Air Max 90 red limited edition',
        'trail running shoes waterproof lightweight',
        'casual white leather sneakers for everyday wear',
        'best shoes for marathon training and recovery',
        'Nike Air Max 270 React grey mesh upper',
    ]

    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(corpus)

    q_exact = 'Nike Air Max 90'
    q_semantic = 'comfortable footwear for long runs'

    print(f'BM25 results for "{q_exact}":')
    for rank, (idx, s) in enumerate(bm25.search(q_exact, k=4)):
        print(f'  [{rank+1}] score={s:.3f}  {corpus[idx][:60]}')

    print(f'\nBM25 results for "{q_semantic}":')
    for rank, (idx, s) in enumerate(bm25.search(q_semantic, k=4)):
        print(f'  [{rank+1}] score={s:.3f}  {corpus[idx][:60]}')
    """),

    md(r"""
    ### 5b Dense retrieval from scratch
    """),

    code(r"""
    # 5b. Simulated dense embeddings (in practice, use sentence-transformers).
    # We simulate by creating topic vectors: each doc gets an embedding
    # that encodes its semantic meaning using predefined axes.
    D = 32  # embedding dimension

    def make_embedding(text, rng, base_seed=None):
        words = text.lower().split()
        vec = rng.normal(0, 0.1, D)
        # Inject signal along semantic axes based on key words.
        running_score = sum(1 for w in words if w in ['running', 'marathon', 'training', 'distance', 'comfortable'])
        nike_score    = sum(1 for w in words if w in ['nike', 'air', 'max', '90', '270'])
        casual_score  = sum(1 for w in words if w in ['casual', 'everyday', 'leather', 'white', 'wear'])
        vec[0] += running_score * 0.8
        vec[1] += nike_score   * 0.8
        vec[2] += casual_score * 0.8
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    corpus_embs = np.stack([make_embedding(doc, rng) for doc in corpus])

    def dense_search(query_emb, corpus_embs, k=5):
        sims = corpus_embs @ query_emb
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(i, float(sims[i])) for i in idx]

    # Query embeddings with matching semantic axes.
    q_exact_emb  = np.array([0.1, 0.9, 0.0] + [0.0]*(D-3), dtype=np.float32)  # Nike-like
    q_exact_emb /= np.linalg.norm(q_exact_emb)
    q_sem_emb    = np.array([0.9, 0.1, 0.0] + [0.0]*(D-3), dtype=np.float32)  # running-like
    q_sem_emb   /= np.linalg.norm(q_sem_emb)

    print('Dense results for "Nike Air Max 90" (exact product):')
    for rank, (idx, s) in enumerate(dense_search(q_exact_emb, corpus_embs, k=4)):
        print(f'  [{rank+1}] score={s:.3f}  {corpus[idx][:60]}')

    print('\nDense results for "comfortable footwear for long runs" (semantic):')
    for rank, (idx, s) in enumerate(dense_search(q_sem_emb, corpus_embs, k=4)):
        print(f'  [{rank+1}] score={s:.3f}  {corpus[idx][:60]}')
    """),

    md(r"""
    ### 5c Reciprocal Rank Fusion (RRF) from scratch
    """),

    code(r"""
    # 5c. RRF: fuse BM25 and dense ranked lists.
    def rrf_fusion(ranked_lists, k=60):
        scores = defaultdict(float)
        for ranked in ranked_lists:
            for rank, (doc_idx, _score) in enumerate(ranked):
                scores[doc_idx] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # For each query, fuse BM25 and dense results.
    for q_text, q_emb, label in [
        ('Nike Air Max 90', q_exact_emb, 'exact product code'),
        ('comfortable footwear for long runs', q_sem_emb, 'semantic intent'),
    ]:
        bm25_results  = bm25.search(q_text, k=8)
        dense_results = dense_search(q_emb, corpus_embs, k=8)
        rrf_results   = rrf_fusion([bm25_results, dense_results], k=60)
        print(f'\nHybrid RRF — "{q_text}" ({label}):')
        for rank, (idx, score) in enumerate(rrf_results[:4]):
            print(f'  [{rank+1}] rrf={score:.5f}  {corpus[idx][:60]}')
    """),

    md(r"""
    ### 5d Alpha-weighted score fusion from scratch
    """),

    code(r"""
    # 5d. Alpha-weighted fusion: normalise scores to [0,1], blend with alpha.
    def alpha_fusion(bm25_results, dense_results, alpha=0.5):
        # Build score dicts.
        bm25_dict  = {idx: s for idx, s in bm25_results}
        dense_dict = {idx: s for idx, s in dense_results}
        all_ids = set(bm25_dict) | set(dense_dict)

        # Min-max normalise each system independently.
        def minmax(d):
            if not d: return d
            lo, hi = min(d.values()), max(d.values())
            rng = hi - lo
            return {k: (v - lo) / rng if rng > 1e-9 else 0.5 for k, v in d.items()}

        bm25_norm  = minmax(bm25_dict)
        dense_norm = minmax(dense_dict)

        fused = {
            idx: alpha * dense_norm.get(idx, 0.0) + (1 - alpha) * bm25_norm.get(idx, 0.0)
            for idx in all_ids
        }
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    for q_text, q_emb, label in [
        ('Nike Air Max 90', q_exact_emb, 'exact product code'),
        ('comfortable footwear for long runs', q_sem_emb, 'semantic intent'),
    ]:
        bm25_res  = bm25.search(q_text, k=8)
        dense_res = dense_search(q_emb, corpus_embs, k=8)
        alpha_res = alpha_fusion(bm25_res, dense_res, alpha=0.5)
        print(f'\nAlpha-fusion (alpha=0.5) — "{q_text}" ({label}):')
        for rank, (idx, score) in enumerate(alpha_res[:4]):
            print(f'  [{rank+1}] score={score:.4f}  {corpus[idx][:60]}')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — BM25 TF saturation curve.
    tf_values = np.linspace(0, 20, 200)
    k1_values = [0.5, 1.2, 1.5, 2.0]
    fig, ax = plt.subplots()
    for k1 in k1_values:
        sat = tf_values * (k1 + 1) / (tf_values + k1)
        ax.plot(tf_values, sat, label=f'k1={k1}')
    ax.axline((0, 0), slope=1, color='gray', ls=':', label='Linear (no saturation)')
    ax.set_xlabel('Raw term frequency f(t,d)')
    ax.set_ylabel('Saturated TF contribution')
    ax.set_title('Figure 1 — BM25 TF saturation curves for different k1')
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** BM25's TF saturation curve. Without saturation (dotted line),
    a term appearing 20× contributes 20× more than appearing 1×. BM25 caps this:
    with $k_1=1.5$, a term appearing 5× contributes only $\sim 2.4\times$ more than
    appearing once. This prevents long documents with repeated terms from dominating.
    Higher $k_1$ = slower saturation (more reward for higher TF); $k_1 \to 0$ =
    pure binary presence/absence. Default $k_1=1.5$ is robust across most corpora.
    """),

    code(r"""
    # Figure 2 — Length normalisation: effect of b parameter.
    avgdl = 10  # average doc length
    dl_values = np.arange(1, 31)   # doc lengths 1 to 30 tokens
    fig, ax = plt.subplots()
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        norms = 1 - b + b * dl_values / avgdl
        ax.plot(dl_values, norms, label=f'b={b}')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Document length |d|'); ax.set_ylabel('Length normalisation factor')
    ax.set_title(f'Figure 2 — BM25 length normalisation (avgdl={avgdl})')
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** Length normalisation factor $1 - b + b \cdot |d|/\text{avgdl}$.
    **$b=0$** (flat line at 1.0): no length normalisation — long documents get no
    penalty. **$b=1$** (steepest slope): full normalisation — scores fully adjusted
    for length, so a document twice the average length has its TF halved. **$b=0.75$**
    (default) is a compromise: penalises long documents but doesn't over-penalise.
    Key insight: if your corpus has highly variable document lengths (tweets vs.
    research papers), tune $b$ upward; if all documents are roughly the same length,
    $b$ matters less.
    """),

    code(r"""
    # Figure 3 — RRF vs alpha-weighted: recall@k across alpha grid.
    # Synthetic ground truth: doc 0 (Nike Air Max 90) and doc 3 (Nike Air Max 90 red)
    # are the relevant results for the "Nike Air Max 90" exact query.
    relevant_exact = {0, 3}    # exact query ground truth
    relevant_sem   = {2, 4, 6} # semantic query ground truth (comfortable running)

    alphas = np.linspace(0, 1, 21)
    recalls_exact, recalls_sem = [], []

    for alpha in alphas:
        for query_text, query_emb, relevant, recall_list in [
            ('Nike Air Max 90', q_exact_emb, relevant_exact, recalls_exact),
            ('comfortable footwear for long runs', q_sem_emb, relevant_sem, recalls_sem),
        ]:
            bm25_res  = bm25.search(query_text, k=8)
            dense_res = dense_search(query_emb, corpus_embs, k=8)
            fused     = alpha_fusion(bm25_res, dense_res, alpha=alpha)
            top3_ids  = set(idx for idx, _ in fused[:3])
            recall_list.append(len(relevant & top3_ids) / len(relevant))

    # RRF recall (fixed, not alpha-dependent).
    rrf_recall_exact = len(relevant_exact & set(
        idx for idx, _ in rrf_fusion([bm25.search('Nike Air Max 90', k=8),
                                       dense_search(q_exact_emb, corpus_embs, k=8)])[:3]
    )) / len(relevant_exact)
    rrf_recall_sem = len(relevant_sem & set(
        idx for idx, _ in rrf_fusion([bm25.search('comfortable footwear for long runs', k=8),
                                       dense_search(q_sem_emb, corpus_embs, k=8)])[:3]
    )) / len(relevant_sem)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, recalls, rrf_r, title in [
        (axes[0], recalls_exact, rrf_recall_exact, 'Exact query: "Nike Air Max 90"'),
        (axes[1], recalls_sem,   rrf_recall_sem,   'Semantic query: "comfortable long runs"'),
    ]:
        ax.plot(alphas, recalls, 'o-', label='Alpha-weighted')
        ax.axhline(rrf_r, color='red', ls='--', label=f'RRF (fixed, R@3={rrf_r:.2f})')
        ax.set_xlabel('alpha (0=BM25 only, 1=dense only)')
        ax.set_ylabel('Recall@3')
        ax.set_title(f'Figure 3 — {title}')
        ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** Two contrasting queries illustrate the alpha-tuning challenge.
    **Left (exact query):** Peak recall at low $\alpha$ (BM25-heavy) — exact token
    matching is essential for "Nike Air Max 90". At $\alpha=1$ (pure dense), recall
    collapses because the dense encoder can't distinguish "Air Max 90" from "Air Max 270"
    without fine-tuning. **Right (semantic query):** Peak recall at higher $\alpha$
    (dense-heavy) — semantic understanding is needed for "comfortable long runs".
    **RRF** (dashed line) achieves a robust compromise on both queries without
    requiring alpha tuning — this is why RRF is the production default. Use
    alpha-weighted only when you have labelled data to tune $\alpha$ per domain.
    """),

    code(r"""
    # Figure 4 — IDF across the corpus: which terms discriminate?
    all_terms = sorted(bm25.df.keys())
    idfs = [bm25.idf(t) for t in all_terms]
    sorted_pairs = sorted(zip(idfs, all_terms), reverse=True)[:15]
    top_idfs, top_terms = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(list(top_terms)[::-1], list(top_idfs)[::-1], color='steelblue')
    ax.set_xlabel('IDF score')
    ax.set_title('Figure 4 — IDF scores: highest-discriminating terms in corpus')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** IDF scores for the product corpus. High-IDF terms (top) appear
    in very few documents — they are the most discriminating. "90", "ultraboost",
    "waterproof", "marathon" each appear in only 1–2 documents, so they strongly
    signal relevance when a query contains them. Low-IDF terms like "running" and
    "shoe" appear in many documents and provide less discrimination. This is why
    BM25 is effective for exact-match queries: rare product codes and model numbers
    get very high IDF and dominate the score for matching documents.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Vocabulary mismatch** | BM25 misses synonyms | Query uses "sneaker", doc says "shoe" | Add query expansion; use hybrid so dense covers it |
    | **Dense exact-match failure** | "SKU-48291" not retrieved | Embedding model encodes as generic token | Ensure BM25 weight in hybrid; consider BM25-only for IDs |
    | **Alpha not tuned** | Hybrid worse than BM25 alone | Alpha=0.5 default wrong for domain | Grid-search alpha on labelled dev set |
    | **BM25 corpus drift** | Recall drops after adding new docs | BM25 IDF not recomputed | Refit BM25 periodically; or use online BM25 |
    | **Score scale mismatch in alpha-fusion** | One system dominates | Raw scores not normalised | Always min-max normalise before alpha fusion; or use RRF |
    | **OOV terms** | BM25 scores 0 for unseen terms | Term not in index | Add stemming/lemmatisation; use BM25+ variant |
    | **Duplicate docs** | Same doc retrieved twice (BM25 and dense) | Both systems return it | De-duplicate by doc ID before returning results |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 LangChain EnsembleRetriever (guarded).
    try:
        from langchain.retrievers import BM25Retriever, EnsembleRetriever  # noqa: F401
        lines = [
            'from langchain.retrievers import BM25Retriever, EnsembleRetriever',
            'from langchain_core.documents import Document',
            '',
            '# Wrap corpus as LangChain Documents.',
            'docs = [Document(page_content=t) for t in corpus]',
            '',
            '# BM25 retriever (sparse).',
            'bm25_retriever = BM25Retriever.from_documents(docs)',
            'bm25_retriever.k = 10',
            '',
            '# Dense retriever (bi-encoder via FAISS, not shown here).',
            '# dense_retriever = ...',
            '',
            '# Ensemble: 40% BM25, 60% dense.',
            'ensemble = EnsembleRetriever(',
            '    retrievers=[bm25_retriever, dense_retriever],',
            '    weights=[0.4, 0.6]',
            ')',
            'results = ensemble.invoke("Nike Air Max 90")',
        ]
        print('\n'.join(lines))
    except ImportError:
        lines = [
            '[langchain not installed — production pattern]:',
            '  from langchain.retrievers import BM25Retriever, EnsembleRetriever',
            '  from langchain_core.documents import Document',
            '  docs = [Document(page_content=t) for t in corpus]',
            '  bm25_ret = BM25Retriever.from_documents(docs); bm25_ret.k = 10',
            '  ensemble = EnsembleRetriever(',
            '      retrievers=[bm25_ret, dense_ret], weights=[0.4, 0.6])',
            '  results = ensemble.invoke("Nike Air Max 90")',
        ]
        print('\n'.join(lines))
    """),

    code(r"""
    # 8.2 rank_bm25 library (guarded).
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
        lines = [
            'from rank_bm25 import BM25Okapi',
            'tokenised = [doc.lower().split() for doc in corpus]',
            'bm25 = BM25Okapi(tokenised, k1=1.5, b=0.75)',
            'scores = bm25.get_scores(["nike", "air", "max", "90"])',
            'top_n = bm25.get_top_n(["nike", "air", "max", "90"], corpus, n=5)',
        ]
        print('\n'.join(lines))
    except ImportError:
        lines = [
            '[rank_bm25 not installed — pattern]:',
            '  pip install rank-bm25',
            '  from rank_bm25 import BM25Okapi',
            '  tokenised = [doc.lower().split() for doc in corpus]',
            '  bm25 = BM25Okapi(tokenised, k1=1.5, b=0.75)',
            '  scores = bm25.get_scores(query_tokens)',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — E-commerce Product Search

    **Scenario.** An e-commerce retailer has 2M product listings. Customers search
    with a mix of semantic intent ("lightweight running shoe for marathons") and
    exact model queries ("Nike Air Max 90 Infrared").

    **Problem.** Pure dense retrieval: misses exact model name queries (the embedding
    distance between "Air Max 90" and "Air Max 270" is small). Pure BM25: fails on
    semantic queries like "comfy sneakers for commuting" when the documents say
    "cushioned walking footwear for urban environments".

    **Solution.** Hybrid search with BM25 + dense + RRF:
    - BM25 index: 2M products, k1=1.5, b=0.75. Rebuilt nightly.
    - Dense index: 2M products, HNSW (Lesson RAG-01), text-embedding-3-small.
    - RRF fusion (k=60) at query time.
    - Top-100 candidates from each system → RRF → top-20 → reranker (Lesson RAG-07).

    **Results (A/B test):**
    - BM25 only: Recall@10 = 0.71, conversion rate 3.1%
    - Dense only: Recall@10 = 0.76, conversion rate 3.4%
    - Hybrid RRF: Recall@10 = 0.89, conversion rate 4.1% (+32% lift)
    """),

    md(r"""
    ## 10 · Production Considerations

    - **BM25 refit cadence.** IDF values change as the corpus grows. Refit BM25 every
      24 hours on the full corpus or incrementally update document frequencies for new
      docs. Stale IDF means newly added rare terms get incorrect (too high) IDF.
    - **Tokenisation alignment.** BM25 and the dense retriever must tokenise queries
      identically. Mismatched lowercasing, stemming, or stop-word removal breaks
      recall on exact-match queries.
    - **RRF k parameter.** Default k=60 is robust. For high-precision retrieval
      (legal, medical), try k=10 to weight rank-1 documents more heavily. For
      low-precision discovery tasks, k=100 gives more uniform weighting.
    - **Corpus-level vs. collection-level BM25.** In a multi-tenant system, BM25
      should be fit per tenant (IDF reflects their document distribution). Shared
      IDF across tenants leaks cross-tenant corpus statistics.
    - **Language-specific tokenisation.** For CJK (Chinese, Japanese, Korean), BM25
      requires character-level or jieba/MeCab tokenisation; whitespace splitting fails.
    - **Hybrid index serving.** Elasticsearch (8.x+) supports native hybrid search
      (BM25 + kNN vector) in a single query. Qdrant and Weaviate have sparse+dense
      hybrid natively. Pinecone requires client-side fusion.
    - **Latency.** BM25 search on 10M docs: ~5ms. Dense ANN (HNSW): ~2ms. RRF fusion
      (client-side, 200 candidates): <1ms. Total latency budget: ~10ms.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Retrieval type comparison:**

    | Method | Exact match | Semantic match | Latency | Memory | Tuning needed |
    |---|---|---|---|---|---|
    | BM25 only | Excellent | Poor | Very fast | Low | k1, b |
    | Dense only | Poor | Excellent | Fast (ANN) | High | None (pretrained) |
    | Hybrid RRF | Good | Good | Slightly slower | High | k (default ok) |
    | Hybrid alpha | Excellent (at right α) | Excellent (at right α) | Slightly slower | High | α per domain |
    | SPLADE (learned sparse) | Excellent | Good | Medium | Medium | Fine-tuning |

    **Fusion strategy comparison:**

    | Strategy | Score-scale sensitive | Requires tuning | Handles missing results | Recommended for |
    |---|---|---|---|---|
    | RRF | No (rank-based) | No (k=60 default) | Yes (rank=N+1) | Default — no labelled data |
    | Alpha-weighted | Yes (need normalisation) | Yes (α per domain) | Needs fill value | When labelled dev set available |
    | Linear combination | Yes | Yes (weights) | Needs fill value | Rarely: use alpha-weighted instead |
    | Max score | No | No | Yes | Aggressive recall: take the best of either |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"Why not just use dense retrieval for everything?"* → Dense models fail on
      exact-match queries: rare product codes, serial numbers, proper nouns not seen
      during training. BEIR benchmark shows BM25 still wins on 6/18 datasets in 2021.
      Hybrid covers both failure modes.
    - *"What is RRF and why is it better than just averaging scores?"* → RRF is
      rank-based fusion: score = $\sum_r 1/(k + \text{rank}_r)$. It's scale-agnostic —
      BM25 scores [0,20] and cosine scores [-1,1] cannot be directly averaged without
      normalisation. RRF avoids this by only using rank order, and the constant k=60
      prevents a single rank-1 outlier from dominating.

    **Deep-dive questions**
    - *"Explain BM25's k1 and b parameters and their effect."* → k1 controls TF
      saturation: at k1=0, it's binary (present/absent); at k1=2, more TF = more
      score (up to a cap). b controls length normalisation: b=0 = no normalisation;
      b=1 = fully normalise by document length relative to corpus average. Tuning:
      longer documents → lower b; diverse length corpus → higher b.
    - *"How would you tune the hybrid fusion for a new domain?"* → Collect 50–200
      representative queries with relevance labels. Grid-search alpha in [0,1] and
      k in [20, 40, 60, 100] to maximise Recall@10 or MRR@10. Alternatively, use
      RRF with default k=60 as a strong no-tune baseline.

    **Whiteboard question**
    - "Design a product search system for 10M SKUs that handles both 'Nike Air Max 90'
      and 'lightweight running shoe' queries. Specify: retrieval system, fusion method,
      parameters, and how you'd measure success."

    **Common mistakes:** choosing alpha=0.5 without tuning (wrong for most domains);
    not normalising scores before alpha fusion; forgetting to de-duplicate results;
    not checking tokenisation alignment between BM25 and dense retriever.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **BM25 vs TF-IDF.** Name the two key innovations BM25 adds over TF-IDF.
       Explain TF saturation with a concrete example.
    2. **IDF formula.** Write BM25 IDF from memory. What does it measure? Why +0.5?
    3. **RRF formula.** Write the RRF score formula. What does the constant k do?
       What happens if k is very small?
    4. **Scale-agnostic.** Explain why RRF is preferred over alpha-weighted when you
       have no labelled data.
    5. **Exact-match failure.** Give a concrete example of a query where dense
       retrieval fails but BM25 succeeds. Explain why.
    6. **b parameter.** A corpus of legal documents has very variable lengths (50 to
       50,000 words). Should you set b higher or lower than 0.75? Why?
    7. **Alpha tuning.** How would you find the optimal alpha for a new domain?
       What metric would you optimise?
    8. **Production latency.** Your hybrid system has BM25 (5ms), dense ANN (3ms),
       RRF fusion (1ms). What is the total p99 latency? How can you reduce it?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Show that BM25 degenerates to binary term matching when $k_1 \to 0$. What
       happens to the TF saturation formula as $k_1 \to 0$?
    2. Two documents contain the query term 5 times each. Document A has 20 tokens;
       document B has 100 tokens. With $k_1=1.5$, $b=0.75$, $\text{avgdl}=50$,
       compute the BM25 TF component for each. Which scores higher and why?

    **Beginner → Intermediate (coding)**
    3. Implement **BM25+** (a variant of BM25 that avoids giving zero scores to
       documents that don't contain all query terms). The formula adds a small floor
       $\delta$ to the IDF. Evaluate its impact on recall@5 for queries where not all
       terms appear in any single document.
    4. Implement **query expansion** for BM25: before scoring, add synonyms for each
       query term from a small pre-defined dictionary (e.g. shoe→sneaker→footwear).
       Measure Recall@3 improvement on the toy corpus for the "comfortable long runs" query.

    **Intermediate (analysis)**
    5. **RRF k ablation**: compute Recall@3 for both queries across k values
       [10, 20, 40, 60, 100, 200]. Plot the results. Is RRF sensitive to k on this
       toy corpus?
    6. **Corpus contamination**: add 50 duplicate documents (copies of doc 0) to the
       BM25 corpus. How does this affect IDF for "nike", "air", "max"? What happens
       to Recall@3 for the exact query? What does this tell you about corpus quality?

    **Senior (design)**
    7. *System design:* a news search system has 50M articles across 20 languages.
       Users search in any language. Design a multilingual hybrid search system:
       how do you handle BM25 for each language (separate index per language?), how
       do you handle cross-lingual dense retrieval (multilingual encoder?), and how
       do you fuse results from different language indexes?
    8. *Interview:* "Our product search uses pure BM25 and our data science team
       wants to add dense retrieval. They're proposing to replace BM25 entirely.
       What is your recommendation and why?" (Expected: recommend hybrid, not
       replacement; explain exact-match failure of dense; propose RRF; describe A/B
       test design.)
    """),

    md(r"""
    ---
    ### Summary
    Hybrid search combines **BM25** (exact token matching, IDF-weighted) with
    **dense vector search** (semantic similarity) to cover both failure modes.
    **Reciprocal Rank Fusion (RRF)** is the production default for fusion — it
    requires no tuning, is scale-agnostic, and is robust across domains.
    **Alpha-weighted fusion** is more precise when labelled data is available.
    Every production RAG system should be hybrid: dense alone fails on product
    codes and proper nouns; BM25 alone fails on semantic intent.

    **Related lesson:** `RAG-07 · Reranking` — the two-stage architecture where fast hybrid
    retrieval (Recall@100) is followed by a slow but accurate cross-encoder reranker
    (Precision@10). Why cross-encoders are more accurate and how to build the pipeline.
    """),
]

build("06_rag/06_hybrid_search.ipynb", cells)
