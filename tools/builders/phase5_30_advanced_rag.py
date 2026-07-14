"""Builder for Notebook 30 — Advanced RAG Architectures."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 30 · Advanced RAG Architectures
    ### Phase 5 — Retrieval-Augmented Generation · *ML/AI Senior Mastery Curriculum*

    > Naive RAG (single-stage retrieval + generation) fails on ambiguous queries,
    > indirect questions, and topics distributed across many documents. This notebook
    > covers the architectural patterns that fix these failure modes: query
    > transformation (HyDE, multi-query, step-back), selective retrieval (Self-RAG),
    > and hierarchical indexing (RAPTOR). After this notebook you will be able to
    > diagnose which naive RAG failure mode a system exhibits and prescribe the
    > correct architectural fix.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Naive RAG failure modes**: the 5 core failure categories and their root causes.
    - **HyDE** (Hypothetical Document Embeddings): generate a hypothetical answer,
      embed it, use the embedding for retrieval. Why it works on distributional shift.
    - **Multi-query retrieval**: paraphrase the query N times, retrieve for each,
      fuse with RRF. Implement from scratch.
    - **Step-back prompting**: abstract the query to a higher level, retrieve broader
      context first. From scratch.
    - **Self-RAG**: decide whether to retrieve at all; generate retrieval tokens
      inline. The pattern and when it matters.
    - **RAPTOR** (Recursive Abstractive Processing for Tree-Organised Retrieval):
      cluster chunks, summarise clusters, build a hierarchy. Implement a 2-level
      version from scratch.
    - Production patterns with LangChain (guarded).

    **Why it matters**
    - In 2023–2024, query transformation became the highest-value optimisation for
      production RAG beyond chunking and reranking. HyDE consistently improves
      Recall@10 by 10–20% on indirect queries. Multi-query improves recall on
      queries with multiple intents. RAPTOR enables answering summary-level questions
      that single-chunk retrieval cannot handle.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Naive RAG limitations were documented early.** The original RAG paper (Lewis
    et al., 2020) noted that single-step retrieval fails when the query is not
    expressed in the same vocabulary as the relevant passages (the vocabulary gap).

    **HyDE (Gao et al., 2022).** Hypothetical Document Embeddings: use an LLM to
    generate a hypothetical relevant document for the query, then embed *that* for
    retrieval. The hypothetical document is in the same register as indexed documents
    (prose about the topic), while the original query may be a short question. This
    bridges the query-document distributional shift.

    **Multi-query retrieval (LangChain, 2023).** Independently reformulate the query
    N ways. Each reformulation emphasises a different aspect of the information need.
    Retrieve for all, merge with RRF. Reduces sensitivity to the exact phrasing of
    the original query.

    **Step-back prompting (Zheng et al., 2023).** Before answering a specific question,
    generate a more abstract "step-back" question and retrieve broader context. For
    "What is the boiling point of ethanol at 2 atm?" → step-back: "What are the
    properties of ethanol?" → retrieve foundational knowledge first.

    **Self-RAG (Asai et al., 2023).** Fine-tune an LLM to generate special retrieval
    tokens: `[Retrieve]`, `[No Retrieve]`, `[Relevant]`, `[Irrelevant]`. The model
    decides on the fly whether retrieval is needed and whether retrieved passages are
    useful — avoiding retrieval on questions the LLM can answer from parametric memory.

    **RAPTOR (Sarthi et al., 2024).** Build a tree of summaries: chunk-level (leaves)
    → cluster summaries (internal nodes) → full-document summary (root). At query
    time, retrieve from all levels simultaneously. Enables answering both specific
    and high-level questions from the same index.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **The 5 naive RAG failure modes:**

    | Failure | Query example | Root cause | Fix |
    |---|---|---|---|
    | Vocabulary gap | "cardiac arrhythmia treatment" → doc says "heart rhythm disorder therapy" | Query not in same vocab as docs | HyDE, query expansion |
    | Multi-intent query | "compare Python and Java for web dev" | Single query → single perspective retrieved | Multi-query retrieval |
    | Specific → general | "what is ethanol's boiling point at 2atm?" | Specific fact requires foundational knowledge | Step-back prompting |
    | Parametric knowledge sufficient | "What is 2+2?" | Retrieval adds noise | Self-RAG (skip retrieval) |
    | Summary-level question | "What are the main themes of this book?" | No single chunk covers the full scope | RAPTOR hierarchical index |

    **HyDE intuition.** Imagine searching a medical journal database with the query
    "can aspirin prevent heart attacks?". The indexed documents say "aspirin reduces
    platelet aggregation, decreasing myocardial infarction risk". The query and
    document are semantically similar but not lexically. HyDE generates: "Aspirin
    (acetylsalicylic acid) has been shown to reduce cardiovascular events by inhibiting
    platelet function..." — which is lexically and semantically similar to the indexed
    documents. The HyDE embedding is a better retrieval key than the query embedding.

    **RAPTOR intuition.** Think of a corporate knowledge base as a tree:
    - Leaf nodes: individual chunks (specific facts, code snippets).
    - Level-1 nodes: summaries of clusters of related chunks.
    - Level-2 node: overall document summary.
    At query time: "What does this company do?" → retrieve from level-2 summary.
    "How does the payment flow work?" → retrieve from level-1 cluster summary.
    "What is the API endpoint for checkout?" → retrieve from leaf chunks.
    """),

    code(r"""
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import math

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (9, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Shared corpus: enterprise knowledge base (technical + HR + product docs).
    CORPUS = [
        'Python is a high-level interpreted language known for readability and simplicity.',
        'Java is a statically typed compiled language popular for enterprise web applications.',
        'Web development frameworks in Python include Django, Flask, and FastAPI.',
        'Java Spring Boot is widely used for building microservices and REST APIs.',
        'Python has extensive data science libraries including NumPy, Pandas and Scikit-learn.',
        'Java ecosystem includes Maven, Gradle for build management and JUnit for testing.',
        'REST APIs use HTTP methods GET POST PUT DELETE to interact with resources.',
        'GraphQL provides a flexible query language as an alternative to REST.',
        'Database connection pooling improves performance in web applications.',
        'Containerisation with Docker simplifies deployment across different environments.',
        'Kubernetes orchestrates containers providing scaling and self-healing capabilities.',
        'CI/CD pipelines automate building testing and deploying software changes.',
        'Employee onboarding includes IT setup payroll registration and badge activation.',
        'Annual performance reviews occur in December with peer feedback collected in November.',
        'The engineering team uses Jira for project management and Confluence for documentation.',
        'Product roadmap for Q3 includes search improvements and mobile app launch.',
        'Customer support tickets are routed based on severity: P1 critical P2 high P3 normal.',
    ]

    print(f'Corpus: {len(CORPUS)} documents')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 HyDE: distributional shift

    Let $E(q)$ be the embedding of query $q$ and $E(d)$ the embedding of document $d$.
    The retrieval score is $\cos(E(q), E(d))$.

    The distributional shift problem: queries and documents differ in style. A question
    ("what is X?") has different embedding distribution than an explanation ("X is...").

    HyDE generates a hypothetical document $\hat{d}$ via LLM: $\hat{d} = \text{LLM}(q)$.
    Retrieval: $\cos(E(\hat{d}), E(d))$ — now both are in the "document" distribution.

    Expected improvement: let $\sigma^2_{\text{shift}}$ be the variance due to
    query-document distribution shift. HyDE reduces this by mapping the query into
    the document space before computing similarity.

    ### 4.2 Multi-query retrieval with RRF

    Generate $N$ query reformulations $\{q_1, \dots, q_N\}$. Retrieve top-$K$ for each.
    Fuse with RRF:
    $$\text{score}(d) = \sum_{i=1}^{N} \frac{1}{k + \text{rank}_i(d)}$$

    where $k=60$ and $\text{rank}_i(d) = N+1$ if $d$ is not retrieved by query $i$.
    The union of retrieved sets covers $NK$ candidates (before deduplication), increasing
    the probability that any relevant document is included in at least one ranked list.

    ### 4.3 RAPTOR: recursive clustering

    Level 0 (leaves): chunks $\{c_1, \dots, c_n\}$.
    Level 1: cluster into $K$ groups using k-means on embeddings. Summarise each cluster:
    $s_k = \text{LLM}(\text{concat}(c \in \text{cluster}_k))$.
    Level 2: if needed, cluster summaries again.

    At query time, retrieve from all levels simultaneously. The RAPTOR score for a
    relevant document found at level $l$ with similarity $\sigma$:
    $$\text{score}(d, l) = \sigma \cdot w_l$$
    where $w_l$ is a level weight (often $w_0=1, w_1=1, w_2=0.9$ — leaf and cluster
    summaries are equally valuable; root summary has lower specificity).
    """),

    md(r"""
    ## 5 · Implementations from Scratch

    ### 5a Shared utilities
    """),

    code(r"""
    # 5a. Shared utilities: embedding, BM25-style term matching, RRF fusion.

    def embed(text, dim=32):
        # Deterministic word-hash embedding (stand-in for a neural encoder).
        words = text.lower().split()
        vec = np.zeros(dim)
        for w in words:
            h = abs(hash(w)) % (10**9 + 7)
            vec[h % dim] += 1.0
            vec[(h * 31 + 7) % dim] += 0.5
            vec[(h * 17 + 3) % dim] += 0.3
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-9 else vec

    CORPUS_EMBS = np.stack([embed(d) for d in CORPUS])  # (N, 32)

    def cosine_search(query_emb, k=5):
        sims = CORPUS_EMBS @ query_emb
        idx = np.argpartition(sims, -min(k, len(CORPUS)))[-min(k, len(CORPUS)):]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(int(i), float(sims[i])) for i in idx]

    def rrf_fuse(ranked_lists, k=60):
        scores = defaultdict(float)
        for ranked in ranked_lists:
            for rank, (doc_idx, _) in enumerate(ranked):
                scores[doc_idx] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print('Utilities ready.')
    print('Baseline retrieval for "Python web frameworks":')
    q_emb = embed('Python web frameworks')
    for rank, (idx, s) in enumerate(cosine_search(q_emb, k=5)):
        print(f'  [{rank+1}] sim={s:.3f}  {CORPUS[idx][:65]}')
    """),

    md(r"""
    ### 5b HyDE — Hypothetical Document Embeddings
    """),

    code(r"""
    # 5b. HyDE: simulate LLM generating a hypothetical document for the query.
    # In production, the LLM generates a real passage. Here we simulate by
    # constructing a hypothetical passage that uses document-style language.

    def hypothetical_document(query):
        # Simulate LLM output: rephrase the query as a document-style answer.
        # In production: return llm.invoke(f"Write a passage that answers: {query}")
        query_lower = query.lower()
        if 'python' in query_lower and 'web' in query_lower:
            return ('Python web development frameworks include Django which is a batteries '
                    'included framework Flask which is lightweight and FastAPI for async APIs. '
                    'Python is widely adopted for web development due to readability.')
        elif 'java' in query_lower and 'web' in query_lower:
            return ('Java Spring Boot is the dominant framework for enterprise web development. '
                    'Java provides static typing Maven build management and JUnit testing. '
                    'It is commonly used for microservices and REST API development.')
        elif 'container' in query_lower or 'docker' in query_lower:
            return ('Containerisation with Docker packages applications with their dependencies '
                    'into portable containers. Kubernetes orchestrates these containers providing '
                    'auto-scaling load balancing and self-healing for production deployments.')
        else:
            # Generic: return query in statement form.
            return query.replace('?', '').replace('what is ', '').replace('how does ', '') + ' is an important concept in software engineering.'

    def hyde_search(query, k=5):
        hyp_doc = hypothetical_document(query)
        hyp_emb = embed(hyp_doc)
        results = cosine_search(hyp_emb, k=k)
        return results, hyp_doc

    # Compare: direct query vs HyDE for an indirect query.
    queries = [
        'What options exist for building a web server with the snake language?',  # indirect: "snake language" = Python
        'Best JVM-based solutions for REST microservices?',                        # indirect: JVM-based = Java
    ]
    for q in queries:
        print(f'\nQuery: "{q}"')
        direct_results = cosine_search(embed(q), k=3)
        hyde_results, hyp = hyde_search(q, k=3)
        print(f'  HyDE doc: {hyp[:80]}...')
        print('  Direct | HyDE:')
        for (di, ds), (hi, hs) in zip(direct_results, hyde_results):
            d_marker = '*' if di != hi else ' '
            print(f'    Direct [{ds:.3f}] {CORPUS[di][:45]:45s} | HyDE [{hs:.3f}] {CORPUS[hi][:45]}')
    """),

    md(r"""
    ### 5c Multi-query retrieval
    """),

    code(r"""
    # 5c. Multi-query: generate N paraphrases, retrieve for each, fuse with RRF.

    def generate_paraphrases(query, n=3):
        # Simulate LLM paraphrasing. In production: call an LLM.
        # Each paraphrase emphasises a different aspect of the query.
        templates = [
            query,  # original
            ' '.join(reversed(query.split()[:4])) + ' ' + ' '.join(query.split()[4:]),  # reordered keywords
            'What are the key features of ' + query.replace('compare ', '').replace('?', '') + '?',
            'Provide an overview of ' + query.replace('compare ', '').replace('?', ''),
        ]
        return templates[:n]

    def multi_query_search(query, n_queries=3, k_per_query=5):
        paraphrases = generate_paraphrases(query, n=n_queries)
        all_ranked = []
        for pq in paraphrases:
            results = cosine_search(embed(pq), k=k_per_query)
            all_ranked.append(results)
        fused = rrf_fuse(all_ranked, k=60)
        return fused, paraphrases

    q = 'compare Python and Java for web development'
    fused_results, paraphrases = multi_query_search(q, n_queries=3, k_per_query=5)

    print(f'Multi-query for: "{q}"')
    print('Paraphrases used:')
    for i, p in enumerate(paraphrases):
        print(f'  [{i+1}] {p}')

    # Compare single-query vs multi-query.
    single = cosine_search(embed(q), k=5)
    single_ids = set(idx for idx, _ in single)
    multi_ids  = set(idx for idx, _ in fused_results[:5])
    new_in_multi = multi_ids - single_ids
    print(f'\nSingle-query top-5 IDs: {single_ids}')
    print(f'Multi-query  top-5 IDs: {multi_ids}')
    print(f'New docs found by multi-query: {new_in_multi}')
    if new_in_multi:
        for idx in new_in_multi:
            print(f'  >> {CORPUS[idx]}')
    """),

    md(r"""
    ### 5d Step-back prompting
    """),

    code(r"""
    # 5d. Step-back: abstract the query, retrieve broad context, then specific context.

    def step_back_query(query):
        # Simulate LLM step-back question generation.
        q = query.lower()
        if 'python' in q or 'java' in q or 'framework' in q or 'web' in q:
            return 'What are the main programming languages and their use cases?'
        elif 'docker' in q or 'container' in q or 'kubernetes' in q:
            return 'What is software containerisation and deployment?'
        elif 'onboard' in q or 'employee' in q or 'hr' in q:
            return 'What are the company policies and procedures for employees?'
        elif 'database' in q or 'sql' in q:
            return 'What are the core concepts of database management?'
        else:
            words = query.split()
            return 'What is the general context and background of ' + ' '.join(words[:3]) + '?'

    def step_back_search(query, k=3):
        abstract_q = step_back_query(query)
        # Retrieve broad context with step-back query.
        broad_results  = cosine_search(embed(abstract_q), k=k)
        # Retrieve specific context with original query.
        specific_results = cosine_search(embed(query), k=k)
        # Combine: abstract context first (for LLM grounding), then specific.
        combined_ids = []
        seen = set()
        for idx, _ in broad_results + specific_results:
            if idx not in seen:
                combined_ids.append(idx)
                seen.add(idx)
        return combined_ids, abstract_q

    q = 'What is the boiling point of FastAPI at high load?'   # trick: LLM must understand FastAPI is a web framework
    combined, abstract_q = step_back_search(q, k=3)
    print(f'Original query:    "{q}"')
    print(f'Step-back query:   "{abstract_q}"')
    print('Combined results (broad + specific):')
    for rank, idx in enumerate(combined[:5]):
        print(f'  [{rank+1}] {CORPUS[idx][:75]}')
    """),

    md(r"""
    ### 5e Self-RAG: selective retrieval
    """),

    code(r"""
    # 5e. Self-RAG pattern: decide whether to retrieve based on query type.
    # In production, a fine-tuned LLM generates special tokens. Here we simulate
    # with a rule-based classifier as a pedagogical stand-in.

    FACTUAL_KEYWORDS  = ['who', 'what', 'when', 'where', 'how many', 'list', 'name']
    MATH_PATTERNS     = [r'\d+\s*[\+\-\*/\^]\s*\d', r'calculate', r'compute', r'what is \d']
    GENERAL_KNOWLEDGE = ['capital of', 'speed of light', 'definition of', 'who invented',
                         'color of', 'how many planets']

    def should_retrieve(query):
        q = query.lower()
        # Pure math: no retrieval needed.
        for pat in MATH_PATTERNS:
            if re.search(pat, q):
                return False, 'math_query'
        # General world knowledge (parametric): no retrieval.
        for gk in GENERAL_KNOWLEDGE:
            if gk in q:
                return False, 'general_knowledge'
        # Corpus-specific factual query: retrieve.
        for kw in FACTUAL_KEYWORDS:
            if q.startswith(kw) or f' {kw} ' in q:
                return True, 'factual_corpus_query'
        # Default: retrieve for safety.
        return True, 'default_retrieve'

    queries = [
        ('What are Python web frameworks?', True),          # should retrieve
        ('What is 2 + 2?', False),                          # no retrieval needed
        ('What is the capital of France?', False),           # general knowledge
        ('Who is the CEO of our company?', True),           # corpus-specific
        ('How many planets are in the solar system?', False), # general knowledge
    ]

    print('Self-RAG decision:')
    for q, expected in queries:
        retrieve, reason = should_retrieve(q)
        status = 'OK' if retrieve == expected else 'MISMATCH'
        print(f'  [{status}] Retrieve={str(retrieve):5s} ({reason:25s}): {q}')
    """),

    md(r"""
    ### 5f RAPTOR — 2-level hierarchical index
    """),

    code(r"""
    # 5f. RAPTOR: cluster leaf chunks, summarise clusters, build 2-level index.

    def simple_kmeans(embeddings, k, n_iters=10):
        # Minimal k-means: initialise with random centroids.
        idx = rng.choice(len(embeddings), k, replace=False)
        centroids = embeddings[idx].copy()
        labels = np.zeros(len(embeddings), dtype=int)
        for _ in range(n_iters):
            dists = ((embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            labels = np.argmin(dists, axis=1)
            for ki in range(k):
                mask = labels == ki
                if mask.any():
                    centroids[ki] = embeddings[mask].mean(0)
        return labels, centroids

    def summarise_cluster(doc_indices):
        # Simulate LLM summarisation: extract key words from the cluster documents.
        all_words = []
        for idx in doc_indices:
            all_words.extend(CORPUS[idx].lower().split())
        # Keep most frequent content words (poor man's extractive summary).
        stop = {'is', 'a', 'an', 'the', 'for', 'and', 'or', 'in', 'of', 'to',
                'with', 'are', 'used', 'by', 'at', 'as', 'from', 'on', 'its', 'that'}
        counts = defaultdict(int)
        for w in all_words:
            w = re.sub(r'[^a-z]', '', w)
            if w and w not in stop:
                counts[w] += 1
        top_words = [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])[:10]]
        return f'Cluster summary: {" ".join(top_words)}'

    # Build RAPTOR 2-level index.
    K_CLUSTERS = 4
    labels, centroids = simple_kmeans(CORPUS_EMBS, k=K_CLUSTERS)

    # Level 0: original corpus (leaves).
    # Level 1: cluster summaries.
    cluster_summaries = []
    cluster_members   = []
    for ki in range(K_CLUSTERS):
        members = [i for i, l in enumerate(labels) if l == ki]
        summary = summarise_cluster(members)
        cluster_summaries.append(summary)
        cluster_members.append(members)

    cluster_embs = np.stack([embed(s) for s in cluster_summaries])

    print('RAPTOR 2-level index:')
    for ki in range(K_CLUSTERS):
        print(f'\n  Cluster {ki}: {len(cluster_members[ki])} docs')
        print(f'    Summary: {cluster_summaries[ki]}')
        print(f'    Docs: {[CORPUS[i][:50] for i in cluster_members[ki][:2]]}...')

    def raptor_search(query, k_leaves=3, k_clusters=2):
        q_emb = embed(query)
        # Retrieve from leaf level.
        leaf_results = cosine_search(q_emb, k=k_leaves)
        # Retrieve from cluster summary level.
        cluster_sims = cluster_embs @ q_emb
        top_cluster_idx = np.argpartition(cluster_sims, -k_clusters)[-k_clusters:]
        top_cluster_idx = top_cluster_idx[np.argsort(cluster_sims[top_cluster_idx])[::-1]]
        # Expand cluster results to member docs.
        cluster_expanded = []
        for ci in top_cluster_idx:
            for member_idx in cluster_members[ci][:2]:
                cluster_expanded.append((member_idx, float(cluster_sims[ci]) * 0.9))
        # Merge leaf + cluster results with RRF.
        fused = rrf_fuse([leaf_results, cluster_expanded])
        return fused

    print('\nRAPTOR search: "What are the deployment and infrastructure options?"')
    for rank, (idx, s) in enumerate(raptor_search(
        'What are the deployment and infrastructure options?', k_leaves=3, k_clusters=2)[:5]):
        print(f'  [{rank+1}] rrf={s:.5f}  {CORPUS[idx][:65]}')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — HyDE: query embedding vs hypothetical document embedding.
    fig, ax = plt.subplots(figsize=(10, 5))
    dim = 32
    q_text = 'snake language web server options'
    hyp_text = hypothetical_document('snake language web server options')
    q_emb_raw = embed(q_text)
    hyp_emb   = embed(hyp_text)

    # Show first 16 dimensions.
    dims = np.arange(16)
    ax.bar(dims - 0.2, q_emb_raw[:16],   width=0.4, label='Query embedding',          alpha=0.7, color='steelblue')
    ax.bar(dims + 0.2, hyp_emb[:16],     width=0.4, label='HyDE (hypothetical doc)',   alpha=0.7, color='darkorange')
    # Show corpus doc embeddings for context.
    corp_emb = embed(CORPUS[2])  # Python web frameworks doc
    ax.plot(dims, corp_emb[:16], 'gs--', ms=6, label='Corpus doc (Python frameworks)', alpha=0.8)
    ax.set_xlabel('Embedding dimension (first 16 of 32)')
    ax.set_ylabel('Embedding value')
    ax.set_title('Figure 1 — HyDE: query embedding vs hypothetical document embedding')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    q_sim     = float(q_emb_raw @ corp_emb)
    hyde_sim  = float(hyp_emb   @ corp_emb)
    print(f'Cosine sim (query → corpus doc):      {q_sim:.4f}')
    print(f'Cosine sim (HyDE doc → corpus doc):   {hyde_sim:.4f}')
    print(f'HyDE improvement: {hyde_sim - q_sim:+.4f}')
    """),

    md(r"""
    **Figure 1.** HyDE bridges the query-document distribution gap. The raw query
    embedding (blue) has a different activation pattern from the indexed corpus document
    (green dots). The HyDE hypothetical document embedding (orange) more closely
    matches the corpus document embedding — resulting in higher cosine similarity.
    This happens because both the hypothetical document and the indexed documents are
    *statements about a topic*, while the query is a *question about a topic* — a
    different linguistic register. HyDE maps the query into the statement register
    before computing similarity.
    """),

    code(r"""
    # Figure 2 — Multi-query: coverage improvement with N paraphrases.
    true_relevant = {2, 3, 8}   # Python frameworks, Java Spring, DB connection pooling

    n_query_range = [1, 2, 3, 4]
    recalls = []
    for n in n_query_range:
        fused, _ = multi_query_search('compare Python and Java for web development', n_queries=n)
        top5_ids = set(idx for idx, _ in fused[:5])
        recalls.append(len(top5_ids & true_relevant) / len(true_relevant))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_query_range, recalls, 'D-', ms=10, color='seagreen', lw=2)
    ax.set_xlabel('Number of query paraphrases (N)')
    ax.set_ylabel('Recall@5 on ground-truth relevant docs')
    ax.set_title('Figure 2 — Multi-query retrieval: recall improves with N paraphrases')
    ax.set_xticks(n_query_range)
    ax.set_ylim(0, 1.05)
    for n, r in zip(n_query_range, recalls):
        ax.annotate(f'{r:.2f}', (n, r), textcoords='offset points', xytext=(5, 3), fontsize=10)
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** Multi-query retrieval recall improves with the number of paraphrases
    (up to a point). Each additional paraphrase covers a different aspect of the query:
    paraphrase 1 might retrieve Python framework documents; paraphrase 2 might retrieve
    Java documents; paraphrase 3 might retrieve comparison documents. After 3–4
    paraphrases, marginal returns diminish (the new paraphrases no longer surface new
    relevant documents). The cost: $N \times$ retrieval calls and $N \times$ LLM calls
    for paraphrase generation. For production, N=3 is usually the sweet spot.
    """),

    code(r"""
    # Figure 3 — RAPTOR cluster visualisation (PCA to 2D).
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(CORPUS_EMBS)
    except ImportError:
        # Fallback: manual 2-component projection.
        embs_2d = CORPUS_EMBS[:, :2]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple']
    for ki in range(K_CLUSTERS):
        mask = labels == ki
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                   s=120, color=colors[ki], label=f'Cluster {ki}', zorder=3, alpha=0.8)
        for i in np.where(mask)[0]:
            ax.annotate(str(i), (embs_2d[i, 0], embs_2d[i, 1]),
                        fontsize=7, color=colors[ki], alpha=0.7)

    # Plot cluster summary embeddings.
    try:
        summ_2d = pca.transform(cluster_embs)
    except Exception:
        summ_2d = cluster_embs[:, :2]
    ax.scatter(summ_2d[:, 0], summ_2d[:, 1],
               s=300, marker='*', color=[colors[ki] for ki in range(K_CLUSTERS)],
               edgecolors='black', lw=0.8, zorder=5, label='Cluster summaries')
    ax.set_title('Figure 3 — RAPTOR cluster structure (PCA projection)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 3.** RAPTOR cluster structure in 2D (PCA projection). Each colour
    represents one k-means cluster of corpus documents. Stars are the cluster summary
    embeddings (level-1 nodes). The RAPTOR retrieval query goes to both the leaf level
    (numbered dots) and the cluster summary level (stars). Documents in the same cluster
    are topically related — retrieving the star (cluster summary) is equivalent to
    retrieving a "meta-document" that covers all the leaves in that cluster. This
    enables answering high-level questions ("What infrastructure topics are covered?")
    that no single chunk can answer.
    """),

    code(r"""
    # Figure 4 — Architecture comparison: retrieval patterns.
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))

    for ax, (title, query_path, notes) in zip(axes, [
        ('Naive RAG', ['Query', 'Index\n(chunks)', 'Top-k\nchunks', 'LLM'], '1 retrieval call'),
        ('HyDE', ['Query', 'LLM\n(hyp. doc)', 'Index\n(chunks)', 'Top-k\nchunks', 'LLM'], '1 LLM + 1 retrieval'),
        ('Multi-query\n+ RRF', ['Query', 'LLM\n(N rephrases)', 'N×Retrieve', 'RRF\nfuse', 'Top-k\nchunks', 'LLM'],
         'N LLM + N retrieval calls'),
    ]):
        n = len(query_path)
        y = np.linspace(0.9, 0.1, n)
        for i, (step, yi) in enumerate(zip(query_path, y)):
            color = 'steelblue' if i == 0 else ('darkorange' if 'LLM' in step else 'seagreen')
            ax.text(0.5, yi, step, ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.7, ec='gray'))
            if i < n - 1:
                ax.annotate('', xy=(0.5, y[i+1] + 0.04), xytext=(0.5, yi - 0.04),
                            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'{title}\n({notes})', fontsize=9)
    plt.suptitle('Figure 4 — RAG architecture comparison')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** Three RAG architectures at a glance. **Naive RAG** (left): direct
    query → retrieval → generation. Simple but fails on indirect queries. **HyDE**
    (centre): adds one LLM call to generate a hypothetical document before retrieval.
    The extra LLM call adds ~100–300ms but significantly improves retrieval for
    indirect queries. **Multi-query** (right): N LLM calls for paraphrasing + N
    retrieval calls + RRF fusion. Highest recall but highest latency (N×500ms for
    paraphrase generation). Trade-off: use HyDE for single indirect query improvement;
    use multi-query when the original query may have multiple valid interpretations.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **HyDE hallucination** | Hypothetical doc is factually wrong; retrieves wrong passages | LLM makes up facts in the hypothetical | Post-filter: verify retrieved passages support the hypothetical |
    | **Multi-query divergence** | Paraphrases explore off-topic aspects | LLM over-creative paraphrasing | Constrain paraphrase prompt; verify paraphrases are related |
    | **Step-back over-generalises** | Broad context drowns specific answer | Abstract query too broad | Blend step-back and direct retrieval; use both in context |
    | **Self-RAG skips needed retrieval** | Factual error from parametric memory | Classifier wrong on edge cases | Log skip decisions; review cases where RAG skipped |
    | **RAPTOR cluster pollution** | Unrelated docs in same cluster | k-means local minima | Increase n_iters; try multiple random seeds; validate cluster coherence |
    | **Latency budget violation** | HyDE/multi-query too slow | Each adds LLM call | Cache common query transformations; use smaller LLM for paraphrasing |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8. LangChain advanced RAG patterns (guarded).
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever  # noqa: F401
        lines = [
            'from langchain.retrievers.multi_query import MultiQueryRetriever',
            'from langchain_core.language_models import BaseChatModel',
            '',
            '# Multi-query retrieval.',
            'retriever = MultiQueryRetriever.from_llm(',
            '    retriever=base_retriever,   # any LangChain retriever',
            '    llm=llm,                     # used for paraphrase generation',
            ')',
            'results = retriever.invoke("What are Python web frameworks?")',
            '',
            '# HyDE (requires custom runnable chain):',
            'from langchain.chains import HypotheticalDocumentEmbedder',
            'hyde = HypotheticalDocumentEmbedder.from_llm(',
            '    llm=llm, base_embeddings=embeddings, custom_prompt=hyde_prompt)',
            'hyde_retriever = hyde.as_retriever()',
            '',
            '# RAPTOR: use LangChain document loader + custom tree builder.',
            '# No out-of-box RAPTOR in LangChain as of 2024; implement with',
            '# VectorStore.from_documents() at each tree level.',
        ]
        print('\n'.join(lines))
    except ImportError:
        lines = [
            '[langchain not installed — production patterns]:',
            '',
            '  # Multi-query:',
            '  from langchain.retrievers.multi_query import MultiQueryRetriever',
            '  retriever = MultiQueryRetriever.from_llm(retriever=base_ret, llm=llm)',
            '',
            '  # HyDE:',
            '  from langchain.chains import HypotheticalDocumentEmbedder',
            '  hyde = HypotheticalDocumentEmbedder.from_llm(llm=llm, base_embeddings=embs)',
            '',
            '  # Step-back: chain two LLM calls manually:',
            '  abstract_q = llm.invoke(step_back_prompt.format(q=query))',
            '  context = retriever.invoke(abstract_q) + retriever.invoke(query)',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Enterprise Knowledge Base

    **Scenario.** A 5,000-person company has an internal knowledge base with 200K
    documents: HR policies, engineering docs, product wikis, meeting notes. Employees
    ask questions ranging from "how do I book holiday?" (specific, HR) to "what is
    our data strategy?" (high-level, strategic).

    **Failures of naive RAG:**
    - "What are our hiring practices?" → retrieves "interview scheduling" but misses
      "diversity hiring goals" (different vocabulary).
    - "What is the company vision?" → no single chunk covers the full strategic picture.
    - "How do I request hardware?" → answered by parametric LLM knowledge (wrong
      for company-specific process).

    **Advanced RAG solution:**
    - **HyDE**: for indirect vocabulary queries (25% of queries).
    - **RAPTOR 2-level**: cluster by department (HR, Engineering, Product, Finance).
      Level-1 summaries enable "What does Engineering team focus on?" queries.
    - **Self-RAG filter**: skip retrieval for general knowledge questions (saves 30%
      of retrieval calls and reduces noise).
    - **Multi-query**: for ambiguous queries (e.g., "Tell me about onboarding").

    **Results (simulated A/B):**
    - Naive RAG: answer accuracy 61%, user satisfaction 3.1/5.
    - Advanced RAG: answer accuracy 79%, user satisfaction 4.0/5.
    - Main driver: RAPTOR for high-level queries (+12pp) and HyDE for vocabulary
      gap queries (+6pp).
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Latency budget.** HyDE: +100–300ms (one LLM call). Multi-query (N=3): +300–900ms.
      RAPTOR retrieval: same as single-stage (no extra calls at query time). Choose
      techniques based on your latency SLA.
    - **HyDE prompt engineering.** The hypothetical document prompt matters: "Write
      a Wikipedia paragraph that answers: {query}" outperforms "Answer: {query}".
      The prompt should specify the expected document style and domain.
    - **Multi-query deduplication.** After fusing N ranked lists, deduplicate by
      document ID before passing to the reranker. Otherwise the same document appears
      multiple times in the context sent to the LLM.
    - **RAPTOR maintenance.** When new documents are added, incrementally add to the
      leaf level. Periodically re-cluster (weekly or when cluster coherence drops below
      a threshold). Full RAPTOR rebuild for major corpus changes.
    - **Self-RAG logging.** Log every "skip retrieval" decision with the query and
      the generated answer. Review weekly for cases where the LLM confidently hallucinated
      because retrieval was skipped.
    - **Caching.** Cache HyDE and step-back query transformations for frequent queries.
      LRU cache with TTL=1h. Multi-query paraphrases can be cached per original query.
    - **Evaluation.** Advanced RAG components require **end-to-end evaluation** with
      an LLM judge (Notebook 40) — component-level metrics (retrieval Recall@k) may
      not capture generation quality improvements from RAPTOR summaries.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Query transformation comparison:**

    | Technique | Latency overhead | Recall gain | When it helps | When it hurts |
    |---|---|---|---|---|
    | Naive RAG | Baseline | Baseline | Simple factual queries | Indirect or ambiguous queries |
    | HyDE | +1 LLM call | +10–20% recall | Vocabulary gap; indirect queries | Factual queries (hallucination risk) |
    | Multi-query | +N LLM calls | +5–15% recall | Multi-intent queries | Latency-critical paths |
    | Step-back | +1 LLM call | Varies | Specific + foundational context needed | Well-scoped factual queries |
    | Self-RAG | Slight reduction | 0% (avoids noise) | LLM has strong parametric knowledge | Domain-specific facts |
    | RAPTOR | 0 (index-time cost) | +15–25% on summary queries | High-level and specific same corpus | Homogeneous flat corpus |

    **Architecture selection guide:**

    | Query type | Recommended architecture |
    |---|---|
    | Direct factual, same vocab | Naive RAG |
    | Indirect, vocabulary gap | HyDE |
    | Ambiguous, multi-intent | Multi-query + RRF |
    | Specific + foundational context | Step-back |
    | Mix of specific + summary queries | RAPTOR |
    | Parametric knowledge often sufficient | Self-RAG filter |
    | Complex, high-stakes | Multi-query + RAPTOR + Reranker (Notebook 28) |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What is HyDE and why does it improve retrieval?"* → HyDE generates a hypothetical
      relevant document via LLM, embeds it, and uses the hypothetical embedding for
      retrieval. It works because queries and documents are in different linguistic
      registers (questions vs. statements). The hypothetical document is in the same
      register as indexed documents, reducing distributional shift.
    - *"When would you use multi-query retrieval vs. HyDE?"* → Multi-query for queries
      with multiple intents or when the user's vocabulary is uncertain (retrieve from
      multiple angles). HyDE for single queries where the vocabulary gap between query
      and document style is the main failure mode. HyDE is one LLM call; multi-query
      is N calls — use HyDE when latency matters more than exhaustive coverage.

    **Deep-dive questions**
    - *"Explain RAPTOR's advantage over flat indexing for summary questions."* →
      In flat indexing, "What is the overall data strategy?" requires retrieving
      and synthesising 50+ leaf chunks — the LLM context window overflows. RAPTOR
      pre-computes cluster summaries at index time. The cluster summary embedding
      captures the gist of multiple related chunks, enabling a single retrieval to
      surface the synthesised context.
    - *"What is the Self-RAG retrieval decision and how would you implement it in
      production?"* → Self-RAG generates a `[Retrieve]` or `[No Retrieve]` token
      before each generation step. Implementation: fine-tune the LLM with these
      special tokens (requires labelled data). Simpler alternative: classify query
      type (factual, math, general knowledge, domain-specific) and skip retrieval
      for non-domain queries.

    **System design question**
    - "Design the RAG pipeline for a legal research assistant: 10M case documents,
      queries range from specific case citations to strategic questions ('What is
      the court's stance on software patents?'). Specify: chunking (Notebook 29),
      indexing, query transformation, retrieval, and reranking (Notebook 28)."

    **Common mistakes:** using HyDE for factual queries where LLM hallucination in
    the hypothetical doc corrupts retrieval; not deduplicating multi-query results;
    building RAPTOR with too many cluster levels (diminishing returns after 2–3
    levels); omitting end-to-end evaluation.
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **5 naive RAG failure modes.** Name them and give a one-sentence example query
       for each.
    2. **HyDE mechanism.** Describe the 3 steps of HyDE. Why is the hypothetical
       document embedding a better retrieval key than the query embedding?
    3. **Multi-query fusion.** How does RRF combine N ranked lists? What is the
       advantage over taking the union and re-ranking by score?
    4. **Step-back.** Give a concrete example of a query and its step-back version.
       What context does the step-back query retrieve that the direct query misses?
    5. **Self-RAG decision.** What are the two Self-RAG generation tokens? In a
       production system without fine-tuning, how would you approximate this?
    6. **RAPTOR tree.** What is at level 0? Level 1? Level 2 (if present)? How does
       the query reach level-1 nodes without searching level-0 nodes?
    7. **Latency.** You have a 200ms SLA. HyDE adds 200ms. Multi-query (N=3) adds
       600ms. Which techniques can you afford? What alternatives exist?
    8. **Evaluation.** Why is Recall@10 insufficient to evaluate advanced RAG components
       like RAPTOR and step-back? What metric should you use instead?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Explain why HyDE works better for the query "cardiac medication regimes" than
       for the query "what is aspirin?". (Hint: think about parametric knowledge and
       vocabulary gap.)
    2. RAPTOR with $K=4$ clusters and $N=100$ leaf chunks produces 4 cluster summaries.
       If the LLM has a 4K context window and each chunk is 200 tokens, how many leaves
       can you fit directly in context vs. how many cluster summaries?

    **Beginner → Intermediate (coding)**
    3. Implement **cached HyDE**: use a Python dict as a cache mapping query → hypothetical
       document embedding. If a near-identical query has been seen before (cosine sim > 0.95),
       return the cached embedding. Measure the cache hit rate on 20 similar queries.
    4. Implement **step-back + direct fusion**: for each query, retrieve k=3 docs with
       the step-back query and k=3 docs with the direct query. Merge with RRF (weight
       step-back 0.4, direct 0.6). Compare Recall@5 to direct-only on the toy corpus.

    **Intermediate (analysis)**
    5. Ablation study: for the enterprise knowledge base scenario, measure Recall@5 for
       each advanced RAG technique individually (HyDE, multi-query N=3, step-back,
       RAPTOR 2-level) and their combination. Which contributes most? Which adds least?
    6. Implement a **RAPTOR 3-level index**: add a level-2 node that summarises all
       level-1 cluster summaries into a single root document summary. At query time,
       retrieve from level 0, level 1, and level 2 with different weights. When does
       the root summary help?

    **Senior (design)**
    7. *System design:* design the full advanced RAG pipeline for a financial research
       assistant covering 50M news articles and 5M earnings reports. Some queries are
       specific ("What did Apple report in Q3 2024?") and some are thematic ("How has
       semiconductor supply affected tech earnings?"). Specify each component and the
       query routing logic.
    8. *Interview:* "Our RAG system answers 'What did our CEO say about our strategy?'
       incorrectly — it retrieves press releases instead of the internal strategic memo.
       The memo uses different vocabulary than the press releases. Design a solution using
       at most 2 of the techniques from this notebook." (Expected: HyDE to bridge vocabulary
       gap; or multi-query to cover both vocabulary sets; justify the choice with latency
       and complexity tradeoffs.)
    """),

    md(r"""
    ---
    ### Summary
    Advanced RAG architectures address the 5 failure modes of naive single-stage
    retrieval. **HyDE** bridges the query-document vocabulary gap with one LLM call.
    **Multi-query** improves recall for multi-intent queries. **Step-back** provides
    foundational context for specific queries. **Self-RAG** avoids retrieval noise
    for parametric queries. **RAPTOR** enables both specific and summary-level retrieval
    from a hierarchical index built at indexing time. Choose techniques based on the
    failure modes in your corpus and latency constraints.

    **Phase 5 complete!** You now have the full RAG stack: embeddings (NB 21) →
    similarity search (NB 25) → vector databases (NB 26) → hybrid search (NB 27) →
    reranking (NB 28) → chunking (NB 29) → advanced architectures (NB 30).

    **Next: Phase 6 — Agentic AI.** `31 · Agent Fundamentals` — what agents are,
    the ReAct loop (Reason + Act), tool calling, and the core agent architecture
    pattern that underlies all modern AI agents.
    """),
]

build("phase5_rag/30_advanced_rag.ipynb", cells)
