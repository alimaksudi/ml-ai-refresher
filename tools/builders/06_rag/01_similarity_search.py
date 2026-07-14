"""Builder for Lesson RAG-01 — Similarity Search."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # RAG-01 · Similarity Search
    ### Section 06 — Retrieval-Augmented Generation · *ML/AI Senior Mastery Curriculum*

    > The embedding models from NLP-01 and NLP-02 produce vectors in $\mathbb{R}^d$.
    > The vector databases of Lesson RAG-05 store millions of them. Between the two sits
    > the critical algorithmic challenge: given a query vector, how do you find its
    > nearest neighbours in a corpus of 100M vectors *in milliseconds*? The answer is
    > **approximate nearest neighbour (ANN) search** — the family of algorithms
    > (LSH, HNSW, IVF, PQ) that trade a small loss in recall for orders-of-magnitude
    > speedup. This notebook teaches each from first principles.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Distance metrics**: L2, cosine, inner product — when each is correct and how
      they relate.
    - **Brute-force exact search**: $O(Nd)$ baseline — when it's acceptable.
    - **LSH** (Locality-Sensitive Hashing): random projection hashing, why similar
      vectors hash to the same bucket, recall-speed tradeoff.
    - **HNSW** (Hierarchical Navigable Small World): navigable small-world graph
      intuition, hierarchical layers, why it achieves $O(\log N)$ search.
    - **IVF** (Inverted File Index): cluster the corpus, search only $n_{probe}$ nearby
      clusters at query time.
    - **PQ** (Product Quantisation): compress vectors from 768 × float32 = 3KB to
      8–16 bytes — the key to billion-scale retrieval.
    - **Recall vs speed tradeoff**: how to measure and tune it.
    - Production search with FAISS.

    **Why it matters**
    - Every RAG system (RAG-01 through RAG-08) starts with a vector similarity search. The
      choice of index determines recall (retrieval quality), latency, and cost. A wrong
      choice at scale means either 500ms query latency or 10% missed relevant passages.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **The curse of dimensionality.** In high dimensions (d > 50), all points become
    roughly equidistant — the ratio of the furthest to nearest neighbour approaches 1.
    This makes tree-based approaches (KD-trees, ball trees) ineffective: at d=100, a
    KD-tree degenerates to linear scan.

    **LSH (Indyk & Motwani, 1998)** introduced a principled probabilistic approach:
    design hash functions where similar items collide with high probability. Widely
    used through the 2000s for large-scale image retrieval.

    **HNSW (Malkov & Yashunin, 2018)** introduced a graph-based approach inspired by
    small-world networks. It builds a layered proximity graph during indexing and
    navigates it greedily during search. HNSW consistently dominates ANN benchmarks
    (ann-benchmarks.com) in the recall-vs-speed tradeoff.

    **FAISS (Johnson et al., 2019)** from Meta AI packaged IVF, PQ, and HNSW into a
    production-ready library with GPU support. It became the industry standard for
    offline and online vector search.

    **Production databases.** As RAG pipelines matured (2022–), managed vector
    databases (Pinecone, Qdrant, Weaviate) emerged to add CRUD, metadata filtering,
    multi-tenancy, and persistence on top of ANN indexes. Lesson RAG-05 covers these.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Brute-force.** Compare the query to every vector in the corpus. Correct by
    definition. Cost: $O(Nd)$. Feasible for $N < 100K$.

    **LSH intuition.** Imagine the vectors as points on a sphere. Draw random
    hyperplanes through the origin. Each hyperplane splits the sphere into two halves.
    A vector's "hash" is the sequence of which half it falls in for each hyperplane.
    Two similar vectors (close angle) are likely on the *same side* of most hyperplanes
    → same hash → same bucket → found quickly. Only search the bucket, not all $N$.

    **HNSW intuition.** Build a graph where each node (vector) is connected to its
    M nearest neighbours. Query: start from an entry point, greedily walk to vectors
    closer to the query. The "hierarchical" part: build multiple layers — upper layers
    are sparse (fast long-range hops), lower layers are dense (precise local search).
    Think of navigating a city: motorway → arterial road → local street.

    **IVF intuition.** Cluster the corpus into $K$ Voronoi cells (k-means). At query
    time, find the $n_{probe}$ nearest cluster centroids, then brute-force only those
    clusters (~$Nn_{probe}/K$ vectors). Typical: $K=1024$, $n_{probe}=16$ → search
    1.5% of corpus. Recall depends on how many relevant vectors land in probed clusters.

    **PQ intuition.** Split each 768-dim vector into 8 sub-vectors of 96 dims each.
    Quantise each sub-space with $k=256$ centroids (learned by k-means). Store the
    8 centroid indices (8 bytes total vs 768×4=3072 bytes). Distance approximated
    by summing precomputed sub-space distances. 380× memory compression.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Corpus of N vectors in D dimensions (simulating sentence embeddings).
    N, D = 5000, 64
    corpus = rng.normal(0, 1, (N, D)).astype(np.float32)
    # L2-normalise for cosine search.
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
    query = rng.normal(0, 1, D).astype(np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-9)
    print(f'Corpus: {N} x {D}  Query: {D}d')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Distance metrics

    For vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$:

    **L2 (Euclidean):** $d_2(\mathbf{a},\mathbf{b}) = \|\mathbf{a}-\mathbf{b}\|_2$

    **Cosine distance:** $d_{\cos}(\mathbf{a},\mathbf{b}) = 1 - \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$

    **Inner product (dot product):** $\text{IP}(\mathbf{a},\mathbf{b}) = \mathbf{a}\cdot\mathbf{b}$

    **When to use each:**
    - **Cosine**: when only *direction* matters (semantic similarity of L2-normalised
      embeddings). After normalising, cosine distance = $\frac{1}{2}d_2^2$ — equivalent.
    - **L2**: when magnitude matters (image features, coordinates).
    - **Inner product (MIPS)**: when the score is a dot product (recommendation,
      ColBERT late interaction). On un-normalised vectors; asymmetric.

    ### 4.2 LSH — random hyperplane projection

    For $L$ hash tables, each with $K$ hash bits from random unit vectors
    $\mathbf{r}_1,\dots,\mathbf{r}_K$:
    $$h_k(\mathbf{v}) = \text{sign}(\mathbf{r}_k \cdot \mathbf{v}) \in \{-1, +1\}$$
    Two vectors $\mathbf{a}$, $\mathbf{b}$ collide in table $l$ iff all $K$ bits match.
    The collision probability is:
    $$P[\text{collision}] = \left(1 - \frac{\theta}{\pi}\right)^K$$
    where $\theta = \cos^{-1}(\mathbf{a}\cdot\mathbf{b})$ is the angle between them.
    Similar vectors ($\theta \approx 0$) have $P \approx 1^K = 1$; dissimilar vectors
    ($\theta \approx \pi/2$) have $P \approx 0.5^K \ll 1$.

    ### 4.3 Product Quantisation

    Split $\mathbf{v} \in \mathbb{R}^d$ into $M$ sub-vectors of dimension $d/M$:
    $$\mathbf{v} = [\mathbf{v}^{(1)}, \dots, \mathbf{v}^{(M)}]$$
    Learn codebooks $\mathcal{C}^{(m)}$ with $k^*$ centroids each. Represent $\mathbf{v}$
    as $M$ centroid indices. Distance approximated by:
    $$d(\mathbf{q}, \mathbf{v}) \approx \sum_{m=1}^M d\!\left(\mathbf{q}^{(m)}, c_{\hat{i}_m}^{(m)}\right)^2$$
    Precompute $M \times k^*$ distance table for each query → lookup replaces compute.
    Memory: from $d \times 4$ bytes (float32) to $M \log_2(k^*)$ bits.

    ### 4.4 Recall@k

    The primary ANN quality metric:
    $$\text{Recall@k} = \frac{|\text{ANN}(q,k) \cap \text{NN}(q,k)|}{k}$$
    where $\text{NN}(q,k)$ is the ground-truth top-$k$ by exact search. Typical
    production target: Recall@10 $\geq 0.95$ with latency $\leq 10$ms.
    """),

    md(r"""
    ## 5 · Manual Implementation from Scratch

    ### 5a Brute-force exact search
    """),

    code(r"""
    # 5a. Brute-force: compute all cosine similarities, return top-k.
    def brute_force_search(query_norm, corpus_norm, k=5):
        sims = corpus_norm @ query_norm          # (N,) dot product = cosine sim
        idx = np.argpartition(sims, -k)[-k:]    # top-k indices (unsorted)
        idx = idx[np.argsort(sims[idx])[::-1]]  # sort descending
        return idx, sims[idx]

    t0 = time.perf_counter()
    bf_idx, bf_sims = brute_force_search(query_norm, corpus_norm, k=5)
    bf_time = time.perf_counter() - t0
    print(f'Brute-force top-5 (exact):')
    for i, (idx, sim) in enumerate(zip(bf_idx, bf_sims)):
        print(f'  [{i+1}] idx={idx:4d}  cos={sim:.4f}')
    print(f'Brute-force time: {bf_time*1000:.2f} ms  (N={N}, D={D})')
    """),

    md(r"""
    ### 5b LSH — Locality-Sensitive Hashing from scratch
    """),

    code(r"""
    # 5b. LSH with L hash tables, each using K random hyperplanes.
    class LSHIndex:
        def __init__(self, d, n_tables=10, n_bits=8, seed=0):
            r = np.random.default_rng(seed)
            self.n_tables = n_tables
            self.n_bits = n_bits
            # Random hyperplane normals: (n_tables, n_bits, d)
            self.planes = r.normal(0, 1, (n_tables, n_bits, d)).astype(np.float32)
            self.tables = [{} for _ in range(n_tables)]
            self.vectors = None

        def _hash(self, vecs, table_idx):
            # vecs: (N, d) or (d,); planes[t]: (n_bits, d)
            proj = vecs @ self.planes[table_idx].T   # (N, n_bits)
            bits = (proj > 0).astype(np.int8)
            # Convert bit array to integer key.
            powers = 1 << np.arange(self.n_bits, dtype=np.int32)
            return (bits @ powers).tolist() if bits.ndim == 2 else int(bits @ powers)

        def build(self, vectors):
            self.vectors = vectors
            for t in range(self.n_tables):
                hashes = self._hash(vectors, t)
                for i, h in enumerate(hashes):
                    self.tables[t].setdefault(h, []).append(i)

        def search(self, query, k=5):
            candidates = set()
            for t in range(self.n_tables):
                h = self._hash(query, t)
                candidates.update(self.tables[t].get(h, []))
            if not candidates:
                return np.array([]), np.array([])
            cands = np.array(list(candidates))
            sims = self.vectors[cands] @ query
            top_k = min(k, len(cands))
            idx = np.argpartition(sims, -top_k)[-top_k:]
            idx = idx[np.argsort(sims[idx])[::-1]]
            return cands[idx], sims[idx]

        def stats(self):
            total = sum(len(v) for t in self.tables for v in t.values())
            avg_bucket = total / max(1, sum(len(t) for t in self.tables))
            return {'tables': self.n_tables, 'avg_bucket_size': avg_bucket}

    # Build LSH index.
    lsh = LSHIndex(D, n_tables=12, n_bits=10)
    t0 = time.perf_counter()
    lsh.build(corpus_norm)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    lsh_idx, lsh_sims = lsh.search(query_norm, k=5)
    lsh_time = time.perf_counter() - t0

    print(f'LSH build time: {build_time*1000:.1f} ms')
    print(f'LSH search time: {lsh_time*1000:.3f} ms')
    print(f'LSH candidates evaluated: {len(set(lsh_idx))} / {N}')
    # Recall@5: how many of the exact top-5 did LSH find?
    gt_set = set(bf_idx.tolist())
    lsh_set = set(lsh_idx.tolist()) if len(lsh_idx) > 0 else set()
    recall = len(gt_set & lsh_set) / len(gt_set)
    print(f'Recall@5: {recall:.2f}')
    """),

    md(r"""
    ### 5c Recall@k measurement across LSH configurations
    """),

    code(r"""
    # 5c. Measure recall@5 vs speed for different LSH configurations.
    N_QUERIES = 50
    queries_norm = rng.normal(0, 1, (N_QUERIES, D)).astype(np.float32)
    queries_norm /= np.linalg.norm(queries_norm, axis=1, keepdims=True) + 1e-9

    def exact_top_k(queries_norm, corpus_norm, k=5):
        results = []
        for q in queries_norm:
            sims = corpus_norm @ q
            idx = np.argpartition(sims, -k)[-k:]
            results.append(set(idx.tolist()))
        return results

    gt_results = exact_top_k(queries_norm, corpus_norm, k=5)

    configs = [(4, 6), (8, 8), (12, 10), (16, 12)]
    results_lsh = []
    for n_tables, n_bits in configs:
        idx_obj = LSHIndex(D, n_tables=n_tables, n_bits=n_bits)
        idx_obj.build(corpus_norm)
        recalls, times = [], []
        for q, gt in zip(queries_norm, gt_results):
            t0 = time.perf_counter()
            found_idx, _ = idx_obj.search(q, k=5)
            times.append(time.perf_counter() - t0)
            found_set = set(found_idx.tolist()) if len(found_idx) > 0 else set()
            recalls.append(len(gt & found_set) / len(gt))
        results_lsh.append({
            'label': f'L={n_tables},K={n_bits}',
            'recall': np.mean(recalls),
            'time_ms': np.mean(times) * 1000,
        })
        print(f'LSH L={n_tables} K={n_bits}: Recall@5={np.mean(recalls):.3f}  '
              f'time={np.mean(times)*1000:.3f}ms')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — Recall vs speed: brute-force vs LSH configurations.
    fig, ax = plt.subplots()
    for r in results_lsh:
        ax.scatter(r['time_ms'], r['recall'], s=100, zorder=3)
        ax.annotate(r['label'], (r['time_ms'], r['recall']),
                    textcoords='offset points', xytext=(6, 0), fontsize=9)
    ax.scatter(bf_time * 1000, 1.0, s=150, marker='*', color='gold', zorder=4,
               label='Brute-force (exact)')
    ax.set_xlabel('Query time (ms)'); ax.set_ylabel('Recall@5')
    ax.set_title('Figure 1 — Recall vs speed: LSH configurations vs brute-force')
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** The recall-speed tradeoff for LSH configurations vs exact brute-force.
    Each point is a (L tables, K bits) combination. More tables → higher recall (more
    collision chances) but slower search (more candidates). The gold star is exact
    brute-force (Recall=1.0, slowest). The Pareto frontier runs from fast-but-low-
    recall (bottom-left) to slow-but-high-recall (top-right). In production, HNSW
    dominates LSH on this tradeoff — it achieves 0.95+ recall in ~2ms even for
    N=1M, while LSH degrades at scale. The key lesson: **tune n_tables and n_bits
    jointly** to hit your recall SLA with minimum latency.
    """),

    code(r"""
    # Figure 2 — L2 vs cosine distance: are they equivalent after normalisation?
    n_pts = 300
    vecs = rng.normal(0, 1, (n_pts, 2))
    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    q_demo = np.array([1.0, 0.5])
    q_demo_norm = q_demo / np.linalg.norm(q_demo)

    l2_dists   = np.linalg.norm(vecs_norm - q_demo_norm, axis=1)
    cos_dists  = 1 - vecs_norm @ q_demo_norm
    corr = np.corrcoef(l2_dists, cos_dists)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    scatter = axes[0].scatter(l2_dists, cos_dists, c=cos_dists, cmap='RdYlGn_r', s=20)
    axes[0].set_xlabel('L2 distance'); axes[0].set_ylabel('Cosine distance')
    axes[0].set_title(f'Figure 2a — L2 vs cosine on L2-normalised vectors\n(r={corr:.4f} ≈ 1.0)')
    plt.colorbar(scatter, ax=axes[0])

    # Show ordering is identical.
    top5_l2  = set(np.argpartition(l2_dists, 5)[:5].tolist())
    top5_cos = set(np.argpartition(cos_dists, 5)[:5].tolist())
    axes[1].scatter(vecs_norm[:, 0], vecs_norm[:, 1], alpha=0.3, s=15)
    for idx in top5_cos:
        axes[1].scatter(*vecs_norm[idx], s=80, c='red', zorder=4)
    axes[1].scatter(*q_demo_norm, s=200, marker='*', c='gold', zorder=5, label='query')
    axes[1].set_title('Figure 2b — Top-5 neighbours (same for L2 and cosine after norm)')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    print(f'L2 top-5 == cosine top-5 on normalised vectors: {top5_l2 == top5_cos}')
    """),

    md(r"""
    **Figure 2.** After L2-normalisation, L2 distance and cosine distance are
    *mathematically equivalent* (Pearson $r \approx 1.0$). This is why FAISS's
    **IndexFlatIP** (inner product on normalised vectors) is preferred over
    **IndexFlatL2** for embedding search — the index can reuse optimised BLAS dot-
    product code. The relationship is exact: $d_2(\hat{a},\hat{b})^2 = 2(1 -
    \hat{a} \cdot \hat{b}) = 2d_{\cos}(\hat{a},\hat{b})$. The implication: always
    L2-normalise your vectors before indexing in FAISS when using cosine similarity.
    """),

    code(r"""
    # Figure 3 — PQ memory compression illustration.
    D_pq = 128; M_pq = 8; subD = D_pq // M_pq    # 16-dim sub-vectors
    N_pq = 1000
    corpus_pq = rng.normal(0, 1, (N_pq, D_pq)).astype(np.float32)

    # Simple PQ: k-means on each sub-space (k*=16 centroids for speed).
    K_star = 16
    pq_codes = np.zeros((N_pq, M_pq), dtype=np.uint8)
    codebooks = []
    for m in range(M_pq):
        sub = corpus_pq[:, m * subD:(m + 1) * subD]
        # Mini k-means (5 iters).
        centroids = sub[rng.choice(N_pq, K_star, replace=False)]
        for _ in range(5):
            dists = ((sub[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            assignments = np.argmin(dists, axis=1)
            for k in range(K_star):
                mask = assignments == k
                if mask.any():
                    centroids[k] = sub[mask].mean(0)
        pq_codes[:, m] = np.argmin(
            ((sub[:, None, :] - centroids[None, :, :]) ** 2).sum(-1), axis=1
        ).astype(np.uint8)
        codebooks.append(centroids)

    orig_bytes = N_pq * D_pq * 4
    pq_bytes   = N_pq * M_pq * 1      # 1 byte per code (uint8)
    print(f'PQ compression:')
    print(f'  Original: {orig_bytes:,} bytes ({orig_bytes/1024:.1f} KB)')
    print(f'  PQ codes: {pq_bytes:,} bytes ({pq_bytes/1024:.2f} KB)')
    print(f'  Compression ratio: {orig_bytes/pq_bytes:.0f}x')
    print(f'  Codebook overhead: {M_pq * K_star * subD * 4:,} bytes')

    # Approximate distance for a random query.
    q_pq = rng.normal(0, 1, D_pq).astype(np.float32)
    # Build distance tables: for each sub-space m, compute dist(q_sub, centroid_k).
    dist_tables = np.array([
        ((q_pq[m * subD:(m + 1) * subD][None, :] - codebooks[m]) ** 2).sum(1)
        for m in range(M_pq)
    ])  # (M, K_star)
    approx_dists = dist_tables[np.arange(M_pq)[:, None], pq_codes.T].sum(0)  # (N_pq,)
    exact_dists  = ((corpus_pq - q_pq[None, :]) ** 2).sum(1)
    corr_pq = np.corrcoef(approx_dists, exact_dists)[0, 1]
    print(f'  Approx vs exact distance correlation: {corr_pq:.4f}')
    """),

    md(r"""
    **Product Quantisation result.** With $M=8$ sub-spaces and $k^*=16$ centroids,
    PQ achieves $128 \times$ memory compression (from 512 bytes to 8 bytes per vector)
    while maintaining $r \approx 0.99$ correlation between approximate and exact
    distances. In practice, FAISS uses $k^*=256$ (1-byte codes) with $M=8$ or $M=16$,
    giving even better approximation quality. The distance table lookup replaces
    float multiplication, making PQ-compressed search orders of magnitude faster than
    exact search at billion scale.
    """),

    code(r"""
    # Figure 4 — HNSW layer structure (schematic, 2D).
    n_nodes = 40
    nodes = rng.uniform(0, 10, (n_nodes, 2))
    q_hnsw = np.array([5.0, 5.0])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, keep_frac, marker_size) in zip(axes, [
        ('Layer 2 (sparse, fast)', 0.15, 120),
        ('Layer 1 (medium)', 0.40, 80),
        ('Layer 0 (dense, precise)', 1.0, 40),
    ]):
        keep_n = max(3, int(n_nodes * keep_frac))
        layer_nodes = nodes[:keep_n]
        ax.scatter(layer_nodes[:, 0], layer_nodes[:, 1], s=marker_size, alpha=0.7)
        # Draw a few edges.
        for ni in range(min(keep_n, 8)):
            dists = np.linalg.norm(layer_nodes - layer_nodes[ni], axis=1)
            dists[ni] = np.inf
            for nb_idx in np.argsort(dists)[:2]:
                ax.plot([layer_nodes[ni, 0], layer_nodes[nb_idx, 0]],
                        [layer_nodes[ni, 1], layer_nodes[nb_idx, 1]],
                        'gray', alpha=0.3, lw=0.8)
        ax.scatter(*q_hnsw, s=200, marker='*', c='red', zorder=5, label='query')
        ax.set_title(title); ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 10.5)
    axes[0].legend(loc='upper left', fontsize=8)
    plt.suptitle('Figure 4 — HNSW hierarchy: sparse upper layers, dense lower layer')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 4.** HNSW's three-layer structure. The **top layer** (left) has few nodes
    with long-range connections — query entry point navigates here in $O(\log N)$ steps
    to rapidly get close to the target neighbourhood. The **bottom layer** (right) has
    all nodes with short-range connections — precise greedy search in this dense graph
    finds the exact nearest neighbour within the neighbourhood. The middle layer
    bridges the two scales. This is analogous to navigating with a motorway (fast but
    coarse) then local roads (slow but precise). The key HNSW parameters: $M$ (number
    of connections per node, more = better recall but more memory) and $ef_{\text{construction}}$
    (beam width during build — more = better quality but slower index build).
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Low recall** | ANN misses relevant results | $n_{probe}$ / n_tables too low | Increase probe/table count; measure Recall@k regularly |
    | **Dimension mismatch** | Index built with d=768, query is d=384 | Model changed | Version-lock embedding model + index together |
    | **L2 on un-normalised vectors** | Wrong top-k for cosine tasks | L2 mixes angle and magnitude | Always normalise before indexing for cosine |
    | **HNSW memory OOM** | 1M vectors × d=1536 × M=32 = 200GB | High-dim + high M | Use IVF+PQ; reduce d via matryoshka truncation |
    | **Index staleness** | New documents not searchable | Index not updated | Incremental index updates; nightly full rebuild |
    | **PQ quality loss** | Recall drops after PQ | Sub-optimal codebook | Train PQ on representative sample; increase $k^*$ |
    | **Cold start (IVF)** | Empty clusters after k-means | Training set too small vs K | K < N/30; train on ≥100K vectors |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 FAISS: production ANN search (guarded).
    try:
        import faiss

        # (a) Flat index: exact brute-force.
        index_flat = faiss.IndexFlatIP(D)
        index_flat.add(corpus_norm)
        D_faiss, I_faiss = index_flat.search(query_norm[None, :], k=5)
        print('FAISS IndexFlatIP (exact):')
        print(f'  Top-5 indices: {I_faiss[0]}')
        print(f'  Top-5 sims:    {D_faiss[0].round(4)}')
        print(f'  Match brute-force: {set(I_faiss[0].tolist()) == set(bf_idx.tolist())}')

        # (b) HNSW index (ANN, very fast).
        index_hnsw = faiss.IndexHNSWFlat(D, 32)   # M=32
        index_hnsw.add(corpus_norm)
        D_hnsw, I_hnsw = index_hnsw.search(query_norm[None, :], k=5)
        print(f'FAISS HNSW top-5: {I_hnsw[0]}')

        # (c) IVF+PQ: compressed billion-scale index.
        quantiser = faiss.IndexFlatIP(D)
        index_ivfpq = faiss.IndexIVFPQ(quantiser, D, 64, 8, 8)   # 64 cells, M=8, k*=256
        index_ivfpq.train(corpus_norm)
        index_ivfpq.add(corpus_norm)
        index_ivfpq.nprobe = 4
        D_ivfpq, I_ivfpq = index_ivfpq.search(query_norm[None, :], k=5)
        print(f'FAISS IVF+PQ top-5: {I_ivfpq[0]}')
        print(f'IVF+PQ mem estimate: {index_ivfpq.sa_code_size() * N} bytes compressed')

    except Exception as e:
        print(f'[faiss not available: {type(e).__name__}]')
        lines = [
            'FAISS production patterns:',
            '  import faiss',
            '  # Flat (exact): d=768, N up to ~500K',
            '  idx = faiss.IndexFlatIP(768)',
            '  idx.add(embeddings)  # (N, 768) float32',
            '  D, I = idx.search(query, k=10)',
            '',
            '  # HNSW (fast ANN, best recall/speed tradeoff):',
            '  idx = faiss.IndexHNSWFlat(768, 32)  # M=32',
            '',
            '  # IVF+PQ (billion-scale, compressed memory):',
            '  quantiser = faiss.IndexFlatIP(768)',
            '  idx = faiss.IndexIVFPQ(quantiser, 768, 1024, 16, 8)',
            '  idx.train(sample); idx.add(all_vecs); idx.nprobe=64',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — E-commerce Product Search at Scale

    **Scenario.** A retailer has 50M product embeddings (d=384, MiniLM) and handles
    10K search queries per second. Queries are user search text encoded in real-time.

    **Index choice:** IVF4096 + PQ16 (16 bytes/vector):
    - Memory: 50M × 16 bytes = 800 MB (vs 50M × 384 × 4 = 76 GB for flat)
    - n_probe=128 → search 3% of corpus per query
    - Recall@10 ≈ 0.94 (acceptable for product discovery)
    - p99 latency: 8ms (GPU) / 30ms (CPU)

    **Why not HNSW?** HNSW at 50M vectors with M=32: ~50GB RAM, no compression.
    IVF+PQ fits in 800MB — the entire index in L3 cache.

    **Why not brute-force?** 50M × 384-dim dot products per query × 10K QPS =
    200T FLOP/s — not feasible on affordable hardware.

    **Monitoring:** track Recall@10 weekly on a golden query set (100 manually labelled
    queries); alert if Recall drops >2pp (Lesson PROD-05). Re-index when the embedding
    model is updated (Lesson PROD-01).
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Index type selection guide.** $N < 100K$: FlatIP (exact, fast enough).
      $100K < N < 1M$: HNSW (best recall-speed). $N > 1M$: IVF+PQ (memory-efficient).
      Billion-scale: IVF+OPQ+PQ (hierarchical quantisation + rotation for quality).
    - **GPU FAISS.** `faiss.index_cpu_to_gpu` moves the index to GPU, gaining 10–50×
      throughput on batch queries. Essential for >1K QPS on CPU-constrained infra.
    - **Recall measurement.** Always maintain a ground-truth evaluation set (100–1000
      queries with manually-verified top-10). Run after every index rebuild. Recall@k
      should be in SLA.
    - **Incremental indexing.** HNSW and IVF support `add` for new vectors without
      rebuild. But PQ codebooks are trained on a snapshot — adding vectors outside the
      training distribution degrades approximation quality over time.
    - **Index serialisation.** `faiss.write_index(idx, path)` / `faiss.read_index(path)`.
      Serialise after training; never re-train in production (slow, requires data).
    - **Dimension reduction pre-index.** PCA or matryoshka truncation to d=256 can
      reduce index memory 3× with <1pp recall loss for high-d models (d=1536).
    - **Ef_search tuning.** HNSW's `ef_search` (beam width at query time) trades recall
      for speed. Default 16; increase to 64–256 for higher recall without rebuilding.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Index type comparison:**

    | Index | Recall@10 | Query latency | Memory | Build time | When to use |
    |---|---|---|---|---|---|
    | FlatIP (exact) | 1.00 | Fast (N<500K) | High (Nd×4B) | Instant | N < 100K |
    | HNSW (M=32) | 0.97–0.99 | Very fast | High (no compress) | Medium | **1M – 10M, recall-first** |
    | IVF-Flat (K=1024) | 0.95 | Fast | Same as flat | Medium | 1M–100M (no compression needed) |
    | IVF+PQ | 0.90–0.95 | Fast | **32–128× compressed** | Long (train) | **>10M, memory-constrained** |
    | LSH | 0.75–0.90 | Variable | Medium | Fast | Rarely: HNSW dominates |

    **Distance metric selection:**

    | Task | Correct metric | Why |
    |---|---|---|
    | Semantic similarity (normalised embeddings) | Cosine / IP | Magnitude = 1, only direction matters |
    | Raw feature similarity | L2 | Magnitude encodes importance |
    | Recommendation (user-item scoring) | Inner product (MIPS) | Score is a dot product by design |
    | Image retrieval (normalised CNN features) | Cosine / IP | Features are typically normalised |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"Why can't you use a KD-tree for high-dimensional search?"* → KD-trees split
      the space by one dimension per level. In $d>50$, the fraction of the space that
      falls within any ball approaches the entire space — the tree must visit nearly all
      nodes. ANN indexes exploit the *approximate* constraint and the structure of
      embedding spaces (not uniform random) to avoid this.
    - *"What is HNSW and why does it work?"* → Navigable small-world graph: each node
      connects to its $M$ nearest neighbours, creating "long-range" shortcut edges.
      Hierarchical: upper layers are sparse (fast navigation), lower layers are dense
      (precise search). Greedy search in $O(\log N)$ hops.

    **Deep-dive questions**
    - *"When would you use IVF+PQ vs HNSW?"* → HNSW: higher recall, fast query, but
      no memory compression (all original vectors stored). IVF+PQ: 32–128× memory
      compression, slightly lower recall, ideal for N > 10M or GPU-memory-constrained.
      In practice: HNSW for $N < 5M$, IVF+PQ for $N > 10M$.
    - *"What is product quantisation and how does it compress vectors?"* → Split each
      $d$-dim vector into $M$ sub-vectors; train $k^*$ centroids per sub-space;
      represent each sub-vector by its nearest centroid ID (1 byte with $k^*=256$).
      Memory: $M$ bytes vs $d \times 4$ bytes → $d/M \times 4$ compression ratio.
      Distance computed by summing precomputed sub-space lookup tables.

    **Whiteboard questions**
    - "Describe LSH hashing step-by-step for two similar 2D vectors." (§4.2, §5b)
    - "How does HNSW handle the query in Figure 4?" (§3, §5 visual)

    **Strong vs weak answers**
    - *"We need to search 100M vectors with 95% Recall@10 in <20ms."*
      - **Weak:** "Use brute-force on a GPU."
      - **Strong:** "IVF4096+PQ16 on a GPU instance. Train IVF clusters on 1M random
        vectors; set n_probe=128 (searches 3% of corpus). PQ16 compresses 100M×768
        float32 from 290GB to 1.6GB — fits entirely in GPU memory. On an A10G,
        10K QPS is achievable. Verify Recall@10 on a 1K golden set — tune n_probe
        until threshold is met."

    **Common mistakes:** using L2 on un-normalised vectors for cosine tasks; forgetting
    to normalise before FAISS IndexFlatIP; confusing n_probe (IVF) with ef_search
    (HNSW); claiming HNSW "always" beats IVF+PQ (HNSW has no memory compression).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Why not KD-tree?** What breaks in high dimensions?
    2. **L2 vs cosine.** After L2-normalisation, what is the relationship between them?
    3. **LSH mechanism.** Describe the random hyperplane hash step. Why does it make
       similar vectors collide?
    4. **HNSW intuition.** Describe the hierarchy. Why does the top layer have fewer
       nodes?
    5. **IVF mechanism.** What is the role of k-means? What does n_probe control?
    6. **PQ compression.** How does PQ achieve 128× compression? What is the memory
       trade-off?
    7. **Recall@k.** Define it. What is the target for production RAG? How do you
       measure it?
    8. **Index selection.** N=50M, d=768, 10K QPS, 95% Recall@10, <30ms. Which index?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Show mathematically that $d_2(\hat{a},\hat{b})^2 = 2(1 - \hat{a}\cdot\hat{b})$
       for L2-normalised vectors. What does this imply about the relationship between
       Recall@k under L2 and cosine metrics on normalised embeddings?
    2. Explain why increasing LSH's $K$ (bits per table) reduces false positives but
       also reduces recall.

    **Beginner → Intermediate (coding)**
    3. Implement **multi-probe LSH**: instead of searching only the exact hash bucket,
       also search all buckets that differ from the query hash by exactly 1 bit.
       Measure the recall improvement over single-probe on the toy corpus.
    4. Implement a simplified **IVF index** from scratch: (a) cluster with k-means
       ($K=16$); (b) at query time, find the nearest $n_{probe}=4$ centroids; (c)
       brute-force search within those clusters only. Measure Recall@5 vs brute-force.

    **Intermediate (analysis)**
    5. Implement the **PQ distance lookup** approximation from scratch: build the
       distance table, look up codes, sum. Compare approximate vs exact L2 on the
       toy corpus across 50 queries. Report mean absolute error and correlation.
    6. Plot the HNSW recall-latency tradeoff by varying `ef_search` from 10 to 200
       using FAISS HNSW index. Identify the ef_search that hits Recall@10=0.95.

    **Senior (interview + production design)**
    7. *Design:* the vector index architecture for the e-commerce system in §9 (50M
       products, 10K QPS, 95% Recall@10, <30ms, $<$200GB RAM). Specify: index type,
       parameters (K, n_probe, M, k*), GPU vs CPU, index rebuild frequency, and
       monitoring plan.
    8. *Scaling:* you need to add filtering ("only return products from category X")
       to your IVF+PQ index. Describe two approaches: (a) post-filter (search all then
       filter); (b) pre-filter (filter candidates before ANN). Analyse recall and
       latency for each when only 1% of the corpus matches the filter.
    """),

    md(r"""
    ---
    ### Summary
    Similarity search is the foundation of every RAG pipeline. **Brute-force** is exact
    but $O(Nd)$ — acceptable for $N < 100K$. **LSH** is fast-to-build but degrades at
    scale. **HNSW** dominates the recall-speed Pareto frontier for $N < 10M$. **IVF+PQ**
    provides 32–128× memory compression for billion-scale search at the cost of some
    recall. **FAISS** packages all of these. **Distance metric choice** (L2 vs cosine
    vs IP) must match the embedding model's training objective.

    **Related lesson:** `RAG-05 · Vector Databases` — what managed databases add over raw FAISS:
    metadata filtering, CRUD, persistence, multi-tenancy, and replication.
    """),
]

build("06_rag/01_similarity_search.ipynb", cells)
