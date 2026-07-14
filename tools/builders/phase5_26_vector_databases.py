"""Builder for Notebook 26 — Vector Databases."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 26 · Vector Databases
    ### Phase 5 — Retrieval-Augmented Generation · *ML/AI Senior Mastery Curriculum*

    > Notebook 25 taught similarity search algorithms (LSH, HNSW, IVF+PQ) and FAISS.
    > FAISS is a pure search library — it has no persistence, no CRUD, no metadata
    > filtering, no multi-tenancy, and no replication. **Vector databases** add all
    > of these production concerns on top of ANN search indexes. This notebook teaches
    > you to build a minimal vector database from scratch, then maps it to production
    > systems (Pinecone, Qdrant, Weaviate, pgvector).
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Why vector databases exist**: what FAISS alone cannot do in production.
    - **VDB architecture**: storage layer, index layer, metadata layer, query planner.
    - **From scratch**: implement a minimal in-memory VDB with `insert`, `delete`,
      `upsert`, `search`, `filter_search`, and `batch_upsert`.
    - **Metadata filtering**: pre-filter vs post-filter approaches, performance tradeoffs.
    - **Production indexing pipeline**: encode → upsert → search → reindex.
    - **System comparison**: Pinecone vs Qdrant vs Weaviate vs pgvector — when to choose.
    - **CRUD semantics**: why deletes and updates require special handling in ANN indexes.
    - **Namespaces and multi-tenancy**: isolating tenants in a shared index.

    **Why it matters**
    - Every production RAG system (Notebooks 27–30) requires a vector database.
      Choosing the wrong system means either performance collapse at scale or inability
      to do metadata-filtered search. Understanding the internals means you can tune
      it for your SLA, debug recall degradation, and design the indexing pipeline.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **2017–2020: The embedding explosion.** BERT (2018) and sentence transformers
    (2019) made high-quality embeddings cheap to produce. Teams began storing millions
    of them in raw NumPy arrays or flat files — workable for prototypes, unworkable
    for production (no CRUD, no filtering, no persistence).

    **2019–2021: FAISS + wrappers.** Meta AI released FAISS (2019). Early production
    systems were FAISS + PostgreSQL for metadata — the "poor man's vector DB". Join
    query: filter in Postgres, fetch candidate IDs, do FAISS vector lookup on IDs.
    Effective but operationally complex.

    **2021–2022: Purpose-built VDBs emerge.** Pinecone (2021, SaaS), Weaviate (2020,
    open-source), Qdrant (2021, open-source) emerged with unified APIs for upsert,
    filtered search, and persistence. They replaced the FAISS+Postgres pattern.

    **2023–: Horizontal scaling.** As RAG pipelines became standard (LangChain 2023,
    LlamaIndex 2023), VDB usage exploded. Systems evolved to add: distributed sharding,
    GPU indexing, multi-vector support (ColBERT), sparse+dense hybrid indexes, and
    quantised storage (Notebook 27 hybrid search).

    **2024: pgvector maturity.** PostgreSQL's pgvector extension reached production
    maturity for organisations already running Postgres — avoiding a new operational
    dependency. For N < 10M vectors at moderate QPS, pgvector is often the correct
    answer.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **What FAISS cannot do.** FAISS is a pure in-memory search library:
    - **No persistence**: the index lives in RAM; restart = rebuild from scratch.
    - **No CRUD**: delete requires rebuilding the index (HNSW has no remove API).
    - **No metadata**: no way to say "only return vectors where `category=shoes`".
    - **No multi-tenancy**: all vectors in one index; can't isolate tenant A from B.
    - **No high availability**: no replication, no failover.

    **VDB architecture layers.**
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │  Client API  (gRPC / REST / Python SDK)                     │
    ├─────────────────────────────────────────────────────────────┤
    │  Query Planner  (parse filter, decide pre/post-filter)      │
    ├─────────────────────────────────────────────────────────────┤
    │  Metadata Store  (key-value or document store for fields)   │
    ├─────────────────────────────────────────────────────────────┤
    │  Vector Index   (HNSW / IVF+PQ / Flat)                     │
    ├─────────────────────────────────────────────────────────────┤
    │  Storage Layer  (WAL + segments on disk; mmap into RAM)     │
    └─────────────────────────────────────────────────────────────┘
    ```

    **Metadata filter strategies.**
    - **Post-filter**: search top-$K' > K$ from the vector index, then filter by
      metadata, return top-$K$ survivors. Risk: if only 1% of vectors match the filter,
      you need $K' = 100K$ — slow and wasteful.
    - **Pre-filter**: evaluate the metadata condition first (bitmap / inverted index),
      restrict the ANN search to matching IDs. Risk: breaks HNSW's graph structure —
      must use brute-force within the filtered set or maintain per-segment indexes.
    - **Segment-level pre-filter** (Qdrant/Weaviate): split the index into segments
      by metadata value; at query time, only search relevant segments. Most practical.
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    import heapq
    import json
    import time
    from collections import defaultdict

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (9, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Cosine similarity (the default metric)

    For L2-normalised vectors $\hat{a}, \hat{b}$:
    $$\text{sim}(\hat{a}, \hat{b}) = \hat{a} \cdot \hat{b} \in [-1, 1]$$

    ### 4.2 Post-filter expected recall

    Let $p$ = fraction of corpus matching the metadata filter (selectivity). Post-filter
    over-fetch factor $f$ required to get $k$ results with probability $> 1 - \delta$:
    $$f \geq \frac{k}{p} \cdot \frac{1}{\text{Recall}_{\text{ANN}}}$$

    For $p = 0.01$, $k = 10$, $\text{Recall}_{\text{ANN}} = 0.95$:
    $f \geq 1053$ — you must retrieve 1053 candidates from the ANN index to get 10
    post-filtered results. This is why low-selectivity filters need pre-filtering.

    ### 4.3 HNSW soft-delete

    HNSW graphs cannot efficiently remove nodes (removing a node breaks its neighbours'
    connections). Standard approach: **tombstone** the node in metadata (mark `deleted=True`);
    skip tombstoned results at query time. Rebuild the index periodically to reclaim space.
    Space overhead = $M \cdot d_{\text{deleted}} \cdot 4$ bytes per deleted vector.

    ### 4.4 Namespace isolation

    A **namespace** is a key that partitions vectors into non-overlapping search domains.
    Implementation options:
    1. **Separate index per namespace**: complete isolation, more memory.
    2. **Shared index + metadata pre-filter**: simpler, but filter cost scales with $N$.
    3. **Segment sharding by namespace**: Weaviate/Qdrant default. Segments = named
       sub-indexes. Query routes only to relevant segments.
    """),

    md(r"""
    ## 5 · Minimal Vector Database from Scratch

    This implementation includes: `insert`, `upsert`, `delete`, `get`, `search`
    (brute-force cosine on normalised vectors), `filter_search` (pre-filter then search),
    `batch_upsert`, `namespace` support, `stats`, and JSON persistence.
    """),

    code(r"""
    # VectorDB: namespaced, metadata-filterable, CRUD, JSON-serialisable.
    class VectorDB:
        def __init__(self, dim):
            self.dim = dim
            # namespace -> {id: {vector, metadata, deleted}}
            self._store = defaultdict(dict)

        # ── Write operations ──────────────────────────────────────────────────────

        def insert(self, id_, vector, metadata=None, namespace='default'):
            if id_ in self._store[namespace]:
                raise ValueError(f'ID {id_!r} already exists. Use upsert to overwrite.')
            self._upsert_internal(id_, vector, metadata, namespace)

        def upsert(self, id_, vector, metadata=None, namespace='default'):
            self._upsert_internal(id_, vector, metadata, namespace)

        def _upsert_internal(self, id_, vector, metadata, namespace):
            vec = np.asarray(vector, dtype=np.float32)
            if vec.shape != (self.dim,):
                raise ValueError(f'Expected dim={self.dim}, got {vec.shape}')
            norm = np.linalg.norm(vec)
            vec_norm = vec / norm if norm > 1e-9 else vec
            self._store[namespace][id_] = {
                'vector': vec_norm,
                'metadata': metadata or {},
                'deleted': False,
            }

        def delete(self, id_, namespace='default'):
            # Soft-delete: tombstone in place (mirrors HNSW graph constraint).
            if id_ not in self._store[namespace]:
                raise KeyError(f'ID {id_!r} not found in namespace {namespace!r}')
            self._store[namespace][id_]['deleted'] = True

        def get(self, id_, namespace='default'):
            rec = self._store[namespace].get(id_)
            if rec is None or rec['deleted']:
                return None
            return {'id': id_, 'metadata': rec['metadata'], 'vector': rec['vector'].tolist()}

        def batch_upsert(self, records, namespace='default'):
            # records: list of (id, vector, metadata) tuples.
            for item in records:
                if len(item) == 3:
                    id_, vec, meta = item
                else:
                    id_, vec = item; meta = {}
                self.upsert(id_, vec, meta, namespace)
            return len(records)

        # ── Read operations ───────────────────────────────────────────────────────

        def search(self, query, k=5, namespace='default', include_metadata=True):
            q = np.asarray(query, dtype=np.float32)
            norm = np.linalg.norm(q)
            q_norm = q / norm if norm > 1e-9 else q

            store = self._store[namespace]
            if not store:
                return []

            # Brute-force cosine: build matrix of active vectors.
            ids, vecs = [], []
            for id_, rec in store.items():
                if not rec['deleted']:
                    ids.append(id_)
                    vecs.append(rec['vector'])

            if not ids:
                return []

            vecs_mat = np.stack(vecs, axis=0)      # (N_active, dim)
            sims = vecs_mat @ q_norm               # (N_active,)
            k = min(k, len(ids))
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

            results = []
            for i in top_idx:
                result = {'id': ids[i], 'score': float(sims[i])}
                if include_metadata:
                    result['metadata'] = store[ids[i]]['metadata']
                results.append(result)
            return results

        def filter_search(self, query, k=5, namespace='default',
                          filter_fn=None, include_metadata=True):
            # Pre-filter by metadata predicate, then cosine search within filtered set.
            q = np.asarray(query, dtype=np.float32)
            norm = np.linalg.norm(q)
            q_norm = q / norm if norm > 1e-9 else q

            store = self._store[namespace]
            ids, vecs = [], []
            for id_, rec in store.items():
                if rec['deleted']:
                    continue
                if filter_fn is not None and not filter_fn(rec['metadata']):
                    continue
                ids.append(id_)
                vecs.append(rec['vector'])

            if not ids:
                return []

            vecs_mat = np.stack(vecs, axis=0)
            sims = vecs_mat @ q_norm
            k = min(k, len(ids))
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

            results = []
            for i in top_idx:
                result = {'id': ids[i], 'score': float(sims[i])}
                if include_metadata:
                    result['metadata'] = store[ids[i]]['metadata']
                results.append(result)
            return results

        # ── Utilities ─────────────────────────────────────────────────────────────

        def compact(self, namespace='default'):
            # Physically remove soft-deleted records (like HNSW index rebuild).
            before = len(self._store[namespace])
            self._store[namespace] = {
                k: v for k, v in self._store[namespace].items() if not v['deleted']
            }
            after = len(self._store[namespace])
            return {'removed': before - after, 'remaining': after}

        def stats(self):
            result = {}
            for ns, store in self._store.items():
                active = sum(1 for v in store.values() if not v['deleted'])
                deleted = sum(1 for v in store.values() if v['deleted'])
                result[ns] = {'active': active, 'deleted': deleted, 'total': len(store)}
            return result

        def save(self, path):
            data = {}
            for ns, store in self._store.items():
                data[ns] = {
                    id_: {
                        'vector': rec['vector'].tolist(),
                        'metadata': rec['metadata'],
                        'deleted': rec['deleted'],
                    }
                    for id_, rec in store.items()
                }
            with open(path, 'w') as f:
                json.dump({'dim': self.dim, 'data': data}, f)

        @classmethod
        def load(cls, path):
            with open(path) as f:
                raw = json.load(f)
            db = cls(raw['dim'])
            for ns, store in raw['data'].items():
                for id_, rec in store.items():
                    db._store[ns][id_] = {
                        'vector': np.array(rec['vector'], dtype=np.float32),
                        'metadata': rec['metadata'],
                        'deleted': rec['deleted'],
                    }
            return db

    print('VectorDB class defined.')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # 6a. Demo: insert, search, filter_search, delete.
    D = 16
    db = VectorDB(dim=D)

    # Insert product catalogue embeddings.
    categories = ['shoes', 'shoes', 'bags', 'bags', 'shoes', 'bags', 'electronics', 'electronics']
    prices      = [120, 85, 350, 220, 65, 180, 999, 450]
    for i in range(8):
        vec = rng.normal(0, 1, D)
        db.insert(f'prod_{i}', vec, metadata={
            'category': categories[i],
            'price': prices[i],
            'in_stock': i % 3 != 0,
        })

    print('--- Unfiltered search (top 3) ---')
    q = rng.normal(0, 1, D)
    for r in db.search(q, k=3):
        print(f"  {r['id']:8s}  score={r['score']:.4f}  {r['metadata']}")

    print('\n--- Filtered search: category=shoes, price<=100 ---')
    def shoes_filter(meta):
        return meta.get('category') == 'shoes' and meta.get('price', 9999) <= 100

    for r in db.filter_search(q, k=3, filter_fn=shoes_filter):
        print(f"  {r['id']:8s}  score={r['score']:.4f}  {r['metadata']}")

    print('\n--- Delete prod_1, check stats ---')
    db.delete('prod_1')
    print('Stats:', db.stats())

    print('\n--- Compact (remove tombstones) ---')
    print('Compact result:', db.compact())
    print('Stats after compact:', db.stats())
    """),

    code(r"""
    # 6b. Namespace isolation demo.
    db2 = VectorDB(dim=D)
    # Tenant A
    for i in range(5):
        db2.upsert(f'doc_{i}', rng.normal(0, 1, D), {'tenant': 'A'}, namespace='tenant_A')
    # Tenant B
    for i in range(5):
        db2.upsert(f'doc_{i}', rng.normal(0, 1, D), {'tenant': 'B'}, namespace='tenant_B')

    q_ns = rng.normal(0, 1, D)
    print('--- Tenant A search ---')
    for r in db2.search(q_ns, k=2, namespace='tenant_A'):
        print(f"  {r['id']}  {r['metadata']}")

    print('\n--- Tenant B search (completely isolated) ---')
    for r in db2.search(q_ns, k=2, namespace='tenant_B'):
        print(f"  {r['id']}  {r['metadata']}")

    print('\nStats:', db2.stats())
    """),

    code(r"""
    # 6c. Persistence: save to disk, reload, verify.
    import tempfile, os
    tmp_path = os.path.join(tempfile.gettempdir(), 'vdb_test.json')
    db2.save(tmp_path)
    db3 = VectorDB.load(tmp_path)
    print(f'Saved and reloaded. Stats: {db3.stats()}')

    # Verify a record round-trips correctly.
    orig = db2.get('doc_0', namespace='tenant_A')
    loaded = db3.get('doc_0', namespace='tenant_A')
    match = np.allclose(orig['vector'], loaded['vector'])
    print(f'Vector round-trip match: {match}')
    os.remove(tmp_path)
    """),

    code(r"""
    # Figure 1 — Pre-filter vs post-filter: recall and latency as selectivity varies.
    N_demo = 2000
    D_demo = 32
    vecs_demo = rng.normal(0, 1, (N_demo, D_demo)).astype(np.float32)
    vecs_demo /= np.linalg.norm(vecs_demo, axis=1, keepdims=True) + 1e-9

    db_bench = VectorDB(dim=D_demo)
    fracs = [round(i * 0.1, 1) for i in range(1, 11)]  # 10% to 100%
    for i in range(N_demo):
        # Assign category based on index so we control selectivity precisely.
        cat = 'A' if i < int(N_demo * 0.1) else 'B'  # start with 10% A
        db_bench.upsert(f'v{i}', vecs_demo[i], {'cat': cat})

    q_bench = rng.normal(0, 1, D_demo).astype(np.float32)

    selectivities, post_times, pre_times = [], [], []
    for frac in fracs:
        # Update selectivity: first frac*N_demo vectors get cat='A'.
        n_A = max(1, int(N_demo * frac))
        for i in range(N_demo):
            db_bench.upsert(f'v{i}', vecs_demo[i], {'cat': 'A' if i < n_A else 'B'})

        # Post-filter: search top-200, filter by cat='A'.
        t0 = time.perf_counter()
        post_results = db_bench.search(q_bench, k=200)
        post_filtered = [r for r in post_results if r['metadata'].get('cat') == 'A'][:5]
        post_times.append(time.perf_counter() - t0)

        # Pre-filter: filter_search with cat='A'.
        t0 = time.perf_counter()
        pre_results = db_bench.filter_search(q_bench, k=5, filter_fn=lambda m: m.get('cat') == 'A')
        pre_times.append(time.perf_counter() - t0)

        selectivities.append(frac)

    fig, ax = plt.subplots()
    ax.plot(selectivities, [t * 1000 for t in post_times], 'o-', label='Post-filter (over-fetch 200)')
    ax.plot(selectivities, [t * 1000 for t in pre_times], 's-', label='Pre-filter (filter then search)')
    ax.set_xlabel('Filter selectivity (fraction matching)'); ax.set_ylabel('Query time (ms)')
    ax.set_title('Figure 1 — Pre-filter vs post-filter latency vs selectivity')
    ax.legend()
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 1.** At high selectivity (most vectors match the filter), post-filter is
    fine — over-fetching 200 to get 5 is only a 40× overhead but recall is high.
    At low selectivity (5–10% match), pre-filter is faster because it restricts the
    search space early. In production, **query planners** (Qdrant, Weaviate) estimate
    selectivity from metadata statistics and switch strategies dynamically. The key
    operational insight: always add an inverted index on high-selectivity filter fields
    (category, tenant_id, date range) so the metadata pre-filter is $O(1)$ or $O(\log N)$
    rather than a full table scan.
    """),

    code(r"""
    # Figure 2 — Batch upsert throughput vs. batch size.
    D_tp = 64
    db_tp = VectorDB(dim=D_tp)
    batch_sizes = [1, 10, 50, 100, 500, 1000]
    throughputs = []

    for bs in batch_sizes:
        recs = [(f'id_{j}', rng.normal(0, 1, D_tp), {'batch': bs})
                for j in range(bs)]
        t0 = time.perf_counter()
        for _ in range(max(1, 1000 // bs)):
            db_tp = VectorDB(dim=D_tp)   # fresh each time
            db_tp.batch_upsert(recs)
        elapsed = (time.perf_counter() - t0) / max(1, 1000 // bs)
        tp = bs / elapsed
        throughputs.append(tp)
        print(f'Batch size {bs:5d}: {tp:8.0f} vectors/s')

    fig, ax = plt.subplots()
    ax.semilogx(batch_sizes, throughputs, 'D-', ms=8)
    ax.set_xlabel('Batch size'); ax.set_ylabel('Throughput (vectors/s)')
    ax.set_title('Figure 2 — Batch upsert throughput vs. batch size')
    plt.tight_layout()
    plt.show()
    """),

    md(r"""
    **Figure 2.** Batch upsert throughput scales roughly linearly with batch size up
    to a point, then plateaus (Python overhead dominates). In production VDBs, batching
    is critical: Pinecone recommends batch sizes of 100–1000; Qdrant uses async batch
    ingestion that buffers writes, sorts by segment, and flushes to WAL. The key
    lesson: **never upsert one vector at a time** in production — 100 vectors/batch
    yields 10–100× better throughput than single-vector upserts.
    """),

    code(r"""
    # Figure 3 — Filter selectivity vs recall (post-filter, k=5, over-fetch=20).
    N_f = 1000; D_f = 32; K = 5; OVERFETCH = 20
    vecs_f = rng.normal(0, 1, (N_f, D_f)).astype(np.float32)
    vecs_f /= np.linalg.norm(vecs_f, axis=1, keepdims=True) + 1e-9

    def simulate_recall(selectivity, n_queries=30):
        # Mark first selectivity*N_f items as 'A', rest as 'B'.
        n_A = max(1, int(N_f * selectivity))
        cats = ['A'] * n_A + ['B'] * (N_f - n_A)

        db_f = VectorDB(dim=D_f)
        for i in range(N_f):
            db_f.upsert(f'v{i}', vecs_f[i], {'cat': cats[i]})

        recalls = []
        for _ in range(n_queries):
            q = rng.normal(0, 1, D_f).astype(np.float32)
            # Ground truth: pre-filter then exact.
            gt = db_f.filter_search(q, k=K, filter_fn=lambda m: m.get('cat') == 'A')
            gt_ids = set(r['id'] for r in gt)
            if not gt_ids:
                continue
            # Post-filter simulation: search top-OVERFETCH, filter.
            candidates = db_f.search(q, k=OVERFETCH)
            post_ids = set(r['id'] for r in candidates if r['metadata'].get('cat') == 'A')
            recalls.append(len(gt_ids & post_ids) / len(gt_ids))
        return float(np.mean(recalls)) if recalls else 0.0

    selectivity_range = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]
    recalls_post = [simulate_recall(s) for s in selectivity_range]

    fig, ax = plt.subplots()
    ax.plot(selectivity_range, recalls_post, 'o-', color='steelblue', ms=8)
    ax.axhline(0.95, color='red', ls='--', label='Target Recall@5=0.95')
    ax.set_xlabel('Filter selectivity (fraction matching)')
    ax.set_ylabel(f'Post-filter Recall@{K} (over-fetch={OVERFETCH})')
    ax.set_title(f'Figure 3 — Post-filter recall vs selectivity (over-fetch={OVERFETCH}x)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    for s, r in zip(selectivity_range, recalls_post):
        print(f'  selectivity={s:.0%}  recall@{K}={r:.3f}')
    """),

    md(r"""
    **Figure 3.** Post-filter recall at fixed over-fetch=$20\times$ collapses as
    selectivity drops below 10%. With 5% selectivity, the expected number of
    matching candidates in the top 20 results is $20 \times 0.05 = 1$, far below
    $k=5$. The solution: **adaptive over-fetch** (increase OVERFETCH as selectivity
    drops) or **pre-filter** (filter first, search within matching set). Production
    systems like Qdrant use a query planner that estimates selectivity from per-field
    histograms and automatically switches strategy.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Index staleness** | New docs not searchable | Upsert buffer not flushed to index | Monitor index lag; set max flush interval |
    | **Delete space leak** | Memory grows after bulk deletes | Soft-deletes never compacted | Schedule periodic compaction / index rebuild |
    | **Filter recall collapse** | Filtered search misses results | Post-filter with low selectivity | Switch to pre-filter; tune over-fetch adaptively |
    | **Namespace pollution** | Tenant A sees Tenant B's vectors | Missing namespace in query | Always pass namespace; validate in API layer |
    | **Dim mismatch** | Upsert fails silently | Embedding model changed | Version-lock model+index; validate dim at ingest |
    | **Codebook drift (PQ)** | Recall degrades over months | New vectors outside training distribution | Re-train codebook quarterly; monitor recall@k |
    | **Replication lag** | Stale reads after recent upsert | Async replication delay | Read from primary for consistency-critical ops |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 Production VDB clients — guarded imports.
    # These patterns show the production API; the logic mirrors our scratch VDB.

    # ── 8.1a: Pinecone pattern ────────────────────────────────────────────────────
    try:
        import pinecone  # noqa: F401
        lines_pinecone = [
            'import pinecone',
            'pc = pinecone.Pinecone(api_key="...")',
            'index = pc.Index("my-index")',
            '',
            '# Upsert (batch recommended)',
            'vectors = [{"id": "v1", "values": embedding, "metadata": {"cat": "shoes"}}]',
            'index.upsert(vectors=vectors, namespace="tenant_A")',
            '',
            '# Filtered search',
            'results = index.query(',
            '    vector=query_embedding, top_k=10, namespace="tenant_A",',
            '    filter={"cat": {"$eq": "shoes"}, "price": {"$lte": 200}},',
            '    include_metadata=True',
            ')',
            '',
            '# Delete by ID',
            'index.delete(ids=["v1"], namespace="tenant_A")',
        ]
        print('\n'.join(lines_pinecone))
    except ImportError:
        lines_pinecone = [
            '[pinecone not installed — production pattern]:',
            '  from pinecone import Pinecone',
            '  pc = Pinecone(api_key="...")',
            '  idx = pc.Index("my-index")',
            '  idx.upsert(vectors=[{"id": "v1", "values": emb, "metadata": {"cat": "shoes"}}])',
            '  results = idx.query(vector=q, top_k=10, filter={"cat": {"$eq": "shoes"}})',
        ]
        print('\n'.join(lines_pinecone))
    """),

    code(r"""
    # 8.1b Qdrant client pattern.
    try:
        from qdrant_client import QdrantClient  # noqa: F401
        lines_qdrant = [
            'from qdrant_client import QdrantClient',
            'from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue',
            '',
            'client = QdrantClient(":memory:")  # or url="http://localhost:6333"',
            'client.create_collection("products", vectors_config=VectorParams(size=768, distance=Distance.COSINE))',
            '',
            '# Upsert with payload (metadata)',
            'points = [PointStruct(id=i, vector=vec, payload={"cat": "shoes", "price": 120})]',
            'client.upsert("products", points=points)',
            '',
            '# Filtered search',
            'results = client.search(',
            '    "products", query_vector=q, limit=10,',
            '    query_filter=Filter(must=[FieldCondition(key="cat", match=MatchValue(value="shoes"))])',
            ')',
        ]
        print('\n'.join(lines_qdrant))
    except ImportError:
        lines_qdrant = [
            '[qdrant_client not installed — production pattern]:',
            '  from qdrant_client import QdrantClient',
            '  from qdrant_client.models import Distance, VectorParams, PointStruct',
            '  client = QdrantClient(url="http://localhost:6333")',
            '  client.create_collection("col", vectors_config=VectorParams(size=768, distance=Distance.COSINE))',
            '  client.upsert("col", points=[PointStruct(id=0, vector=emb, payload={"cat": "A"})])',
            '  results = client.search("col", query_vector=q, limit=10)',
        ]
        print('\n'.join(lines_qdrant))
    """),

    code(r"""
    # 8.1c pgvector (PostgreSQL extension) pattern.
    lines_pgvector = [
        'Production pgvector pattern (psycopg2 + pgvector):',
        '',
        '  CREATE EXTENSION IF NOT EXISTS vector;',
        '  CREATE TABLE products (',
        '      id BIGINT PRIMARY KEY,',
        '      embedding vector(768),',
        '      category TEXT, price NUMERIC',
        '  );',
        '  CREATE INDEX ON products USING hnsw (embedding vector_cosine_ops)',
        '      WITH (m = 16, ef_construction = 64);',
        '',
        '  -- Filtered vector search:',
        '  SELECT id, category, 1 - (embedding <=> %(q)s::vector) AS score',
        '  FROM products',
        '  WHERE category = %(cat)s',
        '  ORDER BY embedding <=> %(q)s',
        '  LIMIT 10;',
        '',
        '  -- Python (psycopg2):',
        '  cursor.execute(sql, {"q": embedding.tolist(), "cat": "shoes"})',
    ]
    print('\n'.join(lines_pgvector))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Multi-Tenant Document Search

    **Scenario.** A B2B SaaS company has 500 enterprise customers. Each customer
    has up to 100K internal documents embedded with `text-embedding-3-small` (d=1536).
    Employees search documents via natural language query.

    **Requirements:**
    - Tenant isolation: company A cannot see company B's documents.
    - Metadata filters: document type (contract/report/email), date range, author.
    - 200ms p99 search latency at 1K QPS (across all tenants).
    - CRUD: documents added/deleted in real-time.
    - Cost target: < $5K/month infrastructure.

    **Solution: Qdrant on dedicated instances, sharded by tenant cluster**
    - 500 tenants × 100K docs × 1536 dims × 4 bytes = 300 GB if shared index.
    - Shard into 10 Qdrant instances by tenant_id hash (10 × 30 GB = 30 GB/node).
    - Each Qdrant node uses HNSW (M=16, ef_construction=128) — high recall.
    - Metadata: `payload` filter on `doc_type` and `date_range` (Qdrant inverted index).
    - CRUD: Qdrant supports `delete_vectors` and `upsert` without index rebuild.
    - Recall@10: 0.97 measured on 500 golden queries per tenant monthly.
    - Latency: p50=18ms, p99=145ms (within SLA).
    - Cost: 10 × 8-core 32GB nodes = $3.2K/month (on-prem) or $4.8K (cloud-managed).

    **Why not Pinecone?** At 500 tenants × 100K docs each = 50M total vectors, Pinecone's
    managed cost would be ~$15K/month. Qdrant self-hosted is 3× cheaper. Why not pgvector?
    At 1K QPS across 10 shards, pgvector HNSW is viable — but Qdrant has better native
    vector-first query planning for complex payload filters.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **WAL (Write-Ahead Log).** Vector inserts must be written to WAL before being
      acknowledged, ensuring durability. If the node crashes, the WAL is replayed on
      restart. Pinecone and Qdrant handle this; pgvector inherits PostgreSQL's WAL.
    - **Replication.** For HA, run 2–3 replicas per shard. Primary handles writes;
      replicas serve reads. Async replication = possible stale reads (acceptable for
      search; not for checkout flows).
    - **Index build latency.** Inserting 1M vectors takes 10–30 minutes to build the
      HNSW graph. Use `IndexFlat` first, migrate to HNSW once ingestion is complete.
    - **Hot reload.** Modern VDBs (Qdrant, Weaviate) support live collection switching
      (blue-green index rebuild) — index the new version while the old serves traffic,
      then swap atomically. Zero-downtime reindex.
    - **Embedding model updates.** When you update the embedding model, you must
      re-embed all documents and rebuild the index. This is a 2–48 hour operation for
      large corpora. Plan the cutover: use the new index for new documents, maintain
      the old index until re-embedding is complete.
    - **Cost levers.** PQ compression (Notebook 25) reduces memory 16–128× at cost of
      2–5pp recall. Matryoshka embeddings (truncate d=1536 → d=256) reduce cost 6× with
      <2pp recall loss. Profile which matters more for your SLA.
    - **Monitoring.** Alert on: Recall@10 on golden queries (weekly), index lag
      (upsert-to-searchable time), p99 latency, memory pressure, disk usage.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **System comparison:**

    | System | Type | Index | Filtered search | CRUD | Self-host | Best for |
    |---|---|---|---|---|---|---|
    | **Pinecone** | Managed SaaS | HNSW+PQ | Yes (payload filter) | Yes (instant) | No | Fast startup, managed ops |
    | **Qdrant** | Open-source | HNSW | Yes (payload + quantised) | Yes | Yes | High-performance, self-hosted |
    | **Weaviate** | Open-source | HNSW | Yes (GraphQL filter) | Yes | Yes | Multi-modal, semantic schema |
    | **pgvector** | PG extension | HNSW / IVF | Yes (SQL WHERE) | Yes (SQL) | Yes | Already on Postgres |
    | **FAISS** | Library | All | No | No | Yes | Research, offline pipelines |
    | **Milvus** | Open-source | IVF+PQ | Yes | Yes | Yes | Billion-scale, cloud-native |

    **Metadata filter tradeoffs:**

    | Strategy | Recall (low selectivity) | Recall (high selectivity) | Latency | Notes |
    |---|---|---|---|---|
    | Post-filter | Low (misses results) | High | Fast | Use when >20% match |
    | Pre-filter (brute) | High | High | Slow (full scan) | Use when <5% match |
    | Adaptive (query planner) | High | High | Optimal | Qdrant/Weaviate default |
    | Segment sharding | High | High | Optimal | Best for categorical filters |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"What does a vector database add over FAISS?"* → Persistence (WAL + disk storage),
      CRUD (upsert/delete without full rebuild), metadata filtering (inverted index +
      query planner), multi-tenancy (namespaces/sharding), high availability
      (replication + failover), and operational tooling (monitoring, backup).
    - *"How do you implement filtered vector search?"* → Pre-filter vs post-filter
      tradeoff (§3, §9). Optimal: adaptive query planner estimates selectivity from
      metadata histograms, switches strategy dynamically. For categorical filters:
      segment-level sharding so each segment is one category value, route query to
      relevant segments only.

    **Deep-dive questions**
    - *"How does Qdrant handle deletes?"* → Soft-delete (tombstone) in HNSW — the node
      remains in the graph but is excluded from results. This avoids the expensive
      HNSW graph repair required by true deletes. Periodic compaction removes tombstoned
      vectors and rebuilds affected segments.
    - *"How would you design the indexing pipeline for a RAG system with 10M documents?"*
      → See the architecture pattern: encode (batch, GPU), upsert (batches of 500,
      Qdrant/Pinecone), metadata index on categorical fields, HNSW with ef_construction=200
      for quality, monitor recall@10 on golden set, plan re-embedding on model updates.
    - *"What is the namespace isolation mechanism and what are its limits?"* → Namespace
      maps to separate sub-index (Pinecone) or collection shard (Qdrant). Limits: cross-
      namespace search requires multiple queries + merge; very high namespace counts (e.g.
      1M users) require per-namespace index management overhead.

    **System design question**
    - "Design the vector storage layer for a legal document search system: 5M contracts,
      50 enterprise clients (strict isolation required), keyword + semantic search, 99.9%
      SLA." → Qdrant with one collection per client (50 collections × 100K docs each);
      HNSW index per collection; sparse+dense hybrid (Notebook 27) for keyword support;
      p99 latency target 200ms; 2 replicas per shard for HA; daily index health checks.

    **Common mistakes:** confusing FAISS with a vector database ("FAISS is a search
    library, not a database"); not planning for embedding model updates (must re-index);
    post-filter with low selectivity (recall collapse); ignoring CRUD semantics
    (HNSW soft-delete overhead accumulates over time).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **FAISS vs VDB.** Name 5 things a vector database provides that raw FAISS cannot.
    2. **Pre-filter vs post-filter.** Explain both approaches. When does post-filter fail
       and what is the failure mode?
    3. **Soft-delete.** Why does HNSW use soft-delete instead of hard-delete? What is
       the operational consequence, and how do you fix it?
    4. **Namespace isolation.** What is a namespace in a VDB? How does it differ from
       a metadata filter?
    5. **Batch upsert.** Why is single-vector upsert inefficient? What is the production
       alternative?
    6. **Filter selectivity.** You have 1M vectors, 1% match the filter, k=10. How many
       candidates must post-filter retrieve to get all 10 results (at 95% ANN recall)?
    7. **Embedding model update.** Your team upgrades the embedding model. What must
       happen to the VDB and how do you ensure zero downtime?
    8. **pgvector vs Pinecone.** When would you choose pgvector over Pinecone for a
       production RAG system?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. Explain why a vector database cannot simply expose FAISS's `IndexHNSWFlat`
       directly as a production service. List at least 5 missing capabilities and
       explain why each matters.
    2. The `VectorDB.compact()` method physically removes soft-deleted records. When
       should you schedule compaction in a production system? What are the tradeoffs
       of running it too frequently vs. too infrequently?

    **Beginner → Intermediate (coding)**
    3. Add an **inverted index** to `VectorDB` for a single metadata field (e.g. `category`):
       maintain a `dict[str, set[str]]` mapping each category value to a set of active
       IDs. Use it in `filter_search` to avoid scanning all vectors when the filter is
       `category == 'shoes'`. Measure the speedup at $N=10000$ vectors.
    4. Implement **adaptive over-fetch** in the `filter_search` method: estimate
       selectivity from the inverted index (ratio of matching IDs to total), then set
       $k_{\text{over-fetch}} = k / \text{selectivity}$, search that many unfiltered
       candidates, and return the top-$k$ matching ones. Compare recall@5 to fixed
       over-fetch at selectivities of 5%, 20%, and 50%.

    **Intermediate (analysis)**
    5. Extend `VectorDB` with a **TTL** (time-to-live) field in metadata: records with
       `expires_at < now()` are treated as deleted in `search` and `filter_search`.
       Add a `purge_expired()` method that hard-deletes all expired records.
    6. Implement **batch search** (multiple queries in one call) in `VectorDB`. Compare
       the vectorised implementation (stack query matrix, batch dot product) to looping
       over individual searches. Report throughput at batch sizes 1, 10, 50.

    **Senior (design)**
    7. *System design:* a news aggregator embeds 10M articles per month (rolling 90-day
       window = 30M active). Articles expire after 90 days. Query: "find recent articles
       similar to this one, published in the last 7 days". Design the VDB architecture:
       index type, filtering strategy for date range, TTL/expiry mechanism, and monthly
       re-indexing plan.
    8. *Interview question:* "We need to search across all our users' documents (each user
       has 0–50K docs, 10M users total). Each user can only see their own docs. We have a
       $10K/month infrastructure budget." Compare: (a) one index per user; (b) shared
       index with namespace filter; (c) sharded index (1000 shards, 10K users per shard).
       Evaluate recall, latency, cost, and operational complexity for each.
    """),

    md(r"""
    ---
    ### Summary
    Vector databases add persistence, CRUD, metadata filtering, multi-tenancy, and HA
    on top of ANN search indexes. **Pre-filter vs post-filter** is the central design
    decision for filtered search — choose adaptively based on selectivity. **Soft-delete**
    is the standard HNSW deletion mechanism; schedule periodic compaction. For $N < 5M$
    and already on Postgres, **pgvector** is often the right choice. For $N > 10M$ or
    complex multi-tenant requirements, **Qdrant** or **Weaviate** self-hosted is
    cost-effective; **Pinecone** for fully managed.

    **Next:** `27 · Hybrid Search` — combining dense vector search with sparse BM25/TF-IDF
    to handle exact keyword matches (product codes, proper nouns) that dense search misses.
    The Reciprocal Rank Fusion (RRF) algorithm for merging ranked lists.
    """),
]

build("phase5_rag/26_vector_databases.ipynb", cells)
