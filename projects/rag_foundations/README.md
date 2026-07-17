# Measured RAG Foundations

This project is the measurement spine for the retrieval curriculum. Its standard
checkpoints use a local, versioned corpus and labelled query/evidence set; no network,
paid model, vector database server, mock judge, or generated labels are required. An
optional pinned neural reranker extension is isolated from those offline checkpoints.

Implemented comparisons:

- fixed-size, sentence, and structure-aware chunking;
- TF-IDF lexical retrieval;
- LSA dense retrieval using TF-IDF plus truncated SVD;
- hybrid retrieval with reciprocal rank fusion;
- BM25, dense LSA, RRF, and alpha-fusion comparison on the same labels;
- recall@k, MRR, nDCG@k, zero-result rate, abstention, latency, and failure slices.

LSA is a genuine dense statistical representation, but it is **not** a neural
sentence embedding. A later ablation may add a pinned local embedding model while
keeping this offline baseline.

```bash
pip install -r projects/rag_foundations/requirements.txt
make rag-foundations-evaluate
make rag-foundations-test
make grounded-rag-evaluate
make grounded-rag-checkpoint
make vector-store-evaluate
make vector-store-checkpoint
make hybrid-rag-evaluate
make hybrid-rag-checkpoint
make rag-system-evaluate
make rag-system-checkpoint
make reranking-evaluate
make reranking-checkpoint
```

Use `HYBRID_MASTERY_EVIDENCE.md`, `RAG_EVALUATION_MASTERY_EVIDENCE.md`, and
`RERANKING_MASTERY_EVIDENCE.md` as mentor scoring references after attempting each
teach-back independently.

The committed retrieval report is written to `artifacts/evaluation.json`; the
grounded-answer report is written to `artifacts/grounded_evaluation.json`. Corpus,
query, and answer hashes make evaluation-data changes visible.

The grounded-answer extension adds gold answers, required answer terms, extractive
answers, evidence citations, abstention, answer correctness, evidence support,
citation validity, component failure taxonomy, and stale/injection/authorization
security cases. Its deliberately simple baseline exposes the difference between
retrieving relevant evidence and selecting the correct evidence to answer.

The vector-store extension fits the same deterministic LSA representation, keeps
NumPy exact search as the control, persists the vectors and payloads in Qdrant local
mode, closes and reopens the index, and reruns the labelled retrieval set. It also
tests idempotent upserts and mandatory freshness, unsafe-content, and authorization
filters. The committed report is `artifacts/vector_store_evaluation.json`.

Qdrant local mode validates database behavior without a server or network. This tiny
corpus does **not** provide a credible distributed-HNSW throughput benchmark. Use the
same labelled queries against a deployed collection before making scale or ANN claims.

The hybrid-search extension uses the versioned `data/hybrid_corpus.json` and
`data/hybrid_queries.json` benchmark. Its five diagnostic slices cover an exact
identifier, a paraphrase, mixed intent, a no-gain control, and an unanswerable query.
It compares BM25 and four-component dense LSA with RRF and five alpha-weighted
settings on the same structure-aware chunks and labels. The report records candidate
lists, stable evidence IDs, abstention, and local latency in
`artifacts/hybrid_evaluation.json`. These local timings teach measurement mechanics;
they are not production capacity claims.

The EVAL-03 extension compares lexical, dense LSA, and hybrid RRF systems under one
versioned corpus, query, answer, metric, and threshold contract. It keeps labelled
retrieval precision/recall separate from required-term correctness, extractive support,
citation validity, and abstention. Required-term correctness and extractive support are
explicitly named as proxies. Its committed report is
`artifacts/rag_system_evaluation.json`.

The RAG-07 extension freezes each hybrid RRF candidate set before comparing its
original order with a query-aware pair scorer. Passage labels are split into
development and held-out evaluation queries. Only development labels choose the
blend weight. The deterministic scorer uses character n-gram TF-IDF so the benchmark
remains offline and transparent; it is explicitly **not** presented as a neural
cross-encoder. Its committed report is `artifacts/reranking_evaluation.json`.

To measure a real neural cross-encoder without silently downloading a model during
the standard checkpoint, run it explicitly:

```bash
make neural-reranking-evaluate \
  RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
  RERANKER_REVISION=c5ee24cb16019beea0893ab7796b1df96625c6b8
```

Add `LOCAL_FILES_ONLY=1` when that exact revision is already cached. Neural results
belong to that model, revision, machine, candidate set, and label hash; they are not
universal quality or latency claims.
