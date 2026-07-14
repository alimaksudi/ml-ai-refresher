# Measured RAG Foundations

This project is the measurement spine for the retrieval curriculum. It uses a local,
versioned corpus and labelled query/evidence set; no network, paid model, vector
database, mock judge, or generated labels are required.

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
```

Use `HYBRID_MASTERY_EVIDENCE.md` as a mentor scoring reference after attempting the
hybrid-search teach-back independently.

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
