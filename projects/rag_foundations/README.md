# Measured RAG Foundations

This project is the measurement spine for the retrieval curriculum. It uses a local,
versioned corpus and labelled query/evidence set; no network, paid model, vector
database, mock judge, or generated labels are required.

Implemented comparisons:

- fixed-size, sentence, and structure-aware chunking;
- TF-IDF lexical retrieval;
- LSA dense retrieval using TF-IDF plus truncated SVD;
- hybrid retrieval with reciprocal rank fusion;
- recall@k, MRR, nDCG@k, zero-result rate, abstention, latency, and failure slices.

LSA is a genuine dense statistical representation, but it is **not** a neural
sentence embedding. A later ablation may add a pinned local embedding model while
keeping this offline baseline.

```bash
make rag-foundations-evaluate
make rag-foundations-test
make grounded-rag-evaluate
make grounded-rag-checkpoint
```

The committed retrieval report is written to `artifacts/evaluation.json`; the
grounded-answer report is written to `artifacts/grounded_evaluation.json`. Corpus,
query, and answer hashes make evaluation-data changes visible.

The grounded-answer extension adds gold answers, required answer terms, extractive
answers, evidence citations, abstention, answer correctness, evidence support,
citation validity, component failure taxonomy, and stale/injection/authorization
security cases. Its deliberately simple baseline exposes the difference between
retrieving relevant evidence and selecting the correct evidence to answer.
