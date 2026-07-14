# RAG-06 Hybrid Search — Mentor Evidence and Scoring Guide

This document records the automated evidence and a concise mentor reference for the
hybrid-search mastery gate. It does **not** certify learner mastery. The learner must
still explain the decisions without copying this guide and complete a fresh failure
analysis.

## Reproducible evaluation contract

Run:

```bash
make hybrid-rag-checkpoint
```

The committed report uses:

- corpus: `hybrid_corpus.json`, SHA-256
  `4b0d5f2fc2d84e90f573c5ba2cb489580cf6cf331a33b407e0c8f48517bd7e9e`;
- queries: `hybrid_queries.json`, SHA-256
  `256827125f092861ca9f05ad8053fdc6e6994bb0d478e4096c009fc34f227a2e`;
- structure-aware chunks, `top_k=3`, candidate depth 9;
- BM25, four-component dense LSA, RRF, and alpha values 0, 0.25, 0.5,
  0.75, and 1.

Every method must use these same hashes, labels, chunks, and retrieval depths. A
comparison is invalid if more than the retriever or fusion rule changes.

## Manual RRF check

For mixed-intent query `h03`, `devices.zx410` is rank 3 in BM25 and rank 4 in dense
LSA. With `k=60`:

```text
RRF(devices.zx410) = 1 / (60 + 3) + 1 / (60 + 4)
                    = 1 / 63 + 1 / 64
                    = 0.03150 approximately
```

`benefits.parental` is rank 5 in BM25 and rank 8 in dense LSA:

```text
RRF(benefits.parental) = 1 / 65 + 1 / 68
                       = 0.03009 approximately
```

A document absent from a candidate list receives no term from that list. Assigning
an invented worst rank would implement a different algorithm and could promote a
document that the branch never retrieved.

## Diagnostic query evidence

| Query | Intended diagnostic | BM25 | Dense LSA | RRF | Conclusion |
|---|---|---:|---:|---:|---|
| `h01` | Exact identifier | RR 1.0 | RR 0.5 | RR 1.0 | BM25 preserves `ZX-410` better |
| `h02` | Paraphrase | RR 0.5 | RR 1.0 | RR 1.0 | Dense LSA ranks the policy first |
| `h03` | Mixed intent, two relevant sections | Recall@3 0.5 | Recall@3 0.0 | Recall@3 1.0 | RRF recovers both evidence sections |
| `h04` | No-gain control | RR 1.0 | RR 1.0 | RR 1.0 | Fusion adds no ranking value |
| `h05` | Unanswerable | Abstain | Abstain | Abstain | No branch invents evidence |

The aggregate result is BM25 Recall@3 0.875, dense LSA 0.75, and RRF 1.0. RRF's
MRR is 0.875 and nDCG@3 is approximately 0.923. Five diagnostic queries justify a
larger evaluation, not a production deployment claim.

## Policy, identity, and abstention explanation

Authorization, tenant, freshness, and safety filters must constrain **both** branch
candidate sets before fusion. Filtering only the dense branch can let forbidden BM25
results enter the union; filtering only after fusion can expose forbidden IDs in logs
and leave fewer than `top_k` safe results.

The branches keep their own index-consistent preprocessing: BM25 uses its lexical
tokenization while the dense branch uses the representation's tokenizer. Equivalent
policy does not mean identical tokenization.

Fusion and deduplication use immutable chunk IDs. Text equality is insufficient
because two distinct sources can contain identical text. Each result retains document,
section, source/version, and branch position so a reviewer can reconstruct the rank.
Candidate depth 9 gives fusion evidence beyond the final three results. The system
abstains before fusion output when neither branch supplies a meaningful candidate.

## Latency explanation

The saved latency values are local teaching measurements over fifteen small sections.
They do not establish production throughput, ANN recall, tail latency, or capacity.
When sparse and dense retrieval run concurrently, a first latency model is:

```text
total latency ≈ max(sparse branch, dense branch) + fusion + orchestration
```

Production decisions require measured end-to-end percentiles under realistic index
size, concurrency, filters, network distance, and failure conditions.

## Scored failure analysis reference

`h03` is the required component failure analysis:

1. **Evidence trace — 2/2:** BM25 returns only `devices.zx410` in its final three;
   dense returns neither relevant section. Both relevant sections remain inside the
   deeper candidate lists.
2. **Metric interpretation — 2/2:** BM25 Recall@3 is 0.5, dense is 0.0, and RRF is
   1.0. MRR alone would hide the missing second evidence section, so recall is the
   decision metric for this multi-evidence query.
3. **Controlled recommendation — 2/2:** keep candidate depth 9 and RRF for the next
   evaluation; do not change chunking, labels, and fusion simultaneously.
4. **Limitation — 2/2:** the five-query diagnostic set is intentionally small and the
   dense branch is LSA rather than a neural encoder.

Reference score: **8/8**. To pass independently, the learner must reproduce the
reasoning on a fresh query or changed candidate depth and explain any different result.
