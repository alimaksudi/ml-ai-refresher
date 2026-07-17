# RAG-07 Reranking — Beginner Audit, Evidence, and Scoring Guide

This document records the learner audit and reproducible evidence for RAG-07. It
does **not** certify learner mastery. The learner must complete a fresh held-out trace
diagnosis without copying this guide.

## Beginner audit outcome

RAG-07 belongs after RAG-06 and EVAL-03. Hybrid retrieval must first produce a
candidate set, and component evaluation must show that useful evidence is present but
poorly ordered. The earlier lesson was correctly located but was not beginner-ready:

- it assumed earlier lessons had established Recall@100, although the measured course
  benchmarks use smaller declared cutoffs;
- it introduced top-100/top-10, BERT internals, monoBERT, duoBERT, ColBERT, APIs,
  GPU serving, and production SLAs before a small fixed-list reorder;
- it called a hand-written overlap function a cross-encoder even though no neural
  model jointly encoded the text pair;
- it repeated BM25, dense retrieval, and RRF implementations instead of reusing the
  tested project spine;
- it presented synthetic attention, simulated latency, invented business metrics,
  and universal tuning rules as though they were measured evidence;
- it used section labels to discuss passage ranking and did not protect an evaluation
  split from tuning;
- it showed only improvement narratives, with no regression, no-gain case, or
  candidate-miss ceiling.

The rebuilt lesson starts with three cards, reorders them manually, calculates one
reciprocal-rank change, and only then introduces pair scoring and a neural
cross-encoder. Production details are an extension path.

## Prerequisite and placement decision

| Requirement | Taught earlier? | Why RAG-07 needs it |
|---|---|---|
| Candidate retrieval and stable evidence IDs | Yes, RAG-03 through RAG-06 | Reranking may reorder only the retrieved IDs |
| Hybrid RRF and candidate depth | Yes, RAG-06 | The benchmark freezes this first-stage candidate list |
| Recall, MRR, and nDCG | Yes, EVAL-03 and its prerequisites | Recall defines the ceiling; MRR/nDCG detect ordering changes |
| Train/validation/test separation and leakage | Yes, ML foundations and EVAL-03 | Development labels may tune alpha; evaluation labels may not |
| Bi-encoder intuition | Yes, NLP-02 and RAG-01 | It explains why first-stage representations can be precomputed |
| Cross-encoder joint scoring | Introduced here | It is the new concept, so it must not be assumed |

Immediate predecessor: **EVAL-03 RAG evaluation**. Immediate successor after mastery:
**RAG-08 Advanced RAG**. No sequence move is recommended.

Dependency map:

```text
Hybrid candidate retrieval
→ required before reranking
→ because a reranker can only reorder candidates that retrieval supplied

Candidate recall
→ required before MRR or nDCG interpretation
→ because missing evidence places a ceiling on every downstream ordering

Development/evaluation separation
→ required before reranker tuning
→ because choosing weights on evaluation labels leaks the answers into the decision

Pair-scoring intuition
→ required before neural cross-encoder internals
→ because the learner should understand the operation before the architecture
```

## Reproducible benchmark

Run:

```bash
make reranking-checkpoint
```

The report freezes sentence-level candidates from hybrid RRF at `candidate_k=15`.
Passage labels, not broad section labels, define the desired ordering. Queries `q01`
through `q08` plus unanswerable `q17` form the development split. Queries `q09`
through `q16` plus unanswerable `q18` remain held out for final evaluation.

The offline teaching reranker uses character n-gram TF-IDF pair similarity blended
with reciprocal candidate position. Only development MRR chooses alpha from
`0.0, 0.1, ..., 1.0`; the selected value is `0.7`. This scorer is transparent and
deterministic, but it is **not** a neural cross-encoder.

| Held-out metric | Original hybrid order | Local pair reranker | What it supports |
|---|---:|---:|---|
| Candidate passage recall | 0.8125 | 0.8125 | Candidate membership is fixed |
| Candidate hit rate | 0.8750 | 0.8750 | One query has no labelled passage to rerank |
| MRR | 0.5667 | 0.6042 | First relevant passage moves earlier on average |
| nDCG@5 | 0.6249 | 0.6436 | Top-five ordering improves slightly |
| Top-1 accuracy | 0.3750 | 0.3750 | The local scorer does not improve every target |
| Unanswerable abstention | 1.0000 | 1.0000 | Reranking does not override the retrieval gate |

This is a modest held-out improvement, not evidence that reranking always helps.
`q09` improves to the labelled answer passage at rank one. `q13` regresses from rank
one to rank two. `q14` is a candidate miss, so neither the local nor a neural
reranker can recover its absent answer passages.

## Optional real neural measurement

The optional pinned run uses
`cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`. On this machine and the same held-out
candidates it measured MRR `0.6500`, nDCG@5 `0.6874`, and top-1 accuracy `0.5000`.
Mean local reranking time in the committed cold-start run was about `94.0 ms` per
query. That time is a local teaching measurement, not a service latency or capacity
claim.

The original BERT reranking paper demonstrates query-passage scoring on established
retrieval benchmarks; its published result belongs to those data and conditions, not
this course corpus: <https://arxiv.org/abs/1901.04085>. The maintained
Sentence Transformers documentation describes `CrossEncoder` pair scoring, its
top-k reranking role, and the fact that raw MS MARCO logits are not automatically
probabilities: <https://www.sbert.net/docs/cross_encoder/usage/usage.html>.

## Mastery scoring guide

Score a fresh evaluation row from 0–2 on each item:

1. verifies corpus, query, and passage-label hashes;
2. distinguishes candidate membership from ordering and identifies the first failure;
3. calculates and interprets one MRR or nDCG change;
4. proposes a one-variable experiment without using evaluation labels for tuning;
5. states one limitation: label coverage, domain mismatch, uncalibrated scores, tiny
   data, or local-only latency.

Pass: **8/10**. A passing answer must not call the local character scorer a
cross-encoder, claim reranking improves recall, or treat a raw relevance logit as a
calibrated probability.
