# RAG Foundation Mastery Checkpoint

## Required evidence

1. Explain every gold evidence ID for five sampled queries.
2. Compare all nine chunking/retriever combinations.
3. Report recall@k, MRR, nDCG, answerable zero-result rate, unanswerable abstention,
   latency, and direct/paraphrase/multi-concept slices.
4. Diagnose at least three failed queries from component rows.
5. Change one variable only—chunk strategy or retriever—and predict the result first.
6. Do not claim LSA is a neural embedding or retrieval relevance is answer correctness.
7. Recommend the simplest baseline supported by the results.

## Teach-back rubric

Score each from 0–2:

1. Why must evidence IDs survive chunking?
2. How do recall@k, MRR, and nDCG differ?
3. Why can high retrieval recall coexist with a wrong answer?
4. Why should raw lexical and dense scores not be added directly?
5. What does an unanswerable query test?
6. Why is a labelled query set part of the system, not an afterthought?
7. What can LSA capture, and what can it not capture?
8. Which measured result justifies the next component?

Pass requires all automated tests, at least **13/16**, and no zero on questions 1,
2, 3, or 6. Advanced RAG remains blocked until this gate passes.

## Grounded-answer extension

Before vector databases or reranking, explain why retrieval recall is higher than
extractive answer correctness, diagnose one retrieval failure and three answer
failures, validate every citation, and pass all stale, injection, and authorization
cases.

## Persistent vector-store extension

Before hybrid search, the learner must:

1. Explain what persistence, CRUD, payload filters, and idempotent upserts add beyond
   a NumPy or FAISS search index.
2. Rebuild the Qdrant-local index, close it, reopen it, and reproduce the same ranked
   results.
3. Prove NumPy exact and Qdrant-local exact retrieval have **100% section-ranking
   parity** on the labelled set and retain at least **90% recall@5**.
4. Demonstrate that public search excludes stale, unsafe, and restricted records,
   while an authorized search can retrieve the restricted record.
5. Explain why a local 70-point exact-search run cannot establish production ANN
   latency, capacity, replication, or recall.
6. Diagnose dimension mismatch, duplicate indexing, stale manifests, and metadata
   filter failures from the saved report.

Pass requires `make vector-store-checkpoint`, all six explanations, and no failure on
items 2–5. RAG-06 remains blocked until this gate passes.

## Hybrid-search extension

Before RAG evaluation, the learner must:

1. Run `make hybrid-rag-evaluate` and compare BM25, dense LSA, RRF, and every tested
   alpha on the same corpus hash, query hash, chunks, labels, `top_k`, and candidate
   depth.
2. Calculate one RRF result manually and explain why a document missing from one
   candidate list receives no contribution from that list.
3. Use component rows to show that BM25 ranks `h01` better, dense LSA ranks `h02`
   better, RRF recovers both relevant sections for `h03`, and fusion adds no ranking
   gain for `h04`. Explain the metric used for each comparison.
4. Explain why both branches must apply equivalent authorization, freshness, and
   safety policies before fusion while keeping branch-specific tokenization.
5. Demonstrate stable-ID deduplication, evidence provenance, candidate-depth control,
   and abstention when neither branch provides meaningful evidence.
6. State why the local latency values are teaching measurements and model concurrent
   sparse/dense execution as `max(branch latency) + fusion and orchestration`, not as
   an unconditional sum.

Pass requires `make hybrid-rag-checkpoint`, all six explanations, and a scored query
failure analysis. The automated report must retain the four diagnostic behaviors in
item 3 and abstain on `h05`. EVAL-03 remains blocked until this gate passes.

## EVAL-03 component-evaluation extension

Before reranking, the learner must:

1. Run `make rag-system-evaluate` and verify the corpus, query, and answer hashes.
2. Explain why context precision, context recall, answer correctness, evidence support,
   citation validity, and abstention measure different contracts.
3. State why required-term correctness and extractive text containment are proxies,
   not semantic correctness or entailment.
4. Use component rows to diagnose one retrieval failure, one answer-selection failure,
   and one abstention failure without changing gold labels.
5. Explain why perfect support can coexist with poor correctness and why unanswerable
   cases do not receive context precision.
6. Treat the committed quality thresholds as a teaching policy. Propose domain-specific
   thresholds only after human-labelled calibration and state the cost of false passes
   and false failures.

Pass requires `make rag-system-checkpoint`, all six explanations, and a fresh trace
diagnosis scoring at least 8/10. RAG-07 remains blocked until this gate passes.
