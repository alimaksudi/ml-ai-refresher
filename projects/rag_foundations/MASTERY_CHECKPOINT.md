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
