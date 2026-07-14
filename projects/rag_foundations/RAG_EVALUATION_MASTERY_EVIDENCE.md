# EVAL-03 RAG Evaluation — Mentor Evidence and Scoring Guide

This guide records reproducible component evidence and model explanations for the
EVAL-03 checkpoint. It does **not** certify learner mastery. The learner must diagnose
a fresh trace and explain the contracts without copying this document.

## Reproducible contract

Run:

```bash
make rag-system-checkpoint
```

The report compares lexical, dense LSA, and hybrid RRF with identical corpus, query,
answer, sentence-chunking, `top_k=5`, metric, and teaching-threshold contracts. Verify
the three SHA-256 values in `artifacts/rag_system_evaluation.json` before comparing
runs.

## Six required explanations

1. **Different component contracts.** Context precision measures the relevant share
   of returned evidence. Context recall measures labelled evidence coverage. Answer
   correctness tests the answer label. Evidence support tests grounding in cited text.
   Citation validity tests ID integrity. Abstention tests the answer/refuse decision.
2. **Proxy honesty.** Required-term correctness can miss negation and contradiction.
   Extractive containment proves copied text, not entailment or truth. They are useful
   deterministic regression proxies only when named that way.
3. **First-failure diagnosis.** A retrieval failure means relevant evidence never
   reached the answerer. An answer failure means evidence was available but rank-one
   selection did not satisfy the answer label. An incorrect refusal is recorded by
   abstention accuracy even when retrieval is the first failure.
4. **Perfect support versus poor correctness.** The extractive answerer always cites
   the passage it copies, so support and citation validity are 1.0. It can still copy
   the wrong passage; dense LSA correctness is only 0.444.
5. **Unanswerable precision.** With no relevant evidence set, context precision is not
   the contract. This project stores it as `null` and scores whether the system refused.
6. **Threshold policy.** The committed thresholds demonstrate regression mechanics.
   They are not universal SLAs. A real policy needs domain risk, human-labelled
   calibration, false-pass cost, false-failure cost, slices, and rollback rules.

## Component evidence

| System | Precision@5 | Recall@5 | Correctness proxy | Support proxy | Citation validity | Abstention | Success rate | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Lexical | 0.175 | 0.875 | 0.389 | 1.0 | 1.0 | 0.944 | 0.389 | Fail |
| Dense LSA | 0.188 | 0.938 | 0.444 | 1.0 | 1.0 | 0.944 | 0.444 | Pass |
| Hybrid RRF | 0.175 | 0.875 | 0.444 | 1.0 | 1.0 | 0.944 | 0.444 | Pass |

Dense LSA is the best retrieval baseline on this dataset. Hybrid RRF does not repeat
its win from the small RAG-06 diagnostic benchmark. The result is dataset-dependent.
The dominant next problem is answer selection, which justifies testing reranking.

## Trace diagnoses

### Retrieval and refusal failure: `q03`

- Retrieved IDs: none.
- Recall@5: 0.0.
- The answerer refuses an answerable question.
- First failed component: retrieval.
- Independent policy result: abstention is incorrect.
- Controlled next step: inspect the paraphrase representation and candidate threshold;
  do not change answer labels or the generator first.

### Answer-selection failure: `q01`

- Relevant evidence appears in the retrieved list, so Recall@5 is 1.0.
- The selected answer is `Test data.` from citation `splits.test`.
- Correctness proxy is 0.0, while support and citation validity are 1.0.
- First failed component: answer selection/ranking, not retrieval or grounding.
- Controlled next step: rerank the existing candidates and check whether
  `splits.validation` moves above `splits.test` without changing retrieval depth.

## Fresh diagnosis scoring rubric

Score a different row from 0–2 on each item:

1. preserves corpus/query/answer and system hashes;
2. identifies the first failed component from trace evidence;
3. interprets every cited metric using its declared contract;
4. proposes a one-variable experiment;
5. states one proxy, label, or scale limitation.

Pass: **8/10**. A passing answer must not call required-term matching semantic
correctness or extractive containment entailment.
