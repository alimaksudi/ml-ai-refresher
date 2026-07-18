# LLM Evaluation Mastery Checkpoint

Complete this checkpoint after EVAL-02 and before NLP-04.

## Evidence to submit

1. An evaluation contract naming the decision, unit, primary metric, guardrails,
   slices, fixed settings, and acceptance rule before results are inspected.
2. A manual token-F1 and ROUGE-L calculation whose Python results match.
3. The real local adaptation report with domain, retention, instruction, and
   preference outcomes kept separate.
4. Per-example paired results and a bootstrap interval with an explicit coverage
   warning.
5. A regression-gate decision that can reject a candidate whose primary metric
   improves when a declared guardrail fails.
6. A metric table that labels lexical overlap, local sentence-embedding retrieval,
   and BERTScore as different constructs.

## Scoring

- Evaluation contract and untouched data: 4 points
- Correct manual metrics and code checks: 4 points
- Paired rows, uncertainty, and slice diagnosis: 4 points
- Predeclared regression gate: 4 points
- Teach-back and limitations: 4 points

Pass with at least **17/20**, including non-zero points in every category. A perfect
automatic score on leaked or unrepresentative data is an automatic retry.

## Teach-back

Without notes, explain why lower perplexity does not prove answer correctness, why
random hash vectors cannot demonstrate semantic similarity, why model comparisons
must preserve prompt pairing, and why bootstrap uncertainty cannot repair biased
coverage. Then defend the primary metric and guardrails in your evaluation contract.
