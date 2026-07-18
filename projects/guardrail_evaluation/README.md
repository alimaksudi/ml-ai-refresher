# Guardrail Evaluation Lab

This project compares a naive keyword blocker with an explicit four-action policy:
`ALLOW`, `REDACT`, `BLOCK`, and `ESCALATE`. It measures exact actions,
over-restriction, critical escapes, and PII redaction. A separate structured-output
suite keeps schema validity, citation validity, source support, and authoritative
correctness independent.

The candidate improves over the keyword baseline but misses a paraphrased credential
request, so the release gate returns `review_or_reject`. The rules are teaching controls,
not a production security boundary or a universal safety policy.

Run `make guardrail-evaluation-test` and `make guardrail-evaluation-run`, then complete
`MASTERY_CHECKPOINT.md`.
