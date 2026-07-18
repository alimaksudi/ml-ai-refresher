# Pretraining and Data Pipeline Checkpoint

Score ten questions 0–2; pass at 17/20.

1. Why must evaluation contamination be checked before training?
2. What does exact deduplication miss?
3. Distinguish pretraining from continued pretraining.
4. Why must the tokenizer remain compatible with the base checkpoint?
5. Explain shifted next-token targets.
6. What does domain validation loss measure?
7. What does the base retention set measure?
8. Why can domain loss improve while the model becomes worse overall?
9. What model, tokenizer, data, split, and optimizer versions must be recorded?
10. Why is this experiment insufficient evidence for broad domain expertise?

Practical repair: mix base replay text into continued pretraining and measure both
domain improvement and retention again. Explain the tradeoff rather than selecting on
one metric.
