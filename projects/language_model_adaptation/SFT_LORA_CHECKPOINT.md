# Instruction Tuning and LoRA Checkpoint

Score ten questions 0–2; pass at 17/20.

1. Which tokens are context and which receive direct SFT labels?
2. Why are targets shifted one position?
3. What does `-100` mean in the loss tensor?
4. Why can response gradients still pass through prompt representations?
5. Distinguish full tuning from LoRA.
6. Why is LoRA's second matrix initialized to zero?
7. What does rank control?
8. Why can frozen base weights still produce behavioral regressions?
9. Why is training loss insufficient evidence of instruction following?
10. When should prompting or RAG be tried before fine-tuning?

Practical repair: remove the response mask, show that prompt reconstruction dominates
the supervised-token count, restore the mask, and rerun the invariant test.
