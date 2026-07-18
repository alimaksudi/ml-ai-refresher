# Transformer Model Families Mastery Assessment

## Required evidence

1. Run all behavioral tests and the controlled training lab.
2. Draw the GPT causal mask, BERT bidirectional padding mask, T5 encoder mask, T5
   decoder causal mask, and T5 cross-attention rectangle.
3. Trace `B × H × T_query × T_key` for every attention operation.
4. Show loss reduction and task accuracy for all four objectives without describing
   synthetic training accuracy as generalization.
5. Break one mask intentionally, predict the failure, detect it, and repair it.

## Teach-back rubric

Score each answer from 0–2.

1. Why can GPT generate naturally while a plain BERT encoder cannot?
2. Why can changing a later token alter an earlier BERT representation but not an
   earlier GPT logit?
3. Why does masked-token prediction require a dedicated mask token and bidirectional
   context?
4. How does padding masking differ from causal masking?
5. What are encoder states, and why does every T5 decoder position need access to them?
6. Distinguish decoder self-attention from encoder-decoder cross-attention using Q, K,
   and V sources.
7. Explain teacher forcing in GPT and T5 and identify the shifted targets.
8. Why is BERT generally a more natural starting point for sentence embeddings?
9. When is encoder-decoder separation useful despite its additional parameters?
10. Choose GPT, BERT, T5, or a simpler non-Transformer baseline for five realistic
    tasks and justify every choice.

Pass requires all automated tests, real loss reduction for every objective, at least
**17/20**, and no zero on questions 1, 2, 5, 6, or 7.
