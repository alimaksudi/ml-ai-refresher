# Pre-RAG Language Model Mastery Assessment

Passing automated tests is necessary but not sufficient. Complete the experiment and
teach-back before starting RAG.

## Required experiment

1. Run the one-batch overfit diagnostic and explain why it is a code-path test rather
   than evidence of generalization.
2. Train the default model and preserve its configuration, seed, corpus hash, learning
   curve, best validation loss, perplexity, and bigram baseline.
3. Generate from one fixed prompt with greedy, temperature, top-k, and top-p decoding.
4. Change exactly one design choice: remove positional embeddings, use one head, use
   one block, or shorten context. Keep all other controls fixed.
5. Explain whether the added Transformer complexity is justified by measured evidence.

## Required tensor trace

For `B=2`, `T=8`, `C=16`, `H=4`, and vocabulary size `V=12`, write the shape of:

- token IDs and targets;
- token and position embeddings;
- combined QKV projection;
- Q, K, and V after splitting heads;
- attention scores and causal mask;
- concatenated attention output;
- final logits;
- flattened logits and targets passed to cross-entropy.

## Teach-back rubric

Score each answer from 0–2.

1. Explain why targets are inputs shifted by one position.
2. Explain what the causal mask prevents and demonstrate it with the automated test.
3. Trace `zero_grad → forward → loss → backward → clip → optimizer.step`.
4. Explain residual connections, layer normalization, and the feed-forward sublayer.
5. Distinguish training loss, validation loss, cross-entropy, and perplexity.
6. Explain why a bigram model is a required baseline.
7. Compare greedy, temperature, top-k, and top-p without calling one universally best.
8. Explain why next-token probability does not imply factual truth.
9. Compare decoder-only GPT, encoder-only BERT, and encoder-decoder T5 by masking,
   training objective, and suitable task.
10. Explain what retrieval adds to a language model and what it does not guarantee.

Pass requires automated tests, the controlled ablation, at least **17/20**, and no zero
on questions 1, 2, 3, 5, or 8. If the model fails to beat the bigram baseline, that does
not automatically fail the student; hiding or misrepresenting the result does.
