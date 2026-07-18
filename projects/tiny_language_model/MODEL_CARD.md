# Tiny Language Model Card

## Model

The project trains a decoder-only, pre-normalization Transformer with learned token and
position embeddings, causal multi-head self-attention, GELU feed-forward layers,
residual connections, final layer normalization, and a tied language-model head.

Default configuration:

- context length: 48 characters;
- hidden width: 64;
- attention heads: 4;
- Transformer blocks: 2;
- tokenizer: deterministic character vocabulary or learned educational BPE;
- objective: next-token cross-entropy;
- optimizer: AdamW with gradient clipping.

Inference supports both full-context recomputation and per-layer key/value caching.
The cache is an exact inference optimization within floating-point tolerance, not a new
model or decoding strategy. It is reset and rebuilt when the learned-position context
window slides.

## Intended use

Use the model to learn and debug the complete training lifecycle. It is intentionally
small enough that a student can trace every tensor and parameter group.

## Out-of-scope use

Do not use this checkpoint for factual answering, safety-sensitive decisions, deployed
generation, or comparison with general-purpose language models. Fluent-looking output
from this model is not evidence of truth or understanding.

## Evaluation

The artifact reports initial and best validation loss, token-level perplexity, bits per
character, a smoothed bigram baseline, token compression, context coverage, parameter
count, full learning history, configuration, seed, and corpus hash. Token-level
perplexity is not compared across tokenizers. Human assessment is defined in
`MASTERY_CHECKPOINT.md`.

The KV-cache benchmark separately records maximum logit difference, greedy-token
equality, cache shapes, median latency, tokens per second, speedup, and estimated cache
bytes. Timing results apply only to the recorded hardware, thread count, model, prompt
lengths, and software environment.
