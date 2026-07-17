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
- tokenizer: deterministic character vocabulary;
- objective: next-token cross-entropy;
- optimizer: AdamW with gradient clipping.

## Intended use

Use the model to learn and debug the complete training lifecycle. It is intentionally
small enough that a student can trace every tensor and parameter group.

## Out-of-scope use

Do not use this checkpoint for factual answering, safety-sensitive decisions, deployed
generation, or comparison with general-purpose language models. Fluent-looking output
from this model is not evidence of truth or understanding.

## Evaluation

The artifact reports initial and best validation loss, perplexity, a smoothed bigram
baseline, parameter count, full learning history, configuration, seed, and corpus hash.
Human assessment is defined in `MASTERY_CHECKPOINT.md`.
