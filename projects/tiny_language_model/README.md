# Tiny Language Model Mastery Project

This project closes the gap between reading Transformer formulas and training a real
language model. It uses no hosted model and requires no API key. The implementation is
small enough to inspect: token IDs enter a two-block decoder, causal self-attention
prevents future-token leakage, cross-entropy supplies the learning signal, and AdamW
updates every trainable parameter.

## What the project proves

Running the project produces evidence for this complete chain:

```text
text → token IDs → shifted windows → embeddings → causal Transformer
→ next-token logits → cross-entropy → backward → AdamW step
→ validation → checkpoint → autoregressive generation
```

It does **not** prove that the small model has broad language ability. The bundled
curriculum-authored corpus is deliberately small and synthetic so mechanics remain
visible and the run remains CPU-friendly.

## Run it

From the repository root:

```bash
make tiny-lm-test
make tiny-lm-train
```

The training command writes model weights, tokenizer vocabulary, configuration,
dataset hash, learning curves, baseline loss, and limitations under `artifacts/`.

## Read the implementation in this order

1. `CharacterTokenizer`: characters become integer token IDs.
2. `make_next_token_windows`: each target is the input shifted left by one token.
3. `CausalSelfAttention`: inspect `B × H × T × D` query, key, and value shapes.
4. `TransformerBlock`: follow both pre-norm residual paths.
5. `TinyLanguageModel.forward`: trace `B × T → B × T × C → B × T × V`.
6. `train`: locate zero-grad, forward, loss, backward, clipping, and optimizer step.
7. `generate`: compare greedy, temperature, top-k, and top-p decoding.

## Expected evidence

- Initial validation loss should be near `log(vocabulary size)` for an untrained model.
- Training and validation loss should fall on the bundled corpus.
- The causal-mask test must show that changing future tokens cannot change earlier logits.
- The one-batch diagnostic must sharply reduce its loss.
- The saved checkpoint must reproduce generation under the same seed and settings.
- The bigram baseline must be reported even if the Transformer does not beat it.

A tiny Transformer is not automatically better than a bigram model on a tiny corpus.
If it loses, report that result and explain whether data size, optimization, or model
complexity is the most plausible cause.

Complete [MASTERY_CHECKPOINT.md](MASTERY_CHECKPOINT.md) before continuing to sentence
embeddings, LLM alignment, prompting, or RAG.
