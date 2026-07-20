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
make tiny-lm-tokenizers
make tiny-lm-kv-cache
```

The training command writes model weights, tokenizer vocabulary, configuration,
partition fingerprints, one-batch diagnostic, learning curves, selected epoch,
baseline loss, environment, elapsed time, artifact hashes, and limitations under
`artifacts/`.
The tokenizer command trains the same decoder once with characters and once with a
from-scratch BPE vocabulary, then writes a controlled comparison report.
The KV-cache command proves cached and uncached logits agree, verifies identical greedy
generation, and benchmarks prompt prefill plus incremental decoding.

## Read the implementation in this order

1. `CharacterTokenizer`: characters become integer token IDs.
2. `make_next_token_windows`: each target is the input shifted left by one token.
3. `CausalSelfAttention`: inspect `B × H × T × D` query, key, and value shapes.
4. `TransformerBlock`: follow both pre-norm residual paths.
5. `TinyLanguageModel.forward`: trace `B × T → B × T × C → B × T × V`.
6. `train`: locate zero-grad, forward, loss, backward, clipping, and optimizer step.
7. `generate`: compare greedy, temperature, top-k, and top-p decoding.

## Character tokenization versus BPE

The character tokenizer gives every character one ID. It is transparent and can encode
known characters without segmentation decisions, but sequences are long.

The BPE tokenizer begins with the same characters. It counts adjacent symbol pairs,
merges the most frequent pair, records that merge, and repeats until reaching the target
vocabulary size. Encoding replays the learned merges in order. This project operates
on the raw character stream—including whitespace—to keep the algorithm lossless and
visible. Production tokenizers add byte fallback, normalization, pre-tokenization,
special tokens, and optimized implementations.

The comparison holds seed, epochs, batch size, token context, hidden width, heads, and
layers fixed. Two consequences cannot be held fixed at the same time:

- BPE changes vocabulary size, so embedding and output-head parameter counts change.
- A fixed number of BPE tokens usually covers more source characters.

Report both effects. Do not compare token-level perplexity across tokenizers because a
“token” is a different unit. Use **bits per original character** for the cross-tokenizer
likelihood comparison.

## Naive generation versus KV caching

Naive generation runs the complete retained context again after every new token. KV
caching runs the prompt once (**prefill**), stores each layer's projected keys and
values, and then processes only one new token per decode step. Queries are not cached:
each new query is consumed once, while old keys and values are consulted repeatedly.

For batch `B`, heads `H`, cached length `T`, and head width `D`, one layer stores keys
and values shaped `(B,H,T,D)`. Cache memory grows with two tensors, every layer, and the
numeric precision. This trades memory for lower repeated computation.

The implementation uses learned absolute positions. When generation slides beyond the
context limit, retained tokens receive new position indices, so the project deliberately
recomputes the retained window instead of reusing stale cache entries.

## Expected evidence

- Initial validation loss should be near `log(vocabulary size)` for an untrained model.
- Training and validation loss should fall on the bundled corpus.
- The causal-mask test must show that changing future tokens cannot change earlier logits.
- The one-batch diagnostic must sharply reduce its loss.
- The saved checkpoint must reproduce generation under the same seed and settings.
- Artifact hashes must match the saved weights and tokenizer.
- The bigram baseline must be reported even if the Transformer does not beat it.
- The tokenizer comparison must report compression, context coverage, parameter count,
  and bits per character; token-level perplexity may only be compared within one
  tokenizer.
- Cached and uncached logits must agree within tolerance, and greedy generation must be
  identical before latency results are interpreted.

A tiny Transformer is not automatically better than a bigram model on a tiny corpus.
If it loses, report that result and explain whether data size, optimization, or model
complexity is the most plausible cause.

Complete [MASTERY_CHECKPOINT.md](MASTERY_CHECKPOINT.md) before continuing to sentence
embeddings, LLM alignment, prompting, or RAG.
