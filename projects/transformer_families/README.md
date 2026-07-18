# Transformer Model Families Lab

This offline project builds three Transformer families from shared PyTorch components:

- a GPT-style causal decoder;
- a BERT-style bidirectional encoder;
- a T5-style encoder-decoder with cross-attention.

It uses no API key, hosted model, `nn.Transformer`, or simulated accuracy. Small
synthetic tasks isolate architecture mechanics; they are diagnostics, not language
quality benchmarks.

## Run

```bash
make transformer-families-test
make transformer-families-train
```

## Shared components and deliberate differences

All families use the same token/position embedding, multi-head attention, pre-norm
residual block, GELU feed-forward layer, width, head count, and layer count.

| Family | Self-attention visibility | Additional attention | Objective |
|---|---|---|---|
| GPT | Earlier/current target tokens | None | Next-token prediction |
| BERT | All unpadded input tokens | None | Masked-token prediction and classification |
| T5 | Encoder: all source; decoder: earlier target | Decoder attends full encoder output | Source-to-target prediction |

Parameter counts are reported rather than forced equal. T5 has both encoder and decoder
stacks, so equal width and depth do not imply equal total capacity.

## Controlled tasks

- GPT predicts the next digit in cyclic sequences.
- BERT reconstructs one masked center digit using left and right context.
- The same BERT encoder classifies ascending versus descending sequences.
- T5 reads a digit sequence and generates its reversal.

The tests prove causal isolation, bidirectional influence, padding-mask correctness,
source influence through cross-attention, decoder causality, and output shapes.

Complete [MASTERY_CHECKPOINT.md](MASTERY_CHECKPOINT.md) before sentence embeddings or
the adaptation pipeline.
