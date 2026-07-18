# Tiny Language Model Data Card

## Source and license

`data/learning_corpus.txt` was written for this repository. It contains short teaching
statements about language-model mechanics and has no external download dependency.

## Intended use

The corpus supports fast verification of tokenization, causal next-token training,
validation, checkpointing, and decoding. It is not intended for benchmarking language
quality, factual knowledge, fairness, safety, or production readiness.

## Split policy

The raw character stream is split contiguously before training windows are created.
No window crosses from training into validation. The split is intentionally not random:
overlapping random windows would leak nearly identical contexts across both partitions.

## Known limitations

- The corpus is tiny, synthetic, English-only, and narrowly focused.
- Repeated terminology makes it easier than natural language.
- Character tokenization produces longer sequences than subword tokenization.
- Validation measures continuation on the same narrow domain, not general language skill.

The tokenizer experiment fits both character and BPE vocabularies on training text
only. Validation text never influences base characters, merge selection, or token IDs.
Unknown validation characters map to an explicit unknown token.
