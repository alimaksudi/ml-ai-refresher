# Transformer Families Diagnostic Data

All examples are generated deterministically from digit-token sequences. No external
data, personal data, API, or download is used.

- GPT receives cyclic ascending sequences and predicts the following token.
- BERT sees the same sequence with its center token replaced by `<MASK>`.
- BERT classification uses ascending and descending sequences with balanced labels.
- T5 receives a five-token source and generates the reversed sequence plus `<EOS>`.

These tasks isolate visibility and information flow. Their repeated patterns make them
unsuitable for estimating language understanding, generalization, fairness, or safety.
