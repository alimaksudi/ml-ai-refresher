# Model Card

The model is a two-layer, 48-dimensional BERT-style bidirectional encoder reused from
`projects/transformer_families`. Mean pooling excludes padding and L2 normalization
maps every sentence to the unit sphere. Training uses MNR/InfoNCE-style cross-entropy
with in-batch negatives and one explicit hard negative per query.

This tiny encoder is for inspecting behavior. It is not pretrained, not multilingual,
and not suitable for production retrieval. A production candidate must be compared on
held-out domain data, latency, memory, privacy, robustness, and current operating cost.
