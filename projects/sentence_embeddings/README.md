# Sentence Embeddings Learning Gate

This project turns the BERT-style encoder from NLP-06 into a sentence bi-encoder. It
uses local customer-support text, a transparent tokenizer, masked mean pooling, L2
normalization, cosine similarity, and Multiple Negatives Ranking loss. No hosted API
or downloaded checkpoint is required.

## What the experiment proves

- query and document towers share exactly the same encoder weights;
- padding does not affect pooled vectors;
- normalized dot product equals cosine similarity;
- in-batch and explicit hard negatives shape the embedding space;
- held-out retrieval is measured against TF-IDF and an untrained Transformer;
- exact train/evaluation text overlap is rejected.

Run:

```bash
make sentence-embeddings-test
make sentence-embeddings-train
```

The dataset is intentionally small and curated. Passing proves that the mechanics and
evaluation contract work; it does not prove broad semantic-search quality.
