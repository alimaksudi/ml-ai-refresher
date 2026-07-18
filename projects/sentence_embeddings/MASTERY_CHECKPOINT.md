# Sentence Embeddings Mastery Checkpoint

Score each answer 0 (missing), 1 (partial), or 2 (correct and justified). Pass at 17/20
and complete the practical repair.

1. Why is an untrained or raw language-model encoder not automatically a good sentence encoder?
2. Show how padding-aware mean pooling is calculated for three valid tokens and two pads.
3. Why does L2 normalization make dot product equal cosine similarity?
4. What does each row and column of the MNR similarity matrix represent?
5. Why can another positive in the batch accidentally become a false negative?
6. Distinguish an easy negative, hard negative, and false negative.
7. Why must train/evaluation pairs be separated before training?
8. Interpret Recall@1, Recall@3, MRR, and positive-negative margin.
9. When can TF-IDF beat a dense encoder, and why is that useful evidence?
10. Why is a bi-encoder fast enough for first-stage retrieval while a cross-encoder is not?

## Practical repair

Break padding-aware pooling by averaging every position. Add enough padding to one
sentence to change its nearest neighbor, explain the failure, restore the mask, and
show the invariant test passing. Then add one false negative to a batch and explain its
effect on the loss.
