# Deep Learning Mastery Checkpoint

This checkpoint tests the Section 04 learning contract on the bundled 8×8 handwritten
digits dataset. It compares a naive majority classifier, scaled logistic regression,
a PyTorch MLP, and a small CNN. Both neural checkpoints are selected using validation
log loss, and the test set is touched only for the final comparison.

```bash
make deep-learning-train
make deep-learning-test
```

The artifact records the data fingerprint, split, seed, framework version, learning
curve, selected epoch, baseline, final metrics, input contract, and required dropout
ablation. See [MASTERY_CHECKPOINT.md](MASTERY_CHECKPOINT.md) for the human assessment.
