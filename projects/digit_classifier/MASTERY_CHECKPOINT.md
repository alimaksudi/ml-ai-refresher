# Deep Learning Mastery Assessment

Automated tests are necessary but not sufficient. Submit a short experiment report
and answer each teach-back prompt from 0–2.

## Required experiment

1. Run the reproducible MLP checkpoint.
2. Run the same configuration with dropout set to `0.0`.
3. Compare validation curves across at least three fixed seeds.
4. Do not choose any configuration using test performance.
5. Report accuracy, macro F1, log loss, best epoch, run variation, and training time.
6. Explain whether the neural model earns its complexity over logistic regression.

## Teach-back rubric

1. Trace `(B,64) → (B,64) → (B,10)` and identify each parameter shape.
2. Explain logits, softmax, cross-entropy, and why the loss accepts logits.
3. Explain `zero_grad`, forward, backward, optimizer step, and evaluation mode.
4. Diagnose underfit, overfit, optimization failure, and data leakage from evidence.
5. Distinguish initialization, normalization, clipping, weight decay, dropout, and early stopping.
6. Explain why the best validation checkpoint—not the last epoch—is tested.
7. Explain what the dropout ablation establishes and what it cannot establish.
8. Defend whether an MLP is appropriate for spatial image data compared with a CNN.

Pass requires all automated tests, at least **13/16**, and no zero on questions 2,
3, or 6. The final answer must include one limitation and one falsifiable next experiment.
