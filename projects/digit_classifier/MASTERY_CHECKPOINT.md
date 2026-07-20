# Deep Learning Mastery Assessment

Passing tests shows that the experiment runs. Mastery means you can explain what the
evidence says, why the experiment is fair, and what you would try next.

## Evidence to inspect

Open `artifacts/metadata.json` after running:

```bash
make deep-learning-train
```

Use these sections:

- `selection.dropout_ablation` compares dropout `0.0` and `0.15`;
- `selection.shift_augmentation_ablation` compares CNN training with and without shifts;
- `selection.mlp_runs` and `selection.cnn_runs` contain all three seeds;
- `selection.run_variation` summarizes how much the validation result moves;
- `representative_histories` shows the learning curves of the saved models;
- `final_test` contains the one-time final comparison;
- `selection.complexity_decision` records whether a neural model earned its cost.

The code runs these experiments for you. Your job is to interpret them—not to copy
numbers without explaining what they mean.

## Short experiment report

Write a report that answers all six questions:

1. Which dropout value won on validation log loss? Was the accuracy conclusion the same?
2. Did shift augmentation help? Give one reason it could help and one reason it could hurt.
3. Which model family varied more across seeds? Use the reported standard deviations.
4. Did logistic regression, the MLP, or the CNN deserve to be the default? Defend the choice.
5. Why would choosing the configuration with the best test score make the result unreliable?
6. Give one limitation and one falsifiable next experiment. State what result would change your conclusion.

## Teach-back rubric

Score each answer from 0–2.

1. Trace the MLP shapes `(B, 64) → (B, 64) → (B, 10)` and name each parameter shape.
2. Explain logits, softmax, cross-entropy, and why cross-entropy receives logits.
3. Explain the order: clear gradients, forward pass, loss, backward pass, optimizer step.
4. Diagnose underfitting, overfitting, optimization failure, and leakage from evidence.
5. Distinguish initialization, scaling, clipping, weight decay, dropout, and early stopping.
6. Explain why we save the best validation checkpoint instead of the final epoch.
7. Explain what a dropout ablation can establish—and what one split cannot establish.
8. Explain what the CNN assumes about pixels that the MLP does not.

## Passing standard

To pass:

- all automated tests pass;
- the teach-back score is at least **13/16**;
- questions 2, 3, and 6 each score at least 1;
- the report includes evidence from multiple seeds;
- the final answer includes one limitation and one falsifiable next experiment.

Do not memorize which model won this particular run. Remember the method: hold the
split still, vary training randomness, choose with validation data, and open the test
set only after the decisions are complete.
