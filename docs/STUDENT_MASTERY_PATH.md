# Student Mastery Path and Remediation Guide

This is the learner-facing route. Semantic lesson IDs are stable identifiers; follow
the canonical route card generated near the top of each notebook.

## How to study one module

Use two passes rather than attempting every senior detail at once.

### Core pass — required

1. Read objectives and intuition.
2. Work through the mathematical foundation with a tiny numeric example.
3. Predict code output before running it.
4. Run and modify the implementation.
5. Reproduce one failure mode.
6. Complete the Required Core Mastery Gate.
7. Continue only at 8/10 with successful teach-back.

### Extension pass — revisit after the section gate

- detailed history;
- complete derivations beyond the core formula;
- production architecture;
- senior interview material;
- long challenges in Section 14.

Trying to master both passes simultaneously is optional and will substantially
increase the stated workload.

## Gate A — Mathematical language and coding readiness

After PRE-01 through PRE-05, without notes:

- evaluate an expression while respecting operation order;
- rearrange a one-variable equation;
- read a graph as an input-output relationship;
- explain derivative and probability intuition;
- trace NumPy matrix shapes;
- write and test a small function;
- load, validate, filter, group, and safely join a DataFrame;
- use a traceback to locate an error.

If weak in notation or algebra, repeat PRE-01 and PRE-02. If code is the blocker,
repeat PRE-03 and PRE-05 using a new five-row dataset.

## Gate B — Mathematical ML readiness

After FND-01 and FND-02:

- distinguish scalar, vector, matrix, and their shapes;
- compute a dot product and explain its meaning;
- distinguish probability, likelihood, expectation, and sample statistic;
- explain sampling variation and why a point estimate is incomplete.

Do not proceed if matrix multiplication is still memorized as a rule with no shape
meaning, or if conditional probability is confused with intersection.

## Gate C — First valid ML experiment

After FND-03, CML-01, FND-04, CML-02, MLE-01, and MLE-02:

- define prediction unit, target, prediction time, and naive baseline;
- split before fitting learned transformations;
- distinguish loss from metric;
- explain a gradient update using squared loss;
- fit regression and classification baselines;
- choose a metric and threshold from the decision cost;
- explain why the test set is not a tuning tool.

Remediate data-contract problems in FND-03, optimization problems in CML-01/FND-04,
and evaluation problems in MLE-01/MLE-02. Do not compensate with a more complex model.

## Gate D — Classical ML mastery

Complete the wine-classifier checkpoint. A passing score requires valid evidence,
not merely accuracy. Then revisit optional derivations and senior extensions from
CML-01 through MLE-05 and complete MLE-06.

## Gate E — Deep Learning mastery

Complete DL-01 through DL-05 and the digit checkpoint. You must be able to trace
tensor shapes, write train/eval loops, gradient-check, diagnose learning curves, and
justify why a neural model does or does not beat a simpler baseline.

## Retention schedule

- Next day: repeat teach-back without notes.
- One week: redo the independent task with different data.
- End of section: complete the cumulative gate before reviewing solutions.
- One month: diagnose one intentionally broken experiment from scratch.

Record attempts, scores, misconception, remediation module, and retry date. Page
completion is not mastery evidence.

Use [`INTEGRATION_PROJECTS.md`](INTEGRATION_PROJECTS.md) after each corresponding
gate. Lesson exercises build one skill; integration projects test whether skills work
together.
