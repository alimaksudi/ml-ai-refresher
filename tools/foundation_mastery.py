"""Beginner-first mastery material for the route through classical ML.

The existing builders retain their deeper refresher material.  This module adds
the small, fully worked core that a new learner must complete before that
material.  Values are plain data so ``nbbuild`` remains the only notebook
assembly implementation.
"""

TOPIC_GUIDES = {
    "PRE-01": {
        "problem": "Read mathematical instructions without guessing what a symbol asks you to do.",
        "analogy": "A formula is a recipe: values are ingredients, operators are actions, and the result is the finished quantity. Unlike a recipe, a formula also states exact relationships.",
        "use": "Use arithmetic and notation to describe quantities, rates, errors, and repeated calculations.",
        "avoid": "Do not manipulate symbols before checking their meaning, units, and order of operations.",
        "alternative": "For one tiny calculation, a sentence or labelled table may be clearer than a formula.",
        "memory": "Name every symbol, keep the units, and perform one operation at a time.",
    },
    "PRE-02": {
        "problem": "Describe how one quantity changes when another quantity changes.",
        "analogy": "A function is a vending machine: an input goes in, one rule acts on it, and an output comes out. Real functions do not physically contain objects, so the analogy stops at input, rule, and output.",
        "use": "Use algebra and graphs for repeatable relationships whose inputs and outputs are defined.",
        "avoid": "Do not assume every relationship is a straight line or that a graph proves cause and effect.",
        "alternative": "Use a lookup table when there are only a few fixed cases and no useful rule.",
        "memory": "A function is one rule that maps each allowed input to one output.",
    },
    "PRE-03": {
        "problem": "Turn a calculation into code that can be rerun, inspected, and checked.",
        "analogy": "A notebook is a laboratory bench: cells are experiments and the kernel is the current workspace. Old material left on the bench resembles hidden notebook state.",
        "use": "Use Python for repeatable logic and NumPy for rectangular numerical data.",
        "avoid": "Do not trust a notebook that only works when cells run in a secret order.",
        "alternative": "Use paper for the first tiny calculation and a script or test for reusable work.",
        "memory": "Predict the value and shape, run the code, then check both.",
    },
    "PRE-04": {
        "problem": "Reason about quantities that change and outcomes that are uncertain.",
        "analogy": "A derivative is the steepness under your feet; probability is the composition of a bag you cannot see inside. A local slope need not describe the whole hill, and probability does not predict one guaranteed draw.",
        "use": "Use derivatives for local change and probability for repeated uncertain outcomes.",
        "avoid": "Do not treat a local slope as a global rule or an expected value as a promised outcome.",
        "alternative": "Use finite differences for a numerical slope and observed counts for an empirical probability.",
        "memory": "A derivative describes local change; probability describes uncertainty across possible outcomes.",
    },
    "PRE-05": {
        "problem": "Move from isolated code cells to a trustworthy table-processing workflow.",
        "analogy": "A data function is a checkpoint: it verifies what enters, performs one job, and records what leaves. It cannot decide whether a plausible value is true without a business rule.",
        "use": "Use pandas for named tabular columns, grouping, filtering, and checked joins.",
        "avoid": "Do not silently drop invalid rows, hide exceptions, or join on keys of unknown uniqueness.",
        "alternative": "Use plain Python for tiny stateful logic and NumPy for dense numerical arrays.",
        "memory": "Validate the input contract before transforming the table.",
    },
    "FND-01": {
        "problem": "Represent many measurements and transformations without losing track of position or shape.",
        "analogy": "A vector is an ordered packing list and a matrix is a machine that combines lists. The analogy does not replace the row-by-column multiplication rule.",
        "use": "Use vectors and matrices for features, parameters, batches, projections, and linear transformations.",
        "avoid": "Do not multiply arrays until their shapes and the intended operation are explicit.",
        "alternative": "Use labelled scalar equations when only one or two quantities are involved.",
        "memory": "Shapes tell you which values can meet; the dot product multiplies matching positions and adds.",
    },
    "FND-02": {
        "problem": "Use a limited sample to describe uncertainty and make cautious claims about a larger process.",
        "analogy": "A sample is one spoonful of soup: it can inform you about the pot only if it was taken fairly. One spoonful cannot reveal every ingredient or guarantee the next spoonful.",
        "use": "Use probability models and statistics when observations vary and uncertainty matters.",
        "avoid": "Do not report a point estimate without its sample, assumptions, and uncertainty.",
        "alternative": "Use descriptive counts and plots when inference beyond the observed data is unnecessary.",
        "memory": "A statistic describes the sample; uncertainty limits what it says about the population.",
    },
    "FND-03": {
        "problem": "Turn a raw table into documented training data without learning from future or test information.",
        "analogy": "Data preparation is an inspection line: identify, measure, repair, and record each issue. An unusual item is not automatically defective.",
        "use": "Use EDA and cleaning to understand schema, quality, distributions, and target availability.",
        "avoid": "Do not fit cleaning statistics on validation, test, or future data.",
        "alternative": "Reject the batch when a safe repair cannot be justified.",
        "memory": "Inspect first, split at the right boundary, and learn transformations from training data only.",
    },
    "FND-04": {
        "problem": "Find parameter values that make a declared loss smaller when no direct solution is practical.",
        "analogy": "Gradient descent is walking downhill in fog by feeling the local slope. The analogy stops because an optimizer can move in thousands of parameter directions at once.",
        "use": "Use gradient methods for differentiable objectives, especially large iterative models.",
        "avoid": "Do not optimize before defining the loss, baseline, and evidence used to judge the result.",
        "alternative": "Use a closed-form or direct solver when it is stable, available, and affordable.",
        "memory": "Define the loss first; then step against its slope and verify that the loss falls.",
    },
    "CML-01": {
        "problem": "Predict a numerical outcome with a simple, explainable relationship between features and target.",
        "analogy": "A fitted line is a straight ruler placed through a cloud of points. It summarizes a trend but cannot bend to follow every pattern.",
        "use": "Use linear regression as a baseline when effects are approximately additive and the target is numerical.",
        "avoid": "Do not use it unchanged for strong nonlinearities, unstable extrapolation, or unhandled outliers.",
        "alternative": "Predict the training mean as the simplest baseline; use trees or transformed features for nonlinear patterns.",
        "memory": "Linear regression chooses coefficients that make squared residuals small.",
    },
    "CML-02": {
        "problem": "Estimate the probability of a class while keeping predictions between zero and one.",
        "analogy": "The sigmoid is a smooth gate: any score enters, but the output stays between zero and one. The output is a model estimate, not a guaranteed frequency unless calibration is checked.",
        "use": "Use logistic regression for a strong, interpretable classification baseline.",
        "avoid": "Do not confuse a probability estimate with the final business decision or assume a 0.5 threshold is always correct.",
        "alternative": "Use a simple class-frequency baseline first; use trees when the boundary needs strong interactions.",
        "memory": "Logistic regression turns a linear score into a probability, then a separate threshold turns it into a decision.",
    },
    "MLE-01": {
        "problem": "Measure model errors in a way that matches the real cost of false alarms and missed cases.",
        "analogy": "A metric is a scoreboard: different scoreboards reward different behavior. No single score explains every kind of mistake.",
        "use": "Choose metrics after defining the target, decision, class balance, and error costs.",
        "avoid": "Do not choose a model from accuracy alone or evaluate on its training examples.",
        "alternative": "Inspect the confusion matrix and per-example errors before compressing them into one number.",
        "memory": "Choose the error that matters before choosing the metric.",
    },
    "MLE-02": {
        "problem": "Estimate performance on future unseen cases without allowing answers to leak into training.",
        "analogy": "A test set is a sealed examination. Looking at its answers while studying turns it into training material.",
        "use": "Use held-out validation that matches time, entity, and serving boundaries.",
        "avoid": "Do not randomly split dependent rows or fit preprocessing before the split.",
        "alternative": "Use a simple train-test split before cross-validation; use group or time splits when rows are related.",
        "memory": "Anything learned from held-out data makes the evaluation less held out.",
    },
    "PROD-04": {
        "problem": "Know exactly what changed between experiments and reproduce the result later.",
        "analogy": "An experiment record is a laboratory notebook: hypothesis, materials, procedure, and result belong together. Recording a run does not make its evaluation valid.",
        "use": "Track data version, split, code, parameters, seed, metric, artifact, and conclusion for every serious run.",
        "avoid": "Do not tune on the final test set or compare runs whose data and metric definitions differ.",
        "alternative": "A small CSV or JSON log is enough before adopting a tracking platform.",
        "memory": "A result without its data, code, configuration, and metric definition is not reproducible.",
    },
    "MLE-03": {
        "problem": "Represent raw information in a form whose meaning and scale a chosen model can use.",
        "analogy": "Feature engineering is translation: the facts stay the same, but their representation becomes understandable to the model. A better translation cannot repair missing or false facts.",
        "use": "Choose transformations from the model's assumptions and fit them on training data only.",
        "avoid": "Do not encode categories as arbitrary ordered numbers or use target information outside a training fold.",
        "alternative": "Start with raw validated features and a baseline before adding transformations.",
        "memory": "A feature is useful when its representation matches the model and remains available at prediction time.",
    },
    "CML-03": {
        "problem": "Learn readable if-then rules for nonlinear boundaries and feature interactions.",
        "analogy": "A tree is a sequence of twenty-questions decisions. Unlike a person, it chooses questions greedily from measured training improvement.",
        "use": "Use a decision tree for nonlinear tabular patterns, interactions, and rule inspection.",
        "avoid": "Do not trust an unrestricted tree; small data changes can produce very different rules.",
        "alternative": "Use logistic regression for a stable linear boundary or a forest for lower variance.",
        "memory": "A tree repeatedly chooses the split that most reduces impurity on the current data.",
    },
    "CML-04": {
        "problem": "Reduce the instability of one decision tree without giving up flexible nonlinear rules.",
        "analogy": "A forest is a panel of varied judges whose votes are averaged. Many judges help less when they all saw the same evidence and make the same errors.",
        "use": "Use random forests as reliable tabular baselines with nonlinearities and interactions.",
        "avoid": "Do not use them when tiny latency, smooth extrapolation, or a compact equation is required.",
        "alternative": "Use one pruned tree for maximum interpretability or boosting for lower bias.",
        "memory": "Bootstrap rows, vary candidate features, grow many trees, and average their predictions.",
    },
    "CML-05": {
        "problem": "Build a strong predictor by adding small models that correct the current model's errors.",
        "analogy": "Boosting is an editor making several small revision passes; each pass focuses on what remains wrong. Too many passes can start editing noise.",
        "use": "Use gradient boosting for high-quality tabular prediction after a validated baseline exists.",
        "avoid": "Do not begin with XGBoost internals before understanding residual correction and validation-based stopping.",
        "alternative": "Use linear models for simplicity or random forests for a less tuning-sensitive ensemble baseline.",
        "memory": "Boosting adds small corrections sequentially; validation decides when to stop.",
    },
}


ADVANCED_SECTION_NOTES = {
    "FND-01": "Eigenvalues, SVD, conditioning, and compression are the advanced path. Master the vector, matrix, shape, dot-product, and projection bridge first.",
    "FND-02": "MLE, MAP, the CLT, bootstrap inference, and formal testing are the advanced path. Master events, distributions, samples, mean, variance, and conditional probability first.",
    "FND-04": "Condition-number rates, momentum, and Adam are the advanced path. Master loss, slope, learning rate, and a few hand-computed gradient steps first.",
    "PROD-04": "Gaussian processes, Expected Improvement, and Bayesian optimization are optional advanced optimization material—not prerequisites for experiment tracking.",
    "CML-05": "The second-order XGBoost derivation is an advanced extension. Master ordinary residual boosting and early stopping first.",
}


# Each bridge is deliberately small.  It supplies the missing hand calculation
# before the existing deeper derivation and implementation.
CORE_MASTERY = {
    "PRE-01": r"""
## Beginner Core · Symbols, comparisons, units, and safe calculation

Before calculating, write the quantity and its unit. The symbols $<$ and $>$
mean “less than” and “greater than.” The symbols $\le$ and $\ge$ include equality.
For example, $0\le p\le1$ says that probability $p$ may be zero, one, or anything
between them. An interval such as $[0,1]$ includes both ends; $(0,1)$ excludes them.

A set is a collection. If $A=\{2,4,6\}$, then $4\in A$ reads “four belongs to A,”
while $5\notin A$ reads “five does not belong to A.” Conditions may use **and**
(both must hold), **or** (at least one holds), and **not** (the condition is false).

Scientific notation separates size from precision:

$$
3.2\times10^4=32{,}000,
\qquad
4.5\times10^{-3}=0.0045.
$$

Read the first expression as “three point two times ten to the fourth.” The
exponent tells how many decimal places to move. Approximation $\approx$ means
“close enough for the stated precision,” not exactly equal.

**Unit check:** $60\text{ km}/2\text{ h}=30\text{ km/h}$. Kilometres cannot be
added directly to hours. Keeping units beside values catches many silent errors.
""",
    "PRE-02": r"""
## Beginner Core · Functions, inequalities, systems, and matrix shapes

For $f(x)=2x+3$, the allowed input is the **domain** and the possible output is
the **range**. Substituting $x=4$ gives $f(4)=2(4)+3=11$. A nonlinear function,
such as $g(x)=x^2$, changes slope; a straight line cannot represent it everywhere.

Solve an inequality as you solve an equation, except multiplying or dividing by
a negative reverses the sign:

$$
-2x<6
\quad\Longrightarrow\quad
x>-3.
$$

Check with $x=0$: $-2(0)=0<6$, so the result is plausible.

Two equations can be solved together. From $x+y=5$ and $x-y=1$, adding the
equations gives $2x=6$, so $x=3$ and $y=2$.

For a matrix $X$ with 2 rows and 3 columns and vector $w$ with 3 entries,

$$
X=
\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},
\qquad
w=\begin{bmatrix}1\\0\\-1\end{bmatrix},
\qquad
Xw=\begin{bmatrix}-2\\-2\end{bmatrix}.
$$

Each output is one row-by-column dot product. The inner sizes match: $(2\times3)
(3\times1)\rightarrow(2\times1)$.
""",
    "PRE-03": r"""
## Beginner Core · Python objects, indexing, broadcasting, and reproducibility

Import a module once, then call a named tool from it: `import numpy as np`.
A tuple is an ordered fixed grouping; a dictionary maps names to values. These
appear later in array shapes, tree nodes, configurations, and metric reports.

Indexing selects existing values. Slicing selects a range. A boolean mask keeps
positions whose condition is true. NumPy **broadcasting** repeats a compatible
smaller shape without copying it conceptually. For a matrix of shape `(2, 3)`
minus a vector of shape `(3,)`, the vector is applied to each row. Broadcasting
is invalid when aligned dimensions are neither equal nor 1.

Arrays can share memory. Use `.copy()` when an independent value is required.
Inside a function, local names normally do not change outer names, but mutating
a passed list, dictionary, array, or DataFrame can change the same underlying
object. Return a new result unless mutation is intentional and documented.

**Debugging routine:** read the final traceback line, find the last line of code
you own, print type and shape at that boundary, and reduce the case until one
assumption fails. Test the failure with `assert` or an expected exception.
""",
    "PRE-04": r"""
## Beginner Core · Change first, uncertainty second

### Part A — derivatives

For $f(x)=x^3$, the power rule gives $f'(x)=3x^2$. At $x=2$, the local slope is
$3(2^2)=12$. The chain rule handles a function inside another function. If
$q(x)=(2x+1)^2$, the outer slope is $2(2x+1)$ and the inner slope is $2$, so
$q'(x)=4(2x+1)$. At $x=1$, that slope is $12$.

### Part B — probability

Let a fair die have sample space $S=\{1,2,3,4,5,6\}$. If $A$ means “even,” then
$P(A)=3/6=0.5$ and $P(\text{not }A)=1-P(A)=0.5$. Events are independent when
learning one does not change the probability of the other. Mutually exclusive
events cannot happen together; they are not generally independent.

A probability table must sum to one. For a value $X$ that is 0 with probability
0.75 and 4 with probability 0.25,

$$
\mathbb E[X]=0(0.75)+4(0.25)=1.
$$

Read this as “the expected value of X is one.” It is a long-run weighted average,
not a guaranteed single result.
""",
    "PRE-05": r"""
## Beginner Core · A checked table workflow

Use `dataframe.loc[row_condition, column_names]` for label-based selection and
`dataframe.iloc[row_positions, column_positions]` for position-based selection.
Prefer column names because positions can change silently.

For grouping, first state the unit of one row, the grouping key, and the output
unit. If four orders are `(A, 10)`, `(A, 20)`, `(B, 5)`, `(B, 15)`, grouping by
customer and summing produces A = 30 and B = 20.

Before a join, state key cardinality. In a many-to-one join, many measurement
rows may match one entity row. If the supposedly unique entity table contains
two copies of a key, a join may multiply rows. Use `validate="many_to_one"` and
check the row count.

Parse dates explicitly, preserve the original raw value when repair is uncertain,
and use categorical values only after checking allowed labels. A safe loader
returns both the cleaned table and a quality report; it does not silently erase
evidence of invalid input.
""",
    "FND-01": r"""
## Beginner Core · Vector and matrix calculations before decomposition

A scalar is one number, a vector is an ordered list, and a matrix is a rectangular
table. For $x=(2,3)$ and $w=(4,5)$, the dot product is

$$
x\cdot w=(2\times4)+(3\times5)=8+15=23.
$$

Read it as “x dot w equals the sum of matching products.” Both vectors have shape
$(2,)$; the output is one scalar. It is useful for a weighted score, but matching
positions must have matching meanings.

For $A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$ and $x=\begin{bmatrix}5\\6\end{bmatrix}$,

$$
Ax=\begin{bmatrix}1(5)+2(6)\\3(5)+4(6)\end{bmatrix}
=\begin{bmatrix}17\\39\end{bmatrix}.
$$

The transpose $A^\top=\begin{bmatrix}1&3\\2&4\end{bmatrix}$ swaps rows and columns.
The identity matrix leaves a vector unchanged. An inverse, when it exists, undoes
a square transformation. Dependent columns make an inverse impossible and a
solution non-unique.

Projection asks for the reachable vector closest to a target. Least squares uses
this idea when no line can pass through every noisy observation. Eigenvalues, SVD,
and condition numbers follow only after these operations are comfortable.
""",
    "FND-02": r"""
## Beginner Core · Population, sample, distributions, and uncertainty

A **population** is the full process we care about; a **sample** is the observed
subset. A parameter describes a population, while a statistic is calculated from
a sample. Sampling method matters as much as sample size.

A PMF assigns probability to discrete values. A PDF describes continuous density;
probability is area over a range. A CDF answers “what probability lies at or below
this value?” A Bernoulli variable has one trial and values 0 or 1. A Binomial
variable counts successes across repeated independent Bernoulli trials. A Gaussian
distribution describes a symmetric bell shape using mean and variance.

For sample values $(2,4,6)$, the mean is 4. Population-style variance divides the
squared deviations by 3, giving $8/3$. Sample variance divides by $n-1=2$, giving
4 when estimating population spread from this sample.

If a sample mean is 10 and its standard error is 1, an approximate 95% interval is

$$
10\pm1.96(1)=[8.04,11.96].
$$

The interval procedure has long-run coverage under its assumptions. It does not
say that 95% of individual observations lie inside it. A p-value is the chance of
data at least this extreme if the null model were true; it is not the probability
that the null hypothesis is true.
""",
    "FND-03": r"""
## Beginner Core · From question to trustworthy training table

Use this order: define the prediction time and one-row meaning; identify target
and features; inspect schema and duplicates; separate train, validation, and test
by the correct boundary; learn cleaning statistics from training only; apply the
frozen transformations elsewhere; document every decision.

Nominal categories have names but no order, ordinal categories have a real order,
and numerical columns have meaningful arithmetic. An identifier is not automatically
a useful numerical feature. Check missingness by column and by important groups.

An outlier is an observation far from most others. It may be a measurement error,
a rare valid case, or the most important case in the dataset. Investigate its
source before removing or clipping it. Preserve a quality report containing row
counts, schema, duplicates, missingness, ranges, category levels, and repair counts.

EDA generates questions and detects problems; it does not justify repeatedly
changing the pipeline until the final test score improves. Keep the final test
partition sealed for later evaluation.
""",
    "FND-04": r"""
## Beginner Core · Loss before gradient descent

A **loss** turns prediction error into a number to minimize. A metric reports
usefulness; it need not be differentiable or equal to the training loss. For one
parameter $w$, data point $x=2$, target $y=6$, and prediction $\hat y=wx$, use
squared loss $L(w)=(wx-y)^2$.

At $w=1$, prediction is 2 and loss is 16. The derivative is

$$
\frac{dL}{dw}=2(wx-y)x=2(2-6)(2)=-16.
$$

Read it as “the slope of loss with respect to w is negative sixteen.” With learning
rate $\eta=0.1$, one update is

$$
w_{new}=w-\eta\frac{dL}{dw}=1-0.1(-16)=2.6.
$$

The new prediction is 5.2 and loss is $0.64$, so this step helped. A learning rate
that is too large can increase the loss. Stop using declared evidence such as a
small gradient, little validation improvement, or a step budget—not merely because
training loss decreased once. Momentum and Adam are extensions of this core loop.
""",
    "CML-01": r"""
## Beginner Core · One line, every calculation visible

Suppose hours studied are $x=(1,2,3)$ and scores are $y=(3,5,7)$. Predicting the
training mean, 5, is the no-feature baseline. The line $\hat y=2x+1$ predicts
$(3,5,7)$.

Residual means actual minus predicted: $r_i=y_i-\hat y_i$. Here residuals are
$(0,0,0)$ and mean squared error is

$$
\operatorname{MSE}=\frac{0^2+0^2+0^2}{3}=0.
$$

Read it as “MSE is the average squared residual.” $y_i$ has target units, residuals
have target units, and MSE has squared target units. RMSE returns to target units.
The intercept 1 is the prediction at $x=0$; the slope 2 means the prediction rises
by two score units for one additional hour, within the supported range. This is an
association, not proof that studying caused the change.

First compare the fitted line with the mean baseline on held-out data. Then inspect
residuals for curvature, changing spread, and unusual points. The closed-form solver
belongs to this lesson; gradient fitting belongs after FND-04.
""",
    "CML-02": r"""
## Beginner Core · Score, probability, and decision are different objects

If probability $p=0.8$, odds are $p/(1-p)=0.8/0.2=4$: one event is estimated four
times as likely as the alternative. Log-odds are $\log(4)\approx1.386$.

For linear score $z=1$, the sigmoid gives

$$
\sigma(1)=\frac{1}{1+e^{-1}}\approx0.731.
$$

Read it as “sigmoid of one is about zero point seven three one.” $e$ is the natural
exponential base, $z$ is any real score, and the output lies strictly between zero
and one. For a positive label $y=1$, binary cross-entropy for this case is
$-\log(0.731)\approx0.313$. A confidently wrong probability receives a much larger
loss.

A threshold is a separate policy. At threshold 0.5, 0.731 becomes class 1; at 0.8,
it becomes class 0. Select the threshold from validation evidence and error costs.
Always compare with a class-frequency baseline and report held-out errors.
""",
    "MLE-01": r"""
## Beginner Core · Calculate every classification metric from one table

For 10 cases, suppose $TP=3$, $TN=4$, $FP=2$, and $FN=1$. These counts sum to 10.

$$
\text{Accuracy}=\frac{3+4}{10}=0.70,
\quad
\text{Precision}=\frac{3}{3+2}=0.60,
\quad
\text{Recall}=\frac{3}{3+1}=0.75.
$$

Accuracy asks “what fraction was correct?” Precision asks “among predicted
positives, what fraction was positive?” Recall asks “among actual positives, what
fraction did we find?” Specificity is $4/(4+2)=0.667$. F1 is the harmonic mean of
precision and recall: $2(0.60)(0.75)/(0.60+0.75)=0.667$.

If a denominator is zero, the metric is undefined; choose and document a library
policy rather than hiding it. Inspect counts before a single score. Threshold curves
compare possible decision policies; AUC measures ranking, not calibration or a
deployed threshold. For numerical targets, begin with MAE and RMSE in target units
and compare against a mean or median baseline.
""",
    "MLE-02": r"""
## Beginner Core · Split before cross-validation

Start with three roles. Training data fits parameters and transformations.
Validation data chooses models, features, thresholds, and hyperparameters. Test data
is used once for the final scoped estimate.

```text
all eligible historical data
        |
        +-- training: learn model and preprocessing
        +-- validation: make development choices
        +-- test: final untouched estimate
```

For 10 ordered time rows, training on rows 1–6, validating on 7–8, and testing on
9–10 respects time. A random split might train on row 10 and evaluate row 3, which
answers the wrong future-prediction question. If several rows belong to one patient,
keep the entire patient on one side.

Cross-validation repeats the train-validation boundary; it does not remove the need
for a final test set or correct grouping. Every learned step—imputation, scaling,
feature selection, target encoding—must fit inside each training fold. Nested CV is
an advanced option when the tuned procedure itself needs an unbiased estimate.
""",
    "PROD-04": r"""
## Beginner Core · Track one honest experiment before tuning

An experiment record must include: question and hypothesis; data snapshot and row
eligibility; split strategy; code version; feature and model configuration; random
seed; metric definition; result and uncertainty; artifact location; conclusion and
limitations.

Begin with a JSON or CSV record. Compare only runs that share the same data boundary
and metric definition. Change one declared factor when the goal is to learn its
effect. A run ID is a label, not evidence.

```text
hypothesis -> frozen data/split -> configuration -> run -> metric -> conclusion
                    |                              |
                    +---------- recorded ----------+
```

Grid search and random search are model-selection procedures and belong inside the
validation design. The final test set is not a tuning dashboard. Bayesian optimization
is an optional later method for choosing configurations efficiently; Gaussian-process
formulas are not required to understand experiment tracking.
""",
    "MLE-03": r"""
## Beginner Core · Choose a representation for the model

Suppose color has values red, blue, and green. Encoding them as 1, 2, and 3 invents
an order and distances. One-hot encoding uses three binary columns instead. For an
unseen category at prediction time, use an explicit unknown policy such as all-zero
with `handle_unknown="ignore"`, and monitor its rate.

Standardization uses training mean and standard deviation. It is helpful for
distance-based, gradient-based, and regularized linear models; it is usually
unnecessary for ordinary tree splits. It is not universally required.

For hour-of-day, 23 and 0 are neighbors. Sine and cosine encode that circular
relationship. Interaction features allow a linear model to represent “the effect of
one feature depends on another.” Each feature must be available with the same meaning
at prediction time.

Compare every transformation against a raw-feature baseline using the same validation
design. Target encoding and feature selection must occur inside folds; keep them on
the advanced path until one-hot and train-only preprocessing are mastered.
""",
    "CML-03": r"""
## Beginner Core · Calculate one split by hand

Consider labels $y=(0,0,1,1)$ sorted by feature $x=(1,2,3,4)$. The parent has class
proportions $(0.5,0.5)$, so Gini impurity is

$$
G_{parent}=1-(0.5^2+0.5^2)=0.5.
$$

Split at $x\le2.5$. The left labels are $(0,0)$ and right labels are $(1,1)$, so both
child impurities are zero. Weighted child impurity is $(2/4)0+(2/4)0=0$, and gain is
$0.5-0=0.5$. Compare this with other candidate midpoints before choosing it.

Read Gini as “one minus the sum of squared class proportions.” It ranges from zero
for a pure binary node to 0.5 for an even binary mix. The greedy choice is local and
can change when data changes. Duplicate feature rows with conflicting labels may
prevent pure leaves. Grow a small tree, evaluate held-out performance, and control
depth or leaf size before interpreting its rules.
""",
    "CML-04": r"""
## Beginner Core · Bootstrap, vary, and vote

For training rows $(A,B,C,D)$, one bootstrap draw might be $(B,D,B,A)$. Sampling is
with replacement, so B appears twice and C is out-of-bag for this tree. Repeat the
draw for each tree and consider a random subset of features at each split.

If three trees predict probabilities $(0.8,0.6,0.2)$, the forest probability is
$(0.8+0.6+0.2)/3=0.533$. With threshold 0.5, the class is 1. Averaging reduces
unstable variation only when tree errors are not perfectly correlated.

OOB predictions use only trees that omitted that row. They are a useful internal
estimate, not an automatic replacement for time-aware, group-aware, or final test
evaluation. Feature-subset defaults differ across libraries and tasks, so treat them
as configurations, not mathematical constants. More trees usually stabilize the
estimate; validation still checks the full pipeline.
""",
    "CML-05": r"""
## Beginner Core · Three residual-correction rounds before XGBoost

For targets $y=(2,4,6)$, start with their mean $F_0=4$. Residuals are
$y-F_0=(-2,0,2)$. Suppose a tiny stump predicts corrections $(-1,0,1)$ and learning
rate is $\nu=0.5$. Then

$$
F_1=F_0+0.5(-1,0,1)=(3.5,4,4.5).
$$

New residuals are $(-1.5,0,1.5)$, smaller than before. Each later tree predicts the
current residual pattern, and the learning rate limits each correction. Measure
training and validation loss after every round; stop at the best validation round.

This is gradient boosting with squared loss. Other losses replace ordinary residuals
with negative loss gradients. XGBoost adds regularized second-order approximations,
efficient split search, missing-value handling, and systems optimizations. Those
details are an advanced extension after the correction loop, shrinkage, depth, and
early stopping are understood.
""",
}


CORE_CODE = {
    "PRE-03": """import numpy as np\n\nvalues = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\ncolumn_means = np.array([2.5, 3.5, 4.5])\ncentered = values - column_means\n\nprint(\"values shape:\", values.shape)\nprint(\"means shape :\", column_means.shape)\nprint(\"centered:\\n\", centered)\nassert centered.shape == (2, 3)\nassert np.allclose(centered.mean(axis=0), 0.0)""",
    "FND-01": """import numpy as np\n\nmatrix = np.array([[1, 2], [3, 4]])\nvector = np.array([5, 6])\nmanual_output = np.array([1 * 5 + 2 * 6, 3 * 5 + 4 * 6])\nlibrary_output = matrix @ vector\n\nprint(\"matrix shape:\", matrix.shape)\nprint(\"vector shape:\", vector.shape)\nprint(\"manual output:\", manual_output)\nprint(\"NumPy output :\", library_output)\nassert np.array_equal(manual_output, library_output)""",
    "FND-04": """weight = 1.0\ninput_value = 2.0\ntarget = 6.0\nlearning_rate = 0.1\n\nfor step in range(3):\n    prediction = weight * input_value\n    loss = (prediction - target) ** 2\n    gradient = 2 * (prediction - target) * input_value\n    print(f\"step={step} weight={weight:.3f} prediction={prediction:.3f} loss={loss:.3f} gradient={gradient:.3f}\")\n    weight = weight - learning_rate * gradient""",
    "CML-01": """import numpy as np\n\nhours = np.array([1.0, 2.0, 3.0])\nscores = np.array([3.0, 5.0, 7.0])\npredictions = 2.0 * hours + 1.0\nresiduals = scores - predictions\nmean_squared_error = np.mean(residuals ** 2)\n\nprint(\"predictions:\", predictions)\nprint(\"residuals  :\", residuals)\nprint(\"MSE        :\", mean_squared_error)\nassert mean_squared_error == 0.0""",
    "CML-02": """import numpy as np\n\nlinear_score = 1.0\nprobability = 1.0 / (1.0 + np.exp(-linear_score))\npositive_label_loss = -np.log(probability)\n\nprint(f\"linear score: {linear_score:.3f}\")\nprint(f\"probability : {probability:.3f}\")\nprint(f\"loss for y=1: {positive_label_loss:.3f}\")\nassert 0.0 < probability < 1.0""",
    "MLE-01": """true_positive, true_negative = 3, 4\nfalse_positive, false_negative = 2, 1\n\naccuracy = (true_positive + true_negative) / 10\nprecision = true_positive / (true_positive + false_positive)\nrecall = true_positive / (true_positive + false_negative)\nf1 = 2 * precision * recall / (precision + recall)\n\nprint(f\"accuracy={accuracy:.3f}\")\nprint(f\"precision={precision:.3f}\")\nprint(f\"recall={recall:.3f}\")\nprint(f\"f1={f1:.3f}\")""",
    "CML-03": """parent_gini = 1 - (0.5 ** 2 + 0.5 ** 2)\nleft_gini = 0.0\nright_gini = 0.0\nweighted_children = (2 / 4) * left_gini + (2 / 4) * right_gini\ngain = parent_gini - weighted_children\n\nprint(\"parent Gini      :\", parent_gini)\nprint(\"weighted children:\", weighted_children)\nprint(\"impurity gain    :\", gain)\nassert gain == 0.5""",
    "CML-04": """import numpy as np\n\ntree_probabilities = np.array([0.8, 0.6, 0.2])\nforest_probability = tree_probabilities.mean()\nforest_class = int(forest_probability >= 0.5)\n\nprint(\"tree probabilities :\", tree_probabilities)\nprint(f\"forest probability : {forest_probability:.3f}\")\nprint(\"forest class       :\", forest_class)\nassert forest_class == 1""",
    "CML-05": """import numpy as np\n\ntargets = np.array([2.0, 4.0, 6.0])\npredictions = np.full(3, targets.mean())\nlearning_rate = 0.5\ncorrections = [np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.0, 0.5])]\n\nfor round_number, correction in enumerate(corrections, start=1):\n    residuals_before = targets - predictions\n    predictions = predictions + learning_rate * correction\n    residuals_after = targets - predictions\n    print(f\"round {round_number}\")\n    print(\"  residuals before:\", residuals_before)\n    print(\"  predictions      :\", predictions)\n    print(\"  residuals after :\", residuals_after)""",
}
