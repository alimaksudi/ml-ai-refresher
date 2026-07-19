"""Build CML-04: a beginner-first Random Forest mastery lesson."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # CML-04 · Random Forest

    **Prerequisites:** CML-03 · Decision Trees and MLE-02 · Validation and Data Leakage  
    **Estimated study time:** 8–10 hours, including practice  
    **Next lesson:** CML-05 · Gradient Boosting and XGBoost

    One decision tree can learn useful nonlinear rules, but a small change in the
    training rows may produce a different tree. A random forest reduces that
    instability by combining many deliberately different trees.

    The goal is not to memorize `RandomForestClassifier`. The goal is to trace where
    every tree's rows came from, why its candidate features differ, how votes become
    a prediction, and which evidence is safe to use during development.

    ### Scope boundary

    This lesson teaches binary classification forests. It intentionally defers:

    - feature importance and SHAP to MLE-05;
    - imbalanced-learning methods to MLE-04;
    - formal model monitoring to PROD-05;
    - boosting to CML-05;
    - regression forests and advanced hyperparameter search to later practice.

    We use wrong-row counts as the first comparison and keep a final test partition
    sealed until one model configuration has been chosen.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - demonstrate that one decision tree changes across training samples;
    - average several predictions manually;
    - create and inspect one bootstrap sample;
    - distinguish bootstrap draws, unique rows, repeated rows, and OOB rows;
    - explain bagging as bootstrap training followed by aggregation;
    - explain why similar trees make similar mistakes;
    - explain why random feature subsets create useful diversity;
    - calculate one forest vote and probability;
    - implement forest orchestration around decision trees;
    - calculate OOB predictions without allowing a tree to predict rows it trained on;
    - use `RandomForestClassifier` with explicit growth controls;
    - compare a baseline, one tree, and one forest on validation data;
    - evaluate the selected forest once on a sealed test partition;
    - decide when a forest is preferable to one tree or logistic regression.

    ### Learning path

    ```mermaid
    flowchart LR
        A[One tree changes] --> B[Average predictions]
        B --> C[Bootstrap rows]
        C --> D[Bag many trees]
        D --> E[Notice correlated trees]
        E --> F[Randomize candidate features]
        F --> G[Random forest]
        G --> H[OOB evidence]
        H --> I[Validation and sealed test]
    ```

    Dependency map:

    Decision-tree splits  
    → required before bootstrap ensembles  
    → because every forest member is still a decision tree.

    Train/validation/test boundaries  
    → required before OOB evaluation  
    → because OOB evidence is useful only when its role and limitations are clear.
    """),

    md(r"""
    ## 2 · The practical problem: one tree is unstable

    Imagine predicting whether a machine will need maintenance. A deep tree might
    ask about vibration first. If a few training machines change, another fitted tree
    might ask about temperature first and make different decisions for the same
    validation machines.

    This is **instability**: small changes in training data cause meaningful changes
    in the fitted model. It is often called high variance. Here, variance means
    variation across models trained on different samples—not the spread of one
    feature column.

    We will train several trees on different samples from the same training
    partition. The validation partition stays unchanged so their decisions are
    directly comparable.
    """),

    code(r"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    # Create enough features to make feature-subset randomness visible later.
    all_features, all_labels = make_classification(
        n_samples=720,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        class_sep=1.0,
        flip_y=0.08,
        random_state=42,
    )

    # First seal 20% for the final test. It is not used during lesson development.
    development_features, sealed_test_features, development_labels, sealed_test_labels = train_test_split(
        all_features,
        all_labels,
        test_size=0.20,
        stratify=all_labels,
        random_state=42,
    )

    # Split the remaining rows into training and validation partitions.
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        development_features,
        development_labels,
        test_size=0.25,
        stratify=development_labels,
        random_state=42,
    )

    print("training rows:", len(train_labels))
    print("validation rows:", len(validation_labels))
    print("sealed test rows:", len(sealed_test_labels))
    print("test status: sealed — not used for model choice")

    assert len(train_labels) + len(validation_labels) + len(sealed_test_labels) == 720
    assert set(np.unique(all_labels)) == {0, 1}
    """),

    code(r"""
    sample_generator = np.random.default_rng(7)
    sampled_tree_decisions = []
    sampled_root_features = []

    # Each tree receives a different sample of training rows, but the same validation rows.
    for sample_number in range(6):
        sampled_indices = sample_generator.integers(0, len(train_labels), len(train_labels))
        sampled_tree = DecisionTreeClassifier(max_depth=5, random_state=sample_number)
        sampled_tree.fit(train_features[sampled_indices], train_labels[sampled_indices])

        sampled_tree_decisions.append(sampled_tree.predict(validation_features))
        sampled_root_features.append(int(sampled_tree.tree_.feature[0]))

    sampled_tree_decisions = np.asarray(sampled_tree_decisions)
    validation_vote_counts = sampled_tree_decisions.sum(axis=0)
    disagreement_rows = np.sum((validation_vote_counts > 0) & (validation_vote_counts < 6))

    print("root feature chosen by each tree:", sampled_root_features)
    print("validation rows with at least one disagreement:", int(disagreement_rows))
    print("first 12 trees-by-row decisions:\n", sampled_tree_decisions[:, :12])

    assert sampled_tree_decisions.shape == (6, len(validation_labels))
    assert disagreement_rows > 0
    """),

    md(r"""
    ## 3 · Averaging turns several opinions into one

    Suppose three trees predict class-1 probabilities:

    $$
    0.80,\quad 0.55,\quad 0.20
    $$

    Their average is:

    $$
    \bar p=\frac{0.80+0.55+0.20}{3}=0.516\overline{6}
    $$

    **Symbols:** $p_m$ is tree $m$'s class-1 probability; $M$ is the number of
    trees; $\bar p$ is their average.

    With a declared threshold of 0.5, the forest predicts class 1. For hard votes
    $[1,1,0]$, two of three trees also vote for class 1.

    Analogy: ask several mechanics independently. One unusual opinion has less power
    after the group combines its evidence. The analogy has a limit: trees are not
    truly independent experts because they learn from overlapping data.
    """),

    code(r"""
    tree_probabilities = np.array([0.80, 0.55, 0.20])
    forest_probability = tree_probabilities.mean()
    forest_decision = int(forest_probability >= 0.5)

    hard_votes = np.array([1, 1, 0])
    positive_vote_fraction = hard_votes.mean()

    print("tree probabilities:", tree_probabilities)
    print("forest probability:", round(forest_probability, 4))
    print("hard-vote fraction:", round(positive_vote_fraction, 4))
    print("forest decision:", forest_decision)

    assert np.isclose(forest_probability, 1.55 / 3)
    assert forest_decision == 1
    """),

    md(r"""
    ## 4 · Bootstrap sampling creates different training sets

    A **bootstrap sample** draws the same number of positions as the training set,
    with replacement. “With replacement” means a row may be selected again.

    For five rows with indices $[0,1,2,3,4]$, one possible sample is:

    $$
    [1,3,3,0,4]
    $$

    - row 3 appears twice;
    - rows 0, 1, and 4 appear once;
    - row 2 never appears and is out of bag;
    - there are five draws but only four unique training rows.

    For $n$ training rows, the chance that one particular row is omitted from all
    $n$ draws is:

    $$
    \left(1-\frac{1}{n}\right)^n\longrightarrow e^{-1}\approx0.368
    $$

    So a large bootstrap contains about 63.2% unique rows and leaves about 36.8% OOB
    for that tree. The tree still receives $n$ draws because repeated rows count.
    """),

    code(r"""
    original_indices = np.arange(5)
    bootstrap_indices = np.array([1, 3, 3, 0, 4])

    # Count how many times each original row appears in the bootstrap draw.
    draw_counts = np.bincount(bootstrap_indices, minlength=len(original_indices))
    unique_bootstrap_indices = np.flatnonzero(draw_counts > 0)
    out_of_bag_indices = np.flatnonzero(draw_counts == 0)

    bootstrap_table = pd.DataFrame(
        {
            "row_index": original_indices,
            "times_drawn": draw_counts,
            "is_out_of_bag": draw_counts == 0,
        }
    )

    print(bootstrap_table.to_string(index=False))
    print("number of draws:", len(bootstrap_indices))
    print("unique rows used:", unique_bootstrap_indices)
    print("OOB rows:", out_of_bag_indices)

    assert len(bootstrap_indices) == 5
    assert np.array_equal(unique_bootstrap_indices, [0, 1, 3, 4])
    assert np.array_equal(out_of_bag_indices, [2])
    """),

    md(r"""
    ## 5 · Bagging means bootstrap, train, and aggregate

    **Bagging** is short for bootstrap aggregating:

    1. draw one bootstrap sample;
    2. train one model on that sample;
    3. repeat many times;
    4. aggregate their predictions.

    We first allow every tree to consider every feature. This is bagged trees, not
    yet a random forest. Separating the steps makes the forest's extra feature
    randomness easier to understand.

    ```mermaid
    flowchart TD
        A[Training rows] --> B1[Bootstrap sample 1]
        A --> B2[Bootstrap sample 2]
        A --> B3[Bootstrap sample M]
        B1 --> T1[Tree 1]
        B2 --> T2[Tree 2]
        B3 --> T3[Tree M]
        T1 --> V[Average probabilities]
        T2 --> V
        T3 --> V
        V --> D[Apply decision threshold]
    ```
    """),

    code(r"""
    def fit_bagged_trees(
        feature_matrix,
        labels,
        number_of_trees,
        maximum_features,
        random_seed,
    ):
        '''Fit bootstrapped trees and retain the row history needed for OOB checks.'''
        bootstrap_generator = np.random.default_rng(random_seed)
        fitted_trees = []
        bootstrap_history = []

        for tree_number in range(number_of_trees):
            # Draw exactly n positions with replacement.
            sampled_indices = bootstrap_generator.integers(
                low=0,
                high=len(labels),
                size=len(labels),
            )

            # The tree implementation comes from CML-03; this lesson orchestrates the ensemble.
            tree = DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=3,
                max_features=maximum_features,
                random_state=random_seed + tree_number,
            )
            tree.fit(feature_matrix[sampled_indices], labels[sampled_indices])

            fitted_trees.append(tree)
            bootstrap_history.append(sampled_indices)

        return fitted_trees, bootstrap_history


    def average_positive_probabilities(fitted_trees, feature_matrix):
        '''Average the class-1 probability produced by every fitted tree.'''
        probability_rows = [
            tree.predict_proba(feature_matrix)[:, 1]
            for tree in fitted_trees
        ]
        return np.mean(probability_rows, axis=0)


    bagged_trees, bagged_bootstrap_history = fit_bagged_trees(
        train_features,
        train_labels,
        number_of_trees=40,
        maximum_features=None,
        random_seed=10,
    )
    bagged_validation_probabilities = average_positive_probabilities(
        bagged_trees,
        validation_features,
    )
    bagged_validation_decisions = (bagged_validation_probabilities >= 0.5).astype(int)
    bagged_validation_wrong = int(np.sum(bagged_validation_decisions != validation_labels))

    print("bagged trees:", len(bagged_trees))
    print("validation rows predicted incorrectly:", bagged_validation_wrong)
    print("first five averaged probabilities:", bagged_validation_probabilities[:5].round(3))

    assert len(bagged_trees) == 40
    assert np.all((bagged_validation_probabilities >= 0) & (bagged_validation_probabilities <= 1))
    """),

    md(r"""
    ## 6 · Similar trees leave a correlation problem

    Bootstrap samples differ, but they overlap. If one feature is much stronger than
    the others, many trees may still choose it near the root. Their mistakes then
    remain similar.

    Correlation $\rho$ measures how similarly two tree predictions vary. Under the
    simplified assumptions that every tree has variance $\sigma^2$ and every pair
    has the same correlation $\rho$, the variance of the average of $M$ trees is:

    $$
    \operatorname{Var}(\bar T)
    =\rho\sigma^2+\frac{1-\rho}{M}\sigma^2
    $$

    **Symbols:** $T_m$ is one tree's prediction; $\bar T$ is their average; $M$ is
    the number of trees; $\sigma^2$ is one tree's variance; $\rho$ is pairwise
    correlation.

    Numerical example with $M=5$:

    - if $\rho=0$, variance becomes $0.2\sigma^2$;
    - if $\rho=0.6$, variance becomes $0.68\sigma^2$.

    More trees reduce one part of the variance, but strong correlation leaves a
    floor. This formula is a useful model of the mechanism, not a promise that every
    finite validation curve decreases monotonically.
    """),

    code(r"""
    def relative_average_variance(number_of_trees, pairwise_correlation):
        '''Return ensemble variance divided by one tree's variance.'''
        shared_part = pairwise_correlation
        reducible_part = (1 - pairwise_correlation) / number_of_trees
        return shared_part + reducible_part


    independent_ratio = relative_average_variance(5, 0.0)
    correlated_ratio = relative_average_variance(5, 0.6)

    print("relative variance when correlation is 0.0:", independent_ratio)
    print("relative variance when correlation is 0.6:", correlated_ratio)

    assert np.isclose(independent_ratio, 0.2)
    assert np.isclose(correlated_ratio, 0.68)
    """),

    md(r"""
    ## 7 · Random feature subsets create additional diversity

    A random forest adds one rule to bagging:

    > At each node, the tree may search only a random subset of features.

    If there are eight features and `max_features=3`, one node might consider
    features $[0,3,6]$ while another considers $[1,4,7]$. A strong feature is not
    removed from the dataset; it is simply unavailable at some split searches.

    This can make individual trees slightly weaker, but it can make their errors less
    similar. The ensemble benefits when diversity gained is worth more than individual
    tree strength lost.

    Random feature selection happens at **each split**, not once for the whole
    forest and not once for the whole tree. Common defaults are starting points, not
    mathematical laws.
    """),

    code(r"""
    # Bagging lets every split inspect all eight features.
    bagging_roots = np.array([tree.tree_.feature[0] for tree in bagged_trees])

    # A forest restricts each split to a random subset of features.
    forest_trees, forest_bootstrap_history = fit_bagged_trees(
        train_features,
        train_labels,
        number_of_trees=40,
        maximum_features="sqrt",
        random_seed=10,
    )
    forest_roots = np.array([tree.tree_.feature[0] for tree in forest_trees])

    print("bagging root-feature counts:", np.bincount(bagging_roots, minlength=8))
    print("forest root-feature counts:", np.bincount(forest_roots, minlength=8))
    print("unique bagging roots:", len(np.unique(bagging_roots)))
    print("unique forest roots:", len(np.unique(forest_roots)))

    assert len(bagging_roots) == len(forest_roots) == 40
    assert np.all((forest_roots >= 0) & (forest_roots < train_features.shape[1]))
    """),

    md(r"""
    ## 8 · Build the forest orchestration explicitly

    CML-03 built the tree mechanism. Repeating that entire implementation would hide
    the new idea, so this scratch forest uses a decision tree as a known component
    and implements the ensemble responsibilities itself:

    - bootstrap bookkeeping;
    - one randomized tree per sample;
    - probability aggregation;
    - OOB-only aggregation;
    - a separate final decision threshold.

    This is “from scratch” at the forest layer. The production version later replaces
    these loops with a tested, optimized library implementation.
    """),

    code(r"""
    class LearningRandomForest:
        '''Small educational binary-classification forest with visible bookkeeping.'''

        def __init__(
            self,
            number_of_trees=60,
            maximum_depth=6,
            minimum_leaf_rows=3,
            maximum_features="sqrt",
            random_seed=42,
        ):
            self.number_of_trees = number_of_trees
            self.maximum_depth = maximum_depth
            self.minimum_leaf_rows = minimum_leaf_rows
            self.maximum_features = maximum_features
            self.random_seed = random_seed

        def fit(self, feature_matrix, labels):
            '''Bootstrap the rows, fit randomized trees, and retain OOB masks.'''
            feature_matrix = np.asarray(feature_matrix)
            labels = np.asarray(labels, dtype=int)
            number_of_rows = len(labels)
            bootstrap_generator = np.random.default_rng(self.random_seed)

            self.trees_ = []
            self.bootstrap_indices_ = []
            self.out_of_bag_masks_ = []

            for tree_number in range(self.number_of_trees):
                # Draw n row positions with replacement for this tree.
                sampled_indices = bootstrap_generator.integers(
                    0,
                    number_of_rows,
                    size=number_of_rows,
                )

                # Mark rows never selected by this tree. Only these rows may receive
                # an OOB prediction from this particular tree.
                out_of_bag_mask = np.ones(number_of_rows, dtype=bool)
                out_of_bag_mask[np.unique(sampled_indices)] = False

                # Tree construction was learned in CML-03. max_features="sqrt"
                # adds a fresh random candidate-feature subset at every split.
                tree = DecisionTreeClassifier(
                    max_depth=self.maximum_depth,
                    min_samples_leaf=self.minimum_leaf_rows,
                    max_features=self.maximum_features,
                    random_state=self.random_seed + tree_number,
                )
                tree.fit(feature_matrix[sampled_indices], labels[sampled_indices])

                self.trees_.append(tree)
                self.bootstrap_indices_.append(sampled_indices)
                self.out_of_bag_masks_.append(out_of_bag_mask)

            return self

        def predict_positive_probability(self, feature_matrix):
            '''Average the class-1 probabilities from all trees.'''
            tree_probabilities = np.vstack(
                [tree.predict_proba(feature_matrix)[:, 1] for tree in self.trees_]
            )
            return tree_probabilities.mean(axis=0)

        def predict(self, feature_matrix, threshold=0.5):
            '''Convert averaged probabilities into decisions using a declared threshold.'''
            probabilities = self.predict_positive_probability(feature_matrix)
            return (probabilities >= threshold).astype(int)

        def out_of_bag_positive_probability(self, training_features):
            '''Aggregate only predictions from trees that omitted each training row.'''
            number_of_rows = len(training_features)
            probability_sum = np.zeros(number_of_rows, dtype=float)
            prediction_count = np.zeros(number_of_rows, dtype=int)

            for tree, out_of_bag_mask in zip(self.trees_, self.out_of_bag_masks_):
                if not np.any(out_of_bag_mask):
                    continue

                # This tree predicts only rows absent from its own bootstrap sample.
                oob_probabilities = tree.predict_proba(
                    training_features[out_of_bag_mask]
                )[:, 1]
                probability_sum[out_of_bag_mask] += oob_probabilities
                prediction_count[out_of_bag_mask] += 1

            valid_mask = prediction_count > 0
            averaged_probabilities = np.full(number_of_rows, np.nan)
            averaged_probabilities[valid_mask] = (
                probability_sum[valid_mask] / prediction_count[valid_mask]
            )
            return averaged_probabilities, prediction_count


    learning_forest = LearningRandomForest(number_of_trees=60, random_seed=21)
    learning_forest.fit(train_features, train_labels)

    learning_validation_probabilities = learning_forest.predict_positive_probability(
        validation_features
    )
    learning_validation_decisions = learning_forest.predict(validation_features)
    learning_validation_wrong = int(
        np.sum(learning_validation_decisions != validation_labels)
    )

    print("trees fitted:", len(learning_forest.trees_))
    print("validation rows predicted incorrectly:", learning_validation_wrong)
    print("first five forest probabilities:", learning_validation_probabilities[:5].round(3))

    assert len(learning_forest.trees_) == 60
    assert np.all((learning_validation_probabilities >= 0) & (learning_validation_probabilities <= 1))
    """),

    md(r"""
    ## 9 · OOB evidence reuses omissions safely

    For one training row, use only trees whose bootstrap samples omitted that row.
    Those trees did not fit on it, so their combined prediction is out of bag.

    OOB evaluation is useful, but it is not “free cross-validation” and it does not
    erase the need for a final test:

    - it shares one training dataset across overlapping bootstrap samples;
    - preprocessing learned from all rows before forest fitting would still leak;
    - ordinary row bootstrap is wrong for time-ordered or grouped observations;
    - too few trees can leave some rows with very few OOB predictions;
    - repeated tuning against OOB results turns OOB evidence into development evidence.

    Use OOB as an efficient training-stage estimate when ordinary row resampling
    matches the data-generating process. Keep the final test sealed.
    """),

    code(r"""
    oob_probabilities, oob_prediction_counts = (
        learning_forest.out_of_bag_positive_probability(train_features)
    )
    rows_with_oob_evidence = ~np.isnan(oob_probabilities)
    oob_decisions = (oob_probabilities[rows_with_oob_evidence] >= 0.5).astype(int)
    oob_wrong = int(
        np.sum(oob_decisions != train_labels[rows_with_oob_evidence])
    )

    print("training rows with OOB evidence:", int(rows_with_oob_evidence.sum()))
    print("minimum OOB predictions for a covered row:", int(oob_prediction_counts[rows_with_oob_evidence].min()))
    print("maximum OOB predictions for a covered row:", int(oob_prediction_counts.max()))
    print("OOB rows predicted incorrectly:", oob_wrong)

    assert rows_with_oob_evidence.all()
    assert np.all(oob_prediction_counts > 0)
    assert np.all((oob_probabilities >= 0) & (oob_probabilities <= 1))
    """),

    md(r"""
    ## 10 · More trees stabilize an estimate; they do not guarantee a monotone score

    As more trees are averaged, predictions usually settle. A finite validation
    wrong-row count can still move up or down when one tree is added because some
    probabilities cross the decision threshold.

    Therefore:

    - more trees usually reduce Monte Carlo noise from the forest construction;
    - performance tends to approach a plateau;
    - adding a tree does not mathematically guarantee a lower validation error;
    - `n_estimators` mainly trades stability against training, memory, and latency;
    - depth and minimum leaf size still control what each tree can memorize.

    We inspect validation evidence below. The sealed test remains untouched.
    """),

    code(r"""
    running_probability_sum = np.zeros(len(validation_labels), dtype=float)
    stage_records = []

    # Add trees one at a time and observe the current ensemble on validation data.
    for number_used, tree in enumerate(learning_forest.trees_, start=1):
        running_probability_sum += tree.predict_proba(validation_features)[:, 1]
        running_probabilities = running_probability_sum / number_used
        running_decisions = (running_probabilities >= 0.5).astype(int)
        wrong_rows = int(np.sum(running_decisions != validation_labels))
        stage_records.append({"trees_used": number_used, "validation_wrong": wrong_rows})

    stage_table = pd.DataFrame(stage_records)
    selected_stages = stage_table.iloc[[0, 4, 9, 19, 39, 59]]
    print(selected_stages.to_string(index=False))

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(stage_table["trees_used"], stage_table["validation_wrong"])
    axis.set_xlabel("number of trees averaged")
    axis.set_ylabel("validation rows predicted incorrectly")
    axis.set_title("Validation results can fluctuate before stabilizing")
    axis.grid(alpha=0.3)
    plt.show()

    assert len(stage_table) == 60
    assert stage_table["validation_wrong"].between(0, len(validation_labels)).all()
    """),

    md(r"""
    ## 11 · Use scikit-learn for practical work

    `RandomForestClassifier` implements the same mechanism with optimized tree
    construction, parallelism, input validation, OOB support, and many tested edge
    cases.

    Important controls:

    - `n_estimators`: number of trees;
    - `max_features`: candidate features available at each split;
    - `max_depth`: maximum questions along one path;
    - `min_samples_leaf`: minimum training rows in every leaf;
    - `bootstrap`: whether each tree receives a bootstrap sample;
    - `oob_score`: whether sklearn aggregates OOB predictions;
    - `random_state`: reproducible sampling and split randomness;
    - `n_jobs`: parallel CPU workers.

    Scaling is usually unnecessary because tree splits depend on order, but raw
    category strings still require an appropriate encoding and all learned
    preprocessing must respect the validation boundary.
    """),

    code(r"""
    from sklearn.ensemble import RandomForestClassifier

    sklearn_forest = RandomForestClassifier(
        n_estimators=120,
        max_features="sqrt",
        max_depth=6,
        min_samples_leaf=3,
        bootstrap=True,
        oob_score=True,
        random_state=21,
        n_jobs=-1,
    )
    sklearn_forest.fit(train_features, train_labels)

    sklearn_validation_probabilities = sklearn_forest.predict_proba(validation_features)[:, 1]
    sklearn_validation_decisions = sklearn_forest.predict(validation_features)
    sklearn_validation_wrong = int(
        np.sum(sklearn_validation_decisions != validation_labels)
    )

    print("sklearn validation rows predicted incorrectly:", sklearn_validation_wrong)
    print("sklearn OOB score:", round(sklearn_forest.oob_score_, 3))
    print("forest depth range:", (
        min(tree.get_depth() for tree in sklearn_forest.estimators_),
        max(tree.get_depth() for tree in sklearn_forest.estimators_),
    ))

    assert len(sklearn_forest.estimators_) == 120
    assert 0 <= sklearn_forest.oob_score_ <= 1
    assert np.all((sklearn_validation_probabilities >= 0) & (sklearn_validation_probabilities <= 1))
    """),

    md(r"""
    ## 12 · When to use a forest—and when not to

    | Model | Main shape | Strength | Limitation | Prefer it when |
    |---|---|---|---|---|
    | Logistic regression | One linear probability boundary | Stable, compact baseline | Misses uncreated interactions | Effects are roughly additive or clarity is central |
    | Shallow decision tree | A few if/else paths | Rules can be inspected | One tree is unstable | A compact decision policy matters most |
    | Random forest | Average of randomized trees | Flexible and usually stable | Larger and less directly readable | Tabular nonlinearities and interactions matter |
    | Gradient boosting | Sequential corrections | Often strong tabular performance | More sequential tuning decisions | After CML-05 prerequisites are mastered |

    Avoid or reconsider a forest when:

    - future predictions require extrapolation beyond observed numerical ranges;
    - model size or per-row latency must be extremely small;
    - one compact equation or short rule list is required;
    - observations are time- or group-dependent but the resampling design ignores it;
    - the data representation available in production differs from training.

    A forest does not automatically fix leakage, biased labels, missing future
    features, poor validation design, class imbalance, or causal questions.
    """),

    md(r"""
    ## 13 · Mini-project: Wine forest with a sealed test

    **Goal:** predict whether a wine is cultivar 0 using two chemical measurements.

    **Dataset columns:**

    - `alcohol`: measured alcohol level;
    - `color_intensity`: measured color intensity;
    - target: 1 for cultivar 0, otherwise 0.

    **Prediction time:** after both measurements exist.

    **Workflow:**

    1. create disjoint training, validation, and test partitions;
    2. freeze the training-majority baseline;
    3. fit one controlled decision tree;
    4. fit one pre-declared forest;
    5. compare validation wrong-row counts;
    6. inspect forest structure without feature-importance claims;
    7. evaluate the selected forest once on the sealed test.

    **Success contract:** the forest must beat the frozen baseline on validation;
    all partitions must be disjoint; the test result appears only in the final cell.
    """),

    code(r"""
    from sklearn.datasets import load_wine

    wine = load_wine()
    wine_feature_names = list(wine.feature_names)
    selected_columns = [
        wine_feature_names.index("alcohol"),
        wine_feature_names.index("color_intensity"),
    ]
    wine_features = wine.data[:, selected_columns]
    wine_labels = (wine.target == 0).astype(int)

    # Seal the project test partition before any model comparison.
    wine_development_features, wine_test_features, wine_development_labels, wine_test_labels = train_test_split(
        wine_features,
        wine_labels,
        test_size=0.20,
        stratify=wine_labels,
        random_state=17,
    )
    wine_train_features, wine_validation_features, wine_train_labels, wine_validation_labels = train_test_split(
        wine_development_features,
        wine_development_labels,
        test_size=0.25,
        stratify=wine_development_labels,
        random_state=17,
    )

    # Freeze the baseline from training labels only.
    baseline_class = int(wine_train_labels.mean() >= 0.5)
    baseline_validation_decisions = np.full(len(wine_validation_labels), baseline_class)
    baseline_validation_wrong = int(
        np.sum(baseline_validation_decisions != wine_validation_labels)
    )

    # Fit one deliberately shallow tree so the forest has a compact-rule comparison.
    project_tree = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=5,
        random_state=17,
    )
    project_tree.fit(wine_train_features, wine_train_labels)
    tree_validation_wrong = int(
        np.sum(project_tree.predict(wine_validation_features) != wine_validation_labels)
    )

    # Use one pre-declared forest configuration; do not tune repeatedly on validation.
    project_forest = RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        max_depth=2,
        min_samples_leaf=3,
        bootstrap=True,
        oob_score=True,
        random_state=17,
        n_jobs=-1,
    )
    project_forest.fit(wine_train_features, wine_train_labels)
    forest_validation_wrong = int(
        np.sum(project_forest.predict(wine_validation_features) != wine_validation_labels)
    )

    partition_manifest = pd.DataFrame(
        {
            "partition": ["train", "validation", "test"],
            "rows": [len(wine_train_labels), len(wine_validation_labels), len(wine_test_labels)],
            "used_for_model_choice": [True, True, False],
        }
    )

    print(partition_manifest.to_string(index=False))
    print("baseline validation wrong:", baseline_validation_wrong)
    print("single-tree validation wrong:", tree_validation_wrong)
    print("forest validation wrong:", forest_validation_wrong)
    print("forest OOB score:", round(project_forest.oob_score_, 3))
    print("test status: sealed — no test predictions calculated yet")

    assert len(wine_train_labels) + len(wine_validation_labels) + len(wine_test_labels) == len(wine_labels)
    assert forest_validation_wrong < baseline_validation_wrong
    assert forest_validation_wrong < tree_validation_wrong
    assert len(project_forest.estimators_) == 150
    assert all(tree.get_depth() <= 2 for tree in project_forest.estimators_)
    """),

    code(r"""
    # Final evaluation happens once, after the forest configuration is fixed.
    final_test_probabilities = project_forest.predict_proba(wine_test_features)[:, 1]
    final_test_decisions = (final_test_probabilities >= 0.5).astype(int)
    final_test_wrong = int(np.sum(final_test_decisions != wine_test_labels))

    print("final sealed-test rows:", len(wine_test_labels))
    print("final sealed-test rows predicted incorrectly:", final_test_wrong)
    print("first five final probabilities:", final_test_probabilities[:5].round(3))
    print("interpretation: this is one final estimate, not a new tuning signal")
    print("the validation win supported this forest here; it is not a universal forest guarantee")
    print("if the final result disappoints, report it and gather new evidence—do not retune on this test")

    assert len(final_test_decisions) == len(wine_test_labels)
    assert 0 <= final_test_wrong <= len(wine_test_labels)
    assert np.all((final_test_probabilities >= 0) & (final_test_probabilities <= 1))
    """),

    md(r"""
    ## 14 · Practice, solutions, and mastery checkpoint

    ### Worked example

    Three trees output probabilities $[0.7,0.4,0.9]$.

    $$
    \bar p=\frac{0.7+0.4+0.9}{3}=\frac{2.0}{3}\approx0.667
    $$

    With threshold 0.5, the forest predicts class 1.

    ### Guided practice

    1. For bootstrap draw $[2,2,0,4,2]$ from rows $[0,1,2,3,4]$, list repeated and OOB rows.
    2. Average tree probabilities $[0.2,0.6,0.8,0.4]$ and apply threshold 0.5.
    3. Explain why five identical trees are not a useful crowd.
    4. Explain why feature sampling happens at each split.
    5. Trace which trees may make an OOB prediction for one training row.

    ### Independent practice

    6. Write a bootstrap function returning sampled indices, counts, and an OOB mask.
    7. Add a method that returns all individual tree probabilities.
    8. Compare bagging with `max_features=None` against a forest with `max_features="sqrt"`.
    9. Plot validation wrong-row count as trees are added and explain any increases.
    10. Verify manually that no tree contributes an OOB prediction to one of its sampled rows.

    ### Challenge

    Rebuild the Wine project without copying. Include a baseline, one tree, one
    forest, bootstrap/OOB assertions, validation comparison, and exactly one final
    test evaluation. Do not use feature importance, SHAP, boosting, or the test set
    for model choice.

    ### Self-check

    Answer without notes:

    1. What exact problem does bagging solve?
    2. What is the difference between $n$ bootstrap draws and unique rows?
    3. What extra step turns bagged trees into a random forest?
    4. Why can random features help even when an individual tree becomes weaker?
    5. Which trees may predict a row for OOB evaluation?
    6. Why is OOB evidence not a replacement for every validation design?
    7. Why can validation error rise after adding one tree?
    8. When would you prefer one shallow tree?

    ### Solution and scoring rubric

    1. Draw $[2,2,0,4,2]$: row 2 repeats three times; rows 1 and 3 are OOB.
    2. The mean is 0.5, so the declared `>= 0.5` rule predicts class 1.
    3. Identical trees make identical mistakes, so averaging adds no diversity.
    4. Per-split sampling prevents one strong feature from controlling every tree path.
    5. Only trees whose bootstrap omitted the row may contribute its OOB prediction.

    Score the eight self-check answers at two points each and the challenge at four
    points. Full credit requires both the mechanism and its limitation.

    ### Common mistakes

    - Saying a bootstrap uses only 63.2% as its number of draws; it uses $n$ draws but fewer unique rows.
    - Sampling rows without replacement and calling it bootstrap.
    - Choosing one feature subset for the whole forest instead of at each split.
    - Averaging labels without declaring how ties are handled.
    - Letting a tree make an OOB prediction for a row it trained on.
    - Calling OOB “cross-validation” or using it for grouped/time-dependent data unchanged.
    - Repeatedly tuning against the final test.
    - Claiming more trees guarantee monotonically improving validation results.
    - Treating forest probabilities as automatically calibrated.
    - Using feature importance before learning its biases and alternatives.

    ### Readiness threshold

    Score at least **16/20**, including a correct manual bootstrap, vote calculation,
    random-feature explanation, OOB boundary, and sealed-test workflow.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    You are ready for CML-05 when you can explain this chain without notes:

    bootstrap rows  
    → train many trees  
    → notice correlated errors  
    → randomize candidate features at every split  
    → average probabilities  
    → evaluate with valid OOB/validation boundaries.

    ### Teach it back

    Explain why a forest can be more stable than one tree while every forest member
    is itself an unstable tree. Then explain why averaging cannot repair systematic
    errors shared by all trees.

    ### Memory aid

    **A random forest bootstraps rows, randomizes split candidates, and averages
    diverse trees so their unstable errors cancel.**

    ### Next dependency

    Random-forest aggregation  
    → required before gradient boosting  
    → because CML-05 contrasts independent averaging with sequential error correction.
    """),
]


build("02_classical_ml/04_random_forest.ipynb", cells)
