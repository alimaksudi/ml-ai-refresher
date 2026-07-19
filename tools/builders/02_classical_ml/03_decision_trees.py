"""Build CML-03: Decision Trees."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # CML-03 · Decision Trees

    **Prerequisites:** FND-03 and CML-02  
    **Estimated study time:** 8–10 hours, including practice  
    **Next lesson:** CML-04 · Random Forest

    Logistic regression uses one linear score. A decision tree learns a sequence of
    if/else questions. This can express thresholds and feature interactions without
    manually creating curved features.

    The goal is not to call `DecisionTreeClassifier().fit(...)`. The goal is to know
    exactly why one question was chosen, how training rows move into child nodes, and
    why an unrestricted tree memorizes noise.

    ### Scope boundary

    This lesson teaches binary classification trees with Gini impurity. It defers:

    - formal classification metrics to MLE-01;
    - cross-validation and systematic hyperparameter selection to MLE-02;
    - entropy comparisons and regression trees to an extension;
    - random forests and bagging to CML-04;
    - boosting to CML-05;
    - feature-importance and SHAP claims to MLE-05.

    We count wrong decisions directly instead of using unexplained metric names.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end, you will be able to:

    - explain when a threshold rule may outperform one linear boundary;
    - identify a root, internal node, branch, leaf, and depth;
    - calculate a leaf class probability from training counts;
    - calculate Gini impurity manually;
    - calculate weighted child impurity;
    - calculate impurity decrease for one candidate split;
    - enumerate valid numerical thresholds;
    - explain why tree construction is greedy;
    - build and traverse a small tree from scratch;
    - distinguish training fit from validation evidence;
    - show how unrestricted depth can memorize training rows;
    - explain `max_depth`, `min_samples_split`, and `min_samples_leaf`;
    - explain pre-pruning and post-pruning;
    - compare the scratch tree with sklearn;
    - explain axis-aligned boundaries and tree instability;
    - complete a sealed-test mini-project.

    ### Learning path

    ```mermaid
    flowchart LR
        A[Count labels in parent] --> B[Try one feature threshold]
        B --> C[Count labels in children]
        C --> D[Calculate weighted impurity]
        D --> E[Keep largest impurity decrease]
        E --> F[Repeat inside each child]
        F --> G[Stop and create leaves]
    ```

    A tree does not search every possible complete tree. It repeatedly chooses the
    best split available at the current node.
    """),

    md(r"""
    ## 2 · The practical problem: rules for late-delivery warnings

    Consider six training routes:

    | Distance (km) | Late label |
    | ---: | ---: |
    | 1 | 0 |
    | 2 | 0 |
    | 3 | 1 |
    | 4 | 0 |
    | 5 | 1 |
    | 6 | 1 |

    A simple rule might be:

    ```text
    if distance <= 2.5:
        predict not late
    else:
        predict late
    ```

    This rule gets one of the four right-side training rows wrong. A deeper tree could
    ask another question inside the right side.

    Decision trees are useful when the relationship can be described by nested rules:

    - distance matters differently when rain is present;
    - a measurement becomes risky only above a threshold;
    - one feature matters only after another condition is met.

    A tree can express those interactions naturally. The cost is instability and easy
    overfitting.
    """),

    code(r"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    route_distance_km = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    late_label = np.array([0, 0, 1, 0, 1, 1], dtype=int)

    rule_decisions = (route_distance_km > 2.5).astype(int)
    wrong_rule_rows = rule_decisions != late_label

    rule_table = pd.DataFrame(
        {
            "distance_km": route_distance_km,
            "actual_label": late_label,
            "rule_decision": rule_decisions,
            "wrong": wrong_rule_rows,
        }
    )

    print(rule_table)
    print("wrong training rows:", int(wrong_rule_rows.sum()))

    assert int(wrong_rule_rows.sum()) == 1
    """),

    md(r"""
    ## 3 · Tree anatomy and leaf predictions

    | Term | Meaning |
    | --- | --- |
    | Root | First node containing all training rows |
    | Internal node | Node that asks a feature question |
    | Split | Feature and threshold used by the question |
    | Branch | Path taken after the answer |
    | Child | Node reached from a parent branch |
    | Leaf | Final node with no further split |
    | Depth | Number of split steps from root to a node |

    For a numerical feature, a binary split is commonly:

    $$
    x_j\le t
    $$

    **Symbols:** $x_j$ is feature $j$ for one row; $t$ is the threshold. Rows that
    satisfy the condition go left; the remaining rows go right.

    A classification leaf stores training label counts. If a leaf contains labels
    $[0,1,1,1]$:

    $$
    \hat p(1)=\frac{3}{4}=0.75
    $$

    Its probability estimate is 0.75 for class 1. A majority decision is class 1.
    The probability is based only on training rows that reached the leaf; it is not a
    guarantee for every future row.
    """),

    code(r"""
    leaf_labels = np.array([0, 1, 1, 1])
    leaf_positive_probability = leaf_labels.mean()
    leaf_majority_decision = int(leaf_positive_probability >= 0.5)

    print("leaf labels:", leaf_labels)
    print("class-1 probability:", leaf_positive_probability)
    print("majority decision:", leaf_majority_decision)

    assert np.isclose(leaf_positive_probability, 0.75)
    assert leaf_majority_decision == 1
    """),

    md(r"""
    ## 4 · Gini impurity measures label mixture

    Before choosing a split, we need a number that describes how mixed a node is.
    For class proportions $p_0$ and $p_1$:

    $$
    G=1-p_0^2-p_1^2
    $$

    **Symbols:** $G$ is Gini impurity; $p_0$ and $p_1$ are the proportions of class 0
    and class 1 in the node.

    Important anchors:

    - labels $[0,0,0,0]$: $p_0=1,p_1=0$, so $G=0$;
    - labels $[0,0,1,1]$: $p_0=p_1=0.5$, so $G=0.5$;
    - labels $[0,1,1,1]$: $p_0=0.25,p_1=0.75$, so
      $G=1-0.25^2-0.75^2=0.375$.

    Zero means pure: every training label in the node is the same. For binary labels,
    0.5 is the most mixed value.

    Gini is not the fraction of wrong predictions. It is a split-selection measure
    based on class proportions.
    """),

    code(r"""
    def gini_impurity(labels):
        '''Calculate Gini impurity from a non-empty one-dimensional label array.'''
        label_array = np.asarray(labels, dtype=int)
        if label_array.ndim != 1 or label_array.size == 0:
            raise ValueError("labels must be a non-empty one-dimensional array")

        _, counts = np.unique(label_array, return_counts=True)
        proportions = counts / label_array.size
        return float(1 - np.sum(proportions**2))


    pure_gini = gini_impurity([0, 0, 0, 0])
    mixed_gini = gini_impurity([0, 0, 1, 1])
    three_to_one_gini = gini_impurity([0, 1, 1, 1])

    print("pure Gini:", pure_gini)
    print("balanced Gini:", mixed_gini)
    print("three-to-one Gini:", three_to_one_gini)

    assert np.isclose(pure_gini, 0.0)
    assert np.isclose(mixed_gini, 0.5)
    assert np.isclose(three_to_one_gini, 0.375)
    """),

    md(r"""
    ## 5 · Calculate one candidate split completely

    The parent labels are $[0,0,1,0,1,1]$. Each class appears three times:

    $$
    G_{parent}=1-(3/6)^2-(3/6)^2=0.5
    $$

    Try threshold $t=2.5$:

    - left labels: $[0,0]$, so $G_L=0$;
    - right labels: $[1,0,1,1]$, so $G_R=0.375$.

    Child impurity must be weighted by child size:

    $$
    G_{children}
    =\frac{n_L}{n}G_L+\frac{n_R}{n}G_R
    $$

    $$
    G_{children}
    =\frac{2}{6}(0)+\frac{4}{6}(0.375)=0.25
    $$

    Impurity decrease is:

    $$
    \Delta G=G_{parent}-G_{children}=0.5-0.25=0.25
    $$

    **Symbols:** $n_L$ and $n_R$ are child row counts; $n$ is parent count;
    $G_L$ and $G_R$ are child impurities; $\Delta G$ is the improvement.

    Larger positive impurity decrease is preferred. A split with an empty child is
    invalid because it did not divide the rows.
    """),

    code(r"""
    parent_gini = gini_impurity(late_label)
    left_mask = route_distance_km <= 2.5
    left_labels = late_label[left_mask]
    right_labels = late_label[~left_mask]

    left_gini = gini_impurity(left_labels)
    right_gini = gini_impurity(right_labels)
    weighted_child_gini = (
        len(left_labels) / len(late_label) * left_gini
        + len(right_labels) / len(late_label) * right_gini
    )
    impurity_decrease = parent_gini - weighted_child_gini

    print("parent Gini:", parent_gini)
    print("left labels and Gini:", left_labels, left_gini)
    print("right labels and Gini:", right_labels, right_gini)
    print("weighted child Gini:", weighted_child_gini)
    print("impurity decrease:", impurity_decrease)

    assert np.isclose(parent_gini, 0.5)
    assert np.isclose(weighted_child_gini, 0.25)
    assert np.isclose(impurity_decrease, 0.25)
    """),

    md(r"""
    ## 6 · Candidate thresholds come from neighbouring values

    For sorted unique numerical values, useful candidates are midpoints between
    neighbours. For values $[1,2,3,4,5,6]$:

    $$
    [1.5,2.5,3.5,4.5,5.5]
    $$

    A threshold below the minimum or at/above the maximum sends every row to one side.
    It cannot create two children.

    The search procedure is:

    1. calculate parent impurity;
    2. try every feature;
    3. sort that feature's unique values;
    4. try neighbouring midpoints;
    5. calculate weighted child impurity;
    6. keep the largest impurity decrease.

    This choice is **greedy**: it chooses the best immediate split and does not revisit
    it after deeper branches are built.
    """),

    code(r"""
    def candidate_thresholds(feature_values):
        '''Return midpoints between sorted unique numerical values.'''
        unique_values = np.unique(np.asarray(feature_values, dtype=float))
        if unique_values.size < 2:
            return np.array([], dtype=float)
        return (unique_values[:-1] + unique_values[1:]) / 2


    def evaluate_split(feature_values, labels, threshold):
        '''Return impurity decrease and masks for one numerical threshold.'''
        feature_array = np.asarray(feature_values, dtype=float)
        label_array = np.asarray(labels, dtype=int)
        left = feature_array <= threshold
        right = ~left
        if not left.any() or not right.any():
            raise ValueError("a valid split must create two non-empty children")

        parent = gini_impurity(label_array)
        weighted_children = (
            left.mean() * gini_impurity(label_array[left])
            + right.mean() * gini_impurity(label_array[right])
        )
        return parent - weighted_children, left, right


    threshold_records = []
    for threshold in candidate_thresholds(route_distance_km):
        gain, left, right = evaluate_split(route_distance_km, late_label, threshold)
        threshold_records.append(
            {
                "threshold": threshold,
                "left_rows": int(left.sum()),
                "right_rows": int(right.sum()),
                "impurity_decrease": gain,
            }
        )

    threshold_table = pd.DataFrame(threshold_records)
    best_threshold_row = threshold_table.loc[threshold_table["impurity_decrease"].idxmax()]

    print(threshold_table)
    print("\nbest threshold:", best_threshold_row["threshold"])

    assert np.isclose(best_threshold_row["threshold"], 2.5)
    assert np.isclose(best_threshold_row["impurity_decrease"], 0.25)
    """),

    md(r"""
    ## 7 · Search several features before recursing

    A real node may have several features. We use a matrix $X$ with rows as examples
    and columns as features. The best split stores:

    - feature index and name;
    - threshold;
    - impurity decrease;
    - left and right row masks.

    The split search is computationally more expensive than one linear score because
    it evaluates many feature-threshold pairs.
    """),

    code(r"""
    def find_best_split(feature_matrix, labels, min_samples_leaf=1):
        '''Find the feature and threshold with largest Gini decrease.'''
        matrix = np.asarray(feature_matrix, dtype=float)
        label_array = np.asarray(labels, dtype=int)
        if matrix.ndim != 2 or label_array.ndim != 1:
            raise ValueError("feature_matrix must be 2D and labels must be 1D")
        if len(matrix) != len(label_array):
            raise ValueError("features and labels must have matching row counts")

        best = None
        for feature_index in range(matrix.shape[1]):
            for threshold in candidate_thresholds(matrix[:, feature_index]):
                gain, left, right = evaluate_split(
                    matrix[:, feature_index], label_array, threshold
                )
                if left.sum() < min_samples_leaf or right.sum() < min_samples_leaf:
                    continue
                if best is None or gain > best["impurity_decrease"]:
                    best = {
                        "feature_index": feature_index,
                        "threshold": float(threshold),
                        "impurity_decrease": float(gain),
                        "left_mask": left,
                        "right_mask": right,
                    }
        return best


    rain_indicator = np.array([0, 0, 1, 0, 0, 1], dtype=float)
    route_features = np.column_stack([route_distance_km, rain_indicator])
    feature_names = ["distance_km", "rain_indicator"]

    best_root_split = find_best_split(route_features, late_label)

    print("best feature:", feature_names[best_root_split["feature_index"]])
    print("best threshold:", best_root_split["threshold"])
    print("impurity decrease:", best_root_split["impurity_decrease"])

    assert best_root_split["feature_index"] == 0
    assert np.isclose(best_root_split["threshold"], 2.5)
    """),

    md(r"""
    ## 8 · Build a shallow tree recursively

    Recursion means the same procedure operates inside each child:

    1. decide whether the node must stop;
    2. otherwise find its best split;
    3. build a left child from left rows;
    4. build a right child from right rows.

    A node becomes a leaf when it is pure, reaches maximum depth, has too few rows to
    split safely, or has no positive impurity decrease.

    Leaf decisions use the majority class. Leaf probabilities use the positive-label
    proportion. If a leaf is tied at 0.5, this learning implementation chooses class
    0, matching scikit-learn's lower-class tie rule for labels 0 and 1.
    """),

    code(r"""
    def make_leaf(labels, depth):
        '''Create a leaf containing training counts and a majority decision.'''
        label_array = np.asarray(labels, dtype=int)
        positive_probability = float(label_array.mean())
        return {
            "leaf": True,
            "depth": depth,
            "rows": len(label_array),
            "positive_probability": positive_probability,
            "decision": int(positive_probability > 0.5),
        }


    def build_classification_tree(
        feature_matrix,
        labels,
        depth=0,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        '''Build a small binary Gini tree for learning purposes.'''
        matrix = np.asarray(feature_matrix, dtype=float)
        label_array = np.asarray(labels, dtype=int)

        should_stop = (
            depth >= max_depth
            or len(label_array) < min_samples_split
            or np.unique(label_array).size == 1
        )
        if should_stop:
            return make_leaf(label_array, depth)

        split = find_best_split(matrix, label_array, min_samples_leaf=min_samples_leaf)
        if split is None or split["impurity_decrease"] <= 0:
            return make_leaf(label_array, depth)

        left = split["left_mask"]
        right = split["right_mask"]
        return {
            "leaf": False,
            "depth": depth,
            "rows": len(label_array),
            "feature_index": split["feature_index"],
            "threshold": split["threshold"],
            "impurity_decrease": split["impurity_decrease"],
            "left": build_classification_tree(
                matrix[left], label_array[left], depth + 1,
                max_depth, min_samples_split, min_samples_leaf,
            ),
            "right": build_classification_tree(
                matrix[right], label_array[right], depth + 1,
                max_depth, min_samples_split, min_samples_leaf,
            ),
        }


    def predict_tree_row(node, feature_row):
        '''Traverse one row until reaching a leaf.'''
        current = node
        while not current["leaf"]:
            if feature_row[current["feature_index"]] <= current["threshold"]:
                current = current["left"]
            else:
                current = current["right"]
        return current["decision"], current["positive_probability"]


    def predict_tree(node, feature_matrix):
        results = [predict_tree_row(node, row) for row in np.asarray(feature_matrix)]
        decisions = np.array([result[0] for result in results], dtype=int)
        probabilities = np.array([result[1] for result in results], dtype=float)
        return decisions, probabilities


    scratch_tree = build_classification_tree(route_features, late_label, max_depth=2)
    scratch_decisions, scratch_probabilities = predict_tree(scratch_tree, route_features)

    print("tree:", scratch_tree)
    print("training decisions:", scratch_decisions)
    print("training probabilities:", scratch_probabilities)
    print("wrong training rows:", int(np.sum(scratch_decisions != late_label)))

    assert scratch_tree["leaf"] is False
    # The depth limit leaves one tied leaf, so one training row remains wrong.
    assert int(np.sum(scratch_decisions != late_label)) == 1
    """),

    md(r"""
    ## 9 · Overfitting appears before pruning controls

    A tree can keep adding questions until individual training rows are isolated.
    Training errors may reach zero even when the rules capture noise that will not
    repeat.

    Compare:

    - a shallow tree: fewer rules, may miss real structure;
    - a deep tree: more flexible, may memorize training details.

    We judge this with validation rows that did not choose splits. We count wrong
    decisions directly. Formal metric selection comes later.

    The example below uses a noisy two-feature pattern. Depth is the only changed
    control.
    """),

    code(r"""
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    noisy_features, noisy_labels = make_moons(n_samples=240, noise=0.28, random_state=12)
    X_train, X_validation, y_train, y_validation = train_test_split(
        noisy_features,
        noisy_labels,
        test_size=0.35,
        random_state=42,
        stratify=noisy_labels,
    )

    depth_records = []
    for depth in [1, 2, 3, 5, None]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        training_wrong = int(np.sum(model.predict(X_train) != y_train))
        validation_wrong = int(np.sum(model.predict(X_validation) != y_validation))
        depth_records.append(
            {
                "max_depth": "unlimited" if depth is None else depth,
                "tree_depth": model.get_depth(),
                "leaf_count": model.get_n_leaves(),
                "training_wrong": training_wrong,
                "validation_wrong": validation_wrong,
            }
        )

    depth_table = pd.DataFrame(depth_records)
    print(depth_table)

    assert depth_table.iloc[-1]["training_wrong"] <= depth_table.iloc[0]["training_wrong"]
    assert depth_table.iloc[-1]["leaf_count"] >= depth_table.iloc[0]["leaf_count"]
    """),

    md(r"""
    The unlimited tree normally has fewer training errors and many more leaves. Its
    validation result need not improve. That gap is the reason stopping and pruning
    come before ensembles.
    """),

    md(r"""
    ## 10 · Pre-pruning controls growth

    **Pre-pruning** stops growth while the tree is being built:

    | Control | Question it asks |
    | --- | --- |
    | `max_depth` | How many split levels may exist? |
    | `min_samples_split` | Does this node contain enough rows to attempt a split? |
    | `min_samples_leaf` | Will each new leaf contain enough rows? |
    | `min_impurity_decrease` | Is the immediate improvement large enough? |

    Larger leaves pool more observations into each probability estimate. They are less
    tailored to individual rows but may miss small real groups.

    These controls are development choices. They must not be repeatedly adjusted from
    final-test results.
    """),

    code(r"""
    prepruned_models = {
        "depth_2": DecisionTreeClassifier(max_depth=2, random_state=42),
        "leaf_12": DecisionTreeClassifier(min_samples_leaf=12, random_state=42),
        "unrestricted": DecisionTreeClassifier(random_state=42),
    }

    prepruning_records = []
    for name, model in prepruned_models.items():
        model.fit(X_train, y_train)
        prepruning_records.append(
            {
                "model": name,
                "depth": model.get_depth(),
                "leaves": model.get_n_leaves(),
                "training_wrong": int(np.sum(model.predict(X_train) != y_train)),
                "validation_wrong": int(np.sum(model.predict(X_validation) != y_validation)),
            }
        )

    prepruning_table = pd.DataFrame(prepruning_records)
    print(prepruning_table)

    assert prepruning_table["leaves"].min() < prepruning_table["leaves"].max()
    """),

    md(r"""
    ## 11 · Post-pruning removes weak branches

    **Post-pruning** first grows a larger tree, then removes branches whose training
    improvement is too small relative to their complexity.

    Scikit-learn uses cost-complexity parameter $\alpha$ through `ccp_alpha`:

    $$
    R_\alpha(T)=R(T)+\alpha|T_{leaves}|
    $$

    **Symbols:** $T$ is a tree; $R(T)$ is its training leaf error or impurity cost;
    $|T_{leaves}|$ is its number of leaves; $\alpha$ controls the complexity penalty.

    - $\alpha=0$ applies no pruning penalty;
    - larger $\alpha$ prefers fewer leaves.

    This formula describes the trade-off. Systematic selection of $\alpha$ with
    cross-validation belongs to MLE-02. Here we compare two declared values on the
    development split and keep the final test out.
    """),

    code(r"""
    pruning_records = []
    for alpha in [0.0, 0.01, 0.03]:
        pruned_model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        pruned_model.fit(X_train, y_train)
        pruning_records.append(
            {
                "ccp_alpha": alpha,
                "depth": pruned_model.get_depth(),
                "leaves": pruned_model.get_n_leaves(),
                "training_wrong": int(np.sum(pruned_model.predict(X_train) != y_train)),
                "validation_wrong": int(np.sum(pruned_model.predict(X_validation) != y_validation)),
            }
        )

    pruning_table = pd.DataFrame(pruning_records)
    print(pruning_table)

    assert pruning_table.iloc[-1]["leaves"] <= pruning_table.iloc[0]["leaves"]
    """),

    md(r"""
    ## 12 · Use sklearn and understand the learned rules

    The library performs the same core operations as our scratch tree:

    1. search feature-threshold candidates;
    2. select the largest impurity decrease;
    3. repeat in child nodes;
    4. stop according to controls;
    5. store class counts and probabilities in leaves.

    `export_text` displays the fitted questions. Read the rules before trusting the
    predictions.

    Two implementations do not have to produce the identical tree. If candidate
    splits have equal impurity decrease, their tie-breaking rules may choose different
    questions. The trees can still have the same depth and number of wrong training
    rows. Matching the root calculation and understanding any later difference is
    more important than forcing every printed rule to match.
    """),

    code(r"""
    from sklearn.tree import export_text

    sklearn_route_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    sklearn_route_tree.fit(route_features, late_label)
    sklearn_route_decisions = sklearn_route_tree.predict(route_features)
    sklearn_route_probabilities = sklearn_route_tree.predict_proba(route_features)[:, 1]

    print(export_text(sklearn_route_tree, feature_names=feature_names))
    print("sklearn decisions:", sklearn_route_decisions)
    print("scratch decisions:", scratch_decisions)
    print("sklearn class-1 probabilities:", sklearn_route_probabilities)
    print("scratch class-1 probabilities:", scratch_probabilities)

    assert sklearn_route_tree.tree_.feature[0] == scratch_tree["feature_index"] == 0
    assert np.isclose(sklearn_route_tree.tree_.threshold[0], scratch_tree["threshold"])
    assert int(np.sum(sklearn_route_decisions != late_label)) == 1
    assert int(np.sum(scratch_decisions != late_label)) == 1
    assert np.all((sklearn_route_probabilities >= 0) & (sklearn_route_probabilities <= 1))
    """),

    md(r"""
    ### What a tree can and cannot express

    **Strengths:**

    - nonlinear threshold rules;
    - feature interactions;
    - no scaling requirement for ordinary numerical splits;
    - inspectable shallow rules;
    - fast prediction after fitting.

    **Limitations:**

    - numerical splits create axis-aligned rectangular regions;
    - stepwise predictions do not extrapolate smoothly;
    - deep trees are hard to inspect;
    - small data changes can change early splits and the whole tree;
    - leaf probabilities can be extreme when leaves are tiny;
    - greedy local choices may miss a better complete tree.

    A single tree's instability motivates CML-04. Random forests average many varied
    trees instead of trusting one fragile structure.
    """),

    md(r"""
    ## 13 · Mini-project: Wine cultivar tree with a sealed test

    **Goal:** classify whether a Wine sample belongs to cultivar class 0 using a
    shallow tree. This is an educational recognition task, not a quality or safety
    decision.

    **Workflow:**

    1. declare the positive class and two features;
    2. create training, validation, and sealed test partitions;
    3. freeze a training-majority decision baseline;
    4. fit one pre-declared depth-3 tree;
    5. count validation errors;
    6. inspect rules and leaf sizes;
    7. preserve the final test without prediction.
    """),

    code(r"""
    from sklearn.datasets import load_wine

    wine_dataset = load_wine(as_frame=True)
    wine_frame = wine_dataset.frame.copy()
    wine_frame.insert(
        0,
        "sample_id",
        [f"wine_{row_number:03d}" for row_number in range(len(wine_frame))],
    )
    wine_frame["is_class_zero"] = (wine_frame["target"] == 0).astype(int)

    project_feature_names = ["alcohol", "color_intensity"]
    project_X = wine_frame[project_feature_names]
    project_y = wine_frame["is_class_zero"]
    project_ids = wine_frame["sample_id"]

    X_development, X_test, y_development, y_test, id_development, id_test = train_test_split(
        project_X,
        project_y,
        project_ids,
        test_size=0.20,
        random_state=42,
        stratify=project_y,
    )
    X_train_project, X_validation_project, y_train_project, y_validation_project, id_train, id_validation = train_test_split(
        X_development,
        y_development,
        id_development,
        test_size=0.25,
        random_state=42,
        stratify=y_development,
    )

    print("training rows:", len(X_train_project))
    print("validation rows:", len(X_validation_project))
    print("sealed test rows:", len(X_test))

    assert len(X_train_project) == 106
    assert len(X_validation_project) == 36
    assert len(X_test) == 36
    assert set(id_train).isdisjoint(id_validation)
    assert set(id_train).isdisjoint(id_test)
    assert set(id_validation).isdisjoint(id_test)
    """),

    code(r"""
    training_majority_decision = int(y_train_project.mean() >= 0.5)
    baseline_validation_decisions = np.full(
        len(y_validation_project),
        training_majority_decision,
    )
    baseline_validation_wrong = int(
        np.sum(baseline_validation_decisions != y_validation_project.to_numpy())
    )

    project_tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )
    project_tree.fit(X_train_project, y_train_project)
    project_validation_decisions = project_tree.predict(X_validation_project)
    project_validation_wrong = int(
        np.sum(project_validation_decisions != y_validation_project.to_numpy())
    )

    leaf_training_counts = np.bincount(
        project_tree.apply(X_train_project),
        minlength=project_tree.tree_.node_count,
    )
    used_leaf_counts = leaf_training_counts[leaf_training_counts > 0]

    print("baseline validation wrong:", baseline_validation_wrong)
    print("tree validation wrong:", project_validation_wrong)
    print("tree depth:", project_tree.get_depth())
    print("leaf count:", project_tree.get_n_leaves())
    print("training rows per used leaf:", used_leaf_counts)
    print("\nlearned rules:\n", export_text(project_tree, feature_names=project_feature_names))

    assert project_validation_wrong < baseline_validation_wrong
    assert project_tree.get_depth() <= 3
    assert used_leaf_counts.min() >= 5
    """),

    code(r"""
    project_partition_manifest = pd.concat(
        [
            pd.DataFrame({"sample_id": id_train, "partition": "train"}),
            pd.DataFrame({"sample_id": id_validation, "partition": "validation"}),
            pd.DataFrame({"sample_id": id_test, "partition": "test"}),
        ],
        ignore_index=True,
    )

    project_result = {
        "positive_class": "original cultivar target equals 0",
        "features": project_feature_names,
        "baseline_validation_wrong": baseline_validation_wrong,
        "tree_validation_wrong": project_validation_wrong,
        "tree_depth": project_tree.get_depth(),
        "leaf_count": project_tree.get_n_leaves(),
        "partition_manifest": project_partition_manifest,
        "test_status": "sealed — no tree decision, probability, or error count calculated",
    }

    print("partition counts:\n", project_partition_manifest["partition"].value_counts())
    print("test status:", project_result["test_status"])

    assert project_partition_manifest["sample_id"].is_unique
    assert len(project_partition_manifest) == len(wine_frame)
    assert project_result["test_status"].startswith("sealed")
    """),

    md(r"""
    ## 14 · Practice, solutions, and mastery checkpoint

    ### Worked example

    A parent contains labels $[0,0,1,1]$, so its Gini is 0.5. A split creates pure
    children $[0,0]$ and $[1,1]$:

    $$
    G_{children}=\frac{2}{4}(0)+\frac{2}{4}(0)=0
    $$

    $$
    \Delta G=0.5-0=0.5
    $$

    This is the largest possible Gini decrease for a balanced binary parent.

    ### Guided practice

    1. Identify the root, branches, and leaves in a three-question rule.
    2. Calculate leaf probability and decision for labels $[0,0,1]$.
    3. Calculate Gini for counts 8 class-0 and 2 class-1.
    4. Calculate weighted child impurity for child sizes 3 and 7.
    5. Compare thresholds 2.5 and 3.5 for the six-route dataset manually.
    6. List candidate thresholds for values $[1,2,4,8]$.

    ### Independent practice

    7. Rebuild `gini_impurity` with input checks.
    8. Rebuild one-feature split search and verify against the threshold table.
    9. Add a second feature and print the chosen feature name and threshold.
    10. Trace two rows through the scratch tree by hand.
    11. Compare depth 1, depth 3, and unlimited trees using training and validation
        wrong-row counts.
    12. Compare pre-pruning and post-pruning tree sizes.

    ### Challenge

    Rebuild the Wine mini-project without copying. Include:

    - a declared positive class and prediction time;
    - a frozen training-majority baseline;
    - disjoint training, validation, and sealed test partitions;
    - one fully manual candidate split;
    - a pre-declared shallow sklearn tree;
    - validation wrong-row comparison;
    - printed rules, depth, leaf count, and leaf training sizes;
    - a partition manifest and at least eight assertions;
    - no test result, random forest, boosting, cross-validation, or feature-importance claim.

    ### Self-check

    For every split, write:

    - parent label counts and Gini;
    - feature name and threshold;
    - left and right row counts;
    - left and right Gini;
    - weighted child Gini;
    - impurity decrease;
    - stopping rule that will eventually create a leaf.
    """),

    md(r"""
    ### Solution and scoring rubric

    1. The root asks the first question; each answer forms a branch; final predictions
       live in leaves.
    2. Positive probability is $1/3$ and majority decision is class 0.
    3. $G=1-0.8^2-0.2^2=0.32$.
    4. Multiply each child impurity by 3/10 or 7/10, then add.
    5. Threshold 2.5 has decrease 0.25; threshold 3.5 has about 0.0556, so 2.5 wins.
    6. Midpoints are $[1.5,3,6]$.
    7. Pure nodes return 0; balanced binary nodes return 0.5.
    8. The six-route best threshold is 2.5.
    9. Search every feature-threshold pair and retain the largest positive decrease.
    10. Apply each condition in order until a leaf is reached.
    11. Training errors usually fall with depth; validation need not improve.
    12. Stronger controls should reduce depth or leaf count.

    Challenge scoring:

    | Skill | Points |
    | --- | ---: |
    | Tree anatomy and leaf probability | 2 |
    | Manual Gini and weighted split | 4 |
    | Correct candidate search | 3 |
    | Recursion and traversal explanation | 2 |
    | Overfitting, stopping, and pruning | 3 |
    | Split-safe baseline and validation comparison | 3 |
    | Rules, leaf evidence, assertions, sealed test | 3 |
    | **Total** | **20** |

    ### Common mistakes

    - Introducing impurity before counting labels.
    - Forgetting to weight child impurity by child size.
    - Choosing the smallest impurity decrease instead of the largest.
    - Allowing an empty child or a leaf smaller than the declared minimum.
    - Treating Gini as the fraction of wrong decisions.
    - Assuming a greedy split produces the globally best complete tree.
    - Judging depth from training fit alone.
    - Tuning controls repeatedly on the final test.
    - Calling every unusual row noise and growing around it.
    - Treating a deep rule path as automatically interpretable.
    - Claiming a split proves causation.
    - Using feature importance before learning its biases and alternatives.

    ### Readiness threshold

    Score at least **16/20**, including a correct manual split, stopping/pruning
    explanation, validation comparison, and sealed final test.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    1. What does a leaf store for binary classification?
    2. What does Gini impurity measure?
    3. Why are child impurities weighted by row count?
    4. How are numerical candidate thresholds created?
    5. What makes tree construction greedy?
    6. Why can training errors reach zero while validation worsens?
    7. How do `max_depth` and `min_samples_leaf` differ?
    8. What is the difference between pre-pruning and post-pruning?
    9. Why are tree boundaries axis-aligned and stepwise?
    10. Why can a small change in training rows alter the whole tree?

    ### Teach it back

    Starting with six route labels, explain:

    **parent counts → parent Gini → candidate threshold → child counts → child Gini →
    weighted impurity → impurity decrease → greedy choice → recursion → stopping →
    leaf probability.**

    Then explain why validation evidence and pruning must appear before random forests.

    ### Memory aid

    **A tree greedily asks the question that most reduces label mixture, then stops
    before its rules become a memory of individual training rows.**

    ### Next dependency

    CML-04 addresses the instability of one tree by training many varied trees and
    averaging their outputs.
    """),
]


build("02_classical_ml/03_decision_trees.ipynb", cells)
