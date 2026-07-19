"""Builder for Lesson FND-03 — Data Workflow, EDA, Cleaning, Pandas, and SQL."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # FND-03 · Data Workflow, EDA, Cleaning, Pandas, and SQL

    *Build trustworthy training data before choosing a model*

    | Lesson detail | Value |
    | --- | --- |
    | Prerequisites | PRE-05 and FND-02; no model-training experience required |
    | Estimated study time | 7–9 hours across two or more sessions |
    | Main outcome | Frame a prediction task and prepare leakage-safe development data |
    | Next lesson | CML-01 · Linear Regression |

    A model sees only the table we give it. If a row means the wrong thing, a future
    value slips into a feature, or preprocessing learns from held-out data, even
    elegant model code can produce misleading evidence.

    This lesson deliberately stops before model fitting. Classification algorithms
    and evaluation metrics have their own later lessons. Here, our job is to build
    data that those lessons can use honestly.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end of this lesson, you will be able to:

    - define the decision, prediction unit, target, prediction time, and features;
    - distinguish a feature, target, identifier, and metadata field;
    - ask whether each feature is genuinely available at prediction time;
    - describe a simple constant baseline without confusing it with a trained model;
    - load and document a real dataset with a beginner-friendly data dictionary;
    - inspect shape, types, ranges, missingness, duplicate keys, and target balance;
    - separate invalid data from unusual but potentially valuable observations;
    - use distributions, grouped summaries, IQR, and correlation cautiously;
    - explain target leakage and train–holdout contamination with concrete examples;
    - distinguish random, stratified, group-based, and time-based split boundaries;
    - create training, validation, and sealed test partitions;
    - calculate median imputation and standardization using training data only;
    - reproduce those transformations with an sklearn transformer Pipeline;
    - write basic SQL for selection, filtering, grouping, missingness, and duplicates;
    - produce a quality report and split manifest that another student can audit.

    You will not train or score a classifier here. That boundary prevents unexplained
    logistic regression and F1 from appearing before their prerequisites.
    """),

    md(r"""
    ## 2 · The practical problem

    A laboratory has chemical measurements from 178 wine samples. Each sample belongs
    to one of three known cultivars—a cultivar is a cultivated plant variety.

    Later, we may build an educational system that suggests a cultivar label from the
    chemical measurements. Before choosing any algorithm, we need to know:

    - what one row represents;
    - which column is the answer we want to predict;
    - when the prediction would be made;
    - which measurements are available at that moment;
    - whether rows are missing, repeated, or outside expected ranges;
    - which rows may teach preprocessing values;
    - which rows must remain untouched for honest evaluation.

    The calculation is not the first decision. The meaning of the calculation is.

    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #4c78a8; border-radius: 10px; padding: 12px 16px; background: #eef5ff; color: #172b4d; text-align: center;"><strong>Decision</strong><br>what action follows?</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 12px 16px; background: #fff4e8; color: #4a2b0b; text-align: center;"><strong>Prediction moment</strong><br>what is known now?</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 12px 16px; background: #eef8ec; color: #173d17; text-align: center;"><strong>One-row meaning</strong><br>what is one example?</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #b279a2; border-radius: 10px; padding: 12px 16px; background: #f8eff7; color: #40213a; text-align: center;"><strong>Data contract</strong><br>what may enter?</div>
    </div>

    The example is intentionally low stakes. Its output is not a judgment about wine
    quality, safety, price, or legal compliance.
    """),

    md(r"""
    ## 3 · Frame the task before opening the dataset

    ### 3.1 The seven-part problem frame

    Write these fields before exploring correlations or choosing a model:

    | Field | Wine example | Why it matters |
    | --- | --- | --- |
    | Decision | Suggest a cultivar label for expert review | A prediction must support an action |
    | Prediction unit | One laboratory-tested wine sample | Defines what one row must represent |
    | Target | Confirmed cultivar class | Defines the answer to learn later |
    | Prediction time | After chemical measurements, before reading the confirmed cultivar | Sets the information boundary |
    | Features | Chemical measurements available by prediction time | Prevents future information from entering |
    | Identifier | A generated `sample_id` | Tracks rows; should not become a numeric feature |
    | Evaluation unit | One independent sample | Determines which rows may be separated safely |

    **Prediction unit** and **row meaning** should agree. If one sample appears in
    several rows, then rows may not be independent. We would need a group-aware split
    that keeps the whole sample on one side.

    ### 3.2 Feature, target, identifier, and metadata

    - A **feature** is information available when the prediction is requested.
    - The **target** is the answer a future model will try to learn.
    - An **identifier** distinguishes records but usually has no reusable numerical
      relationship with the target.
    - **Metadata** describes context, such as site or collection date. It may help
      auditing without being safe or useful as a feature.

    A column can change roles across tasks. Collection date may be metadata for one
    model and a feature for another. The role comes from the decision and prediction
    time, not from the dtype.

    ### 3.3 Start with a baseline idea

    A baseline is the simplest behaviour a future model must improve upon. For a
    three-class target, one constant baseline always predicts the most common class.
    It learns no chemical relationship.

    We will calculate its class proportion after loading the data. Formal model
    comparison and classification metrics come later.
    """),

    md(r"""
    ## 4 · Load the dataset and establish its contract

    We use scikit-learn's bundled copy of the UCI Wine recognition dataset. It needs
    no network connection and contains real chemical analyses rather than generated
    rows.

    The bundled metadata does not document measurement units. We must record that
    gap instead of inventing units. A trustworthy contract can say “unknown” when the
    source documentation is incomplete.
    """),

    code(r"""
    import sqlite3

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_wine

    dataset = load_wine(as_frame=True)
    wine_data = dataset.frame.rename(
        columns={"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"}
    )
    wine_data.insert(
        0,
        "sample_id",
        [f"wine_{row_number:03d}" for row_number in range(len(wine_data))],
    )

    target_column = "target"
    identifier_column = "sample_id"
    feature_columns = [
        column
        for column in wine_data.columns
        if column not in {identifier_column, target_column}
    ]

    print("shape:", wine_data.shape)
    print("feature count:", len(feature_columns))
    print("target names:", dataset.target_names.tolist())
    print(wine_data.head(3))

    assert wine_data.shape == (178, 15)
    assert len(feature_columns) == 13
    assert wine_data[identifier_column].is_unique
    """),

    md(r"""
    One row is one wine sample. There are 178 rows, one identifier, 13 chemical
    features, and one target.

    The target values `0`, `1`, and `2` are category codes. They do not mean that
    cultivar 2 is twice cultivar 1. This is a nominal target: its labels have names
    but no numerical order.

    ### 4.1 A compact data dictionary

    | Column or group | Role | Meaning | Unit in bundled metadata | Initial rule |
    | --- | --- | --- | --- | --- |
    | `sample_id` | Identifier | Generated label for one sample row | Not applicable | Required and unique |
    | `alcohol` | Feature | Alcohol measurement | Not documented | Numeric; investigate impossible values |
    | `malic_acid` | Feature | Malic acid measurement | Not documented | Numeric; missingness must be reported |
    | `ash`, `alcalinity_of_ash` | Features | Ash-related measurements | Not documented | Numeric |
    | `magnesium` | Feature | Magnesium measurement | Not documented | Numeric |
    | Phenol-related columns | Features | Several phenolic measurements | Not documented | Numeric |
    | `color_intensity`, `hue` | Features | Colour-related measurements | Not documented | Numeric |
    | `od280_od315_of_diluted_wines` | Feature | Diluted-wine optical density ratio | Ratio; source details limited | Numeric |
    | `proline` | Feature | Proline measurement | Not documented | Numeric; inspect wide range |
    | `target` | Target | Confirmed cultivar class | Category | Must be one of `0`, `1`, `2` |

    A production contract would need source-approved units and allowed ranges. The
    observed minimum and maximum are useful references, but they are not automatically
    physical validity limits.
    """),

    code(r"""
    data_dictionary = pd.DataFrame(
        {
            "column": wine_data.columns,
            "role": [
                "identifier" if column == identifier_column
                else "target" if column == target_column
                else "feature"
                for column in wine_data.columns
            ],
            "dtype": [str(wine_data[column].dtype) for column in wine_data.columns],
            "unit_status": [
                "not applicable" if column in {identifier_column, target_column}
                else "not documented in bundled metadata"
                for column in wine_data.columns
            ],
            "allows_missing": [False] * len(wine_data.columns),
        }
    )

    print(data_dictionary.head())
    """),

    md(r"""
    The dictionary separates facts we know from facts we still need. Recording a
    documentation gap is safer than silently treating an observed range as universal.
    """),

    md(r"""
    ## 5 · Inspect data quality before changing values

    Start with facts that do not learn a model transformation:

    - row and column counts;
    - names and storage types;
    - missing counts;
    - duplicate identifiers;
    - unexpected target labels;
    - observed ranges and category counts.

    PRE-05 taught the pandas operations. Here we connect them to a prediction task.
    """),

    code(r"""
    print("shape:", wine_data.shape)
    print("duplicate sample IDs:", int(wine_data[identifier_column].duplicated().sum()))
    print("missing cells:", int(wine_data.isna().sum().sum()))
    print("\nfeature ranges:\n", wine_data[feature_columns].agg(["min", "max"]).T)

    expected_targets = {0, 1, 2}
    observed_targets = set(wine_data[target_column].unique())

    assert observed_targets == expected_targets
    assert wine_data[identifier_column].is_unique
    assert wine_data.isna().sum().sum() == 0
    """),

    md(r"""
    ### 5.1 Inject known problems for safe practice

    The source dataset is clean. To practise detection, we create a copy with two
    known problems:

    - 12 missing `malic_acid` values;
    - one repeated `sample_id`, representing duplicate ingestion.

    Because the causes are known, we can verify our checks. We never pretend these
    injected problems came from the original UCI data.
    """),

    code(r"""
    random_generator = np.random.default_rng(42)
    development_source = wine_data.copy()

    missing_positions = random_generator.choice(
        development_source.index,
        size=12,
        replace=False,
    )
    development_source.loc[missing_positions, "malic_acid"] = np.nan

    repeated_record = development_source.iloc[[0]].copy()
    development_source = pd.concat(
        [development_source, repeated_record],
        ignore_index=True,
    )

    duplicate_id_mask = development_source[identifier_column].duplicated(keep=False)

    print("rows after injection:", len(development_source))
    print("missing malic_acid values:", int(development_source["malic_acid"].isna().sum()))
    print("rows with a repeated sample ID:", int(duplicate_id_mask.sum()))
    print(development_source.loc[duplicate_id_mask, ["sample_id", "malic_acid", "target"]])

    assert len(development_source) == 179
    assert int(development_source["malic_acid"].isna().sum()) == 12
    assert int(duplicate_id_mask.sum()) == 2
    """),

    md(r"""
    We can remove the injected duplicate because its identifier proves that the same
    record was ingested twice. Two different sample IDs with identical measurements
    would require investigation; identical feature values alone do not prove an error.
    """),

    code(r"""
    deduplicated_data = (
        development_source
        .drop_duplicates(subset=[identifier_column], keep="first")
        .reset_index(drop=True)
    )

    print("rows after justified deduplication:", len(deduplicated_data))

    assert len(deduplicated_data) == 178
    assert deduplicated_data[identifier_column].is_unique
    assert int(deduplicated_data["malic_acid"].isna().sum()) == 12
    """),

    md(r"""
    ### 5.2 Seal the final test before target-aware exploration

    Basic source checks established the schema, allowed target labels, identifiers,
    and known ingestion problems. Before calculating target proportions, correlations,
    or exploratory plots, we reserve a final test partition.

    One row is one independent sample, there is no time or group field, and the target
    is categorical. A **stratified random split** is reasonable here: random assignment
    separates rows, while stratification keeps each target class represented.

    This choice would be wrong for repeated patients, devices, documents, households,
    or time-ordered events. Section 8 compares those boundaries.
    """),

    code(r"""
    from sklearn.model_selection import train_test_split

    development_data, sealed_test_data = train_test_split(
        deduplicated_data,
        test_size=0.20,
        random_state=42,
        stratify=deduplicated_data[target_column],
    )

    development_data = development_data.sort_index().copy()
    sealed_test_data = sealed_test_data.sort_index().copy()

    print("development rows:", len(development_data))
    print("sealed test rows:", len(sealed_test_data))

    assert set(development_data[identifier_column]).isdisjoint(
        sealed_test_data[identifier_column]
    )
    assert len(development_data) + len(sealed_test_data) == len(deduplicated_data)
    """),

    md(r"""
    From this point onward, we do not calculate test-set summaries, target proportions,
    correlations, preprocessing values, or SQL exploration results. We preserve only
    its identifiers for the partition manifest and its row count for the boundary check.

    The development rows may now teach us about the target and guide development.
    Their most common class provides a constant baseline:
    """),

    code(r"""
    development_target_counts = development_data[target_column].value_counts()
    majority_class = int(development_target_counts.idxmax())
    majority_fraction = float(
        development_target_counts.max() / development_target_counts.sum()
    )

    print("development target counts:\n", development_target_counts.sort_index())
    print("constant prediction:", majority_class)
    print("majority-class fraction:", round(majority_fraction, 3))

    assert 0 < majority_fraction < 1
    assert majority_class in {0, 1, 2}
    """),

    md(r"""
    The baseline ignores every chemical feature. It is a sanity benchmark, not a
    complete evaluation, and it was calculated without opening the sealed test.

    ### 5.3 Build a reusable development-data quality report

    A report should preserve counts and examples, not only a green or red status.
    Missing-value rates use the same boolean-mask idea introduced in PRE-05:

    $$
    r_j = \frac{1}{n}\sum_{i=1}^{n} m_{ij}
    $$

    Symbols:

    - $r_j$ is the missing rate for column $j$;
    - $n$ is the total number of rows;
    - $m_{ij}$ equals 1 when row $i$, column $j$ is missing and 0 otherwise.

    For 3 missing values among 20 development rows:

    $$
    \frac{3}{20} = 0.15 = 15\%
    $$
    """),

    code(r"""
    def build_quality_report(dataframe, identifier, target, features):
        '''Return auditable table-quality facts without repairing the data.'''
        return {
            "rows": int(len(dataframe)),
            "columns": int(dataframe.shape[1]),
            "duplicate_identifier_rows": int(dataframe[identifier].duplicated(keep=False).sum()),
            "missing_count_by_column": dataframe.isna().sum().astype(int).to_dict(),
            "missing_rate_by_column": dataframe.isna().mean().to_dict(),
            "target_count": dataframe[target].value_counts(dropna=False).sort_index().to_dict(),
            "feature_minimum": dataframe[features].min().to_dict(),
            "feature_maximum": dataframe[features].max().to_dict(),
        }


    quality_report = build_quality_report(
        development_data,
        identifier_column,
        target_column,
        feature_columns,
    )

    print("rows:", quality_report["rows"])
    print("malic_acid missing count:", quality_report["missing_count_by_column"]["malic_acid"])
    print("malic_acid missing rate:", round(quality_report["missing_rate_by_column"]["malic_acid"], 4))

    development_missing_count = int(development_data["malic_acid"].isna().sum())

    assert quality_report["rows"] == len(development_data)
    assert quality_report["missing_count_by_column"]["malic_acid"] == development_missing_count
    assert np.isclose(
        quality_report["missing_rate_by_column"]["malic_acid"],
        development_missing_count / len(development_data),
    )
    """),

    md(r"""
    ## 6 · Explore with questions, not decoration

    **Exploratory data analysis**, usually shortened to EDA, describes what is in the
    available data and generates questions. It does not prove that a future model will
    work, and it does not turn an association into a cause.

    Every calculation in this section uses `development_data`. The final test was
    sealed first because its patterns must not guide feature or workflow choices.

    Use this order:

    1. State the question.
    2. Choose the rows and columns that answer it.
    3. Calculate a summary or draw a labelled plot.
    4. Describe what is visible.
    5. State what the result cannot prove.

    ### 6.1 Question: how is alcohol distributed across target classes?
    """),

    code(r"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    development_data["target"].value_counts().sort_index().plot.bar(
        ax=axes[0],
        color="tab:blue",
    )
    axes[0].set_title("Cultivar class counts — development rows only")
    axes[0].set_xlabel("cultivar class code")
    axes[0].set_ylabel("sample rows")

    for class_code, class_rows in development_data.groupby("target"):
        axes[1].hist(
            class_rows["alcohol"],
            bins=10,
            alpha=0.5,
            label=f"class {class_code}",
        )
    axes[1].set_title("Alcohol distribution by known class")
    axes[1].set_xlabel("alcohol measurement — source unit not documented")
    axes[1].set_ylabel("sample rows")
    axes[1].legend()

    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    The class counts differ, and alcohol distributions overlap. This plot suggests
    alcohol may contain some class information, but it cannot prove causation or show
    how well an unseen sample can be classified.

    The plot uses development rows only. The sealed test remains unopened. Later,
    preprocessing values will be learned from the smaller training partition only.

    ### 6.2 Unusual does not automatically mean invalid

    The interquartile range, or IQR, describes the width of the middle half of sorted
    values.

    $$
    \operatorname{IQR} = Q_{0.75} - Q_{0.25}
    $$

    - $Q_{0.25}$ is the first quartile: 25% of values are at or below it.
    - $Q_{0.75}$ is the third quartile: 75% of values are at or below it.

    A common diagnostic boundary is:

    $$
    \text{lower fence} = Q_{0.25} - 1.5\times\operatorname{IQR}
    $$

    $$
    \text{upper fence} = Q_{0.75} + 1.5\times\operatorname{IQR}
    $$

    A value outside a fence is flagged for investigation. It is not automatically a
    measurement error.
    """),

    code(r"""
    small_values = pd.Series([2, 3, 4, 5, 6, 7, 20], name="small_example")
    first_quartile = small_values.quantile(0.25)
    third_quartile = small_values.quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_fence = first_quartile - 1.5 * interquartile_range
    upper_fence = third_quartile + 1.5 * interquartile_range

    print("Q1:", first_quartile)
    print("Q3:", third_quartile)
    print("IQR:", interquartile_range)
    print("fences:", lower_fence, "to", upper_fence)
    print("flagged values:", small_values[(small_values < lower_fence) | (small_values > upper_fence)].tolist())
    """),

    md(r"""
    In this tiny example, 20 is far from the middle values and is flagged. We still
    need source knowledge to decide whether 20 is a typo, a different unit, or a rare
    valid case.
    """),

    code(r"""
    proline = development_data["proline"]
    proline_q1 = proline.quantile(0.25)
    proline_q3 = proline.quantile(0.75)
    proline_iqr = proline_q3 - proline_q1
    proline_lower = proline_q1 - 1.5 * proline_iqr
    proline_upper = proline_q3 + 1.5 * proline_iqr
    proline_flag = (proline < proline_lower) | (proline > proline_upper)

    print("proline fences:", proline_lower, "to", proline_upper)
    print("flagged proline rows:", int(proline_flag.sum()))
    print(development_data.loc[proline_flag, ["sample_id", "proline", "target"]])
    """),

    md(r"""
    A zero flagged count does not prove every value is correct. IQR detects one kind
    of unusualness; it does not understand units, instruments, or laboratory rules.

    ### 6.3 Correlation is a descriptive linear relationship

    FND-02 introduced covariance and correlation. Here we use correlation to ask
    which numerical features move together in this dataset.

    $$
    \rho_{XY} = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
    $$

    - $\rho_{XY}$ is the correlation between variables $X$ and $Y$;
    - covariance describes whether they vary together;
    - $\sigma_X$ and $\sigma_Y$ are their standard deviations.

    Correlation ranges from −1 to 1 when both variables vary. Near zero means little
    linear relationship; it does not rule out a curved relationship.
    """),

    code(r"""
    correlation_features = [
        "alcohol",
        "malic_acid",
        "total_phenols",
        "flavanoids",
        "color_intensity",
        "proline",
    ]
    correlation_matrix = development_data[correlation_features].corr()

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation_features)), correlation_features, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation_features)), correlation_features)
    axis.set_title("Feature correlation — development rows only")

    for row_index in range(len(correlation_features)):
        for column_index in range(len(correlation_features)):
            axis.text(
                column_index,
                row_index,
                f"{correlation_matrix.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    The labelled matrix is slower to build than an unlabelled heatmap, but it lets a
    beginner connect each colour to an actual feature pair. Correlated features are
    not automatically duplicates, and correlation never proves that one causes the
    other.
    """),

    md(r"""
    ## 7 · Leakage: information crossing the prediction boundary

    **Leakage** happens when training or evaluation uses information that would not
    be legitimately available for a new prediction. Leakage can produce impressive
    results for the wrong reason.

    Think of an examination. Studying past questions is allowed. Reading the answer
    key for the examination you will be scored on is not. The analogy has a limit:
    real leakage can happen indirectly through timestamps, aggregates, repeated
    entities, and preprocessing statistics.

    ### 7.1 Target leakage

    Suppose someone adds `confirmed_cultivar_after_review`. It becomes available only
    after the expert confirms the target. It is not a safe feature at prediction time.
    """),

    code(r"""
    leakage_demo = development_data[["sample_id", "target"]].copy()
    leakage_demo["confirmed_cultivar_after_review"] = leakage_demo["target"]
    leakage_matches = leakage_demo["confirmed_cultivar_after_review"].eq(leakage_demo["target"])

    print(leakage_demo.head())
    print("rows where leaked feature equals target:", int(leakage_matches.sum()))

    assert leakage_matches.all()
    """),

    md(r"""
    The column is perfectly related to the target because it is a copy of the answer.
    Its availability time—not its correlation—makes it invalid.

    ### 7.2 Common leakage paths

    | Leakage path | Example | Safer rule |
    | --- | --- | --- |
    | Future feature | Outcome confirmed after prediction time | Use only point-in-time available values |
    | Target-derived aggregate | Average target calculated using the current row | Build aggregates inside training folds only |
    | Repeated entity | Same customer appears in training and test | Keep an entity on one side |
    | Preprocessing contamination | Median calculated using all rows | Fit median on training rows only |
    | Test-set tuning | Repeatedly change choices after test results | Keep final test sealed |

    EDA and leakage control are related but different:

    - EDA asks what the data contains.
    - Validation asks how well a fixed workflow works on unseen examples.
    - Leakage control protects the boundary between learning and evaluation.
    """),

    md(r"""
    ## 8 · Complete the development split before learning transformations

    We use three partitions:

    | Partition | Purpose | May teach preprocessing values? |
    | --- | --- | --- |
    | Training | Learn medians, means, scales, and later model parameters | Yes |
    | Validation | Compare development choices | No |
    | Final test | One final estimate after choices are frozen | No |

    <div style="display: flex; align-items: stretch; justify-content: center; gap: 12px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 14px 18px; background: #eef8ec; color: #173d17; text-align: center;"><strong>Training rows</strong><br>learn transformations<br>and future model</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 14px 18px; background: #fff4e8; color: #4a2b0b; text-align: center;"><strong>Validation rows</strong><br>develop choices<br>without fitting</div>
      <div style="border: 2px solid #e15759; border-radius: 10px; padding: 14px 18px; background: #fff0f0; color: #4a1717; text-align: center;"><strong>Final test rows</strong><br>sealed until<br>the workflow is frozen</div>
    </div>

    ### 8.1 Which splitting method?

    | Split | Use when | Main danger it addresses |
    | --- | --- | --- |
    | Random | Rows are genuinely independent and similarly distributed | Ordinary sampling variation |
    | Stratified | A categorical target's proportions should remain similar | A small class disappearing from a partition |
    | Group-based | Several rows belong to one person, device, household, or document | Entity information leaking across partitions |
    | Time-based | The system predicts later events from earlier information | Future patterns leaking backward |

    Section 5 already sealed the final test using a stratified random boundary. We now
    split the remaining development rows into training and validation. That does not
    make random splitting correct for every dataset.
    """),

    code(r"""
    X_development = development_data[feature_columns]
    y_development = development_data[target_column]
    id_development = development_data[identifier_column]

    X_test = sealed_test_data[feature_columns]
    y_test = sealed_test_data[target_column]
    id_test = sealed_test_data[identifier_column]

    # Use 25% of the remaining 80% for validation: about 20% of the full dataset.
    X_train, X_validation, y_train, y_validation, id_train, id_validation = train_test_split(
        X_development,
        y_development,
        id_development,
        test_size=0.25,
        random_state=42,
        stratify=y_development,
    )

    print("training rows:", len(X_train))
    print("validation rows:", len(X_validation))
    print("sealed test rows:", len(X_test))
    print("training target proportions:\n", y_train.value_counts(normalize=True).sort_index())
    print("validation target proportions:\n", y_validation.value_counts(normalize=True).sort_index())

    all_partition_ids = set(id_train) | set(id_validation) | set(id_test)

    assert set(id_train).isdisjoint(id_validation)
    assert set(id_train).isdisjoint(id_test)
    assert set(id_validation).isdisjoint(id_test)
    assert all_partition_ids == set(deduplicated_data[identifier_column])
    assert len(X_train) + len(X_validation) + len(X_test) == len(deduplicated_data)
    """),

    md(r"""
    The split produces 106 training rows, 36 validation rows, and 36 sealed test rows.
    `random_state=42` makes the assignment reproducible. `stratify` keeps class
    proportions reasonably similar; it does not solve time or entity leakage.

    We printed no test statistics beyond its row count. The test partition now stays
    untouched for the rest of this lesson.
    """),

    md(r"""
    ## 9 · Learn preprocessing from training rows only

    A preprocessing step is **learned** when it calculates and stores information from
    data. Median imputation learns a median. Standardization learns a mean and standard
    deviation. Those values belong to the training partition.

    - `fit` learns and stores values from training data.
    - `transform` applies already learned values without changing them.
    - `fit_transform` fits and then transforms the same training data.

    ### 9.1 Median imputation by hand

    Imputation fills a missing input using an explicit rule. Median imputation uses
    the middle training value. It is less affected by extreme values than the mean,
    but it still invents a replacement and can hide meaningful missingness.
    """),

    code(r"""
    training_malic_acid = X_train["malic_acid"].dropna().sort_values()
    training_median = float(training_malic_acid.median())
    full_data_median = float(deduplicated_data["malic_acid"].median())

    validation_malic_before = X_validation["malic_acid"].copy()
    validation_malic_after = validation_malic_before.fillna(training_median)

    print("training non-missing count:", len(training_malic_acid))
    print("training median:", training_median)
    print("full-data median, shown only for comparison:", full_data_median)
    print("validation missing before:", int(validation_malic_before.isna().sum()))
    print("validation missing after:", int(validation_malic_after.isna().sum()))

    assert validation_malic_after.isna().sum() == 0
    """),

    md(r"""
    Even if the training and full-data medians happen to be close, only the training
    median is legitimate. Leakage is defined by the information boundary, not by how
    much the leaked number changed.

    ### 9.2 Standardization by hand

    Features can use very different numerical scales. Standardization expresses a
    value as its distance from the training mean in training standard-deviation units.

    $$
    z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
    $$

    Symbols:

    - $x$ is one raw feature value;
    - $\mu_{\text{train}}$ is that feature's training mean;
    - $\sigma_{\text{train}}$ is that feature's training standard deviation;
    - $z$ is the standardised, unitless result.

    A result of 0 is at the training mean. A result of 1 is one training standard
    deviation above the mean. If the standard deviation is zero, every training value
    is constant and division is undefined.
    """),

    code(r"""
    training_alcohol_mean = float(X_train["alcohol"].mean())
    training_alcohol_std = float(X_train["alcohol"].std(ddof=0))
    first_validation_alcohol = float(X_validation["alcohol"].iloc[0])

    first_validation_standardized = (
        first_validation_alcohol - training_alcohol_mean
    ) / training_alcohol_std

    print("training alcohol mean:", round(training_alcohol_mean, 4))
    print("training alcohol population-style standard deviation:", round(training_alcohol_std, 4))
    print("first validation alcohol value:", first_validation_alcohol)
    print("standardized with training values:", round(first_validation_standardized, 4))

    assert training_alcohol_std > 0
    """),

    md(r"""
    Standardization does not make bad data valid, remove skew, or guarantee a better
    future model. Some algorithms are sensitive to scale; tree-based algorithms often
    are not. We will make algorithm-specific choices in later lessons.
    """),

    md(r"""
    ## 10 · Package the same transformations in a Pipeline

    A Pipeline applies steps in a fixed order and stores their learned state together.
    Here it contains transformers only—no classifier—because model algorithms have
    not been taught yet.

    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #4c78a8; border-radius: 10px; padding: 12px 16px; background: #eef5ff; color: #172b4d; text-align: center;"><strong>Training table</strong><br>fit starts here</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 12px 16px; background: #fff4e8; color: #4a2b0b; text-align: center;"><strong>Median imputer</strong><br>learn training medians</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 12px 16px; background: #eef8ec; color: #173d17; text-align: center;"><strong>Standard scaler</strong><br>learn training mean and scale</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #b279a2; border-radius: 10px; padding: 12px 16px; background: #f8eff7; color: #40213a; text-align: center;"><strong>Frozen transformer</strong><br>transform validation later</div>
    </div>
    """),

    code(r"""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    preprocessing_pipeline = Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
        ]
    )

    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    X_validation_transformed = preprocessing_pipeline.transform(X_validation)

    malic_acid_position = feature_columns.index("malic_acid")
    learned_malic_acid_median = preprocessing_pipeline.named_steps[
        "median_imputer"
    ].statistics_[malic_acid_position]

    print("transformed training shape:", X_train_transformed.shape)
    print("transformed validation shape:", X_validation_transformed.shape)
    print("manual training median:", training_median)
    print("pipeline training median:", learned_malic_acid_median)
    print("remaining validation missing cells:", int(np.isnan(X_validation_transformed).sum()))

    assert X_train_transformed.shape == X_train.shape
    assert X_validation_transformed.shape == X_validation.shape
    assert np.isclose(learned_malic_acid_median, training_median)
    assert np.isnan(X_validation_transformed).sum() == 0
    assert np.allclose(X_train_transformed.mean(axis=0), 0, atol=1e-12)
    """),

    md(r"""
    The pipeline learned values during `fit_transform(X_train)`. Calling
    `transform(X_validation)` reused those frozen values. It did not recalculate a
    validation median or validation mean.

    The output is a NumPy array with 13 columns in the original feature order. A
    maintained workflow should store and validate those input names and their order.
    Categorical encoding and mixed-type `ColumnTransformer` workflows belong to the
    later feature-engineering lesson.
    """),

    md(r"""
    ## 11 · Express the same quality questions in SQL

    Pandas works with in-memory DataFrames. SQL asks a database to select and combine
    rows. The ideas are the same: one-row meaning, column meaning, missingness, keys,
    and grouping still matter.

    Basic SQL clauses are read in this logical order:

    | Clause | Question |
    | --- | --- |
    | `FROM` | Which table supplies the rows? |
    | `WHERE` | Which rows are allowed? |
    | `GROUP BY` | Which rows belong to one output group? |
    | `SELECT` | Which columns or summaries should be returned? |
    | `HAVING` | Which completed groups are allowed? |
    | `ORDER BY` | How should the result be arranged? |

    SQL is written with `SELECT` first even though understanding begins with `FROM`.
    We use an in-memory SQLite database, so no external server is required. The SQL
    table contains development rows only; the sealed test remains outside the query
    path.
    """),

    code(r"""
    connection = sqlite3.connect(":memory:")

    # Recreate one duplicate inside development data so SQL can detect it without
    # opening or querying the sealed test partition.
    sql_development_source = pd.concat(
        [development_data, development_data.iloc[[0]].copy()],
        ignore_index=True,
    )
    sql_development_source.to_sql(
        "wine_samples",
        connection,
        index=False,
        if_exists="replace",
    )

    selected_rows = pd.read_sql_query(
        '''
        SELECT sample_id, alcohol, malic_acid, target
        FROM wine_samples
        WHERE target = 1
          AND alcohol >= 13
        ORDER BY alcohol DESC
        LIMIT 5
        ''',
        connection,
    )

    print(selected_rows)
    """),

    md(r"""
    This query returns at most five class-1 rows whose alcohol value is at least 13,
    sorted from highest to lowest alcohol value. `WHERE` filters individual rows
    before grouping.

    ### 11.1 `COUNT(*)` and `COUNT(column)` answer different questions

    - `COUNT(*)` counts rows.
    - `COUNT(malic_acid)` counts rows where `malic_acid` is not NULL.
    - `AVG(malic_acid)` ignores NULL values in SQLite.

    SQL uses `IS NULL`, not `= NULL`, because NULL represents missing or unknown.
    """),

    code(r"""
    sql_class_summary = pd.read_sql_query(
        '''
        SELECT target,
               COUNT(*) AS row_count,
               COUNT(malic_acid) AS measured_malic_acid_count,
               AVG(malic_acid) AS mean_observed_malic_acid,
               SUM(CASE WHEN malic_acid IS NULL THEN 1 ELSE 0 END) AS missing_malic_acid_count
        FROM wine_samples
        GROUP BY target
        ORDER BY target
        ''',
        connection,
    )

    print(sql_class_summary)

    assert int(sql_class_summary["row_count"].sum()) == len(sql_development_source)
    assert int(sql_class_summary["missing_malic_acid_count"].sum()) == int(
        sql_development_source["malic_acid"].isna().sum()
    )
    """),

    md(r"""
    One output row now represents one cultivar class. The difference between row count
    and measured count exposes missing values rather than hiding them inside an average.

    ### 11.2 Find duplicate keys with `GROUP BY` and `HAVING`
    """),

    code(r"""
    duplicate_ids_from_sql = pd.read_sql_query(
        '''
        SELECT sample_id, COUNT(*) AS occurrence_count
        FROM wine_samples
        GROUP BY sample_id
        HAVING COUNT(*) > 1
        ORDER BY sample_id
        ''',
        connection,
    )

    print(duplicate_ids_from_sql)

    duplicated_sql_id = sql_development_source.iloc[0]["sample_id"]
    assert duplicate_ids_from_sql.to_dict("records") == [
        {"sample_id": duplicated_sql_id, "occurrence_count": 2}
    ]

    connection.close()
    """),

    md(r"""
    `HAVING` filters completed groups, while `WHERE` filters source rows. The SQL query
    and pandas `duplicated` mask reveal the same contract violation through different
    tools.

    | Task | pandas | SQL |
    | --- | --- | --- |
    | Select columns | `data[["a", "b"]]` | `SELECT a, b` |
    | Filter rows | `data.loc[mask]` | `WHERE condition` |
    | Count rows | `len(data)` or `groupby.size()` | `COUNT(*)` |
    | Count non-missing | `Series.count()` | `COUNT(column)` |
    | Group summary | `groupby().agg()` | `GROUP BY` with aggregate functions |
    | Duplicate key | `duplicated()` | `GROUP BY key HAVING COUNT(*) > 1` |
    """),

    md(r"""
    ## 12 · Decisions, limitations, and common failure modes

    ### When this workflow helps

    Use it when preparing tabular development data, reviewing a new data source,
    reproducing a training dataset, or investigating why a later model changed.

    ### When it is not enough

    - Images, audio, text, graphs, and event streams need structure-specific checks.
    - Causal decisions need experimental or causal reasoning beyond correlation.
    - A clean schema does not prove that labels are correct or the population is fair.
    - A historical split does not guarantee future conditions will remain unchanged.

    ### Common mistakes

    - Opening with an algorithm before defining the decision and prediction time.
    - Treating an identifier as an ordinary numerical feature.
    - Calling observed minimum and maximum values physical validity rules.
    - Removing every IQR-flagged row without source investigation.
    - Deduplicating on all feature values when repeated observations may be legitimate.
    - Letting the same person, machine, or document cross a random split.
    - Using a future or target-derived column because it has strong correlation.
    - Calculating an imputation value or scale before splitting.
    - Calling `fit_transform` separately on validation or test data.
    - Repeatedly checking the final test while changing the workflow.
    - Using SQL `= NULL` or confusing `COUNT(*)` with `COUNT(column)`.
    - Believing a successful notebook run proves the dataset is trustworthy.

    Data-quality evidence and model-quality evidence are different. Schema, range,
    missingness, and volume checks can run without labels. Model performance requires
    predictions and appropriate held-out labels, which later lessons will teach.
    """),

    md(r"""
    ## 13 · Mini-project: create a development-data audit bundle

    The goal is not to train a model. The goal is to hand the next lesson an auditable
    bundle containing:

    - the written problem frame;
    - a data dictionary;
    - a quality report;
    - a partition manifest;
    - a fitted training-only preprocessing pipeline;
    - validation transformations created without refitting;
    - SQL evidence for missing values and duplicate keys.
    """),

    code(r"""
    problem_frame = {
        "decision": "suggest a cultivar label for expert review",
        "prediction_unit": "one laboratory-tested wine sample",
        "target": "confirmed cultivar class",
        "prediction_time": "after chemical measurements and before cultivar confirmation",
        "identifier": identifier_column,
        "evaluation_unit": "one independent wine sample",
    }

    partition_manifest = pd.concat(
        [
            pd.DataFrame({"sample_id": id_train, "partition": "train"}),
            pd.DataFrame({"sample_id": id_validation, "partition": "validation"}),
            pd.DataFrame({"sample_id": id_test, "partition": "test"}),
        ],
        ignore_index=True,
    )

    audit_bundle = {
        "problem_frame": problem_frame,
        "data_dictionary": data_dictionary,
        "quality_report": quality_report,
        "partition_manifest": partition_manifest,
        "preprocessing_pipeline": preprocessing_pipeline,
        "transformed_training_shape": X_train_transformed.shape,
        "transformed_validation_shape": X_validation_transformed.shape,
    }

    print("problem frame:", audit_bundle["problem_frame"])
    print("\npartition counts:\n", partition_manifest["partition"].value_counts())
    print("\nquality-report rows:", quality_report["rows"])
    print("pipeline steps:", list(preprocessing_pipeline.named_steps))

    assert partition_manifest["sample_id"].is_unique
    assert len(partition_manifest) == len(deduplicated_data)
    assert set(partition_manifest["partition"]) == {"train", "validation", "test"}
    assert audit_bundle["quality_report"]["rows"] == len(development_data)
    assert audit_bundle["transformed_training_shape"][1] == len(feature_columns)

    print("\nDevelopment-data audit bundle passed its checks.")
    """),

    md(r"""
    ### Why this bundle matters

    Another student can now answer:

    - what prediction task the data supports;
    - what one row and every column mean;
    - which source problems were detected;
    - exactly which samples belong to each partition;
    - which partition taught preprocessing values;
    - whether validation was transformed without refitting.

    In production, the same ideas grow into versioned schemas, dataset hashes,
    lineage, access controls, partition snapshots, transformation artifacts, and
    monitoring. Those operational systems come later; the reasoning starts here.
    """),

    md(r"""
    ## 14 · Exercises, self-check, and solutions

    **Estimated practice time:** 2–3 hours.

    ### Worked example

    Suppose a customer table has three rows for the same customer. A random row split
    places two rows in training and one in validation. The partitions are different,
    but the customer is not new to the workflow. A group-based split must keep all
    three rows together.

    ### Guided practice

    1. Write the seven-part problem frame for predicting whether a delivery will be
       late. State the exact prediction time.
    2. Extend the Wine data dictionary with a `source_owner` and `description` column.
       Use “unknown” where the bundled source is insufficient.
    3. Calculate the `malic_acid` missing rate in `development_data` manually and with
       pandas. Explain why a rate alone does not justify imputation.
    4. Calculate IQR fences for `color_intensity`. List flagged rows without deleting
       them.
    5. Explain why `confirmed_cultivar_after_review` is unsafe even if it produces
       excellent future model performance.

    ### Independent practice

    6. Create a six-row table containing `machine_id`, `event_time`, two measurements,
       and `failure_next_day`. Design a time-based split and explain why random rows
       would be misleading.
    7. Inject missing values into `magnesium`. Split first, calculate the training
       median manually, and prove a Pipeline learns the same value.
    8. Write SQL that reports row count, non-missing `magnesium` count, and mean
       observed `magnesium` by target. Explain every selected output column.
    9. Add a second duplicate `sample_id`. Detect it in both pandas and SQL, then
       explain what evidence would be needed before removing it.

    ### Challenge

    Build an audit bundle for a different small tabular dataset. It must include:

    - a decision, prediction unit, target, prediction time, identifier, and evaluation
      unit;
    - a data dictionary that distinguishes known facts from documentation gaps;
    - quality checks for missingness, key duplication, target labels, and ranges;
    - a justified random, stratified, group-based, or time-based split that seals the
      final test before target-aware exploration;
    - one labelled development-only distribution plot and one cautious correlation
      analysis;
    - one manually calculated training-only transformation;
    - the equivalent sklearn transformer Pipeline;
    - two SQL validation queries;
    - a partition manifest and at least five assertions;
    - no model fitting and no test-set performance inspection.

    ### Self-check before reading solutions

    For every decision, answer:

    - What would be known at prediction time?
    - What does one row represent?
    - Which rows taught this number?
    - Which evidence was preserved?
    - Could the same entity or future information exist on both sides?
    """),

    md(r"""
    ### Solution and scoring rubric

    1. A valid delivery frame names the decision, one-delivery unit, late/on-time
       target, cutoff time, available features, delivery ID, and independent unit.
    2. Do not invent source owners or measurement meanings. Documentation gaps are
       legitimate contract values.
    3. Divide `development_missing_count` by `len(development_data)` and verify it
       matches `development_data["malic_acid"].isna().mean()`. Imputation also needs
       a missingness reason, downstream purpose, and training-only rule.
    4. Use the first and third quartiles, calculate IQR, then create a boolean mask.
       A flag requests investigation; it is not a deletion instruction.
    5. It becomes available after prediction time and directly contains the answer.
    6. Sort by event time and place later events after earlier events. If machines
       repeat, consider both time and machine boundaries.
    7. Fit only on training rows and compare `SimpleImputer.statistics_` with the
       manual training median using `np.isclose`.
    8. Use `COUNT(*)`, `COUNT(magnesium)`, and `AVG(magnesium)` with `GROUP BY target`.
    9. `duplicated(keep=False)` and `HAVING COUNT(*) > 1` identify repeated keys.
       Source lineage determines whether deletion is justified.

    Challenge scoring:

    | Skill | Points |
    | --- | ---: |
    | Complete and coherent problem frame | 3 |
    | Honest data dictionary and quality report | 3 |
    | Question-led EDA with cautious interpretation | 3 |
    | Split method justified by the evaluation unit | 3 |
    | Manual and Pipeline transformations match | 3 |
    | SQL checks and partition manifest | 3 |
    | Assertions, leakage explanation, and sealed test | 2 |

    Maximum: 20 points.

    **Common mistakes:** defining the target without a prediction time, using an ID as
    a feature, deleting every outlier, splitting repeated entities randomly, fitting
    transformations before splitting, fitting validation separately, inspecting test
    results during development, inventing missing source documentation, and using
    model code to hide an unclear data question.

    **Readiness threshold:** 16/20, including a valid prediction-time boundary, a
    justified split, matching manual and Pipeline preprocessing, and a still-sealed
    final test partition.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    You are ready for CML-01 when you can, without copying this notebook:

    - define the decision, prediction unit, target, prediction time, features,
      identifier, and evaluation unit;
    - explain why feature availability matters more than correlation for leakage;
    - create and interpret a data dictionary and quality report;
    - distinguish missing, invalid, duplicate, and unusual observations;
    - explain what an IQR flag and a correlation value can and cannot establish;
    - choose among random, stratified, group-based, and time-based boundaries;
    - explain the roles of training, validation, and final test partitions;
    - calculate a training median and training standardization by hand;
    - fit a transformer Pipeline on training data and transform validation without
      refitting;
    - translate selection, missingness, grouping, and duplicate checks into SQL;
    - complete the mini-project and score at least 16/20.

    ### Teach it back

    Explain why this sequence is safe:

    **frame the decision → document the table → inspect quality → explore cautiously
    → choose the split boundary → learn transformations from training only → freeze
    them → transform validation → leave final test sealed.**

    Then give one example of a notebook that runs perfectly but violates that sequence.

    ### Memory aid

    **Define what is knowable, split before learning, and preserve the evidence behind
    every data decision.**

    CML-01 will introduce the first prediction algorithm and squared loss. FND-04
    will then explain optimization, while later modules introduce classification and
    evaluation metrics only after their prerequisites exist.
    """),
]


build("01_ml_foundations/03_data_workflow_eda_and_cleaning.ipynb", cells)
