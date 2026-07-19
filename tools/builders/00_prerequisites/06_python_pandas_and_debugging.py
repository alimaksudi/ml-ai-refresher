"""Builder for Lesson PRE-05 — Practical Python, Pandas, Debugging, and Tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # PRE-05 · Practical Python, Pandas, Debugging, and Tests

    *Turn a messy table into evidence you can explain and trust*

    | Lesson detail | Value |
    | --- | --- |
    | Prerequisite | PRE-03; no previous pandas experience required |
    | Estimated study time | 7–9 hours across two or more sessions |
    | Main outcome | Load, inspect, clean, combine, test, and explain a small tabular dataset |
    | Next lesson | FND-03 · Data Workflow, EDA, and Cleaning |

    Running a pandas command is easy. Knowing whether it changed the correct rows,
    preserved the evidence, and produced a trustworthy table is the real skill.

    We will build that skill slowly. One small measurement dataset will stay with us
    from the first CSV file to the final checked report.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end of this lesson, you will be able to:

    - explain rows, columns, labels, indexes, Series, DataFrames, and schemas;
    - create a DataFrame from ordinary Python records;
    - write and load a small CSV using a clear file path;
    - inspect a table before changing it;
    - select rows and columns with names, positions, and boolean masks;
    - explain missing values and choose whether to keep, fill, reject, or investigate;
    - convert text to numeric, date, and categorical data without hiding failures;
    - create columns with vectorized operations and safe assignment;
    - group rows and explain the difference between `size` and `count`;
    - join tables while checking key relationships and unmatched rows;
    - turn data expectations into functions, exceptions, assertions, and tests;
    - read common pandas errors and reduce a failure to a tiny example;
    - make a labelled diagnostic plot only after validating the data; and
    - complete a small cleaning pipeline that returns both data and a quality report.

    This lesson does not teach statistical inference or model preprocessing. Those
    come later. Here we make sure the table itself means what we think it means.
    """),

    md(r"""
    ## 2 · The problem we are trying to solve

    Imagine a laboratory receives measurement files from several sites. One file
    contains values in grams, another in milligrams, and someone typed `bad-date`
    into a date column. A metadata table tells us where each entity came from.

    We need to answer a simple question:

    > What is the average valid measurement for each experimental group?

    The calculation is not the difficult part. The difficult questions come first:

    - Does one row represent one sample, one entity, or one day?
    - Are sample identifiers unique?
    - Which values are missing rather than truly zero?
    - Can text values safely become numbers and dates?
    - Are grams and milligrams being compared as though they were the same unit?
    - Will the metadata join preserve the number of measurement rows?
    - What should happen to invalid rows, and can we still inspect them afterward?

    A spreadsheet can handle a tiny file, but manual edits are difficult to repeat
    and easy to forget. Plain Python can process records, but table operations become
    verbose. Pandas gives names to table operations. Our checks give those operations
    meaning.
    """),

    md(r"""
    ## 3 · A table is a collection of observations

    Before touching pandas, state the **unit of observation**: what one row means.

    In our measurement table:

    > One row represents one recorded sample measurement.

    That sentence controls everything. If one sample appears twice unexpectedly,
    counts and averages can be wrong even when the code runs perfectly.

    | Word | Simple meaning |
    | --- | --- |
    | Row | One observation, such as one sample measurement |
    | Column | One property measured for every row, such as `unit` |
    | Cell | One value at the meeting point of a row and column |
    | Index | Pandas' label for locating rows; it is not automatically a business ID |
    | Series | One labelled column of values |
    | DataFrame | Several labelled Series aligned into a table |
    | Schema | The table's contract: column names, meanings, types, and rules |

    Think of a DataFrame as a labelled tray. Rows are items on the tray, columns are
    the questions asked about every item, and the schema is the tray's packing list.
    The analogy has a limit: pandas will sometimes accept a mixed or invalid column,
    so the tray does not inspect itself.

    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #4c78a8; border-radius: 10px; padding: 12px 16px; background: #eef5ff; color: #172b4d; text-align: center;"><strong>Raw records</strong><br>what arrived</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 12px 16px; background: #fff4e8; color: #4a2b0b; text-align: center;"><strong>Inspect</strong><br>shape, types, rules</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #e15759; border-radius: 10px; padding: 12px 16px; background: #fff0f0; color: #4a1717; text-align: center;"><strong>Validate</strong><br>make problems visible</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 12px 16px; background: #eef8ec; color: #173d17; text-align: center;"><strong>Transform</strong><br>documented decisions</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #b279a2; border-radius: 10px; padding: 12px 16px; background: #f8eff7; color: #40213a; text-align: center;"><strong>Check output</strong><br>data + report</div>
    </div>

    **Memory rule:** never clean a table before you can describe what one row means.
    """),

    md(r"""
    ## 4 · From Python records to a CSV and DataFrame

    ### 4.1 Build a tiny raw dataset

    PRE-03 introduced a list of dictionaries. Each dictionary can represent one row:
    the keys become column names and the values become cells.

    Our raw data deliberately contains realistic problems. Do not fix them yet.
    First preserve what arrived.
    """),

    code(r"""
    from pathlib import Path
    import shutil
    import tempfile

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    raw_records = [
        {"sample_id": "s1", "entity_id": "e1", "group": "control",   "measured_at": "2026-01-01", "value": "500",  "unit": "mg"},
        {"sample_id": "s2", "entity_id": "e1", "group": "control",   "measured_at": "2026-01-02", "value": "1.2",  "unit": "g"},
        {"sample_id": "s3", "entity_id": "e2", "group": "treatment", "measured_at": "bad-date",   "value": "800",  "unit": "mg"},
        {"sample_id": "s4", "entity_id": "e2", "group": "treatment", "measured_at": "2026-01-04", "value": None,   "unit": "mg"},
        {"sample_id": "s5", "entity_id": "e3", "group": "treatment", "measured_at": "2026-01-05", "value": "-10",  "unit": "mg"},
        {"sample_id": "s6", "entity_id": "e4", "group": "control",   "measured_at": "2026-01-06", "value": "2",    "unit": "kg"},
        {"sample_id": "s7", "entity_id": "e3", "group": "treatment", "measured_at": "2026-01-07", "value": "0.9",  "unit": "g"},
        {"sample_id": "s8", "entity_id": "e5", "group": "control",   "measured_at": "2026-01-08", "value": "350",  "unit": "mg"},
    ]

    records_frame = pd.DataFrame(raw_records)

    print("Python object:", type(raw_records).__name__)
    print("pandas object:", type(records_frame).__name__)
    print(records_frame)

    assert records_frame.shape == (8, 6)
    """),

    md(r"""
    The DataFrame has shape `(8, 6)`: eight rows and six columns. Shape always uses
    the order `(rows, columns)`.

    The table looks organised, but appearance is not proof. The `value` column still
    contains text, missing data, and a negative value. The date column also contains
    invalid text.

    ### 4.2 Write and load a CSV through a clear path

    A **path** tells Python where a file lives. `Path` joins folders safely and makes
    the destination visible. We use a temporary lesson folder so this notebook does
    not leave generated data in the repository.

    CSV means **comma-separated values**. A CSV stores text, not rich pandas types.
    Loading it is the beginning of interpretation, not proof that its values are valid.
    """),

    code(r"""
    lesson_directory = Path(tempfile.mkdtemp(prefix="pre05_"))
    measurements_path = lesson_directory / "measurements.csv"

    records_frame.to_csv(measurements_path, index=False)
    raw_measurements = pd.read_csv(measurements_path, dtype={"sample_id": "string", "entity_id": "string"})

    print("CSV path:", measurements_path)
    print("file exists:", measurements_path.exists())
    print("loaded shape:", raw_measurements.shape)
    print(raw_measurements.head(3))

    assert measurements_path.exists()
    assert raw_measurements.shape == (8, 6)
    """),

    md(r"""
    `index=False` prevents pandas' row index from becoming an extra CSV column.
    Explicit `dtype` protects identifier columns from being treated as numbers.
    An identifier such as `0012` is a label; turning it into number `12` loses meaning.

    ### 4.3 Inspect before editing

    Use several views because each answers a different question:

    - `head()` shows a few example rows;
    - `shape` reports row and column counts;
    - `columns` reports available names;
    - `dtypes` reports pandas' current storage types;
    - `info()` combines counts and types;
    - `describe()` summarises values, but only after their types are sensible.

    The dtype `object` often means general Python values or text. It does not prove
    that every value has the intended meaning.
    """),

    code(r"""
    print("shape:", raw_measurements.shape)
    print("columns:", raw_measurements.columns.tolist())
    print("\ndtypes:\n", raw_measurements.dtypes)
    print("\ninfo:")
    raw_measurements.info()
    print("\nmissing cells per column:\n", raw_measurements.isna().sum())
    """),

    md(r"""
    Expected evidence:

    - there are eight rows and six columns;
    - `sample_id` and `entity_id` use pandas' string dtype;
    - `value` may load as `float64` because most values look numeric;
    - one value is missing;
    - `measured_at` is still text, so the bad date has not been detected yet.

    pandas infers types from the observed file. Inference is convenient, but the
    schema must come from the problem, not from pandas' best guess.
    """),

    md(r"""
    ## 5 · Selecting the rows and columns you mean

    Selection is where many quiet mistakes begin. Always ask two questions:

    1. Which rows do I want?
    2. Which columns do I want?

    ### 5.1 Selecting one or several columns

    One pair of brackets returns one Series. A list of names inside the brackets
    returns a DataFrame, even if the list contains only one name.
    """),

    code(r"""
    value_series = raw_measurements["value"]
    value_frame = raw_measurements[["value"]]
    identity_frame = raw_measurements[["sample_id", "entity_id", "group"]]

    print("one column type:", type(value_series).__name__, "shape:", value_series.shape)
    print("one-column table type:", type(value_frame).__name__, "shape:", value_frame.shape)
    print(identity_frame.head(3))

    assert value_series.ndim == 1
    assert value_frame.ndim == 2
    """),

    md(r"""
    ### 5.2 Labels with `loc`, positions with `iloc`

    `loc` uses labels and conditions. `iloc` uses integer positions.

    ```python
    dataframe.loc[row_selector, column_selector]
    dataframe.iloc[row_positions, column_positions]
    ```

    Prefer named columns for data rules. Column positions can change silently after
    someone inserts a new column.
    """),

    code(r"""
    first_two_named = raw_measurements.loc[0:1, ["sample_id", "value", "unit"]]
    first_two_by_position = raw_measurements.iloc[0:2, [0, 4, 5]]

    print("loc selection:\n", first_two_named)
    print("\niloc selection:\n", first_two_by_position)

    assert first_two_named.equals(first_two_by_position)
    """),

    md(r"""
    Notice the slice difference:

    - label slice `0:1` includes both labels `0` and `1`;
    - position slice `0:2` includes position `0` but stops before position `2`.

    That difference is a good reason to keep selection explicit.

    ### 5.3 Boolean masks answer yes-or-no for every row

    Suppose row number $i$ either belongs to the treatment group or does not.
    Define a mask value:

    $$
    m_i =
    \begin{cases}
    1, & \text{if row } i \text{ passes the condition} \\
    0, & \text{otherwise}
    \end{cases}
    $$

    Symbols:

    - $i$ identifies one row;
    - $m_i$ is the mask result for that row;
    - $1$ means keep the row;
    - $0$ means do not keep the row.

    In pandas, these appear as `True` and `False`. Adding booleans counts the `True`
    values because Python treats `True` like 1 and `False` like 0 in this calculation.

    $$
    \text{selected count} = \sum_{i=1}^{n} m_i
    $$

    Here, $n$ is the total number of rows and the sigma symbol means “add all mask
    values.”
    """),

    code(r"""
    treatment_mask = raw_measurements["group"].eq("treatment")
    treatment_rows = raw_measurements.loc[
        treatment_mask,
        ["sample_id", "group", "value", "unit"],
    ]

    print("mask:", treatment_mask.tolist())
    print("selected count:", int(treatment_mask.sum()))
    print(treatment_rows)

    assert treatment_mask.shape[0] == raw_measurements.shape[0]
    assert int(treatment_mask.sum()) == 4
    """),

    md(r"""
    Combine conditions with `&` for “and,” `|` for “or,” and `~` for “not.” Put each
    comparison in parentheses. Python's words `and` and `or` expect one truth value;
    a pandas Series contains one truth value per row.
    """),

    code(r"""
    valid_unit_mask = raw_measurements["unit"].isin(["mg", "g"])
    positive_value_mask = raw_measurements["value"].gt(0)
    candidate_mask = valid_unit_mask & positive_value_mask

    candidates = raw_measurements.loc[candidate_mask, ["sample_id", "value", "unit"]]
    print(candidates)

    assert "s5" not in candidates["sample_id"].tolist()  # Negative value
    assert "s6" not in candidates["sample_id"].tolist()  # Unsupported unit
    """),

    md(r"""
    This is only a candidate mask. We have not checked dates or missing data yet.
    A filter is not “cleaning” unless its rule is justified and rejected evidence is
    preserved.
    """),

    md(r"""
    ## 6 · Missing values, types, and invalid values

    ### 6.1 Missing is not zero

    These meanings are different:

    - `0` means a measured value of zero;
    - an empty string means text is present but contains no visible characters;
    - `None` is Python's general marker for no value;
    - `np.nan` is a floating-point missing marker;
    - `pd.NA` is pandas' missing marker for nullable types.

    Use `isna()` and `notna()` rather than equality comparisons. A missing value is
    not an ordinary value, so comparisons can behave differently from intuition.
    """),

    code(r"""
    missing_value_mask = raw_measurements["value"].isna()
    missing_count = int(missing_value_mask.sum())
    missing_rate = float(missing_value_mask.mean())

    print("missing value count:", missing_count)
    print("missing value rate:", missing_rate)
    print(raw_measurements.loc[missing_value_mask, ["sample_id", "value"]])

    assert missing_count == 1
    assert np.isclose(missing_rate, 1 / 8)
    """),

    md(r"""
    We do not automatically fill the missing measurement with zero or the mean:

    - zero would invent a measurement;
    - the mean would make the group look less variable;
    - dropping the row silently would erase evidence of a collection problem.

    For this lesson, the row will go into a rejected-records table with a reason.

    ### 6.2 Convert types without hiding failures

    Use `errors="coerce"` during inspection to turn unparseable values into missing
    markers. Then count and review those failures. Coercion is not permission to
    ignore them.
    """),

    code(r"""
    typed_measurements = raw_measurements.copy()

    # Preserve the raw columns before interpreting them.
    typed_measurements["value_raw"] = typed_measurements["value"]
    typed_measurements["measured_at_raw"] = typed_measurements["measured_at"]

    typed_measurements["value"] = pd.to_numeric(typed_measurements["value"], errors="coerce")
    typed_measurements["measured_at"] = pd.to_datetime(
        typed_measurements["measured_at"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    invalid_date_mask = typed_measurements["measured_at"].isna()
    invalid_or_missing_value_mask = typed_measurements["value"].isna()

    print("parsed dtypes:\n", typed_measurements[["value", "measured_at"]].dtypes)
    print("invalid dates:", int(invalid_date_mask.sum()))
    print("invalid or missing values:", int(invalid_or_missing_value_mask.sum()))

    assert int(invalid_date_mask.sum()) == 1
    assert int(invalid_or_missing_value_mask.sum()) == 1
    """),

    md(r"""
    The original text columns remain available as evidence. This matters when a
    parser produces a missing value: we can still see whether the source contained
    `bad-date`, blank text, or something else.

    ### 6.3 Categories have an allowed vocabulary

    A categorical column contains labels from a small, meaningful set. Checking the
    allowed labels should happen before converting to pandas' `category` dtype.
    """),

    code(r"""
    allowed_groups = {"control", "treatment"}
    allowed_units = {"mg", "g"}

    invalid_group_mask = ~typed_measurements["group"].isin(allowed_groups)
    invalid_unit_mask = ~typed_measurements["unit"].isin(allowed_units)

    print("invalid groups:", int(invalid_group_mask.sum()))
    print("invalid units:", int(invalid_unit_mask.sum()))

    assert int(invalid_group_mask.sum()) == 0
    assert int(invalid_unit_mask.sum()) == 1

    typed_measurements["group"] = typed_measurements["group"].astype("category")
    """),

    md(r"""
    Converting to `category` can save memory and make the intended vocabulary clearer.
    It does not prove that the vocabulary is correct; the allowed-label rule does.
    """),

    md(r"""
    ## 7 · Transform columns without losing control

    ### 7.1 Copies, views, and safe assignment

    A filtered object may share data with its parent or may be a new object. Guessing
    which one you have can cause confusing changes or warnings.

    Use this pattern when you intend to create an independent table:

    ```python
    subset = original.loc[condition, columns].copy()
    subset.loc[:, "new_column"] = values
    ```

    `.copy()` makes the intention explicit. `.loc` makes the target rows and columns
    explicit.

    ### 7.2 Convert units with a vectorized rule

    We want every valid value in milligrams.

    $$
    \text{value in mg} =
    \begin{cases}
    1000 \times \text{value}, & \text{if unit is g} \\
    \text{value}, & \text{if unit is mg}
    \end{cases}
    $$

    The number 1000 is the conversion factor because one gram equals 1000 milligrams.
    This rule changes the numerical representation, not the physical quantity.
    """),

    code(r"""
    preview_columns = ["sample_id", "value", "unit"]
    conversion_preview = typed_measurements.loc[
        typed_measurements["unit"].isin(allowed_units),
        preview_columns,
    ].copy()

    gram_mask = conversion_preview["unit"].eq("g")
    conversion_preview.loc[:, "value_mg"] = conversion_preview["value"]
    conversion_preview.loc[gram_mask, "value_mg"] = (
        conversion_preview.loc[gram_mask, "value"] * 1000
    )

    print(conversion_preview)

    converted_s2 = conversion_preview.loc[
        conversion_preview["sample_id"].eq("s2"), "value_mg"
    ].iloc[0]
    assert converted_s2 == 1200
    """),

    md(r"""
    This vectorized operation acts on a whole selected column. It is clearer here
    than `.apply()` because the rule is ordinary arithmetic. Use `.apply()` only when
    a clear built-in or vectorized operation does not express the task well.

    ### 7.3 Build explicit validity reasons

    One boolean named `is_valid` is useful, but reasons are more useful. They tell us
    what to repair upstream.
    """),

    code(r"""
    reviewed = typed_measurements.copy()

    reviewed["valid_date"] = reviewed["measured_at"].notna()
    reviewed["valid_value"] = reviewed["value"].notna() & reviewed["value"].gt(0)
    reviewed["valid_group"] = reviewed["group"].isin(allowed_groups)
    reviewed["valid_unit"] = reviewed["unit"].isin(allowed_units)

    rule_columns = ["valid_date", "valid_value", "valid_group", "valid_unit"]
    reviewed["is_valid"] = reviewed[rule_columns].all(axis="columns")

    valid_measurements = reviewed.loc[reviewed["is_valid"]].copy()
    rejected_measurements = reviewed.loc[~reviewed["is_valid"]].copy()

    print("valid sample IDs:", valid_measurements["sample_id"].tolist())
    print("rejected rules:\n", rejected_measurements[["sample_id", *rule_columns]])

    assert len(valid_measurements) == 4
    assert len(rejected_measurements) == 4
    assert len(valid_measurements) + len(rejected_measurements) == len(reviewed)
    """),

    md(r"""
    `all(axis="columns")` asks whether every rule is true within each row. The result
    has one boolean per row.

    We keep both outputs. “Rejected” means “not safe for this calculation,” not
    “worthless forever.” A domain expert may later repair a source value.
    """),

    md(r"""
    ## 8 · Group rows and calculate summaries

    `groupby` follows a simple idea:

    1. **Split** rows by a key such as `group`.
    2. **Apply** a calculation within each group.
    3. **Combine** the results into a new table.

    Before grouping, state:

    - one input row represents one sample measurement;
    - the grouping key is experimental group;
    - one output row represents one experimental group.

    First standardise the valid units, then summarise.
    """),

    code(r"""
    gram_mask = valid_measurements["unit"].eq("g")
    valid_measurements.loc[:, "value_mg"] = valid_measurements["value"]
    valid_measurements.loc[gram_mask, "value_mg"] = (
        valid_measurements.loc[gram_mask, "value"] * 1000
    )

    group_summary = (
        valid_measurements
        .groupby("group", observed=True)
        .agg(
            row_count=("sample_id", "size"),
            measured_count=("value_mg", "count"),
            mean_value_mg=("value_mg", "mean"),
            minimum_value_mg=("value_mg", "min"),
            maximum_value_mg=("value_mg", "max"),
        )
        .reset_index()
    )

    print(group_summary)
    """),

    md(r"""
    The important distinction is:

    - `size` counts rows, including rows whose measured column is missing;
    - `count` counts non-missing values in the chosen column.

    They are equal here because invalid and missing measurements were separated
    before grouping. If they differ unexpectedly, investigate instead of choosing
    whichever count looks nicer.

    Manual check for the control group:

    $$
    \text{control mean}
    = \frac{500 + 1200 + 350}{3}
    = \frac{2050}{3}
    \approx 683.33\text{ mg}
    $$

    The numerator adds three valid control measurements. The denominator is three
    valid control rows. The unit remains milligrams.
    """),

    code(r"""
    control_mean = group_summary.loc[
        group_summary["group"].eq("control"), "mean_value_mg"
    ].iloc[0]

    print("control mean from pandas:", control_mean)
    print("control mean by hand:", (500 + 1200 + 350) / 3)

    assert np.isclose(control_mean, 2050 / 3)
    """),

    md(r"""
    ## 9 · Join tables without multiplying rows

    A join combines columns from two tables by matching keys. Imagine attaching an
    address card to each package. If two address cards exist for the same package ID,
    the join may create two package rows. That is row multiplication.

    ### 9.1 Choose the join by the question

    | Join | Keeps | Use when |
    | --- | --- | --- |
    | Left | Every row from the left table | The left table defines the population |
    | Inner | Only keys present in both tables | Unmatched rows must be excluded intentionally |
    | Right | Every row from the right table | The right table defines the population |
    | Outer | Every key from both tables | Reconciling two sources |

    For enrichment, a left join is usually safer because measurements remain visible
    even when metadata is missing.

    ### 9.2 State key cardinality

    **Cardinality** means how many rows on each side may share a key.

    - one-to-one: each key appears at most once on both sides;
    - many-to-one: many measurement rows may match one metadata row;
    - one-to-many: one left row may match several right rows;
    - many-to-many: keys repeat on both sides, so rows may multiply rapidly.

    Our relationship is many measurements to one entity record.
    """),

    code(r"""
    entity_metadata = pd.DataFrame(
        {
            "entity_id": ["e1", "e2", "e3", "e4"],
            "site": ["north", "south", "north", "west"],
            "device_type": ["A", "B", "A", "B"],
        }
    )

    assert entity_metadata["entity_id"].is_unique

    enriched_measurements = valid_measurements.merge(
        entity_metadata,
        on="entity_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    print(enriched_measurements[["sample_id", "entity_id", "site", "_merge"]])
    print("rows before join:", len(valid_measurements))
    print("rows after join:", len(enriched_measurements))

    assert len(enriched_measurements) == len(valid_measurements)
    """),

    md(r"""
    `validate="many_to_one"` turns the expected relationship into an executable
    check. `indicator=True` adds `_merge` so unmatched keys stay visible.

    Sample `s8` has entity `e5`, which is absent from metadata. The left join keeps
    that measurement and marks it `left_only`. Silently dropping it with an inner
    join would change the population.
    """),

    code(r"""
    unmatched_mask = enriched_measurements["_merge"].eq("left_only")
    unmatched_rows = enriched_measurements.loc[
        unmatched_mask,
        ["sample_id", "entity_id"],
    ]

    print("unmatched metadata rows:\n", unmatched_rows)

    assert unmatched_rows["sample_id"].tolist() == ["s8"]
    """),

    md(r"""
    Null join keys need separate attention. Two missing keys do not prove two rows
    describe the same entity. Decide how missing keys should behave before joining.
    """),

    md(r"""
    ## 10 · Functions turn assumptions into data contracts

    A data contract states what a function accepts and what it promises to return.
    A useful validation function should:

    - receive its inputs explicitly;
    - name the required rule;
    - fail with an informative exception when the rule is broken;
    - avoid changing the caller's DataFrame unexpectedly; and
    - return a value whose meaning is clear.

    Assertions are excellent for internal beliefs and lesson checks. At a public
    input boundary, intentional exceptions are clearer because they explain what the
    caller must fix.
    """),

    code(r"""
    REQUIRED_MEASUREMENT_COLUMNS = {
        "sample_id",
        "entity_id",
        "group",
        "measured_at",
        "value",
        "unit",
    }


    def require_columns(dataframe, required_columns):
        '''Raise ValueError when required column names are absent.'''
        missing_columns = sorted(set(required_columns) - set(dataframe.columns))
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


    def require_unique_key(dataframe, key):
        '''Raise ValueError when a key is missing or duplicated.'''
        if dataframe[key].isna().any():
            raise ValueError(f"Key column {key!r} contains missing values")
        duplicate_mask = dataframe[key].duplicated(keep=False)
        if duplicate_mask.any():
            duplicate_values = dataframe.loc[duplicate_mask, key].astype(str).tolist()
            raise ValueError(f"Key column {key!r} has duplicates: {duplicate_values}")


    require_columns(raw_measurements, REQUIRED_MEASUREMENT_COLUMNS)
    require_unique_key(raw_measurements, "sample_id")
    print("Input contract passed.")
    """),

    md(r"""
    The functions do not know every business rule. They check only what their names
    promise. Small functions are easier to test and reuse than one enormous function
    that loads, cleans, joins, plots, and saves at once.
    """),

    md(r"""
    ## 11 · Debug failures with evidence, then test the repair

    Debugging is not random editing. Use this routine:

    1. Read the final traceback line: exception type and message.
    2. Find the last frame that belongs to your code.
    3. Inspect names, values, shapes, dtypes, and a few rows at that boundary.
    4. Build the smallest input that still fails.
    5. Fix the cause, not merely the message.
    6. Add a test so the same failure becomes easy to recognise.

    ### 11.1 A missing column produces `KeyError`

    The example catches the expected error so the lesson can continue. In real
    debugging, do not catch an error unless you can add useful context or recover
    intentionally.
    """),

    code(r"""
    tiny_frame = pd.DataFrame({"sample_id": ["s1"]})

    try:
        tiny_frame["value"]
    except KeyError as error:
        print("exception type:", type(error).__name__)
        print("message:", error)
        print("available columns:", tiny_frame.columns.tolist())
    """),

    md(r"""
    The cause is a contract mismatch: the code requests `value`, but the input only
    contains `sample_id`. The fix may be correcting a spelling mistake or rejecting
    the input with `require_columns`. Inventing an empty `value` column would only
    hide the missing source data.

    ### 11.2 Duplicate metadata produces `MergeError`
    """),

    code(r"""
    duplicate_metadata = pd.DataFrame(
        {"entity_id": ["e1", "e1"], "site": ["north", "south"]}
    )
    tiny_measurements = pd.DataFrame(
        {"sample_id": ["s1"], "entity_id": ["e1"]}
    )

    try:
        tiny_measurements.merge(
            duplicate_metadata,
            on="entity_id",
            validate="many_to_one",
        )
    except pd.errors.MergeError as error:
        print("exception type:", type(error).__name__)
        print("message:", error)
    """),

    md(r"""
    The error protects us from an ambiguous entity table. Removing `validate` would
    make the code run but would not resolve which site is correct.

    ### 11.3 Small tests describe expected behaviour

    A test has three parts:

    - arrange a tiny input;
    - act by calling the function;
    - assert the expected output or error.

    Tests cannot prove the business rule is wise. They prove the code follows the
    rule we wrote.
    """),

    code(r"""
    # Valid case
    valid_input = pd.DataFrame({"sample_id": ["s1", "s2"], "value": [1.0, 2.0]})
    require_columns(valid_input, {"sample_id", "value"})
    require_unique_key(valid_input, "sample_id")

    # Missing-column case
    try:
        require_columns(valid_input, {"sample_id", "unit"})
        raise AssertionError("Expected require_columns to raise ValueError")
    except ValueError as error:
        assert "unit" in str(error)

    # Duplicate-key case
    duplicated_input = pd.DataFrame({"sample_id": ["s1", "s1"]})
    try:
        require_unique_key(duplicated_input, "sample_id")
        raise AssertionError("Expected require_unique_key to raise ValueError")
    except ValueError as error:
        assert "duplicates" in str(error)

    print("Three small contract tests passed.")
    """),

    md(r"""
    Catch the narrow exception you expect. `except Exception:` can hide programming
    mistakes, keyboard interrupts, and unrelated failures. If a test expects
    `ValueError`, another exception should remain visible.
    """),

    md(r"""
    ## 12 · Plot only after the table is trustworthy

    Our question is: how are valid standardised measurements distributed within each
    group?

    A plot should include:

    - the data population being shown;
    - readable axis labels and units;
    - a title connected to the question; and
    - an explanation of what the plot cannot prove.

    With only four valid rows, this plot is a data check, not strong statistical
    evidence.
    """),

    code(r"""
    figure, axis = plt.subplots(figsize=(7, 4))

    for group_name, group_data in valid_measurements.groupby("group", observed=True):
        axis.scatter(
            [str(group_name)] * len(group_data),
            group_data["value_mg"],
            s=70,
            alpha=0.8,
            label=str(group_name),
        )

    axis.set_title("Valid sample measurements by experimental group")
    axis.set_xlabel("experimental group")
    axis.set_ylabel("measurement (mg)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    plt.show()
    """),

    md(r"""
    Visible pattern: the valid treatment measurement is 900 mg, while valid control
    measurements range from 350 to 1200 mg.

    What the plot cannot establish: it cannot tell us whether treatment causes a
    change. The sample is tiny, and we have not studied sampling, experimental design,
    or uncertainty here.
    """),

    md(r"""
    ## 13 · Mini-project: build a checked measurement pipeline

    Now combine the pieces into a small pipeline. It will return separate outputs:

    - valid measurements ready for the stated summary;
    - rejected measurements with rule results;
    - enriched measurements with join status;
    - a quality report containing counts rather than vague claims.

    The function copies its input, preserves raw values, parses types, evaluates
    explicit rules, standardises units, and never silently removes rejected evidence.
    """),

    code(r"""
    def clean_measurements(dataframe):
        '''Return valid rows, rejected rows, and a quality report.'''
        require_columns(dataframe, REQUIRED_MEASUREMENT_COLUMNS)
        require_unique_key(dataframe, "sample_id")

        reviewed_data = dataframe.copy()
        reviewed_data["value_raw"] = reviewed_data["value"]
        reviewed_data["measured_at_raw"] = reviewed_data["measured_at"]

        reviewed_data["value"] = pd.to_numeric(reviewed_data["value"], errors="coerce")
        reviewed_data["measured_at"] = pd.to_datetime(
            reviewed_data["measured_at"],
            format="%Y-%m-%d",
            errors="coerce",
        )

        reviewed_data["valid_date"] = reviewed_data["measured_at"].notna()
        reviewed_data["valid_value"] = (
            reviewed_data["value"].notna() & reviewed_data["value"].gt(0)
        )
        reviewed_data["valid_group"] = reviewed_data["group"].isin(allowed_groups)
        reviewed_data["valid_unit"] = reviewed_data["unit"].isin(allowed_units)

        rule_names = ["valid_date", "valid_value", "valid_group", "valid_unit"]
        reviewed_data["is_valid"] = reviewed_data[rule_names].all(axis="columns")

        valid_data = reviewed_data.loc[reviewed_data["is_valid"]].copy()
        rejected_data = reviewed_data.loc[~reviewed_data["is_valid"]].copy()

        gram_rows = valid_data["unit"].eq("g")
        valid_data.loc[:, "value_mg"] = valid_data["value"]
        valid_data.loc[gram_rows, "value_mg"] = valid_data.loc[gram_rows, "value"] * 1000

        report = {
            "input_rows": len(dataframe),
            "valid_rows": len(valid_data),
            "rejected_rows": len(rejected_data),
            "missing_cells_by_column": dataframe.isna().sum().astype(int).to_dict(),
            "failed_rules": {
                rule_name: int((~reviewed_data[rule_name]).sum())
                for rule_name in rule_names
            },
        }

        assert report["input_rows"] == report["valid_rows"] + report["rejected_rows"]
        return valid_data, rejected_data, report


    def enrich_with_metadata(valid_data, metadata):
        '''Left-join many measurements to one metadata row per entity.'''
        require_columns(metadata, {"entity_id", "site", "device_type"})
        require_unique_key(metadata, "entity_id")

        enriched_data = valid_data.merge(
            metadata,
            on="entity_id",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if len(enriched_data) != len(valid_data):
            raise ValueError("Metadata join changed the measurement row count")
        return enriched_data
    """),

    code(r"""
    project_valid, project_rejected, quality_report = clean_measurements(raw_measurements)
    project_enriched = enrich_with_metadata(project_valid, entity_metadata)

    project_summary = (
        project_valid
        .groupby("group")
        .agg(sample_count=("sample_id", "size"), mean_value_mg=("value_mg", "mean"))
        .reset_index()
    )

    print("quality report:")
    print(quality_report)
    print("\nrejected rows and failed rules:")
    print(project_rejected[["sample_id", "valid_date", "valid_value", "valid_group", "valid_unit"]])
    print("\ngroup summary:")
    print(project_summary)
    print("\njoin status:")
    print(project_enriched[["sample_id", "entity_id", "site", "_merge"]])

    assert quality_report["input_rows"] == 8
    assert quality_report["valid_rows"] == 4
    assert quality_report["rejected_rows"] == 4
    assert project_valid["sample_id"].is_unique
    assert len(project_enriched) == len(project_valid)
    assert int(project_enriched["_merge"].eq("left_only").sum()) == 1
    assert np.isclose(
        project_valid.loc[project_valid["sample_id"].eq("s2"), "value_mg"].iloc[0],
        1200,
    )

    print("\nAll mini-project checks passed.")
    """),

    md(r"""
    ### Why these choices?

    - **Why pandas instead of plain loops?** Named table operations make selection,
      grouping, and joins easier to inspect. Plain Python remains useful inside small
      validation logic.
    - **Why preserve raw columns?** A failed parser should not erase the original
      evidence.
    - **Why reject rather than fill?** We have no domain rule that justifies inventing
      a measurement or date.
    - **Why vectorized unit conversion?** It directly expresses arithmetic over
      selected rows and avoids unnecessary per-row Python calls.
    - **Why a left join?** The measurement table defines the population, so missing
      metadata must remain visible.
    - **Why return a report?** A clean-looking table alone does not reveal what was
      excluded or why.

    In production, use stable input locations, versioned schemas, structured logs,
    and a proper test runner. Do not store secrets in a notebook. Reusable functions
    belong in importable modules once exploration becomes a maintained workflow.
    """),

    md(r"""
    ## 14 · Exercises, self-check, and solutions

    **Estimated practice time:** 2–3 hours.

    ### Worked example

    Predict the result before running this selection:

    ```python
    project_valid.loc[
        project_valid["value_mg"].ge(900),
        ["sample_id", "group", "value_mg"],
    ]
    ```

    It returns `s2` at 1200 mg and `s7` at 900 mg. The mask has one boolean per valid
    row; `loc` keeps the matching rows and the three named columns.

    ### Guided practice

    1. Print `project_valid.shape`, its column names, and its dtypes. Explain each
       dimension of the shape.
    2. Select control rows with only `sample_id`, `value_mg`, and `unit`. Predict the
       selected row count first.
    3. Calculate missing counts and missing rates for every raw column.
    4. Add an allowed date range from `2026-01-01` through `2026-01-31`. Create a
       boolean rule column instead of silently filtering.
    5. Compare `size` and `count` by grouping the uncleaned table by `group`.

    ### Independent practice

    6. Create a new CSV with five rows. Include one invalid number and one missing
       identifier. Load it from a temporary path and produce a quality report.
    7. Add a `site_region` metadata column and perform a checked many-to-one left
       join. Report unmatched keys and prove the row count did not change.
    8. Write a function that accepts allowed units as an argument instead of reading
       a global variable. Test one valid input and one invalid-unit input.
    9. Create a labelled plot of valid measurements by site. Explain one visible
       pattern and two claims the plot cannot support.

    ### Challenge

    Extend the mini-project so that it:

    - loads measurement and metadata CSV files through explicit `Path` objects;
    - validates required columns and unique keys;
    - retains raw values before conversion;
    - produces one reason column containing all failed rules for each rejected row;
    - reports input, valid, rejected, and unmatched counts;
    - performs a checked many-to-one join;
    - writes valid and rejected outputs to different CSV files; and
    - passes at least four tests, including duplicate metadata and a missing column.

    ### Self-check before reading solutions

    For every operation, say aloud:

    - what one input row means;
    - the expected shape before and after;
    - which rule justifies a changed or rejected value;
    - what evidence remains available if the rule fails.
    """),

    md(r"""
    ### Solution and scoring rubric

    1. The shape has one count for rows and one for columns. Dtypes describe storage,
       not guaranteed business meaning.
    2. Use `.eq("control")` to build the mask and `.loc` with a list of named columns.
       The expected count is three.
    3. `raw_measurements.isna().sum()` gives counts;
       `raw_measurements.isna().mean()` gives fractions.
    4. Parse dates first, then combine `.ge(start)` and `.le(end)` with `&`.
    5. `size` includes every row in a group; `count` ignores missing values in the
       selected column.
    6. Preserve the invalid number, coerce a parsed copy, and report the missing ID
       separately from the failed numeric rule.
    7. Validate the metadata key before `merge`, use `validate="many_to_one"`, add
       `indicator=True`, and compare row counts.
    8. Passing allowed values as an argument makes the function's dependency visible
       and easier to test.
    9. A useful plot has a question, population, labels, units, and a cautious reading.

    Challenge scoring:

    | Skill | Points |
    | --- | ---: |
    | Clear paths and reproducible loading | 2 |
    | Schema and key validation | 3 |
    | Preserved raw evidence and explicit rule reasons | 3 |
    | Correct unit conversion and valid/rejected split | 3 |
    | Checked join and unmatched report | 3 |
    | Separate outputs and four meaningful tests | 4 |
    | Explanation of row meaning, shapes, and tradeoffs | 2 |

    Maximum: 20 points.

    **Common mistakes:** treating the index as a stable ID, confusing one Series with
    a one-column DataFrame, using `and` instead of `&`, omitting parentheses around
    mask conditions, comparing missing values with `==`, editing a filtered view,
    coercing bad values and forgetting to count them, using `count` when row count is
    needed, joining without checking key uniqueness, using an inner join that silently
    removes measurements, catching every exception, and plotting before validating.

    **Readiness threshold:** 16/20, including correct selection, missing-value
    handling, type conversion, unit conversion, checked joining, and at least one
    passing failure-path test.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    You are ready for FND-03 when you can, without copying this notebook:

    - explain what one row represents and distinguish index from identifier;
    - load a CSV through an explicit path and inspect shape, names, examples, types,
      missingness, and duplicate keys;
    - select rows and columns with `loc`, `iloc`, and combined boolean masks;
    - preserve raw values while converting numbers and dates;
    - explain why missing, zero, and invalid are different;
    - create a column safely with `.copy()` and `.loc`;
    - explain `groupby`, `size`, `count`, and `reset_index`;
    - state join type and key cardinality before merging;
    - preserve unmatched rows and prove that an enrichment join did not multiply rows;
    - read a `KeyError` or `MergeError` and build a tiny failing input;
    - write a small function with one successful and one failure-path test; and
    - complete the mini-project with at least 16/20 points.

    ### Teach it back

    Explain the journey of sample `s3` from raw CSV to rejected-records table. Then
    explain the journey of sample `s2` from `1.2 g` to `1200 mg`. For each journey,
    name the evidence, rule, transformation, and check.

    ### Memory aid

    **Name what one row means, preserve what arrived, make every rule visible, and
    check the shape before trusting the result.**

    FND-03 will use these table skills to study distributions, duplicates, data
    quality, problem framing, and leakage-safe modeling workflows.
    """),

    code(r"""
    # Remove the temporary CSV folder created by this lesson.
    shutil.rmtree(lesson_directory, ignore_errors=True)
    print("Temporary lesson files removed:", not lesson_directory.exists())
    """),
]


build("00_prerequisites/06_python_pandas_and_debugging.ipynb", cells)
