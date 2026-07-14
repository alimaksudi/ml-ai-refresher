"""Builder for Notebook 00E — Practical Python, Pandas, Debugging, and Tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # 00E · Practical Python, Pandas, Debugging, and Tests
    ### Prerequisite Phase — Bridge from small code cells to data work

    **Prerequisites:** 00D. **Estimated time:** 5–7 hours including practice.
    This module exists because knowing NumPy syntax is not yet enough to debug a
    data workflow. Complete it before 03A; experienced Python users may pass the
    readiness gate instead.
    """),
    md(r"""
    ## 1 · Learning Objectives

    Write and test small functions; use lists, dictionaries, conditions, loops, and
    exceptions; read tracebacks; load and inspect tabular data; select, filter,
    group, join, and validate pandas data; plot a distribution; and distinguish a
    code failure from a data-contract failure.
    """),
    md(r"""
    ## 2 · Historical Motivation

    Real ML work is mostly ordinary programs around numerical kernels: loading data,
    validating assumptions, transforming columns, logging decisions, and testing
    interfaces. A model cannot rescue a pipeline that silently selects the wrong row,
    changes units, or catches every exception.
    """),
    md(r"""
    ## 3 · Intuition and Visual Understanding

    Treat code as small contracts:

    ```text
    input assumptions → function → output guarantees
           ↓                          ↓
       assertions                  tests
    ```

    Read a traceback from the final line upward: exception type and message first,
    then the last frame you own, then inspect the values and shapes at that boundary.
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    A boolean mask is an indicator vector. If $m_i\in\{0,1\}$ marks whether row $i$
    passes a condition, then $\sum_i m_i$ is the selected-row count. For mask
    `[True, False, True]`, the count is 2. This connection explains filtering,
    missing-value counts, and group aggregation without introducing new mathematics.
    """),
    code(r"""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    records = [
        {"sample_id": "a", "group": "control", "value": 2.0},
        {"sample_id": "b", "group": "treatment", "value": 5.0},
        {"sample_id": "c", "group": "treatment", "value": None},
    ]
    frame = pd.DataFrame(records)
    print(frame)
    """),
    md(r"""
    ## 5 · Manual Implementation from Scratch

    Begin with plain Python before pandas. A function should validate what it cannot
    safely infer and raise an informative error rather than return plausible nonsense.
    """),
    code(r"""
    def safe_mean(values):
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
            raise ValueError("safe_mean needs at least one non-missing value")
        return sum(cleaned) / len(cleaned)

    assert safe_mean([2, None, 4]) == 3.0
    try:
        safe_mean([None])
    except ValueError as error:
        print(type(error).__name__, str(error))
    """),
    md(r"""
    ## 6 · Visualization

    A plot is a diagnostic, not decoration. State the question first, label axes and
    units, then explain one visible pattern and one thing the plot cannot establish.
    """),
    code(r"""
    frame["value"].plot(kind="hist", bins=4, title="Observed value distribution")
    plt.xlabel("value (arbitrary units)")
    plt.tight_layout()
    plt.show()
    """),
    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Catching `Exception` and discarding the error.
    - Mutating a DataFrame without knowing whether a view or copy is involved.
    - Comparing missing values with `== None` instead of using `isna`.
    - Selecting columns by position when names are the contract.
    - Joining on a non-unique key and silently multiplying rows.
    - Treating `object` dtype as a meaningful schema.
    """),
    md(r"""
    ## 8 · Library Implementation

    Use pandas operations deliberately: inspect schema, select named columns, filter
    with masks, aggregate with `groupby`, merge with cardinality validation, and make
    assumptions executable with assertions.
    """),
    code(r"""
    required = {"sample_id", "group", "value"}
    assert required.issubset(frame.columns)
    assert frame["sample_id"].is_unique
    summary = frame.groupby("group", dropna=False)["value"].agg(["count", "mean"])
    print(summary)

    metadata = pd.DataFrame({"sample_id": ["a", "b", "c"], "site": ["x", "x", "y"]})
    joined = frame.merge(metadata, on="sample_id", validate="one_to_one")
    assert len(joined) == len(frame)
    """),
    md(r"""
    ## 9 · Realistic Case Study

    Given a CSV of measurements, write a loader that checks required columns, unique
    IDs, numeric conversion, missingness, and allowed labels. Return a clean table and
    a report; never silently delete invalid rows. Notebook 03A will turn this table
    into a leakage-safe modeling workflow.
    """),
    md(r"""
    ## 10 · Production and Learning Considerations

    Keep environments reproducible, use relative paths from the repository root,
    avoid secrets in notebooks, and move reusable logic into importable functions.
    Tests should use tiny deterministic data and cover valid input plus one failure.
    """),
    md(r"""
    ## 11 · Tradeoff Analysis

    Vectorized pandas is concise and fast for ordinary tables; explicit loops can be
    clearer for stateful logic. Assertions are excellent learning checks but public
    input validation should raise intentional exceptions. Notebooks aid exploration;
    modules and tests aid reuse.
    """),
    md(r"""
    ## 12 · Readiness and Interview Preparation

    You are ready when you can debug a `KeyError`, explain DataFrame shape and dtypes,
    prevent a many-to-many join, write a pure function with a test, and load a table
    without relying on hidden notebook state.
    """),
    md(r"""
    ## 13 · Teach-Back

    Explain the difference between a list, dictionary, NumPy array, and DataFrame.
    Then describe how you would investigate “the row count doubled after enrichment”
    and why restarting the kernel is a useful reproducibility test.
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Worked example:** the merge above uses `validate="one_to_one"`; duplicate a
    metadata key and observe the intentional error.

    **Guided practice (30 min):** add a `unit` column, reject values whose unit is
    neither `mg` nor `g`, and convert grams to milligrams. Hint: write one conversion
    function and use `.apply`. Self-check: `2 g → 2000 mg`.

    **Independent practice (45 min):** create a five-row CSV in a temporary directory,
    load it, check required columns and unique IDs, and return a missingness report.
    Self-check: deliberately remove one column and confirm an informative failure.

    **Challenge (60 min):** join measurements to entity metadata while guaranteeing
    row count and key cardinality; plot one labelled distribution by group.

    <details><summary><strong>Solution and scoring rubric</strong></summary>

    A complete solution uses named columns, `isna`, `is_unique`, `merge(validate=...)`,
    an explicit exception, and at least two assertions. Award 2 points for conversion,
    3 for the loader/report, 3 for safe joining, and 2 for explanation and tests.
    Common mistakes: hidden global variables, broad exception handling, positional
    columns, and accepting a multiplied join. **Readiness threshold: 8/10.**
    </details>
    """),
]

build("phase_minus1_onboarding/00e_python_pandas_debugging.ipynb", cells)
