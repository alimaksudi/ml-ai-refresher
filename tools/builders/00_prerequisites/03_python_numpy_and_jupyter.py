"""Builder for Lesson PRE-03 — Python, NumPy, and Jupyter Foundations."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # PRE-03 · Python, NumPy, and Jupyter Foundations

    *Turn reasoning into code you can rerun, inspect, and trust*

    | Lesson detail | Value |
    | --- | --- |
    | Prerequisites | PRE-01 and PRE-02; no previous programming required |
    | Estimated study time | 6–8 hours across two or more sessions |
    | Main outcome | Write small Python programs and inspect numerical data safely with NumPy |
    | Next lessons | PRE-04 for change; PRE-06 for uncertainty; PRE-05 for pandas workflows |

    This is a typing lesson only in the smallest sense. The real goal is to build a
    reliable mental model: predict what code should do, run it, inspect the result,
    and turn important assumptions into checks.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end of this lesson, you will be able to:

    - explain the difference between notebook text, code, output, and kernel state;
    - use values, variables, strings, lists, tuples, and dictionaries;
    - write comparisons, conditions, loops, comprehensions, and functions;
    - read common errors and raise an intentional error for invalid input;
    - use imports and assertions without treating them as magic;
    - create NumPy arrays and explain `dtype`, `ndim`, `size`, and `shape`;
    - index, slice, filter, reshape, and combine arrays;
    - explain axis operations and simple broadcasting with small numbers;
    - avoid accidental shared-data mutation with `.copy()`;
    - create reproducible random values and compare decimals safely;
    - make a small labelled plot and complete a checked mini-project.

    Formal vectors, dot products, and matrix multiplication belong to FND-01.
    Here, two-dimensional arrays are rectangular data containers: rows are examples
    and columns are measured fields.
    """),

    md(r"""
    ## 2 · The problem we are trying to solve

    A service reports seven request latencies:

    `120, 95, 140, 110, 300, 105, 130` milliseconds.

    We want to:

    - reject impossible values;
    - calculate a typical latency;
    - identify requests above a threshold;
    - repeat the calculation when new data arrives;
    - get the same simulated test data tomorrow;
    - prove that important assumptions still hold.

    Paper is excellent for the first tiny calculation. It becomes fragile when the
    dataset changes. Python records the steps. NumPy applies repeated numerical work
    to whole arrays. Jupyter keeps explanation, code, and visible evidence together.
    """),

    md(r"""
    ## 3 · How a Jupyter notebook actually works

    A notebook has two layers:

    - the **document** stores Markdown cells, code cells, and saved outputs;
    - the **kernel** is a running Python process holding the current variables.

    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #4c78a8; border-radius: 10px; padding: 14px 18px; background: #eef5ff; color: #172b4d; text-align: center;"><strong>Code cell</strong><br>instructions</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 14px 18px; background: #fff4e8; color: #4a2b0b; text-align: center;"><strong>Kernel</strong><br>current memory</div>
      <div style="font-size: 24px; color: #555;">→</div>
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 14px 18px; background: #eef8ec; color: #173d17; text-align: center;"><strong>Output</strong><br>visible evidence</div>
    </div>

    Running a cell changes kernel state. Editing a cell does not automatically rerun
    it. Restarting the kernel clears variables but does not erase notebook text.

    The strongest reproducibility check is **Restart Kernel and Run All**. If that
    fails, the notebook depended on a secret execution order or an undeclared value.
    """),

    code(r"""
    # Run this cell, then the next one. The variable lives in kernel memory.
    request_count = 3
    print("request_count:", request_count)
    """),

    code(r"""
    request_count = request_count + 1
    print("updated request_count:", request_count)
    assert request_count == 4
    """),

    md(r"""
    If you run the second cell twice, the result becomes 5. If you restart and run
    only the second cell, Python raises `NameError` because `request_count` does not
    exist yet. That is notebook state, not randomness.

    **Notebook habit:** when an output surprises you, restart and run from the top
    before changing the code.
    """),

    md(r"""
    ## 4 · Plain Python foundations

    ### 4.1 Values, types, names, and operators

    A **value** is data. A **type** tells Python what kind of data it is and which
    operations make sense.

    | Type | Meaning | Example |
    | --- | --- | --- |
    | `int` | whole number | `12` |
    | `float` | decimal approximation | `12.5` |
    | `str` | text | `"latency"` |
    | `bool` | truth value | `True` |
    | `NoneType` | intentional absence of a value | `None` |

    In Python, `=` means **assign the value on the right to the name on the left**.
    It is not mathematical equality. Comparison uses `==`.

    Use descriptive `snake_case` names such as `average_latency_ms`. The suffix
    keeps the unit visible.
    """),

    code(r"""
    request_total = 8
    failed_requests = 2
    failure_fraction = failed_requests / request_total
    service_name = "checkout"
    passed_target = failure_fraction < 0.30

    print("type of request_total:", type(request_total).__name__)
    print("failure fraction:", failure_fraction)
    print("service:", service_name)
    print("passed target:", passed_target)

    assert request_total == 8
    assert failed_requests != request_total
    """),

    md(r"""
    Python arithmetic mirrors PRE-01: `+`, `-`, `*`, `/`, and `**` mean addition,
    subtraction, multiplication, division, and power. `%` returns a remainder.

    Comparisons produce booleans: `==`, `!=`, `<`, `<=`, `>`, and `>=`. Combine
    conditions with `and`, `or`, and `not`.
    """),

    md(r"""
    ### 4.2 Strings and collections

    A string stores text. An **f-string** inserts evaluated values into readable
    output. Lists, tuples, and dictionaries group values in different ways.

    - A `list` is ordered and mutable: it can change.
    - A `tuple` is ordered and immutable: its positions stay fixed.
    - A `dict` maps meaningful keys to values.

    Python starts positions at zero. In a three-item list, valid positions are
    `0`, `1`, and `2`. A negative index counts backward, so `-1` is the last item.
    """),

    code(r"""
    latencies_ms = [120, 95, 140]
    latency_range_ms = (0, 1_000)
    service_report = {
        "service": "checkout",
        "request_count": len(latencies_ms),
        "unit": "ms",
    }

    print(f"first latency: {latencies_ms[0]} ms")
    print(f"last latency: {latencies_ms[-1]} ms")
    print("first two values:", latencies_ms[0:2])
    print("service name:", service_report["service"])

    latencies_ms.append(110)  # Lists can change.
    print("after append:", latencies_ms)

    assert latency_range_ms[0] == 0
    assert len(latencies_ms) == 4
    """),

    md(r"""
    A slice such as `values[start:stop]` includes `start` but excludes `stop`.
    That half-open rule makes `values[0:2]` contain exactly two items.

    Assigning `second_name = first_name` does not always copy the underlying data.
    For mutable objects such as lists, both names can refer to the same object. We
    will revisit this with NumPy views and copies.
    """),

    md(r"""
    ### 4.3 Conditions choose which path runs

    An `if` statement runs a block only when its condition is true. Indentation is
    syntax: it marks which lines belong to the block.

    Use `elif` for another condition and `else` for every remaining case. Begin
    with the most specific rule when conditions overlap.
    """),

    code(r"""
    latency_ms = 140

    if latency_ms < 0:
        label = "invalid"
    elif latency_ms <= 120:
        label = "within target"
    else:
        label = "slow"

    print(f"{latency_ms} ms is {label}")
    assert label == "slow"
    """),

    md(r"""
    ### 4.4 Loops repeat a clear action

    A `for` loop takes one item at a time from an iterable collection.

    - `range(3)` produces `0, 1, 2`.
    - `enumerate(values)` provides position and value.
    - `zip(a, b)` pairs corresponding items and stops at the shorter collection.
    - A list comprehension builds a new list from a short, readable rule.

    Prefer an ordinary loop when the logic needs several steps or careful debugging.
    """),

    code(r"""
    latencies_ms = [120, 95, 140, 110]
    total_latency_ms = 0

    for position, latency_ms in enumerate(latencies_ms):
        total_latency_ms = total_latency_ms + latency_ms
        print(f"position {position}: running total = {total_latency_ms} ms")

    thresholds_ms = [100, 100, 100, 100]
    comparisons = []
    for latency_ms, threshold_ms in zip(latencies_ms, thresholds_ms):
        comparisons.append(latency_ms > threshold_ms)

    doubled_latencies_ms = [latency_ms * 2 for latency_ms in latencies_ms]

    print("above 100 ms:", comparisons)
    print("doubled:", doubled_latencies_ms)
    assert total_latency_ms == 465
    """),

    md(r"""
    ### 4.5 Functions name reusable behavior

    A function has a contract:

    - parameters describe required inputs;
    - the body performs one focused job;
    - `return` sends a result back;
    - a docstring explains the promise in plain language.

    Names created inside a function are normally **local**. They disappear when the
    call finishes. But mutating a list passed into a function can change the caller's
    object. Prefer returning a new value unless mutation is intentional.
    """),

    code(r"""
    def classify_latency(latency_ms, target_ms=120):
        '''Return a readable label for one non-negative latency.'''
        if latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if latency_ms <= target_ms:
            return "within target"
        return "slow"


    def add_marker_without_mutating(values, marker):
        result = values.copy()
        result.append(marker)
        return result


    original = [95, 140]
    marked = add_marker_without_mutating(original, 120)

    print(classify_latency(140))
    print("original:", original)
    print("new list:", marked)

    assert classify_latency(95) == "within target"
    assert original == [95, 140]
    """),

    md(r"""
    ### 4.6 Imports, errors, exceptions, and assertions

    Python's standard library and installed packages provide reusable modules.
    `import statistics` loads a module; `statistics.mean(...)` identifies the tool's
    owner. An alias such as `import numpy as np` gives a conventional shorter name.

    Errors are information:

    - `NameError`: a name does not exist in current state;
    - `TypeError`: an operation received an incompatible type;
    - `IndexError`: a sequence position does not exist;
    - `KeyError`: a dictionary key does not exist;
    - `ValueError`: the type is acceptable but the value is not.

    Read a traceback from the final line upward: error type and message first, then
    the last referenced line that you own.
    """),

    code(r"""
    import statistics


    def safe_mean(values):
        cleaned_values = [float(value) for value in values if value is not None]
        if not cleaned_values:
            raise ValueError("safe_mean needs at least one numeric value")
        return statistics.mean(cleaned_values)


    print("safe mean:", safe_mean([100, None, 140]))

    try:
        safe_mean([None])
    except ValueError as error:
        print("caught expected error:", type(error).__name__, str(error))

    assert safe_mean([100, 140]) == 120
    """),

    md(r"""
    Catch only errors you can handle meaningfully. Broadly catching `Exception` and
    continuing can turn a visible failure into silent bad data.

    An assertion records an internal belief such as `assert row_count > 0`. It is a
    useful learning and testing check. For invalid public input, raise an intentional
    exception with a useful message.
    """),

    md(r"""
    ## 5 · Worked example and Python checkpoint

    **Worked example:** calculate a checked average from raw latency values using
    plain Python. We will reject `None` and negative values, keep the original list
    unchanged, then calculate the average manually.

    Workflow:

    1. inspect one value at a time;
    2. skip missing values;
    3. reject impossible negatives;
    4. build a new clean list;
    5. sum with a loop;
    6. divide by the count;
    7. assert the expected result.
    """),

    code(r"""
    raw_latencies_ms = [120, None, 95, 140, 110]
    clean_latencies_ms = []

    for latency_ms in raw_latencies_ms:
        if latency_ms is None:
            continue
        if latency_ms < 0:
            raise ValueError("latency cannot be negative")
        clean_latencies_ms.append(float(latency_ms))

    total_latency_ms = 0.0
    for latency_ms in clean_latencies_ms:
        total_latency_ms = total_latency_ms + latency_ms

    average_latency_ms = total_latency_ms / len(clean_latencies_ms)

    print("raw values:", raw_latencies_ms)
    print("clean values:", clean_latencies_ms)
    print("total:", total_latency_ms, "ms")
    print("average:", average_latency_ms, "ms")

    assert raw_latencies_ms[1] is None
    assert clean_latencies_ms == [120.0, 95.0, 140.0, 110.0]
    assert average_latency_ms == 116.25
    """),

    md(r"""
    **Python checkpoint:** before continuing, make sure you can explain every line
    above. Change `95` to `195`, predict the new direction of the average, and rerun.
    Then restore the original value.

    NumPy will shorten repeated numerical work, but it should not hide the workflow
    you just built by hand.
    """),

    md(r"""
    ## 6 · Why NumPy exists and what an array stores

    Python lists can mix types and grow dynamically. That flexibility is useful for
    general programs. Numerical work usually needs many values of one compatible
    type and the same operation repeated across them.

    NumPy arrays add:

    - a data type called `dtype`;
    - a number of dimensions called `ndim`;
    - a total element count called `size`;
    - a layout called `shape`;
    - fast elementwise operations.

    Shape `(4,)` means one dimension containing four values. Shape `(3, 2)` means
    three rows and two columns. This lesson treats that as a rectangular dataset;
    FND-01 will teach the mathematical meaning of vectors and matrices.
    """),

    code(r"""
    import numpy as np

    latency_array_ms = np.array([120, 95, 140, 110], dtype=float)
    request_table = np.array([
        [120, 1],
        [95, 0],
        [140, 1],
    ], dtype=float)

    print("array:", latency_array_ms)
    print("dtype:", latency_array_ms.dtype)
    print("dimensions:", latency_array_ms.ndim)
    print("size:", latency_array_ms.size)
    print("shape:", latency_array_ms.shape)
    print("table shape:", request_table.shape)

    assert latency_array_ms.shape == (4,)
    assert request_table.shape == (3, 2)
    """),

    md(r"""
    NumPy normally chooses one compatible type for the whole array. Mixing numbers
    and text may produce a string or object array that cannot support the intended
    arithmetic. Check `dtype` at data boundaries instead of assuming it is numeric.
    """),

    md(r"""
    ## 7 · Indexing, slicing, and boolean masks

    Array indexing resembles list indexing:

    - `array[0]` selects one value;
    - `array[1:3]` selects positions 1 and 2;
    - `table[0, 1]` selects row 0, column 1;
    - `table[:, 0]` selects every row from column 0.

    A boolean mask has one truth value per position. Applying it keeps values where
    the mask is `True`. This is the NumPy version of a filtering loop.
    """),

    code(r"""
    print("first latency:", latency_array_ms[0])
    print("middle slice:", latency_array_ms[1:3])
    print("first table column:", request_table[:, 0])

    slow_mask = latency_array_ms > 120
    slow_latencies_ms = latency_array_ms[slow_mask]

    print("mask:", slow_mask)
    print("slow values:", slow_latencies_ms)
    print("slow count:", slow_mask.sum())

    assert np.array_equal(slow_latencies_ms, np.array([140.0]))
    """),

    md(r"""
    The mask and the filtered axis must have compatible lengths. A four-value mask
    cannot select from a three-row table. When filtering fails, print both shapes.
    """),

    md(r"""
    ## 8 · Aggregation and axis meaning

    An **aggregation** compresses several values into a summary such as a sum, mean,
    minimum, or maximum.

    For a two-dimensional array:

    - `axis=0` collapses rows and returns one summary per column;
    - `axis=1` collapses columns and returns one summary per row;
    - no axis collapses every value into one summary.

    Do not memorize the axis number alone. Say what survives: “one result per column”
    or “one result per row.”
    """),

    code(r"""
    daily_latency_ms = np.array([
        [100, 120, 140],
        [80, 100, 120],
    ], dtype=float)

    overall_mean_ms = daily_latency_ms.mean()
    mean_per_column_ms = daily_latency_ms.mean(axis=0)
    mean_per_row_ms = daily_latency_ms.mean(axis=1)

    print("shape:", daily_latency_ms.shape)
    print("overall mean:", overall_mean_ms)
    print("one mean per column:", mean_per_column_ms)
    print("one mean per row:", mean_per_row_ms)

    assert overall_mean_ms == 110
    assert np.array_equal(mean_per_column_ms, np.array([90, 110, 130]))
    assert np.array_equal(mean_per_row_ms, np.array([120, 100]))
    """),

    md(r"""
    Manually verify the first column: $(100+80)/2=90$. That one calculation anchors
    what `axis=0` did. In real data, column meaning must be known before averaging;
    averaging an ID column would run successfully and still be meaningless.
    """),

    md(r"""
    ## 9 · Vectorization and broadcasting

    **Vectorization** applies an operation to an entire array without writing the
    Python loop yourself.

    **Broadcasting** lets NumPy combine compatible shapes by conceptually extending
    a smaller array. It does not make every shape compatible.

    Suppose two rows contain three measurements and we want to subtract one baseline
    per column:

    ```text
    data shape       (2, 3)    two rows, three columns
    baseline shape      (3,)   one value for each column
    result shape     (2, 3)    baseline applied to every row
    ```
    """),

    code(r"""
    measurements = np.array([
        [10.0, 20.0, 30.0],
        [12.0, 18.0, 35.0],
    ])
    column_baseline = np.array([10.0, 20.0, 30.0])

    differences = measurements - column_baseline

    print("measurement shape:", measurements.shape)
    print("baseline shape:", column_baseline.shape)
    print("differences:\n", differences)

    assert differences.shape == (2, 3)
    assert np.array_equal(differences[1], np.array([2.0, -2.0, 5.0]))
    """),

    md(r"""
    NumPy compares dimensions from the right. Aligned dimensions must be equal or
    one of them must be `1`. Shapes `(2, 3)` and `(2,)` are incompatible because
    their rightmost dimensions are `3` and `2`.

    Broadcasting is convenient, but a compatible shape does not guarantee the
    meanings are compatible. Confirm that corresponding columns use the same unit.
    """),

    md(r"""
    ## 10 · Reshaping, combining, views, and copies

    `reshape` changes layout without changing the number of elements. Four values
    can become shape `(2, 2)` but not `(3, 2)`.

    `np.concatenate` joins arrays along an existing axis. Joined dimensions must
    match everywhere except the joining axis.

    A NumPy slice is often a **view** sharing data with the original array. Changing
    the view can change the original. Use `.copy()` when independent data is needed.
    """),

    code(r"""
    values = np.array([1, 2, 3, 4])
    reshaped = values.reshape(2, 2)

    first_batch = np.array([[1, 10], [2, 20]])
    second_batch = np.array([[3, 30]])
    combined = np.concatenate([first_batch, second_batch], axis=0)

    view = values[1:3]
    independent_copy = values[1:3].copy()
    independent_copy[0] = 999

    print("reshaped:\n", reshaped)
    print("combined:\n", combined)
    print("original after editing copy:", values)
    print("view shares memory:", np.shares_memory(values, view))

    assert reshaped.shape == (2, 2)
    assert combined.shape == (3, 2)
    assert values[1] == 2
    """),

    md(r"""
    Shape-changing tools are not repairs for misunderstood data. Before reshaping or
    combining, state what one row and one column represent.
    """),

    md(r"""
    ## 11 · Reproducibility, decimal comparison, and plotting

    A pseudo-random generator produces a deterministic sequence from a starting
    **seed**. The same seed and procedure reproduce the same values. This supports
    debugging and fair experiments; it does not make simulated data realistic.

    Decimal values use finite binary approximations. Therefore `0.1 + 0.2` may not
    equal `0.3` exactly. Use `np.isclose` or `np.allclose` when small rounding error
    is acceptable.
    """),

    code(r"""
    import matplotlib.pyplot as plt

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    simulated_a_ms = rng_a.normal(loc=120, scale=10, size=20)
    simulated_b_ms = rng_b.normal(loc=120, scale=10, size=20)

    print("first five simulated values:", simulated_a_ms[:5])
    print("0.1 + 0.2:", 0.1 + 0.2)

    assert np.allclose(simulated_a_ms, simulated_b_ms)
    assert np.isclose(0.1 + 0.2, 0.3)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(simulated_a_ms, marker="o")
    ax.axhline(120, color="tab:red", linestyle="--", label="target: 120 ms")
    ax.set_xlabel("simulated request position")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Reproducible simulated request latency")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()
    """),

    md(r"""
    Expected result: both generators create the same 20 values, the decimal printout
    exposes a tiny representation difference, and the plot shows which simulated
    requests lie above or below the 120 ms reference.

    A chart is evidence only when its axes, units, and comparison are clear. It does
    not explain why a request was slow.
    """),

    md(r"""
    ## 12 · Common mistakes and recovery habits

    - **Cell works only after manual clicking:** restart and run all; find the first
      missing definition or import.
    - **`=` used inside a question:** use `==` for comparison and `=` for assignment.
    - **Wrong item selected:** print the collection and remember indexing begins at 0.
    - **Function changes the caller's list:** return a copy unless mutation is part
      of the documented contract.
    - **Broad exception hides a bug:** catch only the expected error and preserve its
      message.
    - **Array operation gives a strange shape:** print every input shape and state
      what each axis represents.
    - **Broadcasting runs but meaning is wrong:** check units and column order, not
      only shape compatibility.
    - **Slice edit changes original data:** use `.copy()` when independence matters.
    - **Random result cannot be reproduced:** create and pass an explicit generator.
    - **Exact decimal assertion fails:** use `isclose` with a justified tolerance.
    - **Plot looks dramatic:** inspect axis limits, units, and the underlying values.

    Debugging routine: read the last traceback line, locate the last line you own,
    print the smallest relevant value, type, and shape, then reproduce the failure
    with the smallest possible input.
    """),

    md(r"""
    ## 13 · Mini-project: checked latency report

    **Goal:** turn raw request latencies into a small, reproducible report without
    pandas.

    **Input fields:**

    - `request_id`: unique text label;
    - `latency_ms`: non-negative number or `None`;
    - `endpoint`: text category.

    **Workflow:**

    1. Store six records as dictionaries inside a list.
    2. Validate that request IDs are unique.
    3. Skip `None` latency while counting how many values are missing.
    4. Raise `ValueError` for a negative latency.
    5. Convert clean latencies to a NumPy array.
    6. Report count, mean, minimum, maximum, and count above 150 ms.
    7. Use a boolean mask to print the slow values.
    8. Create a labelled plot with a 150 ms reference line.
    9. Add assertions for clean count, shape, and one known summary.

    **Expected output:** a readable summary, slow-request values, missing count, and
    one plot. The raw records must remain unchanged.

    **Evaluation:** correctness matters first, followed by readable names, explicit
    units, useful checks, and an explanation of what the report cannot conclude.
    """),

    md(r"""
    ## 14 · Practice, self-check, and solutions

    **Estimated practice time:** 75–100 minutes.

    ### Worked example

    Given `[80, 120, 160]`, a plain loop calculates total `360` and mean `120`.
    A NumPy mask `values > 120` becomes `[False, False, True]` and selects `[160]`.
    Both approaches answer the same question; NumPy expresses the repeated numeric
    comparison more directly.

    ### Guided practice

    1. Create variables for `successful=18` and `total=20`; print the percentage
       with an f-string and assert it equals 90.
    2. Loop over `[3, 5, 7]` with `enumerate` and calculate the total.
    3. Write `validate_score(score)` that raises `ValueError` outside `[0, 1]`.
    4. Create a float array with shape `(2, 3)` and print its second column.
    5. For that array, calculate one mean per row and explain why the result has
       shape `(2,)`.

    ### Independent practice

    6. Use a dictionary to store a model name, version, and score. Print one readable
       sentence from it.
    7. Filter values greater than 10 from `[4, 12, 7, 20]` using a Python list
       comprehension, then using a NumPy boolean mask.
    8. Reshape `np.arange(6)` to `(2, 3)`, copy its first row, edit the copy, and
       prove the original did not change.
    9. Explain the output shape when shape `(4, 3)` subtracts shape `(3,)`. Then
       explain why subtracting shape `(2,)` fails.

    ### Challenge

    Complete the checked latency mini-project from Section 13 using at least:

    - one function with a clear contract;
    - one dictionary per raw record;
    - one intentional `ValueError` path;
    - one NumPy boolean mask;
    - one seeded generator or reproducibility check;
    - three assertions;
    - one labelled plot.

    ### Self-check before reading solutions

    Predict each value, type, and shape before running code. After running, explain
    any mismatch rather than merely replacing your answer.
    """),

    md(r"""
    ### Solution and scoring rubric

    1. `percentage = successful / total * 100` produces `90.0`.
    2. The positions are `0, 1, 2`; the total is `15`.
    3. Return the score when `0 <= score <= 1`; otherwise raise `ValueError`.
    4. One valid array is `np.array([[1,2,3],[4,5,6]], dtype=float)`. Its second
       column is `[2,5]`.
    5. `array.mean(axis=1)` collapses three columns and leaves one result for each
       of two rows, so shape is `(2,)`.
    6. Access dictionary values by their meaningful keys inside an f-string.
    7. Both methods produce `[12,20]`; the NumPy version first produces a four-value
       boolean mask.
    8. `np.arange(6).reshape(2,3)` is valid because both layouts contain six values.
       Editing `.copy()` does not change the original.
    9. `(3,)` supplies one value per column and broadcasts across four rows, producing
       `(4,3)`. `(2,)` conflicts with the rightmost size `3`.

    Award one point for each guided or independent task, with Questions 5, 8, and 9
    worth two points because they require explanation. Award five challenge points:
    validation, correct summary, mask, reproducibility/checks, and labelled plot.
    Maximum: 17 points.

    **Common mistakes:** running cells out of order, confusing `=` with `==`, hiding
    an exception, using the wrong axis, filtering with a mismatched mask, editing a
    view accidentally, or reporting a number without units.

    **Readiness threshold:** 13/17, including correct function validation, row/column
    selection, axis explanation, broadcasting explanation, and three passing project
    assertions.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    You are ready for PRE-04, PRE-06, and PRE-05 when you can:

    - restart the kernel and run the notebook top to bottom;
    - explain values, types, assignment, comparison, collections, conditions, loops,
      functions, imports, exceptions, and assertions;
    - read a traceback from its final line and isolate the smallest failing input;
    - explain a NumPy array's type, dimensions, size, and shape;
    - index, slice, filter, aggregate, reshape, combine, and copy arrays;
    - explain one axis operation and one broadcast with small numbers;
    - reproduce random values and compare decimals safely;
    - complete the mini-project without hidden state.

    ### Teach it back

    Explain how the checked latency calculation grows from a Python list and loop
    into a NumPy array, mask, summary, and plot. Include one reason the NumPy result
    can be numerically correct but conceptually wrong.

    ### Memory aid

    **Predict the value, type, and shape; run the code; then make the expectation
    executable.**

    Formal matrix multiplication comes later in FND-01, after arrays and shapes feel
    familiar rather than mysterious.
    """),
]


build("00_prerequisites/03_python_numpy_and_jupyter.ipynb", cells)
