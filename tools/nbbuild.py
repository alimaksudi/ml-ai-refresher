"""nbbuild — a tiny helper for assembling Jupyter notebooks from plain Python.

WHY THIS EXISTS
---------------
The *source of truth* for this curriculum lives in section directories under
``tools/builders/`` as
ordinary Python files. Each builder describes a notebook as a list of cells,
using clean triple-quoted strings — no hand-written ``.ipynb`` JSON, no manual
escaping of quotes/newlines, and trivially diff-able in code review.

Running a builder emits a real ``.ipynb`` under ``notebooks/``. The generated
notebooks are *build artifacts*: edit the builder, re-run, regenerate.

USAGE (inside a builder)
------------------------
    from nbbuild import md, code, build

    build("01_ml_foundations/01_linear_algebra_essentials.ipynb", [
        md('''# Title ...'''),
        code('''import numpy as np ...'''),
    ])
"""
from __future__ import annotations

import os
import json
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from foundation_mastery import (
    ADVANCED_SECTION_NOTES,
    CORE_CODE,
    CORE_MASTERY,
    TOPIC_GUIDES,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tests and validation can redirect generated notebooks to a temporary directory
# without mutating the checked-in build artifacts.
NB_DIR = os.environ.get("NB_OUTPUT_DIR", os.path.join(ROOT, "notebooks"))
CURRICULUM_PATH = os.path.join(ROOT, "docs", "CURRICULUM_PATH.json")
SELF_CONTAINED_LESSON_IDS = {"PRE-01", "PRE-02", "PRE-03", "PRE-04", "PRE-05", "PRE-06", "FND-01", "FND-02", "FND-03", "FND-04", "CML-01", "CML-02", "CML-03", "CML-04", "CML-05", "CML-06", "MLE-04", "MLE-05", "EVAL-01", "MLE-06", "DL-01"}


def _curriculum_metadata(rel_path: str) -> dict:
    """Return canonical dependency metadata for a generated notebook."""
    with open(CURRICULUM_PATH, encoding="utf-8") as source:
        modules = json.load(source)["modules"]
    matches = [module for module in modules if module["path"] == rel_path]
    if len(matches) != 1:
        raise ValueError(
            f"{rel_path!r} must have exactly one entry in docs/CURRICULUM_PATH.json"
        )
    module = matches[0]
    return {
        "id": module["id"],
        "prerequisites": module["prerequisites"],
        "gate": module["gate"],
        "canonical_order": modules.index(module) + 1,
    }


def _learner_navigation_cell(rel_path: str):
    """Generate one authoritative student-facing route card from canonical data."""
    with open(CURRICULUM_PATH, encoding="utf-8") as source:
        modules = json.load(source)["modules"]
    index = next(i for i, module in enumerate(modules) if module["path"] == rel_path)
    module = modules[index]
    previous = modules[index - 1]["id"] if index else "entry diagnostic"
    following = modules[index + 1]["id"] if index + 1 < len(modules) else "course completion"
    prerequisites = ", ".join(module["prerequisites"]) or "none"
    time_by_gate = {
        "readiness": "2–4 hours",
        "foundations": "4–7 hours",
        "core_ml": "4–7 hours",
        "classical_ml": "5–8 hours",
        "deep_learning": "5–8 hours",
        "llm_rag": "5–8 hours",
        "agents": "5–8 hours",
        "production": "5–8 hours",
        "final_capstone": "8–16 hours",
    }
    return md(f"""
    > **Canonical learner route · module {module['id']} of {len(modules)}**
    >
    > Required prerequisites: **{prerequisites}** · Previous: **{previous}** ·
    > Next after mastery: **{following}** · Expected first-pass workload:
    > **{time_by_gate.get(module['gate'], '4–8 hours')}**
    >
    > **Core path:** intuition, objectives, foundations, runnable implementation,
    > failure modes, and assessed exercises. **Extension path:** history, production,
    > tradeoffs, and interview material may be revisited after the core pass.
    > Do not continue merely because every cell ran. Continue when you can complete
    > the independent exercise and teach-back without notes. The canonical route in
    > `docs/CURRICULUM_PATH.json` is authoritative when section-local file order and
    > prerequisite order differ.
    """)


def _required_mastery_gate_cell(rel_path: str):
    """Add a consistent core gate; original Section 14 may remain an extension set."""
    with open(CURRICULUM_PATH, encoding="utf-8") as source:
        modules = json.load(source)["modules"]
    module = next(item for item in modules if item["path"] == rel_path)
    return md(f"""
    ## Required Core Mastery Gate · {module['id']}

    Complete this gate before treating the module as finished. The longer exercises
    in Section 14 are extension practice unless the module says otherwise.

    **Worked example:** rerun the smallest worked example in this notebook. Annotate
    every input, output, shape or unit, and the claim the result supports.

    **Guided practice (20–30 min):** change one input in that example. Before running
    it, predict the direction of change and explain why. Use the nearest preceding
    formula or algorithm step as a hint. **Self-check:** compare prediction with the
    result and explain any mismatch rather than editing the prediction afterward.

    **Independent practice (30–45 min):** on fresh tiny data, reproduce the module's
    central operation without copying the completed cell. State assumptions, expected
    output, and one invalid input. **Self-check:** verify with an assertion, a manual
    calculation, or a trusted library implementation.

    **Challenge (30–60 min):** create one failure described in Section 7, detect it
    using evidence, and repair it without changing unrelated variables.

    **Solution and scoring rubric:** 2 points for a correct prediction, 2 for a
    runnable independent implementation, 2 for an objective self-check, 2 for failure
    diagnosis, and 2 for teach-back without notes. Common mistakes: copying before
    attempting, using later-module concepts as unexplained shortcuts, evaluating on
    training data, and continuing because cells ran. **Readiness threshold: 8/10**,
    including both independent implementation and teach-back points. If below 8,
    revisit the named prerequisite in the route card and retry with different data.
    """)


def _lesson_title(cells) -> str:
    for cell in cells:
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                if line.startswith("# "):
                    return line.removeprefix("# ").strip()
    return "this lesson"


def _student_lesson_companion_cell(title: str, module_id: str):
    """State the actual beginner decision before asking reflection questions."""
    guide = TOPIC_GUIDES.get(module_id)
    if not guide:
        return _legacy_student_lesson_companion_cell(title)
    if guide:
        topic_intro = f"""
        ### Practical problem before history

        **{guide['problem']}**

        ### Concept, analogy, and analogy limit

        {guide['analogy']}

        ### Use / avoid / alternatives

        | Decision | Topic-specific answer |
        |---|---|
        | Use it when | {guide['use']} |
        | Avoid it when | {guide['avoid']} |
        | Prefer instead | {guide['alternative']} |
        """
    return md(f"""
    ## Student Lesson Companion · {title}

    Use this companion during the **core pass**. The lesson first gives a concrete
    answer; after studying it, restate that answer in your own words.

    {topic_intro}

    ### Questions to answer after the core pass

    1. What is the smallest useful baseline?
    2. What limitation of that baseline creates the need for this concept?
    3. What evidence would support using the concept?
    4. Which assumption could make the result misleading?
    5. What simpler or safer alternative would you choose when that assumption fails?

    """)


def _legacy_student_lesson_companion_cell(title: str):
    """Keep unaffected modules byte-for-byte compatible with their builders."""
    return md(f"""
    ## Student Lesson Companion · {title}

    Use this companion during the **core pass**. Write short answers before reading
    the extension material; then correct them in a different colour after the lesson.

    ### Practical problem before history

    1. What concrete task or decision are we trying to improve?
    2. Why is it difficult with the data, time, labels, or compute available?
    3. What is the simplest previous baseline?
    4. Where does that baseline fail?
    5. What capability must the new concept add?

    Section 2 must supply enough evidence to answer these questions. Historical detail
    is extension material unless it explains the problem or design constraint.

    ### Concept, analogy, and analogy limit

    After Section 3, explain the concept in one sentence without unexplained jargon.
    Map it to one daily-life analogy **or** one concrete visual example. Then state
    one place where the mapping breaks; an analogy is a bridge, not a proof.

    ### Use / avoid / alternatives

    Complete this decision table from Sections 7, 9, and 11:

    | Decision | Required answer |
    |---|---|
    | Use it when | Three realistic conditions where its assumptions and benefits fit |
    | Avoid it when | Two conditions where it is misleading, unsafe, or wasteful |
    | Prefer instead | At least one simpler baseline and one alternative for a failed assumption |
    | Evidence | Metric, diagnostic, or constraint that supports the choice |

    """)


def _lesson_close_cell(title: str, module_id: str):
    guide = TOPIC_GUIDES.get(module_id)
    if not guide:
        return _legacy_lesson_close_cell(title)
    memory = guide["memory"] if guide else (
        "Start from the problem, trace the mechanism, verify the evidence, and check the assumptions."
    )
    return md(f"""
    ## Lesson Close · Summary, Student Check, and Memory Aid

    ### Five short student checks

    1. What practical problem does **{title}** solve?
    2. What is its central mechanism in simple language?
    3. Which assumption or limitation is easiest to forget?
    4. What output or diagnostic tells you it worked as intended?
    5. When would you choose a simpler or related alternative?

    ### Plain-language summary

    Complete four sentences without notes: **The problem is… The concept works by…
    I would use it when… I would avoid it when…** Compare your answer with the
    objectives, failure modes, tradeoff analysis, and teach-back section.

    ### One-sentence memory aid

    **{memory}**

    Now write your own version in no more than 20 words without looking back.

    The lesson is complete only after the Required Core Mastery Gate, not after the
    final code cell runs.
    """)


def _legacy_lesson_close_cell(title: str):
    return md(f"""
    ## Lesson Close · Summary, Student Check, and Memory Aid

    ### Five short student checks

    1. What practical problem does **{title}** solve?
    2. What is its central mechanism in simple language?
    3. Which assumption or limitation is easiest to forget?
    4. What output or diagnostic tells you it worked as intended?
    5. When would you choose a simpler or related alternative?

    ### Plain-language summary

    Complete four sentences without notes: **The problem is… The concept works by…
    I would use it when… I would avoid it when…** Compare your answer with the
    objectives, failure modes, tradeoff analysis, and teach-back section.

    ### One-sentence memory aid

    **Remember {title}: start from the problem, trace the mechanism, verify the
    evidence, and use it only when its assumptions fit.** Replace this general aid
    with your own topic-specific sentence of no more than 20 words.

    The lesson is complete only after the Required Core Mastery Gate, not after the
    final code cell runs.
    """)


def _clean(text: str) -> str:
    """Dedent a triple-quoted block and normalise trailing whitespace.

    ``textwrap.dedent`` removes the *common* leading indentation, so builders
    can indent cell bodies to match surrounding Python without corrupting the
    content. Blank lines are ignored when computing the common prefix.
    """
    return textwrap.dedent(text).strip("\n") + "\n"


def md(text: str):
    """A markdown cell. Supports GitHub-flavoured markdown, LaTeX ($...$),
    and ```mermaid fenced diagrams (rendered by GitHub and modern JupyterLab)."""
    return new_markdown_cell(_clean(text))


def code(text: str):
    """A code cell containing runnable Python (NumPy / matplotlib / stdlib)."""
    return new_code_cell(_clean(text))


def _notation_support_cell():
    """Compact prerequisite reminder inserted into post-foundation notebooks."""
    return md(r"""
    <details>
    <summary><strong>Mathematics notation support — open when a formula feels dense</strong></summary>

    - $x_i$: item or component number $i$; a subscript is a label, not multiplication.
    - $n$: number of examples; $d$: number of features or dimensions.
    - $\mathbf x$: vector; $X$: matrix; $X^\top$: transpose (rows and columns swap).
    - $\hat y$: an estimate or prediction; a hat marks an estimated quantity.
    - $\sum$: add repeated terms; $\prod$: multiply repeated terms.
    - $\lVert\mathbf x\rVert$: vector length; $|x|$: distance of one number from zero.
    - $\frac{\partial f}{\partial x}$: slope of $f$ as $x$ changes while other inputs
      stay fixed; $\nabla f$: vector containing all parameter slopes.
    - $P(A\mid B)$: probability of $A$ after restricting attention to cases where
      $B$ is true.
    - $\log$ reverses an exponential and turns products into sums.

    Read a formula one operator at a time, write object shapes beside vectors and
    matrices, and substitute a tiny numeric example. Review PRE-01 through PRE-04 for
    worked explanations of these symbols.
    </details>
    """)


def build(rel_path: str, cells, kernel: str = "python3") -> str:
    """Assemble ``cells`` into a notebook and write it to ``notebooks/<rel_path>``."""
    cells = list(cells)
    title = _lesson_title(cells)
    module_id = _curriculum_metadata(rel_path)["id"]
    cells.insert(1, _learner_navigation_cell(rel_path))
    if module_id not in SELF_CONTAINED_LESSON_IDS:
        objective_index = next(
            (index for index, cell in enumerate(cells) if cell.cell_type == "markdown" and "## 1" in cell.source),
            1,
        )
        cells.insert(objective_index + 1, _student_lesson_companion_cell(title, module_id))
        if not rel_path.startswith(("00_prerequisites/", "01_ml_foundations/")):
            insert_at = 1
            for index, cell in enumerate(cells):
                if cell.cell_type == "markdown" and "## 1" in cell.source:
                    insert_at = index + 1
                    break
            cells.insert(insert_at, _notation_support_cell())
        if module_id in CORE_MASTERY:
            foundation_index = next(
                (
                    index
                    for index, cell in enumerate(cells)
                    if cell.cell_type == "markdown"
                    and ("## 4 ·" in cell.source or "## 4." in cell.source)
                ),
                objective_index + 2,
            )
            beginner_cells = [md(CORE_MASTERY[module_id])]
            if module_id in CORE_CODE:
                beginner_cells.append(code(CORE_CODE[module_id]))
            if module_id in ADVANCED_SECTION_NOTES:
                beginner_cells.append(
                    md(
                        f"""
                        > **Advanced-path boundary.** {ADVANCED_SECTION_NOTES[module_id]}
                        > You may read the remaining derivations now, but they are not
                        > required for the first mastery attempt.
                        """
                    )
                )
            cells[foundation_index:foundation_index] = beginner_cells
        cells.append(_lesson_close_cell(title, module_id))
        cells.append(_required_mastery_gate_cell(rel_path))
    nb = new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": kernel,
            },
            "language_info": {"name": "python", "version": "3.x"},
            "curriculum": _curriculum_metadata(rel_path),
        }
    )
    out = os.path.join(NB_DIR, rel_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"wrote {os.path.relpath(out, ROOT)}  ({len(cells)} cells)")
    return out
