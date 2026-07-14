"""Builder for Lesson RAG-04 — Grounded Answer and Citation Evaluation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # RAG-04 · Grounded Answer and Citation Evaluation
    ### Section 06 — Separate retrieval success from answer success

    **Prerequisite:** RAG-03. **Estimated time:** 6–8 hours. The first answerer is
    deliberately extractive: it returns cited evidence rather than free-form LLM text.
    This creates a grounding baseline that later generation must beat without losing
    citation validity, abstention, or security.
    """),
    md(r"""
    ## 1 · Learning Objectives

    Create gold answers and acceptable terms, generate extractive cited answers,
    measure answer correctness/support/citation validity/abstention, distinguish
    component failures, and block stale, injected, and unauthorized evidence.
    """),
    md(r"""
    ## 2 · Historical Motivation

    Retrieval recall answers whether useful evidence appeared somewhere in the ranked
    context. It does not prove the answer selected the right passage, interpreted it
    correctly, cited it honestly, or refused an unsupported question. Treating one
    final “accuracy” number as RAG evaluation hides the component that failed.
    """),
    md(r"""
    ## 3 · Intuition and Visual Understanding

    ```text
    query → ranked evidence → answer selection → answer → citations
                 ↓                 ↓              ↓         ↓
             recall@k        correctness      support    validity
                    unanswerable → abstention
    ```

    A librarian can bring the right shelf but hand you the wrong book. Retrieval and
    answering are related stages, not the same capability.
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    Component success can be expressed as indicators. Citation validity is 1 when
    every cited evidence ID is among retrieved IDs. Extractive support is 1 when the
    answer is contained in cited evidence. Abstention accuracy is the fraction of
    questions whose answer/refusal decision matches answerability.

    For ten cases with nine correct answer/refusal decisions, abstention accuracy is
    $9/10=0.9$. This does not measure whether the nine non-refused answers are correct.
    """),
    md(r"""
    ## 5 · Manual Implementation from Scratch

    Run the deterministic baseline and inspect the difference between high retrieval
    recall and lower answer correctness.
    """),
    code(r"""
    import sys
    from pathlib import Path
    ROOT = next(candidate for candidate in (Path.cwd(), *Path.cwd().parents)
                if (candidate / "projects/rag_foundations/src").exists())
    sys.path.insert(0, str(ROOT / "projects/rag_foundations/src"))
    from rag_foundations.grounded import evaluate_answers, evaluate_security

    report = evaluate_answers(ROOT / "projects/rag_foundations/data")
    print("metrics", report["metrics"])
    print("failures", report["failure_counts"])
    """),
    md(r"""
    ## 6 · Visualization

    A component chart prevents a strong support score from hiding weak answer
    selection. Perfect support is expected for extractive text and is not evidence of
    correctness.
    """),
    code(r"""
    import matplotlib.pyplot as plt
    metric_names = [name for name in report["metrics"] if name != "mean_latency_ms"]
    values = [report["metrics"][name] for name in metric_names]
    plt.bar(metric_names, values); plt.ylim(0, 1.05); plt.xticks(rotation=45, ha="right")
    plt.title("Component metrics reveal different failure modes")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Calling retrieved evidence an answer.
    - Scoring support without validating the citation ID.
    - Counting an extractive quotation as correct merely because it is grounded.
    - Letting untrusted document instructions control the system.
    - Retrieving stale or unauthorized evidence before filtering.
    - Tuning abstention on the final test cases.
    - Reporting averages without per-query component traces.
    """),
    md(r"""
    ## 8 · Library Implementation

    `grounded.py` keeps answer text, evidence IDs, retrieved IDs, metrics, latency,
    and outcome taxonomy in one trace. Later LLM generation must use the same schema
    so its gains and regressions are comparable.
    """),
    md(r"""
    ## 9 · Realistic Case Study

    The curriculum assistant retrieves a relevant section for most questions, yet an
    extractive top-one answer is correct on fewer cases. This shows that ranking the
    correct section somewhere in top-k is not the same as choosing it for the answer.
    """),
    md(r"""
    ## 10 · Production and Learning Considerations

    Apply authorization, freshness, and unsafe-content controls before evidence
    reaches answer generation. Log refusals without logging secrets. Version gold
    answers separately from prompts and never rewrite labels to flatter a system.
    """),
    md(r"""
    ## 11 · Tradeoff Analysis

    Extractive answers maximize support and auditability but may be verbose or select
    the wrong passage. Generative answers can synthesize multiple sources but add
    interpretation, citation, and hallucination failures. Abstention improves safety
    but can reduce useful coverage.
    """),
    md(r"""
    ## 12 · Readiness and Interview Preparation

    Given one trace, identify the first failed component. Recommend a retrieval,
    selection, generation, citation, or policy fix—never “improve the RAG system” as
    an undifferentiated response.
    """),
    md(r"""
    ## 13 · Teach-Back

    Explain retrieval recall, correctness, support, citation validity, and abstention
    with one counterexample for each. Explain why prompt injection in retrieved text
    is data to inspect, not an instruction to follow.
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Worked:** classify one trace into retrieval, answer, attribution, grounding, or
    abstention failure. **Guided:** inspect five failed rows and identify the first
    broken component. **Independent:** add an acceptable answer variant without
    changing evidence labels. **Challenge:** add a deterministic multi-chunk answer
    selector and prove whether correctness improves without lowering support/security.

    <details><summary><strong>Solution and scoring rubric</strong></summary>
    Full credit uses the first failed component, preserves gold labels, validates IDs,
    and reports all component metrics. Award 3 points for taxonomy, 3 for trace-based
    diagnosis, 2 for security, and 2 for controlled ablation. Common mistakes: using
    gold evidence inside the runtime selector and treating exact extraction as correct
    by definition. **Readiness threshold: 8/10.**
    </details>
    """),
]

build("06_rag/04_grounded_answer_evaluation.ipynb", cells)
