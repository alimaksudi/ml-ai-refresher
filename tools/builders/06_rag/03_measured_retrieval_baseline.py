"""Builder for Lesson RAG-03 — Measured Baseline RAG Retrieval."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402

cells = [
    md(r"""
    # RAG-03 · Measured Baseline RAG Retrieval
    ### Section 06 — Establish evidence before infrastructure

    **Prerequisites:** NLP-01, NLP-02, RAG-01, and RAG-02.
    **Estimated time:** 6–8 hours.
    This notebook creates the measurement baseline that vector databases, hybrid
    search, reranking, and advanced RAG must beat. It evaluates retrieval only;
    generation quality is a separate downstream problem.
    """),
    md(r"""
    ## 1 · Learning Objectives

    Version corpus and labels, preserve evidence provenance, compare chunking,
    implement lexical and dense statistical retrieval, fuse rankings, calculate
    recall@k/MRR/nDCG, measure abstention and latency, and diagnose failure slices.
    """),
    md(r"""
    ## 2 · Historical Motivation

    RAG systems often added embeddings, vector stores, rerankers, and agents before
    anyone established whether retrieval improved. Classical information retrieval
    starts with a test collection: documents, queries, relevance judgements, metrics,
    and a baseline. RAG needs the same discipline.
    """),
    md(r"""
    ## 3 · Intuition and Visual Understanding

    ```text
    versioned documents → chunks with evidence IDs → retriever → ranked chunks
             ↑                                              ↓
       labelled queries ← failure review ← metrics and slices
    ```

    If the needed evidence is absent, generation cannot be grounded. If evidence is
    present and the answer is wrong, the generator failed. Keep these diagnoses apart.
    """),
    md(r"""
    ## 4 · Mathematical Foundations

    Recall@k is the fraction of relevant evidence retrieved in the first k results.
    Reciprocal rank is $1/r$ where $r$ is the first relevant rank. DCG discounts a
    relevant result at rank $i$ by $1/\log_2(i+1)$; nDCG divides by the ideal DCG.

    **Small example:** if the only relevant section is ranked third, recall@3 is 1,
    reciprocal rank is 1/3, and nDCG@3 is $1/\log_2(4)=0.5$. Recall says it was found;
    rank-sensitive metrics say the learner had to look past two distractions.
    """),
    md(r"""
    ## 5 · Manual Implementation from Scratch

    Run the project evaluation. The dense baseline is LSA—TF-IDF projected with
    truncated SVD—not a neural sentence embedding. That label matters.
    """),
    code(r"""
    import sys
    from pathlib import Path
    ROOT = next(
        candidate
        for candidate in (Path.cwd(), *Path.cwd().parents)
        if (candidate / "projects/rag_foundations/src").exists()
    )
    sys.path.insert(0, str(ROOT / "projects/rag_foundations/src"))
    from rag_foundations.evaluation import evaluate

    report = evaluate(ROOT / "projects/rag_foundations/data")
    for name, result in report["experiments"].items():
        print(name, "recall", round(result["recall_at_k"], 3),
              "MRR", round(result["mrr"], 3),
              "nDCG", round(result["ndcg_at_k"], 3))
    """),
    md(r"""
    ## 6 · Visualization

    Compare systems on more than one axis. A small ranking improvement may not justify
    worse abstention, latency, operational complexity, or failure on paraphrases.
    """),
    code(r"""
    import matplotlib.pyplot as plt
    names = list(report["experiments"])
    recalls = [report["experiments"][name]["recall_at_k"] for name in names]
    plt.figure(figsize=(10, 4)); plt.bar(range(len(names)), recalls)
    plt.xticks(range(len(names)), names, rotation=60, ha="right")
    plt.ylabel("Recall@5"); plt.ylim(0, 1.05); plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 7 · Failure Modes and Common Mistakes

    - Labelling queries after seeing one system's results.
    - Losing source/section IDs during chunking.
    - Calling LSA a neural semantic encoder.
    - Reporting only an average and hiding paraphrase failures.
    - Treating no lexical overlap as proof a question is unanswerable.
    - Adding raw scores from different retrievers rather than fusing ranks.
    - Evaluating generated answers when retrieval evidence was never measured.
    """),
    md(r"""
    ## 8 · Library Implementation

    The project uses sklearn TF-IDF and TruncatedSVD, then deterministic ranking and
    metric functions. Inspect `projects/rag_foundations/src/rag_foundations/evaluation.py`.
    Every report includes corpus/query hashes and per-query component rows.
    """),
    md(r"""
    ## 9 · Realistic Case Study

    The corpus contains curriculum knowledge with direct, paraphrased, multi-concept,
    and unanswerable questions. The best aggregate system is not automatically the
    deployment choice; inspect which student questions it misses and why.
    """),
    md(r"""
    ## 10 · Production and Learning Considerations

    Version documents, parsers, chunks, query labels, representation, and thresholds.
    Log evidence IDs, scores, latency, and abstention for every query. Re-evaluate
    whenever corpus, chunker, or embedding model changes.
    """),
    md(r"""
    ## 11 · Tradeoff Analysis

    Lexical retrieval is transparent and excellent for exact terms. LSA captures
    latent co-occurrence but is corpus-dependent. Rank fusion is robust to score scale
    but adds another choice. Structure chunks preserve meaning; smaller chunks may rank
    precisely while losing surrounding context.
    """),
    md(r"""
    ## 12 · Readiness and Interview Preparation

    Given one failed query, decide whether the label, chunking, representation,
    ranking, threshold, or corpus is responsible. Propose one change and one metric
    that would confirm or falsify the diagnosis.
    """),
    md(r"""
    ## 13 · Teach-Back

    Explain why retrieval evaluation precedes generation. Contrast recall@k, MRR,
    and nDCG. Explain why an unanswerable set and evidence provenance are safety
    requirements, not optional metrics.
    """),
    md(r"""
    ## 14 · Exercises, Self-Check, and Solutions

    **Worked:** manually rank three sections for one query and calculate all metrics.
    **Guided:** inspect three component failures and assign a cause. **Independent:**
    add two queries without changing corpus text, predict which retriever wins, then
    evaluate. **Challenge:** add a pinned neural embedding model as a new experiment
    without removing lexical/LSA baselines or changing gold labels.

    <details><summary><strong>Solution and scoring rubric</strong></summary>
    Full credit preserves IDs and hashes, separates retrieval from answer claims,
    reports slices, and makes one-variable ablations. Award 3 points for metric math,
    3 for diagnosis, 2 for reproducibility, and 2 for limitations. Common mistakes:
    changing labels to improve scores and claiming the most complex system must win.
    **Readiness threshold: 8/10.**
    </details>
    """),
]

build("06_rag/03_measured_retrieval_baseline.ipynb", cells)
