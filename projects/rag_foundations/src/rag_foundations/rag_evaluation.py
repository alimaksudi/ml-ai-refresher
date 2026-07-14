"""Versioned component-level evaluation for the complete teaching RAG pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluation import file_sha256, load_json
from .grounded import evaluate_answers


DEFAULT_THRESHOLDS = {
    "context_precision_at_k": 0.15,
    "context_recall_at_k": 0.80,
    "answer_correctness_proxy": 0.40,
    "evidence_support_proxy": 1.00,
    "citation_validity": 1.00,
    "abstention_accuracy": 0.90,
}


def context_precision_at_k(
    retrieved_sections: list[str],
    relevant_sections: set[str],
) -> float:
    """Return labelled precision over the retrieved section IDs."""
    if not retrieved_sections:
        return 0.0
    return len(set(retrieved_sections) & relevant_sections) / len(retrieved_sections)


def apply_quality_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> dict:
    """Apply an explicit teaching policy; thresholds are not universal SLAs."""
    violations = [
        {
            "metric": metric,
            "observed": metrics[metric],
            "required": minimum,
        }
        for metric, minimum in thresholds.items()
        if metrics[metric] < minimum
    ]
    return {"passed": not violations, "thresholds": thresholds, "violations": violations}


def evaluate_rag_systems(
    data_dir: Path,
    output_path: Path | None = None,
    strategy: str = "sentence",
    top_k: int = 5,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Compare retrieval modes under one answer, label, and metric contract."""
    corpus_path = data_dir / "corpus.json"
    query_path = data_dir / "queries.json"
    answer_path = data_dir / "answers.json"
    queries = load_json(query_path)["queries"]
    query_by_id = {query["id"]: query for query in queries}
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    systems = {}

    for mode in ("lexical", "dense_lsa", "hybrid_rrf"):
        component_report = evaluate_answers(
            data_dir,
            strategy=strategy,
            mode=mode,
            top_k=top_k,
        )
        rows = []
        for row in component_report["rows"]:
            query = query_by_id[row["query_id"]]
            relevant = set(query["relevant_sections"])
            precision = (
                context_precision_at_k(row["retrieved_sections"], relevant)
                if relevant else None
            )
            rows.append({**row, "context_precision_at_k": precision})

        answerable_rows = [row for row in rows if row["answerable"]]
        metrics = {
            "context_precision_at_k": float(np.mean([
                row["context_precision_at_k"] for row in answerable_rows
            ])),
            "context_recall_at_k": float(np.mean([
                row["retrieval_recall_at_k"] for row in answerable_rows
            ])),
            "answer_correctness_proxy": float(np.mean([
                row["answer_correctness"] for row in rows
            ])),
            "evidence_support_proxy": float(np.mean([
                row["evidence_support"] for row in rows
            ])),
            "citation_validity": float(np.mean([
                row["citation_validity"] for row in rows
            ])),
            "abstention_accuracy": float(np.mean([
                row["abstention_correct"] for row in rows
            ])),
            "successful_case_rate": float(np.mean([
                row["outcome"] == "success" for row in rows
            ])),
            "mean_latency_ms": float(np.mean([row["latency_ms"] for row in rows])),
        }
        systems[mode] = {
            "metrics": metrics,
            "quality_gate": apply_quality_gate(metrics, thresholds),
            "failure_counts": component_report["failure_counts"],
            "rows": rows,
        }

    report = {
        "schema_version": "1.0",
        "strategy": strategy,
        "top_k": top_k,
        "corpus_sha256": file_sha256(corpus_path),
        "queries_sha256": file_sha256(query_path),
        "answers_sha256": file_sha256(answer_path),
        "metric_contract": {
            "context_precision_at_k": "Labelled relevant sections divided by retrieved sections for answerable queries.",
            "context_recall_at_k": "Labelled relevant sections recovered in the first k unique section IDs.",
            "answer_correctness_proxy": "Proxy: all required answer terms occur in the extractive answer; not semantic correctness.",
            "evidence_support_proxy": "Proxy: normalized extractive answer text occurs in cited evidence; not entailment.",
            "citation_validity": "Every citation ID appears among retrieved IDs.",
            "abstention_accuracy": "Answer versus abstain decision matches labelled answerability.",
        },
        "quality_gate_note": "Teaching thresholds for regression mechanics; calibrate from risk and human review before deployment.",
        "latency_note": "Local measurements over a tiny corpus; not production capacity or tail-latency evidence.",
        "systems": systems,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
