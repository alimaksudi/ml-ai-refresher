"""Measured BM25, dense, and hybrid retrieval comparisons."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .evaluation import (
    BM25Index,
    Chunk,
    RetrievalIndex,
    build_chunks,
    deduplicate_sections,
    file_sha256,
    load_json,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)


def minmax_score_fusion(
    sparse_results: list[tuple[Chunk, float]],
    dense_results: list[tuple[Chunk, float]],
    alpha: float,
) -> list[tuple[Chunk, float]]:
    """Fuse the union of two candidate lists after per-list min-max scaling."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if alpha == 0:
        return list(sparse_results)
    if alpha == 1:
        return list(dense_results)

    def normalize(results: list[tuple[Chunk, float]]) -> dict[str, float]:
        if not results:
            return {}
        values = [score for _, score in results]
        lower, upper = min(values), max(values)
        score_range = upper - lower
        if score_range <= 1e-12:
            return {chunk.id: 1.0 for chunk, _ in results}
        return {
            chunk.id: (score - lower) / score_range
            for chunk, score in results
        }

    sparse_normalized = normalize(sparse_results)
    dense_normalized = normalize(dense_results)
    chunks = {chunk.id: chunk for chunk, _ in sparse_results + dense_results}
    fused = {
        chunk_id: (
            (1 - alpha) * sparse_normalized.get(chunk_id, 0.0)
            + alpha * dense_normalized.get(chunk_id, 0.0)
        )
        for chunk_id in chunks
    }
    return [
        (chunks[chunk_id], score)
        for chunk_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    ]


def _metric_summary(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    slices = {}
    for slice_name in sorted({row["slice"] for row in answerable}):
        subset = [row for row in answerable if row["slice"] == slice_name]
        slices[slice_name] = {
            "count": len(subset),
            "recall_at_k": float(np.mean([row["recall_at_k"] for row in subset])),
            "mrr": float(np.mean([row["reciprocal_rank"] for row in subset])),
        }
    return {
        "recall_at_k": float(np.mean([row["recall_at_k"] for row in answerable])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in answerable])),
        "ndcg_at_k": float(np.mean([row["ndcg_at_k"] for row in answerable])),
        "answerable_zero_result_rate": float(np.mean([row["abstained"] for row in answerable])),
        "unanswerable_abstention_rate": float(np.mean([row["abstained"] for row in unanswerable])),
        "slices": slices,
        "rows": rows,
    }


def evaluate_hybrid(
    data_dir: Path,
    output_path: Path | None = None,
    top_k: int = 3,
    candidate_multiplier: int = 3,
    alpha_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    corpus_filename: str = "hybrid_corpus.json",
    query_filename: str = "hybrid_queries.json",
    dense_components: int = 4,
) -> dict:
    """Evaluate all retrievers on one corpus, label set, and chunk strategy."""
    if top_k < 1 or candidate_multiplier < 1:
        raise ValueError("top_k and candidate_multiplier must be positive")
    corpus_path = data_dir / corpus_filename
    query_path = data_dir / query_filename
    corpus = load_json(corpus_path)
    query_data = load_json(query_path)
    chunks = build_chunks(corpus, "structure")
    sparse = BM25Index(chunks)
    dense = RetrievalIndex(chunks, "dense_lsa", dense_components=dense_components)
    candidate_k = top_k * candidate_multiplier

    mode_names = ["bm25", "dense_lsa", "hybrid_rrf"] + [
        f"hybrid_alpha_{alpha:.2f}" for alpha in alpha_values
    ]
    rows_by_mode: dict[str, list[dict]] = {mode: [] for mode in mode_names}
    elapsed_by_mode = {mode: 0.0 for mode in mode_names}

    for query in query_data["queries"]:
        sparse_started = time.perf_counter()
        sparse_results = sparse.search(query["query"], candidate_k)
        sparse_elapsed = time.perf_counter() - sparse_started
        dense_started = time.perf_counter()
        dense_results = [
            (chunk, score)
            for chunk, score in dense.search(query["query"], candidate_k)
            if score > 0.02
        ]
        dense_elapsed = time.perf_counter() - dense_started
        has_evidence = bool(sparse_results) or bool(dense_results)

        mode_results: dict[str, tuple[list[tuple[Chunk, float]], float]] = {
            "bm25": (sparse_results, sparse_elapsed),
            "dense_lsa": (dense_results, dense_elapsed),
        }
        fusion_started = time.perf_counter()
        rrf_results = reciprocal_rank_fusion([sparse_results, dense_results])
        mode_results["hybrid_rrf"] = (
            rrf_results,
            max(sparse_elapsed, dense_elapsed) + time.perf_counter() - fusion_started,
        )
        for alpha in alpha_values:
            fusion_started = time.perf_counter()
            alpha_results = minmax_score_fusion(sparse_results, dense_results, alpha)
            mode_results[f"hybrid_alpha_{alpha:.2f}"] = (
                alpha_results,
                max(sparse_elapsed, dense_elapsed) + time.perf_counter() - fusion_started,
            )

        relevant = set(query["relevant_sections"])
        for mode, (results, elapsed) in mode_results.items():
            elapsed_by_mode[mode] += elapsed
            if not has_evidence:
                retrieved = []
            else:
                retrieved = deduplicate_sections(results, 0.0)[:top_k]
            rows_by_mode[mode].append({
                "query_id": query["id"],
                "slice": query["slice"],
                "retrieved_sections": retrieved,
                "sparse_candidate_sections": [chunk.section_id for chunk, _ in sparse_results],
                "dense_candidate_sections": [chunk.section_id for chunk, _ in dense_results],
                "recall_at_k": recall_at_k(retrieved, relevant, top_k),
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                "ndcg_at_k": ndcg_at_k(retrieved, relevant, top_k),
                "abstained": not retrieved,
                "answerable": bool(relevant),
            })

    experiments = {}
    for mode in mode_names:
        summary = _metric_summary(rows_by_mode[mode])
        summary["mean_latency_ms"] = elapsed_by_mode[mode] * 1000 / len(query_data["queries"])
        experiments[mode] = summary

    report = {
        "schema_version": "1.0",
        "corpus_version": corpus["version"],
        "query_version": query_data["version"],
        "corpus_file": corpus_filename,
        "query_file": query_filename,
        "corpus_sha256": file_sha256(corpus_path),
        "queries_sha256": file_sha256(query_path),
        "chunk_strategy": "structure",
        "dense_representation": f"lsa_{dense_components}_components",
        "top_k": top_k,
        "candidate_k": candidate_k,
        "latency_note": "Local teaching measurements; hybrid branches are modelled as concurrent.",
        "experiments": experiments,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
