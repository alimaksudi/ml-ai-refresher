"""Measured reranking over a fixed first-stage candidate set."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .evaluation import (
    Chunk,
    RetrievalIndex,
    build_chunks,
    file_sha256,
    load_json,
    ndcg_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)


class PairScorer(Protocol):
    """The minimal interface shared by teaching and neural pair scorers."""

    name: str

    def score(self, query: str, candidates: list[Chunk]) -> np.ndarray:
        """Return one query-conditioned score per candidate."""


class CharacterPairScorer:
    """Transparent offline pair scorer; deliberately not called a cross-encoder."""

    name = "character_ngram_pair_scorer"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index_by_id = {chunk.id: index for index, chunk in enumerate(chunks)}
        texts = [f"{chunk.heading} {chunk.text}" for chunk in chunks]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
        self.matrix = Normalizer().fit_transform(self.vectorizer.fit_transform(texts))

    def score(self, query: str, candidates: list[Chunk]) -> np.ndarray:
        query_vector = Normalizer().fit_transform(self.vectorizer.transform([query]))
        indices = [self.index_by_id[candidate.id] for candidate in candidates]
        product = self.matrix[indices] @ query_vector.T
        return np.asarray(product.toarray(), dtype=float).reshape(-1)


class CrossEncoderPairScorer:
    """Optional real neural cross-encoder loaded through sentence-transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        from sentence_transformers import CrossEncoder

        self.name = model_name_or_path
        self.revision = revision
        self.model = CrossEncoder(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )

    def score(self, query: str, candidates: list[Chunk]) -> np.ndarray:
        pairs = [(query, f"{candidate.heading} {candidate.text}") for candidate in candidates]
        return np.asarray(
            self.model.predict(pairs, batch_size=16, show_progress_bar=False),
            dtype=float,
        ).reshape(-1)


def _normalise(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values
    lower, upper = float(values.min()), float(values.max())
    if upper - lower <= 1e-12:
        return np.ones_like(values)
    return (values - lower) / (upper - lower)


def blend_pair_and_candidate_scores(
    pair_scores: np.ndarray,
    candidate_count: int,
    alpha: float,
) -> np.ndarray:
    """Blend normalized pair evidence with the original reciprocal candidate rank."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if len(pair_scores) != candidate_count:
        raise ValueError("pair score count must equal candidate count")
    reciprocal_positions = 1.0 / np.arange(1, candidate_count + 1, dtype=float)
    return alpha * _normalise(pair_scores) + (1 - alpha) * reciprocal_positions


def _rank_candidates(
    query: str,
    lexical: RetrievalIndex,
    dense: RetrievalIndex,
    candidate_k: int,
    evidence_threshold: float,
) -> tuple[list[Chunk], float]:
    lexical_results = lexical.search(query, candidate_k)
    dense_results = dense.search(query, candidate_k)
    evidence_strength = max(lexical_results[0][1], dense_results[0][1])
    if evidence_strength <= evidence_threshold:
        return [], evidence_strength
    fused = reciprocal_rank_fusion([lexical_results, dense_results])[:candidate_k]
    return [chunk for chunk, _ in fused], evidence_strength


def _rerank(
    query: str,
    candidates: list[Chunk],
    scorer: PairScorer,
    alpha: float,
) -> tuple[list[Chunk], list[float], float]:
    started = time.perf_counter()
    pair_scores = scorer.score(query, candidates)
    blended = blend_pair_and_candidate_scores(pair_scores, len(candidates), alpha)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (-blended[index], candidates[index].id),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return (
        [candidates[index] for index in order],
        [float(blended[index]) for index in order],
        elapsed_ms,
    )


def _metrics(rows: list[dict], ranking_key: str, top_k: int) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    return {
        "candidate_passage_recall": float(np.mean([
            row["candidate_passage_recall"] for row in answerable
        ])),
        "candidate_hit_rate": float(np.mean([
            row["candidate_hit"] for row in answerable
        ])),
        "mrr": float(np.mean([
            reciprocal_rank(row[ranking_key], set(row["relevant_passages"]))
            for row in answerable
        ])),
        "ndcg_at_k": float(np.mean([
            ndcg_at_k(row[ranking_key], set(row["relevant_passages"]), top_k)
            for row in answerable
        ])),
        "top_1_accuracy": float(np.mean([
            bool(row[ranking_key])
            and row[ranking_key][0] in set(row["relevant_passages"])
            for row in answerable
        ])),
        "unanswerable_abstention_rate": float(np.mean([
            not row[ranking_key] for row in unanswerable
        ])),
    }


def _evaluate_scorer(
    labelled_queries: list[dict],
    query_by_id: dict[str, dict],
    lexical: RetrievalIndex,
    dense: RetrievalIndex,
    scorer: PairScorer,
    alpha: float,
    candidate_k: int,
    evidence_threshold: float,
) -> list[dict]:
    rows = []
    for label in labelled_queries:
        query_record = query_by_id[label["query_id"]]
        candidates, evidence_strength = _rank_candidates(
            query_record["query"], lexical, dense, candidate_k, evidence_threshold
        )
        baseline_ids = [candidate.id for candidate in candidates]
        if candidates:
            reranked, scores, latency_ms = _rerank(
                query_record["query"], candidates, scorer, alpha
            )
            reranked_ids = [candidate.id for candidate in reranked]
        else:
            scores, latency_ms, reranked_ids = [], 0.0, []
        relevant = set(label["relevant_passages"])
        rows.append({
            "query_id": label["query_id"],
            "split": label["split"],
            "answerable": bool(relevant),
            "relevant_passages": label["relevant_passages"],
            "evidence_strength": float(evidence_strength),
            "candidate_passages": baseline_ids,
            "candidate_passage_recall": (
                len(set(baseline_ids) & relevant) / len(relevant) if relevant
                else float(not baseline_ids)
            ),
            "candidate_hit": bool(set(baseline_ids) & relevant) if relevant else False,
            "baseline_ranking": baseline_ids,
            "reranked_ranking": reranked_ids,
            "reranked_scores": scores,
            "rerank_latency_ms": latency_ms,
        })
    return rows


def evaluate_reranking(
    data_dir: Path,
    output_path: Path | None = None,
    candidate_k: int = 15,
    top_k: int = 5,
    evidence_threshold: float = 0.02,
    alpha_grid: tuple[float, ...] = tuple(step / 10 for step in range(11)),
    neural_model: str | None = None,
    neural_revision: str | None = None,
    local_files_only: bool = False,
) -> dict:
    """Tune on development labels, then compare fixed candidates on held-out queries."""
    if candidate_k < top_k or top_k < 1:
        raise ValueError("candidate_k must be at least top_k, and top_k must be positive")
    corpus_path = data_dir / "corpus.json"
    queries_path = data_dir / "queries.json"
    labels_path = data_dir / "reranking_queries.json"
    corpus = load_json(corpus_path)
    query_data = load_json(queries_path)
    label_data = load_json(labels_path)
    query_by_id = {query["id"]: query for query in query_data["queries"]}
    labels = label_data["queries"]
    if {label["query_id"] for label in labels} != set(query_by_id):
        raise ValueError("reranking labels must cover every canonical query exactly once")

    chunks = build_chunks(corpus, label_data["chunk_strategy"])
    chunk_ids = {chunk.id for chunk in chunks}
    for label in labels:
        if not set(label["relevant_passages"]).issubset(chunk_ids):
            raise ValueError(f"unknown passage label for {label['query_id']}")

    lexical = RetrievalIndex(chunks, "lexical")
    dense = RetrievalIndex(chunks, "dense_lsa")
    local_scorer = CharacterPairScorer(chunks)
    development = [label for label in labels if label["split"] == "development"]
    evaluation = [label for label in labels if label["split"] == "evaluation"]

    tuning = []
    for alpha in alpha_grid:
        rows = _evaluate_scorer(
            development, query_by_id, lexical, dense, local_scorer, float(alpha),
            candidate_k, evidence_threshold,
        )
        tuning.append({
            "alpha": float(alpha),
            "development_mrr": _metrics(rows, "reranked_ranking", top_k)["mrr"],
        })
    selected_alpha = max(tuning, key=lambda item: (item["development_mrr"], -item["alpha"]))["alpha"]

    systems = {}
    for split_name, split_labels in (("development", development), ("evaluation", evaluation)):
        rows = _evaluate_scorer(
            split_labels, query_by_id, lexical, dense, local_scorer, selected_alpha,
            candidate_k, evidence_threshold,
        )
        systems[split_name] = {
            "baseline_metrics": _metrics(rows, "baseline_ranking", top_k),
            "local_pair_reranker_metrics": _metrics(rows, "reranked_ranking", top_k),
            "mean_local_rerank_latency_ms": float(np.mean([
                row["rerank_latency_ms"] for row in rows
            ])),
            "rows": rows,
        }

    neural = None
    if neural_model:
        scorer = CrossEncoderPairScorer(
            neural_model,
            revision=neural_revision,
            local_files_only=local_files_only,
        )
        neural_rows = _evaluate_scorer(
            evaluation, query_by_id, lexical, dense, scorer, 1.0,
            candidate_k, evidence_threshold,
        )
        neural = {
            "model": neural_model,
            "revision": neural_revision,
            "local_files_only": local_files_only,
            "metrics": _metrics(neural_rows, "reranked_ranking", top_k),
            "mean_rerank_latency_ms": float(np.mean([
                row["rerank_latency_ms"] for row in neural_rows
            ])),
            "rows": neural_rows,
        }

    report = {
        "schema_version": "1.0",
        "corpus_sha256": file_sha256(corpus_path),
        "queries_sha256": file_sha256(queries_path),
        "reranking_labels_sha256": file_sha256(labels_path),
        "chunk_strategy": label_data["chunk_strategy"],
        "candidate_system": "hybrid_rrf",
        "candidate_k": candidate_k,
        "top_k": top_k,
        "evidence_threshold": evidence_threshold,
        "local_pair_scorer": {
            "name": local_scorer.name,
            "note": "Character n-gram TF-IDF pair similarity is a transparent teaching reranker, not a neural cross-encoder.",
            "tuning_contract": "Alpha is selected only on development labels; evaluation labels are held out until final scoring.",
            "alpha_grid": tuning,
            "selected_alpha": selected_alpha,
        },
        "metric_contract": {
            "candidate_passage_recall": "Fraction of labelled passages present before reranking; this is the reranker's ceiling.",
            "mrr": "Reciprocal rank of the first labelled passage, averaged over answerable queries.",
            "ndcg_at_k": "Position-discounted gain for labelled passages in the first k ranks.",
            "top_1_accuracy": "Share of answerable queries with a labelled passage at rank one.",
        },
        "latency_note": "Local teaching measurements on a tiny corpus; not throughput, capacity, or tail-latency evidence.",
        "systems": systems,
        "neural_cross_encoder": neural,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
