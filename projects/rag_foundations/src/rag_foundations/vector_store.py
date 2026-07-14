"""Persistent local vector storage and measured exact-search parity evaluation."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from qdrant_client import QdrantClient, models
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .evaluation import (
    Chunk,
    build_chunks,
    file_sha256,
    load_json,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any]


class LsaEncoder:
    """Small deterministic dense encoder used by the offline curriculum project."""

    def __init__(self, max_components: int = 32, seed: int = 42):
        self.max_components = max_components
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
        )
        self.svd: TruncatedSVD | None = None
        self.normalizer = Normalizer()

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        sparse = self.vectorizer.fit_transform(texts)
        components = min(self.max_components, sparse.shape[0] - 1, sparse.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=max(2, components), random_state=self.seed)
        dense = self.svd.fit_transform(sparse)
        return np.asarray(self.normalizer.fit_transform(dense), dtype=np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("fit the encoder before transforming text")
        dense = self.svd.transform(self.vectorizer.transform(texts))
        return np.asarray(self.normalizer.transform(dense), dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self.svd is None:
            raise RuntimeError("fit the encoder before reading its dimension")
        return int(self.svd.n_components)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "LsaEncoder":
        encoder = joblib.load(path)
        if not isinstance(encoder, cls):
            raise TypeError(f"expected {cls.__name__}, found {type(encoder).__name__}")
        return encoder


def _point_id(record_id: str) -> int:
    """Map stable string IDs to Qdrant's supported positive integer ID space."""
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class QdrantVectorStore:
    """Thin Qdrant-local adapter with mandatory safety metadata filters."""

    def __init__(self, path: Path, collection: str, dimension: int, reset: bool = False):
        self.path = path
        self.collection = collection
        self.dimension = dimension
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path / "qdrant"))
        exists = self.client.collection_exists(collection)
        if reset and exists:
            self.client.delete_collection(collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
        configured_dimension = self.client.get_collection(collection).config.params.vectors.size
        if configured_dimension != dimension:
            self.close()
            raise ValueError(
                f"collection dimension {configured_dimension} does not match encoder {dimension}"
            )

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        records = list(records)
        points = []
        seen_point_ids: dict[int, str] = {}
        for record in records:
            point_id = _point_id(record.id)
            if point_id in seen_point_ids and seen_point_ids[point_id] != record.id:
                raise ValueError("stable point ID collision")
            seen_point_ids[point_id] = record.id
            payload = {
                "record_id": record.id,
                "access_level": "public",
                "current": True,
                "unsafe": False,
                **record.metadata,
            }
            points.append(models.PointStruct(id=point_id, vector=record.vector, payload=payload))
        if points:
            self.client.upsert(self.collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector: np.ndarray | list[float],
        k: int = 5,
        user_access: str = "public",
        current_only: bool = True,
        include_unsafe: bool = False,
        score_threshold: float | None = None,
        exact: bool = True,
    ) -> list[SearchResult]:
        must: list[models.Condition] = []
        if user_access != "restricted":
            must.append(models.FieldCondition(
                key="access_level", match=models.MatchValue(value="public")
            ))
        if current_only:
            must.append(models.FieldCondition(
                key="current", match=models.MatchValue(value=True)
            ))
        if not include_unsafe:
            must.append(models.FieldCondition(
                key="unsafe", match=models.MatchValue(value=False)
            ))
        response = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            query_filter=models.Filter(must=must),
            search_params=models.SearchParams(exact=exact),
            limit=k,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return [SearchResult(
            id=str(point.payload["record_id"]),
            score=float(point.score),
            metadata=dict(point.payload),
        ) for point in response.points]

    def delete(self, record_ids: Iterable[str]) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(
                points=[_point_id(record_id) for record_id in record_ids]
            ),
            wait=True,
        )

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def close(self) -> None:
        self.client.close()


def _chunk_records(chunks: list[Chunk], vectors: np.ndarray) -> list[VectorRecord]:
    return [VectorRecord(
        id=chunk.id,
        vector=vector.tolist(),
        metadata={
            "document_id": chunk.document_id,
            "section_id": chunk.section_id,
            "heading": chunk.heading,
            "strategy": chunk.strategy,
        },
    ) for chunk, vector in zip(chunks, vectors)]


def _section_ids(results: list[SearchResult], k: int) -> list[str]:
    sections: list[str] = []
    for result in results:
        section_id = str(result.metadata["section_id"])
        if section_id not in sections:
            sections.append(section_id)
        if len(sections) == k:
            break
    return sections


def _summary(rows: list[dict[str, Any]], elapsed: float) -> dict[str, float]:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    return {
        "recall_at_k": float(np.mean([row["recall_at_k"] for row in answerable])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in answerable])),
        "ndcg_at_k": float(np.mean([row["ndcg_at_k"] for row in answerable])),
        "answerable_zero_result_rate": float(np.mean([row["abstained"] for row in answerable])),
        "unanswerable_abstention_rate": float(np.mean([row["abstained"] for row in unanswerable])),
        "mean_latency_ms": elapsed * 1000 / len(rows),
    }


def _evaluate_rows(
    query_data: dict[str, Any],
    search: Any,
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    rows, elapsed = [], 0.0
    for query in query_data["queries"]:
        started = time.perf_counter()
        retrieved = search(query["query"])
        elapsed += time.perf_counter() - started
        relevant = set(query["relevant_sections"])
        rows.append({
            "query_id": query["id"],
            "slice": query["slice"],
            "retrieved_sections": retrieved,
            "recall_at_k": recall_at_k(retrieved, relevant, top_k),
            "reciprocal_rank": reciprocal_rank(retrieved, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, top_k),
            "abstained": not retrieved,
            "answerable": bool(relevant),
        })
    return rows, elapsed


def policy_filter_check(path: Path) -> dict[str, Any]:
    """Verify public queries cannot retrieve stale, unsafe, or restricted records."""
    store = QdrantVectorStore(path, "policy_check", dimension=2, reset=True)
    records = [
        VectorRecord("public", [1.0, 0.0], {"access_level": "public", "current": True, "unsafe": False}),
        VectorRecord("stale", [1.0, 0.0], {"access_level": "public", "current": False, "unsafe": False}),
        VectorRecord("unsafe", [1.0, 0.0], {"access_level": "public", "current": True, "unsafe": True}),
        VectorRecord("restricted", [1.0, 0.0], {"access_level": "restricted", "current": True, "unsafe": False}),
    ]
    store.upsert(records)
    public_ids = [item.id for item in store.search([1.0, 0.0], k=10)]
    restricted_ids = [item.id for item in store.search([1.0, 0.0], k=10, user_access="restricted")]
    store.close()
    checks = {
        "public_visible": public_ids == ["public"],
        "stale_blocked": "stale" not in public_ids,
        "unsafe_blocked": "unsafe" not in public_ids,
        "restricted_blocked_for_public": "restricted" not in public_ids,
        "restricted_visible_when_authorized": "restricted" in restricted_ids,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "public_results": public_ids, "restricted_results": restricted_ids}


def evaluate_vector_store(
    data_dir: Path,
    index_dir: Path,
    output_path: Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build, persist, reopen, and compare Qdrant-local with exact NumPy search."""
    corpus_path, query_path = data_dir / "corpus.json", data_dir / "queries.json"
    corpus, query_data = load_json(corpus_path), load_json(query_path)
    chunks = build_chunks(corpus, "sentence")
    encoder = LsaEncoder()
    matrix = encoder.fit_transform([chunk.text for chunk in chunks])
    encoder_path = index_dir / "encoder.joblib"
    encoder.save(encoder_path)

    store = QdrantVectorStore(index_dir, "curriculum", encoder.dimension, reset=True)
    records = _chunk_records(chunks, matrix)
    store.upsert(records)
    first_count = store.count()
    store.upsert(records)
    duplicate_count = store.count()

    def exact_search(query: str) -> list[str]:
        query_vector = encoder.transform([query])[0]
        scores = matrix @ query_vector
        order = np.argsort(-scores, kind="stable")
        results = []
        for index in order:
            if scores[index] <= 0.02:
                continue
            section_id = chunks[int(index)].section_id
            if section_id not in results:
                results.append(section_id)
            if len(results) == top_k:
                break
        return results

    def qdrant_search(query: str) -> list[str]:
        query_vector = encoder.transform([query])[0]
        return _section_ids(store.search(
            query_vector, k=top_k * 3, score_threshold=0.02, exact=True
        ), top_k)

    exact_rows, exact_elapsed = _evaluate_rows(query_data, exact_search, top_k)
    qdrant_rows, qdrant_elapsed = _evaluate_rows(query_data, qdrant_search, top_k)
    first_query_vector = encoder.transform([query_data["queries"][0]["query"]])[0]
    before_restart = [item.id for item in store.search(first_query_vector, k=top_k, exact=True)]
    store.close()

    loaded_encoder = LsaEncoder.load(encoder_path)
    reopened = QdrantVectorStore(index_dir, "curriculum", loaded_encoder.dimension)
    after_restart = [item.id for item in reopened.search(
        loaded_encoder.transform([query_data["queries"][0]["query"]])[0], k=top_k, exact=True
    )]
    reopened.close()

    overlaps = []
    exact_matches = []
    for exact_row, qdrant_row in zip(exact_rows, qdrant_rows):
        exact_ids, qdrant_ids = exact_row["retrieved_sections"], qdrant_row["retrieved_sections"]
        if not exact_ids and not qdrant_ids:
            overlaps.append(1.0)
        else:
            denominator = max(1, len(exact_ids))
            overlaps.append(len(set(exact_ids) & set(qdrant_ids)) / denominator)
        exact_matches.append(exact_ids == qdrant_ids)

    policy = policy_filter_check(index_dir)
    manifest = {
        "schema_version": "1.0",
        "corpus_sha256": file_sha256(corpus_path),
        "queries_sha256": file_sha256(query_path),
        "strategy": "sentence",
        "encoder": "tfidf_lsa",
        "dimension": encoder.dimension,
        "points": first_count,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    report = {
        **manifest,
        "top_k": top_k,
        "numpy_exact": {**_summary(exact_rows, exact_elapsed), "rows": exact_rows},
        "qdrant_local_exact": {**_summary(qdrant_rows, qdrant_elapsed), "rows": qdrant_rows},
        "parity": {
            "mean_section_overlap_at_k": float(np.mean(overlaps)),
            "exact_ranking_match_rate": float(np.mean(exact_matches)),
        },
        "persistence": {
            "restart_result_match": before_restart == after_restart,
            "duplicate_upsert_idempotent": first_count == duplicate_count,
            "count_before_duplicate_upsert": first_count,
            "count_after_duplicate_upsert": duplicate_count,
        },
        "policy_filters": policy,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
