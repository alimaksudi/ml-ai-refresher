"""Local, deterministic retrieval baselines and evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import Normalizer


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    section_id: str
    heading: str
    text: str
    strategy: str


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def build_chunks(corpus: dict, strategy: str, words: int = 18) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in corpus["documents"]:
        for section in document["sections"]:
            base = f"{section['heading']}. {section['text']}"
            if strategy == "structure":
                parts = [base]
            elif strategy == "sentence":
                parts = _sentences(base)
            elif strategy == "fixed":
                tokens = base.split()
                step = max(1, words - 4)
                parts = [" ".join(tokens[start : start + words]) for start in range(0, len(tokens), step)]
            else:
                raise ValueError(f"unknown chunk strategy: {strategy}")
            for index, text in enumerate(parts):
                chunks.append(Chunk(
                    id=f"{section['id']}::{strategy}::{index}",
                    document_id=document["id"],
                    section_id=section["id"],
                    heading=section["heading"],
                    text=text,
                    strategy=strategy,
                ))
    return chunks


class RetrievalIndex:
    def __init__(self, chunks: list[Chunk], mode: str, seed: int = 42):
        self.chunks = chunks
        self.mode = mode
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        sparse = self.vectorizer.fit_transform([chunk.text for chunk in chunks])
        self.svd = None
        if mode == "lexical":
            self.matrix = Normalizer().fit_transform(sparse)
        elif mode == "dense_lsa":
            components = min(32, sparse.shape[0] - 1, sparse.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=max(2, components), random_state=seed)
            self.matrix = Normalizer().fit_transform(self.svd.fit_transform(sparse))
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

    def scores(self, query: str) -> np.ndarray:
        vector = self.vectorizer.transform([query])
        if self.svd is not None:
            vector = self.svd.transform(vector)
        vector = Normalizer().fit_transform(vector)
        product = self.matrix @ vector.T
        if hasattr(product, "toarray"):
            product = product.toarray()
        return np.asarray(product, dtype=float).reshape(-1)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
        return [(self.chunks[index], float(scores[index])) for index in order]


class BM25Index:
    """Small deterministic BM25 index used by the hybrid-search lesson."""

    TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25Index requires at least one chunk")
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [self.tokenize(f"{chunk.heading} {chunk.text}") for chunk in chunks]
        self.term_frequencies = [self._term_frequency(tokens) for tokens in self.tokens]
        self.document_frequency: dict[str, int] = {}
        for tokens in self.tokens:
            for term in set(tokens):
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
        self.average_length = float(np.mean([len(tokens) for tokens in self.tokens]))

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return [
            token for token in cls.TOKEN_PATTERN.findall(text.lower())
            if token not in ENGLISH_STOP_WORDS
        ]

    @staticmethod
    def _term_frequency(tokens: list[str]) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def inverse_document_frequency(self, term: str) -> float:
        document_count = len(self.chunks)
        matching_documents = self.document_frequency.get(term, 0)
        return math.log(
            (document_count - matching_documents + 0.5) / (matching_documents + 0.5) + 1
        )

    def scores(self, query: str) -> np.ndarray:
        query_terms = self.tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=float)
        for index, frequencies in enumerate(self.term_frequencies):
            document_length = len(self.tokens[index])
            length_factor = 1 - self.b + self.b * document_length / self.average_length
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                saturated_frequency = (
                    term_frequency * (self.k1 + 1)
                    / (term_frequency + self.k1 * length_factor)
                )
                scores[index] += self.inverse_document_frequency(term) * saturated_frequency
        return scores

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")
        return [
            (self.chunks[index], float(scores[index]))
            for index in order
            if scores[index] > 0
        ][: min(k, len(scores))]


def reciprocal_rank_fusion(rankings: list[list[tuple[Chunk, float]]], k: int = 60) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, 1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks[chunk.id] = chunk
    return [(chunks[key], value) for key, value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, section_id in enumerate(retrieved, 1):
        if section_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(retrieved[:k], 1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


def deduplicate_sections(results: list[tuple[Chunk, float]], threshold: float = 0.0) -> list[str]:
    output = []
    for chunk, score in results:
        if score <= threshold:
            continue
        if chunk.section_id not in output:
            output.append(chunk.section_id)
    return output


def evaluate(data_dir: Path, output_path: Path | None = None, top_k: int = 5) -> dict:
    corpus_path, query_path = data_dir / "corpus.json", data_dir / "queries.json"
    corpus, query_data = load_json(corpus_path), load_json(query_path)
    report = {
        "schema_version": "1.0",
        "corpus_version": corpus["version"],
        "query_version": query_data["version"],
        "corpus_sha256": file_sha256(corpus_path),
        "queries_sha256": file_sha256(query_path),
        "top_k": top_k,
        "experiments": {},
    }
    for strategy in ("fixed", "sentence", "structure"):
        chunks = build_chunks(corpus, strategy)
        lexical = RetrievalIndex(chunks, "lexical")
        dense = RetrievalIndex(chunks, "dense_lsa")
        for mode in ("lexical", "dense_lsa", "hybrid_rrf"):
            rows, elapsed = [], 0.0
            for query in query_data["queries"]:
                started = time.perf_counter()
                lex_results = lexical.search(query["query"], top_k * 3)
                dense_results = dense.search(query["query"], top_k * 3)
                if mode == "lexical":
                    results = lex_results
                    threshold = 0.02
                elif mode == "dense_lsa":
                    results = dense_results
                    threshold = 0.02
                else:
                    results = reciprocal_rank_fusion([lex_results, dense_results])
                    threshold = 0.0
                elapsed += time.perf_counter() - started
                # Abstain when neither base representation has meaningful evidence.
                if max(lex_results[0][1], dense_results[0][1]) <= 0.02:
                    retrieved = []
                else:
                    retrieved = deduplicate_sections(results, threshold)[:top_k]
                relevant = set(query["relevant_sections"])
                rows.append({
                    "query_id": query["id"], "slice": query["slice"],
                    "retrieved_sections": retrieved,
                    "recall_at_k": recall_at_k(retrieved, relevant, top_k),
                    "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                    "ndcg_at_k": ndcg_at_k(retrieved, relevant, top_k),
                    "abstained": not retrieved,
                    "answerable": bool(relevant),
                })
            answerable = [row for row in rows if row["answerable"]]
            unanswerable = [row for row in rows if not row["answerable"]]
            by_slice = {}
            for slice_name in sorted({row["slice"] for row in answerable}):
                subset = [row for row in answerable if row["slice"] == slice_name]
                by_slice[slice_name] = {"count": len(subset), "recall_at_k": float(np.mean([r["recall_at_k"] for r in subset]))}
            report["experiments"][f"{strategy}:{mode}"] = {
                "chunks": len(chunks),
                "mean_latency_ms": elapsed * 1000 / len(rows),
                "recall_at_k": float(np.mean([row["recall_at_k"] for row in answerable])),
                "mrr": float(np.mean([row["reciprocal_rank"] for row in answerable])),
                "ndcg_at_k": float(np.mean([row["ndcg_at_k"] for row in answerable])),
                "answerable_zero_result_rate": float(np.mean([row["abstained"] for row in answerable])),
                "unanswerable_abstention_rate": float(np.mean([row["abstained"] for row in unanswerable])),
                "slices": by_slice,
                "rows": rows,
            }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
