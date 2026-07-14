"""Deterministic extractive answering and component-level RAG evaluation."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .evaluation import (
    Chunk,
    RetrievalIndex,
    build_chunks,
    file_sha256,
    load_json,
    recall_at_k,
    reciprocal_rank_fusion,
)


ABSTENTION = "I cannot answer from the available authorized evidence."


@dataclass(frozen=True)
class GroundedResponse:
    query_id: str
    answer: str
    citations: list[str]
    retrieved_sections: list[str]
    abstained: bool
    latency_ms: float


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def term_correctness(answer: str, required_terms: list[str]) -> float:
    if not required_terms:
        return 1.0 if answer == ABSTENTION else 0.0
    normalised = _normalise(answer)
    return float(all(_normalise(term) in normalised for term in required_terms))


def answer_is_supported(answer: str, cited_chunks: list[Chunk]) -> float:
    if answer == ABSTENTION:
        return 1.0
    evidence = " ".join(chunk.text for chunk in cited_chunks)
    return float(_normalise(answer) in _normalise(evidence))


def _rank(query: str, lexical: RetrievalIndex, dense: RetrievalIndex, mode: str, limit: int):
    lexical_results = lexical.search(query, limit)
    dense_results = dense.search(query, limit)
    if mode == "lexical":
        return lexical_results, lexical_results, dense_results
    if mode == "dense_lsa":
        return dense_results, lexical_results, dense_results
    if mode == "hybrid_rrf":
        return reciprocal_rank_fusion([lexical_results, dense_results]), lexical_results, dense_results
    raise ValueError(f"unknown mode: {mode}")


def answer_query(
    query_id: str,
    query: str,
    lexical: RetrievalIndex,
    dense: RetrievalIndex,
    mode: str,
    top_k: int = 5,
    threshold: float = 0.02,
) -> GroundedResponse:
    started = time.perf_counter()
    ranked, lexical_results, dense_results = _rank(query, lexical, dense, mode, top_k * 3)
    evidence_strength = max(lexical_results[0][1], dense_results[0][1])
    unique: list[tuple[Chunk, float]] = []
    seen = set()
    for chunk, score in ranked:
        if chunk.section_id not in seen:
            unique.append((chunk, score)); seen.add(chunk.section_id)
        if len(unique) == top_k:
            break
    if evidence_strength <= threshold or not unique:
        answer, citations, retrieved = ABSTENTION, [], []
    else:
        # The first baseline is deliberately extractive: answer and citation cannot
        # drift apart through free-form generation.
        answer = unique[0][0].text
        citations = [unique[0][0].section_id]
        retrieved = [chunk.section_id for chunk, _ in unique]
    return GroundedResponse(
        query_id=query_id,
        answer=answer,
        citations=citations,
        retrieved_sections=retrieved,
        abstained=answer == ABSTENTION,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def evaluate_answers(
    data_dir: Path,
    output_path: Path | None = None,
    strategy: str = "sentence",
    mode: str = "dense_lsa",
    top_k: int = 5,
) -> dict:
    corpus_path = data_dir / "corpus.json"
    query_path = data_dir / "queries.json"
    answer_path = data_dir / "answers.json"
    corpus, query_data, answer_data = load_json(corpus_path), load_json(query_path), load_json(answer_path)
    answer_by_id = {item["query_id"]: item for item in answer_data["answers"]}
    if set(answer_by_id) != {item["id"] for item in query_data["queries"]}:
        raise ValueError("answers.json must cover every query exactly once")
    chunks = build_chunks(corpus, strategy)
    lexical, dense = RetrievalIndex(chunks, "lexical"), RetrievalIndex(chunks, "dense_lsa")
    rows = []
    for query in query_data["queries"]:
        response = answer_query(query["id"], query["query"], lexical, dense, mode, top_k)
        gold = answer_by_id[query["id"]]
        relevant = set(query["relevant_sections"])
        retrieved_recall = recall_at_k(response.retrieved_sections, relevant, top_k)
        cited_chunks = [chunk for chunk in chunks if chunk.section_id in response.citations]
        citation_valid = float(all(citation in response.retrieved_sections for citation in response.citations))
        supported = answer_is_supported(response.answer, cited_chunks)
        correctness = term_correctness(response.answer, gold["required_terms"])
        answerable = bool(relevant)
        abstention_correct = float(response.abstained != answerable)
        if answerable and retrieved_recall == 0:
            failure = "retrieval_failure"
        elif answerable and response.abstained:
            failure = "abstention_failure"
        elif not answerable and not response.abstained:
            failure = "abstention_failure"
        elif correctness == 0:
            failure = "answer_failure"
        elif citation_valid == 0:
            failure = "attribution_failure"
        elif supported == 0:
            failure = "grounding_failure"
        else:
            failure = "success"
        rows.append({
            "query_id": query["id"], "slice": query["slice"], "answerable": answerable,
            "answer": response.answer, "citations": response.citations,
            "retrieved_sections": response.retrieved_sections,
            "retrieval_recall_at_k": retrieved_recall,
            "answer_correctness": correctness,
            "evidence_support": supported,
            "citation_validity": citation_valid,
            "abstention_correct": abstention_correct,
            "latency_ms": response.latency_ms,
            "outcome": failure,
        })
    failure_counts = {name: sum(row["outcome"] == name for row in rows) for name in (
        "success", "retrieval_failure", "answer_failure", "attribution_failure",
        "grounding_failure", "abstention_failure"
    )}
    report = {
        "schema_version": "1.0", "strategy": strategy, "mode": mode, "top_k": top_k,
        "corpus_sha256": file_sha256(corpus_path),
        "queries_sha256": file_sha256(query_path),
        "answers_sha256": file_sha256(answer_path),
        "metrics": {
            "retrieval_recall_at_k": float(np.mean([row["retrieval_recall_at_k"] for row in rows if row["answerable"]])),
            "answer_correctness": float(np.mean([row["answer_correctness"] for row in rows])),
            "evidence_support": float(np.mean([row["evidence_support"] for row in rows])),
            "citation_validity": float(np.mean([row["citation_validity"] for row in rows])),
            "abstention_accuracy": float(np.mean([row["abstention_correct"] for row in rows])),
            "mean_latency_ms": float(np.mean([row["latency_ms"] for row in rows])),
        },
        "failure_counts": failure_counts,
        "rows": rows,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def evaluate_security(data_dir: Path) -> dict:
    payload = load_json(data_dir / "security_cases.json")
    sections = payload["sections"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([f"{item['heading']} {item['text']}" for item in sections])
    rows = []
    for case in payload["cases"]:
        scores = (matrix @ vectorizer.transform([case["query"]]).T).toarray().reshape(-1)
        section = sections[int(np.argmax(scores))]
        if not section.get("current", True):
            outcome, citation = "abstain_stale", None
        elif section.get("unsafe_instruction", False):
            outcome, citation = "abstain_unsafe", None
        elif section.get("access_level", "public") == "restricted" and case["user_access"] != "restricted":
            outcome, citation = "abstain_unauthorized", None
        else:
            outcome, citation = "answer", section["id"]
        rows.append({"case_id": case["id"], "outcome": outcome, "citation": citation,
                     "passed": outcome == case["expected_outcome"] and citation == case["expected_section"]})
    return {"version": payload["version"], "pass_rate": float(np.mean([row["passed"] for row in rows])), "rows": rows}
