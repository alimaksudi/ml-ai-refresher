from __future__ import annotations

import json
from pathlib import Path

from rag_foundations.evaluation import build_chunks, evaluate, load_json
from rag_foundations.grounded import ABSTENTION, evaluate_answers, evaluate_security

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_dataset_contract_and_evidence_ids_are_valid():
    corpus = load_json(DATA / "corpus.json")
    queries = load_json(DATA / "queries.json")
    section_ids = [section["id"] for doc in corpus["documents"] for section in doc["sections"]]
    assert len(section_ids) == len(set(section_ids))
    assert len(queries["queries"]) >= 15
    assert any(not query["relevant_sections"] for query in queries["queries"])
    for query in queries["queries"]:
        assert set(query["relevant_sections"]).issubset(section_ids)


def test_every_chunk_preserves_source_provenance():
    corpus = load_json(DATA / "corpus.json")
    for strategy in ("fixed", "sentence", "structure"):
        chunks = build_chunks(corpus, strategy)
        assert chunks
        assert all(chunk.document_id and chunk.section_id and chunk.id for chunk in chunks)
        assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_evaluation_is_deterministic_and_meets_baseline(tmp_path):
    first = evaluate(DATA, tmp_path / "first.json")
    second = evaluate(DATA, tmp_path / "second.json")
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["queries_sha256"] == second["queries_sha256"]
    for key in first["experiments"]:
        for metric in ("recall_at_k", "mrr", "ndcg_at_k", "answerable_zero_result_rate", "unanswerable_abstention_rate"):
            assert first["experiments"][key][metric] == second["experiments"][key][metric]
    best_recall = max(item["recall_at_k"] for item in first["experiments"].values())
    assert best_recall >= 0.90
    assert min(item["unanswerable_abstention_rate"] for item in first["experiments"].values()) >= 0.50
    saved = json.loads((tmp_path / "first.json").read_text())
    assert saved["schema_version"] == "1.0"


def test_report_contains_failure_slices_and_component_rows():
    report = evaluate(DATA)
    for experiment in report["experiments"].values():
        assert {"direct", "paraphrase", "multi_concept"}.issubset(experiment["slices"])
        assert len(experiment["rows"]) == 18
        assert experiment["mean_latency_ms"] >= 0


def test_gold_answers_cover_queries_and_grounded_report_is_component_level(tmp_path):
    queries = load_json(DATA / "queries.json")["queries"]
    answers = load_json(DATA / "answers.json")["answers"]
    assert {item["id"] for item in queries} == {item["query_id"] for item in answers}
    report = evaluate_answers(DATA, tmp_path / "grounded.json")
    assert report["metrics"]["retrieval_recall_at_k"] >= 0.90
    assert report["metrics"]["evidence_support"] == 1.0
    assert report["metrics"]["citation_validity"] == 1.0
    assert report["metrics"]["answer_correctness"] >= 0.40
    assert sum(report["failure_counts"].values()) == len(queries)
    assert any(row["outcome"] == "retrieval_failure" for row in report["rows"])
    assert any(row["outcome"] == "answer_failure" for row in report["rows"])


def test_unanswerable_answers_include_a_real_abstention():
    report = evaluate_answers(DATA)
    unanswerable = [row for row in report["rows"] if not row["answerable"]]
    assert unanswerable
    assert any(row["answer"] == ABSTENTION for row in unanswerable)


def test_security_cases_block_stale_unsafe_and_unauthorized_evidence():
    report = evaluate_security(DATA)
    assert report["pass_rate"] == 1.0
    outcomes = {row["outcome"] for row in report["rows"]}
    assert {"answer", "abstain_stale", "abstain_unsafe", "abstain_unauthorized"}.issubset(outcomes)
