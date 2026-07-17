from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag_foundations.evaluation import build_chunks, evaluate, load_json
from rag_foundations.grounded import ABSTENTION, evaluate_answers, evaluate_security
from rag_foundations.hybrid import evaluate_hybrid, minmax_score_fusion
from rag_foundations.rag_evaluation import (
    apply_quality_gate,
    context_precision_at_k,
    evaluate_rag_systems,
)
from rag_foundations.reranking import (
    blend_pair_and_candidate_scores,
    evaluate_reranking,
)
from rag_foundations.vector_store import (
    QdrantVectorStore,
    VectorRecord,
    evaluate_vector_store,
)

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


def test_hybrid_report_compares_branches_on_the_same_labels(tmp_path):
    first = evaluate_hybrid(DATA, tmp_path / "hybrid-first.json")
    second = evaluate_hybrid(DATA, tmp_path / "hybrid-second.json")
    assert first["candidate_k"] == first["top_k"] * 3
    assert first["corpus_file"] == "hybrid_corpus.json"
    assert first["query_file"] == "hybrid_queries.json"
    assert first["dense_representation"] == "lsa_4_components"
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["queries_sha256"] == second["queries_sha256"]
    expected_modes = {
        "bm25", "dense_lsa", "hybrid_rrf",
        "hybrid_alpha_0.00", "hybrid_alpha_0.25", "hybrid_alpha_0.50",
        "hybrid_alpha_0.75", "hybrid_alpha_1.00",
    }
    assert set(first["experiments"]) == expected_modes
    for mode in expected_modes:
        experiment = first["experiments"][mode]
        assert len(experiment["rows"]) == 5
        assert {
            "exact_identifier", "paraphrase", "mixed_intent", "no_gain_control"
        }.issubset(experiment["slices"])
        for metric in ("recall_at_k", "mrr", "ndcg_at_k"):
            assert experiment[metric] == second["experiments"][mode][metric]
            assert 0 <= experiment[metric] <= 1
        for row in experiment["rows"]:
            assert len(row["retrieved_sections"]) == len(set(row["retrieved_sections"]))
            assert isinstance(row["sparse_candidate_sections"], list)
            assert isinstance(row["dense_candidate_sections"], list)
            if not row["sparse_candidate_sections"] and not row["dense_candidate_sections"]:
                assert row["abstained"]

    rows = {
        mode: {row["query_id"]: row for row in experiment["rows"]}
        for mode, experiment in first["experiments"].items()
    }
    assert rows["bm25"]["h01"]["reciprocal_rank"] > rows["dense_lsa"]["h01"]["reciprocal_rank"]
    assert rows["dense_lsa"]["h02"]["reciprocal_rank"] > rows["bm25"]["h02"]["reciprocal_rank"]
    assert rows["hybrid_rrf"]["h03"]["recall_at_k"] == 1.0
    assert rows["hybrid_rrf"]["h03"]["recall_at_k"] > rows["bm25"]["h03"]["recall_at_k"]
    assert rows["hybrid_rrf"]["h03"]["recall_at_k"] > rows["dense_lsa"]["h03"]["recall_at_k"]
    assert all(rows[mode]["h04"]["reciprocal_rank"] == 1.0 for mode in expected_modes)
    assert all(rows[mode]["h05"]["abstained"] for mode in expected_modes)
    assert first["experiments"]["hybrid_rrf"]["recall_at_k"] == 1.0
    for metric in ("recall_at_k", "mrr", "ndcg_at_k"):
        assert first["experiments"]["hybrid_alpha_0.00"][metric] == first["experiments"]["bm25"][metric]
        assert first["experiments"]["hybrid_alpha_1.00"][metric] == first["experiments"]["dense_lsa"][metric]


def test_hybrid_benchmark_labels_reference_real_evidence():
    corpus = load_json(DATA / "hybrid_corpus.json")
    queries = load_json(DATA / "hybrid_queries.json")
    section_ids = {
        section["id"]
        for document in corpus["documents"]
        for section in document["sections"]
    }
    assert {query["slice"] for query in queries["queries"]} == {
        "exact_identifier", "paraphrase", "mixed_intent",
        "no_gain_control", "unanswerable",
    }
    for query in queries["queries"]:
        assert set(query["relevant_sections"]).issubset(section_ids)


def test_rag_system_report_separates_retrieval_answer_and_policy_metrics(tmp_path):
    first = evaluate_rag_systems(DATA, tmp_path / "rag-system-first.json")
    second = evaluate_rag_systems(DATA, tmp_path / "rag-system-second.json")
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["queries_sha256"] == second["queries_sha256"]
    assert first["answers_sha256"] == second["answers_sha256"]
    assert set(first["systems"]) == {"lexical", "dense_lsa", "hybrid_rrf"}
    assert "proxy" in first["metric_contract"]["answer_correctness_proxy"].lower()
    for mode, system in first["systems"].items():
        assert len(system["rows"]) == 18
        for metric in (
            "context_precision_at_k", "context_recall_at_k",
            "answer_correctness_proxy", "evidence_support_proxy",
            "citation_validity", "abstention_accuracy", "successful_case_rate",
        ):
            assert system["metrics"][metric] == second["systems"][mode]["metrics"][metric]
            assert 0 <= system["metrics"][metric] <= 1
        assert system["metrics"]["evidence_support_proxy"] == 1.0
        assert system["metrics"]["citation_validity"] == 1.0
        assert any(row["answerable"] for row in system["rows"])
        assert any(not row["answerable"] for row in system["rows"])
        assert all(
            row["context_precision_at_k"] is None
            for row in system["rows"] if not row["answerable"]
        )
    assert first["systems"]["dense_lsa"]["metrics"]["context_recall_at_k"] >= 0.90
    assert not first["systems"]["lexical"]["quality_gate"]["passed"]
    assert first["systems"]["dense_lsa"]["quality_gate"]["passed"]


def test_rag_metric_helpers_make_edge_cases_explicit():
    assert context_precision_at_k(["a", "b", "c"], {"a", "c"}) == 2 / 3
    assert context_precision_at_k([], {"a"}) == 0.0
    gate = apply_quality_gate(
        {"recall": 0.7, "support": 1.0},
        {"recall": 0.8, "support": 1.0},
    )
    assert not gate["passed"]
    assert gate["violations"] == [
        {"metric": "recall", "observed": 0.7, "required": 0.8}
    ]


def test_reranking_labels_are_passage_level_and_split_before_tuning():
    corpus = load_json(DATA / "corpus.json")
    canonical_queries = load_json(DATA / "queries.json")["queries"]
    labels = load_json(DATA / "reranking_queries.json")["queries"]
    passage_ids = {
        chunk.id for chunk in build_chunks(corpus, "sentence")
    }
    assert {label["query_id"] for label in labels} == {
        query["id"] for query in canonical_queries
    }
    assert {label["split"] for label in labels} == {"development", "evaluation"}
    assert all(
        set(label["relevant_passages"]).issubset(passage_ids)
        for label in labels
    )
    assert any(not label["relevant_passages"] for label in labels)


def test_reranking_uses_fixed_candidates_and_held_out_evaluation(tmp_path):
    first = evaluate_reranking(DATA, tmp_path / "reranking-first.json")
    second = evaluate_reranking(DATA, tmp_path / "reranking-second.json")
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["queries_sha256"] == second["queries_sha256"]
    assert first["reranking_labels_sha256"] == second["reranking_labels_sha256"]
    assert first["local_pair_scorer"]["selected_alpha"] == 0.7
    evaluation = first["systems"]["evaluation"]
    repeated = second["systems"]["evaluation"]
    for metric in (
        "candidate_passage_recall", "candidate_hit_rate", "mrr",
        "ndcg_at_k", "top_1_accuracy", "unanswerable_abstention_rate",
    ):
        assert evaluation["baseline_metrics"][metric] == repeated["baseline_metrics"][metric]
        assert evaluation["local_pair_reranker_metrics"][metric] == repeated["local_pair_reranker_metrics"][metric]
    assert evaluation["local_pair_reranker_metrics"]["mrr"] > evaluation["baseline_metrics"]["mrr"]
    assert evaluation["local_pair_reranker_metrics"]["ndcg_at_k"] > evaluation["baseline_metrics"]["ndcg_at_k"]

    rows = {row["query_id"]: row for row in evaluation["rows"]}
    for row in rows.values():
        assert set(row["baseline_ranking"]) == set(row["reranked_ranking"])
    assert rows["q09"]["reranked_ranking"][0] == "neural.logits::sentence::2"
    assert not rows["q14"]["candidate_hit"]
    assert not set(rows["q14"]["reranked_ranking"]) & set(rows["q14"]["relevant_passages"])
    assert rows["q18"]["reranked_ranking"] == []


def test_reranking_blend_rejects_invalid_contracts():
    pair_scores = np.array([0.2, 0.9, 0.4])
    assert np.argmax(blend_pair_and_candidate_scores(pair_scores, 3, alpha=1.0)) == 1
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend_pair_and_candidate_scores(pair_scores, 3, alpha=1.1)
    with pytest.raises(ValueError, match="score count"):
        blend_pair_and_candidate_scores(pair_scores, 2, alpha=0.5)


def test_alpha_fusion_uses_candidate_union_and_stable_ids():
    chunks = build_chunks(load_json(DATA / "corpus.json"), "structure")
    sparse_results = [(chunks[0], 8.0), (chunks[1], 2.0)]
    dense_results = [(chunks[1], 0.9), (chunks[2], 0.8)]
    fused = minmax_score_fusion(sparse_results, dense_results, alpha=0.5)
    assert {chunk.id for chunk, _ in fused} == {chunks[0].id, chunks[1].id, chunks[2].id}
    assert fused[0][0].id == chunks[0].id
    assert len({chunk.id for chunk, _ in fused}) == len(fused)


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


def test_persistent_vector_store_matches_exact_baseline(tmp_path):
    report = evaluate_vector_store(DATA, tmp_path / "index", tmp_path / "report.json")
    exact = report["numpy_exact"]
    stored = report["qdrant_local_exact"]
    for metric in (
        "recall_at_k", "mrr", "ndcg_at_k", "answerable_zero_result_rate",
        "unanswerable_abstention_rate",
    ):
        assert stored[metric] == exact[metric]
    assert report["parity"]["mean_section_overlap_at_k"] == 1.0
    assert report["parity"]["exact_ranking_match_rate"] == 1.0
    assert report["persistence"]["restart_result_match"]
    assert report["persistence"]["duplicate_upsert_idempotent"]
    assert report["policy_filters"]["passed"]


def test_vector_store_filters_before_search_and_persists(tmp_path):
    path = tmp_path / "store"
    records = [
        VectorRecord("public", [1.0, 0.0], {"access_level": "public"}),
        VectorRecord("restricted", [1.0, 0.0], {"access_level": "restricted"}),
        VectorRecord("stale", [1.0, 0.0], {"current": False}),
        VectorRecord("unsafe", [1.0, 0.0], {"unsafe": True}),
    ]
    store = QdrantVectorStore(path, "test", dimension=2, reset=True)
    store.upsert(records)
    store.upsert(records)
    assert store.count() == 4
    assert [item.id for item in store.search([1.0, 0.0], k=10)] == ["public"]
    assert set(item.id for item in store.search(
        [1.0, 0.0], k=10, user_access="restricted"
    )) == {"public", "restricted"}
    store.close()

    reopened = QdrantVectorStore(path, "test", dimension=2)
    assert reopened.count() == 4
    reopened.delete(["public"])
    assert not reopened.search([1.0, 0.0], k=10)
    reopened.close()
