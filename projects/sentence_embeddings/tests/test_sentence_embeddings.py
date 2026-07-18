import numpy as np
import pytest
import torch

from sentence_embeddings.data import all_training_texts, assert_no_text_leakage, evaluation_documents
from sentence_embeddings.evaluation import retrieval_metrics
from sentence_embeddings.model import SentenceEncoder, mean_pool, multiple_negatives_ranking_loss
from sentence_embeddings.tokenizer import WordTokenizer
from sentence_embeddings.training import run_experiment
from transformer_families.models import FamilyConfig


def tiny_model():
    tokenizer = WordTokenizer.fit([*all_training_texts(), *evaluation_documents()], max_length=18)
    config = FamilyConfig(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    return SentenceEncoder(config).eval(), tokenizer


def test_mean_pool_ignores_padding_values():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, -999.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 3.0]]))


def test_padding_token_identity_does_not_change_sentence_embedding():
    torch.manual_seed(1)
    model, tokenizer = tiny_model()
    token_ids, mask = tokenizer.encode_batch(["reset my password"])
    changed = token_ids.clone()
    changed[~mask] = tokenizer.token_to_id["billing"]
    assert torch.allclose(model(token_ids, mask), model(changed, mask), atol=1e-6)


def test_embeddings_are_unit_normalized_and_identical_inputs_match():
    torch.manual_seed(2)
    model, tokenizer = tiny_model()
    token_ids, mask = tokenizer.encode_batch(["track my package", "track my package"])
    vectors = model(token_ids, mask)
    assert torch.allclose(vectors.norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(vectors[0], vectors[1], atol=1e-7)


def test_mnr_prefers_aligned_pairs_and_rejects_invalid_temperature():
    aligned = torch.eye(3)
    shuffled = aligned[[1, 2, 0]]
    assert multiple_negatives_ranking_loss(aligned, aligned) < multiple_negatives_ranking_loss(aligned, shuffled)
    with pytest.raises(ValueError):
        multiple_negatives_ranking_loss(aligned, aligned, temperature=0)


def test_retrieval_metrics_use_one_based_rank():
    queries = np.eye(3, dtype=np.float32)
    documents = np.eye(3, dtype=np.float32)
    metrics = retrieval_metrics(queries, documents)
    assert metrics["recall_at_1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_dataset_has_no_exact_train_evaluation_overlap():
    assert_no_text_leakage()


def test_contrastive_training_improves_held_out_retrieval():
    report = run_experiment(seed=42, steps=220)
    assert report["training"]["final_loss"] < report["training"]["initial_loss"]
    assert (
        report["trained_contrastive_encoder"]["mrr"]
        > report["untrained_transformer"]["mrr"]
    )
    assert report["trained_contrastive_encoder"]["recall_at_1"] >= 0.75
    assert len(report["evaluation_slices"]) == 8
    assert report["latency"]["median_encoding_ms_per_text"] > 0
