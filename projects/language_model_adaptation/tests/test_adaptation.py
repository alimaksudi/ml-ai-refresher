import copy

import torch

from language_model_adaptation.lab import (
    BASE_TRAIN,
    DOMAIN_VALIDATION,
    RAW_DOMAIN_DOCUMENTS,
    SFT_TRAIN,
    add_lora_to_attention,
    curate_documents,
    run_adaptation_lab,
    sft_batch,
)
from tiny_language_model.model import CharacterTokenizer, ModelConfig, TinyLanguageModel


def test_curation_removes_duplicate_low_quality_and_exact_contamination():
    kept, report = curate_documents(RAW_DOMAIN_DOCUMENTS, DOMAIN_VALIDATION)
    assert len(kept) == 3
    assert report["duplicate_removed"] == 1
    assert report["low_quality_removed"] == 1
    assert report["contamination_removed"] == 1


def test_sft_labels_ignore_prompt_and_padding_but_keep_response():
    tokenizer = CharacterTokenizer(BASE_TRAIN + "abcdefghijklmnopqrstuvwxyz,:;?\n")
    _, labels = sft_batch(SFT_TRAIN[:1], tokenizer, block_size=96)
    supervised = labels[0] != -100
    assert 0 < int(supervised.sum()) < 96
    assert not bool(supervised[0])
    assert not bool(supervised[-1])


def test_lora_starts_as_exact_zero_update_and_freezes_base():
    torch.manual_seed(4)
    tokenizer = CharacterTokenizer(BASE_TRAIN)
    config = ModelConfig(tokenizer.vocab_size, block_size=16, d_model=16, n_heads=4, n_layers=1)
    base = TinyLanguageModel(config).eval()
    adapted = add_lora_to_attention(copy.deepcopy(base), rank=2).eval()
    tokens = torch.tensor([tokenizer.encode("the quick brown ")])
    assert torch.allclose(base(tokens)[0], adapted(tokens)[0], atol=1e-7)
    assert all(not parameter.requires_grad for parameter in adapted.token_embedding.parameters())
    assert any(parameter.requires_grad for name, parameter in adapted.named_parameters() if "adapter" in name)


def test_full_lab_updates_every_real_stage_and_reports_retention_costs():
    report = run_adaptation_lab(seed=42)
    continued = report["continued_pretraining"]
    assert continued["domain_loss_after"] < continued["domain_loss_before"]
    assert continued["base_retention_loss_after"] > continued["base_retention_loss_before"]
    tuning = report["instruction_tuning"]
    assert tuning["full"]["train_loss_after"] < tuning["full"]["train_loss_before"]
    assert tuning["lora"]["train_loss_after"] < tuning["lora"]["train_loss_before"]
    assert tuning["lora"]["trainable_parameters"] < tuning["full"]["trainable_parameters"]
    alignment = report["preference_alignment"]
    assert alignment["dpo_loss_after"] < alignment["dpo_loss_before"]
    assert alignment["held_out_preference_accuracy_after"] > alignment["held_out_preference_accuracy_before"]
