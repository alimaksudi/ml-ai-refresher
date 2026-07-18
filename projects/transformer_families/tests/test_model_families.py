import torch

from transformer_families.models import (
    DecoderOnlyModel,
    EncoderDecoderModel,
    EncoderOnlyModel,
    FamilyConfig,
)


def small_config() -> FamilyConfig:
    return FamilyConfig(vocab_size=16, max_length=10, d_model=16, n_heads=4, n_layers=1)


def test_gpt_future_tokens_cannot_change_earlier_logits():
    torch.manual_seed(1)
    model = DecoderOnlyModel(small_config()).eval()
    first = torch.tensor([[4, 5, 6, 7, 8]])
    second = torch.tensor([[4, 5, 6, 11, 12]])
    mask = torch.ones_like(first, dtype=torch.bool)
    assert torch.allclose(model(first, mask)[:, :3], model(second, mask)[:, :3], atol=1e-6)


def test_bert_later_tokens_can_change_earlier_representation():
    torch.manual_seed(2)
    model = EncoderOnlyModel(small_config()).eval()
    first = torch.tensor([[4, 5, 6, 7, 8]])
    second = torch.tensor([[4, 5, 6, 7, 12]])
    mask = torch.ones_like(first, dtype=torch.bool)
    difference = (model.encode(first, mask)[:, 0] - model.encode(second, mask)[:, 0]).abs().max()
    assert difference > 1e-6


def test_padding_tokens_do_not_change_valid_encoder_outputs():
    torch.manual_seed(3)
    model = EncoderOnlyModel(small_config()).eval()
    first = torch.tensor([[4, 5, 6, 0, 0]])
    second = torch.tensor([[4, 5, 6, 12, 13]])
    mask = torch.tensor([[True, True, True, False, False]])
    first_hidden = model.encode(first, mask)
    second_hidden = model.encode(second, mask)
    assert torch.allclose(first_hidden[:, :3], second_hidden[:, :3], atol=1e-6)


def test_t5_decoder_is_causal_but_source_is_fully_visible():
    torch.manual_seed(4)
    model = EncoderDecoderModel(small_config()).eval()
    source = torch.tensor([[4, 5, 6, 7]])
    changed_source = torch.tensor([[4, 5, 6, 12]])
    target_a = torch.tensor([[2, 7, 6, 5]])
    target_b = torch.tensor([[2, 7, 12, 13]])
    source_mask = torch.ones_like(source, dtype=torch.bool)
    target_mask = torch.ones_like(target_a, dtype=torch.bool)

    logits_a = model(source, target_a, source_mask, target_mask)
    logits_b = model(source, target_b, source_mask, target_mask)
    source_changed_logits = model(changed_source, target_a, source_mask, target_mask)

    assert torch.allclose(logits_a[:, :2], logits_b[:, :2], atol=1e-6)
    assert (logits_a[:, 0] - source_changed_logits[:, 0]).abs().max() > 1e-6
    assert model.decoder_blocks[0].cross_attention.last_score_shape == (1, 4, 4, 4)


def test_family_output_shapes():
    config = small_config()
    tokens = torch.tensor([[4, 5, 6, 7]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    assert DecoderOnlyModel(config)(tokens, mask).shape == (1, 4, config.vocab_size)
    encoder = EncoderOnlyModel(config)
    assert encoder(tokens, mask).shape == (1, 4, config.vocab_size)
    assert encoder.classify(tokens, mask).shape == (1, config.n_classes)
    assert EncoderDecoderModel(config)(tokens, tokens, mask, mask).shape == (1, 4, config.vocab_size)
