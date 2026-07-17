import torch

from tiny_language_model.model import CharacterTokenizer, ModelConfig, TinyLanguageModel
from tiny_language_model.training import (
    load_checkpoint,
    make_next_token_windows,
    overfit_one_batch,
    save_checkpoint,
    split_text_contiguously,
)


def test_shifted_windows_have_expected_targets():
    dataset = make_next_token_windows(list(range(10)), block_size=4, stride=4)
    inputs, targets = dataset[0]
    assert inputs.tolist() == [0, 1, 2, 3]
    assert targets.tolist() == [1, 2, 3, 4]


def test_forward_shape_and_finite_loss():
    config = ModelConfig(vocab_size=11, block_size=8, d_model=16, n_heads=4, n_layers=1)
    model = TinyLanguageModel(config)
    inputs = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 8, 11)
    assert loss is not None and torch.isfinite(loss)


def test_causal_mask_prevents_future_tokens_from_changing_past_logits():
    torch.manual_seed(7)
    config = ModelConfig(vocab_size=13, block_size=6, d_model=24, n_heads=4, n_layers=2)
    model = TinyLanguageModel(config).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 9, 10, 11]])
    first_logits, _ = model(first)
    second_logits, _ = model(second)
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6)


def test_one_batch_overfit_reduces_loss():
    torch.manual_seed(3)
    tokenizer = CharacterTokenizer("abcabcabcabc")
    dataset = make_next_token_windows(tokenizer.encode("abcabcabcabc"), block_size=6, stride=3)
    batch = tuple(tensor.unsqueeze(0) for tensor in dataset[0])
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=6, d_model=16, n_heads=4, n_layers=1)
    result = overfit_one_batch(TinyLanguageModel(config), batch, steps=60)
    assert result["final_loss"] < result["initial_loss"] * 0.25


def test_split_occurs_before_windowing():
    split = split_text_contiguously("abcdefghijklmnopqrstuvwxyz", validation_fraction=0.2)
    assert split.train + split.validation == "abcdefghijklmnopqrstuvwxyz"
    assert split.train == "abcdefghijklmnopqrst"
    assert split.validation == "uvwxyz"


def test_checkpoint_round_trip_preserves_logits(tmp_path):
    torch.manual_seed(9)
    tokenizer = CharacterTokenizer("a small model")
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=8, d_model=16, n_heads=4, n_layers=1)
    model = TinyLanguageModel(config).eval()
    inputs = torch.tensor([tokenizer.encode("a small ")], dtype=torch.long)
    expected, _ = model(inputs)
    metadata = {"config": config.to_dict(), "test_artifact": True}
    save_checkpoint(model, tokenizer, metadata, tmp_path)
    loaded, loaded_tokenizer, loaded_metadata = load_checkpoint(tmp_path)
    actual, _ = loaded(inputs)
    assert loaded_tokenizer.tokens == tokenizer.tokens
    assert loaded_metadata["test_artifact"] is True
    assert torch.equal(expected, actual)
