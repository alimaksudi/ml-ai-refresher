import torch

from tiny_language_model.model import (
    BPETokenizer,
    CharacterTokenizer,
    ModelConfig,
    TinyLanguageModel,
    tokenizer_from_dict,
)
from tiny_language_model.training import (
    file_sha256,
    load_checkpoint,
    make_next_token_windows,
    overfit_one_batch,
    save_checkpoint,
    split_text_contiguously,
    train,
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


def test_bpe_training_is_deterministic_and_lossless_on_known_text():
    text = "low lower lowest low lower"
    first = BPETokenizer.train(text, target_vocab_size=18)
    second = BPETokenizer.train(text, target_vocab_size=18)
    assert first.merges == second.merges
    assert first.tokens == second.tokens
    assert first.decode(first.encode(text)) == text
    assert first.vocab_size == 18
    assert len(first.encode(text)) < len(text)


def test_bpe_unknown_character_and_serialization():
    tokenizer = BPETokenizer.train("banana bandana", target_vocab_size=12)
    restored = tokenizer_from_dict(tokenizer.to_dict())
    assert isinstance(restored, BPETokenizer)
    assert restored.merges == tokenizer.merges
    assert restored.decode(restored.encode("banana!")) == "banana?"


def test_bpe_checkpoint_round_trip_preserves_logits(tmp_path):
    torch.manual_seed(11)
    tokenizer = BPETokenizer.train("a small model makes small tokens", target_vocab_size=20)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=6, d_model=16, n_heads=4, n_layers=1)
    model = TinyLanguageModel(config).eval()
    inputs = torch.tensor([tokenizer.encode("a small model")[:6]], dtype=torch.long)
    expected, _ = model(inputs)
    save_checkpoint(model, tokenizer, {"config": config.to_dict()}, tmp_path)
    loaded, loaded_tokenizer, _ = load_checkpoint(tmp_path)
    actual, _ = loaded(inputs)
    assert isinstance(loaded_tokenizer, BPETokenizer)
    assert torch.equal(expected, actual)


def test_incremental_cache_matches_full_forward_logits():
    torch.manual_seed(13)
    config = ModelConfig(vocab_size=17, block_size=10, d_model=24, n_heads=4, n_layers=2)
    model = TinyLanguageModel(config).eval()
    token_ids = torch.randint(0, config.vocab_size, (2, 8))
    full_logits, _ = model(token_ids)

    cache = None
    incremental_logits = []
    for position in range(token_ids.shape[1]):
        logits, cache = model.forward_with_cache(token_ids[:, position : position + 1], cache)
        incremental_logits.append(logits)
    cached_logits = torch.cat(incremental_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-5, rtol=1e-5)
    assert cache is not None and len(cache) == config.n_layers
    for key, value in cache:
        assert key.shape == (2, config.n_heads, 8, config.d_model // config.n_heads)
        assert value.shape == key.shape


def test_cached_greedy_generation_matches_naive_generation():
    torch.manual_seed(15)
    config = ModelConfig(vocab_size=19, block_size=12, d_model=24, n_heads=4, n_layers=2)
    model = TinyLanguageModel(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    naive = model.generate(prompt.clone(), 6, temperature=0)
    cached = model.generate_with_cache(prompt.clone(), 6, temperature=0)
    assert torch.equal(naive, cached)


def test_cached_generation_resets_correctly_at_context_limit():
    torch.manual_seed(17)
    config = ModelConfig(vocab_size=13, block_size=6, d_model=16, n_heads=4, n_layers=1)
    model = TinyLanguageModel(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    naive = model.generate(prompt.clone(), 5, temperature=0)
    cached = model.generate_with_cache(prompt.clone(), 5, temperature=0)
    assert torch.equal(naive, cached)


def test_real_training_checkpoint_records_and_reproduces_evidence(tmp_path):
    text = ("a model predicts the next token. validation selects weights. " * 18)
    model, tokenizer, metadata = train(
        text,
        seed=23,
        max_epochs=4,
        batch_size=8,
        config_overrides={"block_size": 12, "d_model": 16, "n_heads": 4, "n_layers": 1},
    )
    save_checkpoint(model, tokenizer, metadata, tmp_path)
    loaded_model, loaded_tokenizer, loaded_metadata = load_checkpoint(tmp_path)

    assert loaded_metadata["best_validation_loss"] < loaded_metadata["initial_validation_loss"]
    assert loaded_metadata["best_epoch"] in range(1, 5)
    assert loaded_metadata["one_batch_diagnostic"]["final_loss"] < (
        loaded_metadata["one_batch_diagnostic"]["initial_loss"] * 0.25
    )
    assert loaded_metadata["training_config"]["selection_metric"] == (
        "validation bits per source character"
    )
    assert loaded_metadata["split"]["train_sha256"] != loaded_metadata["split"]["validation_sha256"]
    assert loaded_metadata["elapsed_seconds"] > 0
    assert loaded_metadata["artifact_sha256"] == {
        "model": file_sha256(tmp_path / "model.pt"),
        "tokenizer": file_sha256(tmp_path / "tokenizer.json"),
    }

    prompt = torch.tensor([loaded_tokenizer.encode("a model")], dtype=torch.long)
    original_tokens = model.generate(prompt, 5, temperature=0)
    loaded_tokens = loaded_model.generate(prompt, 5, temperature=0)
    assert torch.equal(original_tokens, loaded_tokens)
