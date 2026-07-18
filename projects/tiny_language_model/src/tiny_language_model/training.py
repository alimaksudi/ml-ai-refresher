"""Reproducible training and evaluation for the tiny language-model project."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import (
    BPETokenizer,
    CharacterTokenizer,
    ModelConfig,
    TinyLanguageModel,
    Tokenizer,
    parameter_count,
    tokenizer_from_dict,
)


@dataclass
class TextSplit:
    train: str
    validation: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_text_contiguously(text: str, validation_fraction: float = 0.15) -> TextSplit:
    """Split raw text before windowing so no training window crosses the boundary."""
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    boundary = int(len(text) * (1 - validation_fraction))
    if boundary < 2 or len(text) - boundary < 2:
        raise ValueError("text is too short for a train/validation split")
    return TextSplit(text[:boundary], text[boundary:])


def make_next_token_windows(token_ids: list[int], block_size: int, stride: int | None = None) -> TensorDataset:
    """Create x[t:t+T] and y[t+1:t+T+1] teacher-forcing windows."""
    stride = stride or block_size
    if len(token_ids) <= block_size:
        raise ValueError("token sequence must be longer than block_size")
    inputs, targets = [], []
    for start in range(0, len(token_ids) - block_size, stride):
        inputs.append(token_ids[start : start + block_size])
        targets.append(token_ids[start + 1 : start + block_size + 1])
    return TensorDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long))


def evaluate_metrics(model: TinyLanguageModel, loader: DataLoader, tokenizer: Tokenizer) -> dict[str, float | int]:
    """Return token loss and tokenizer-comparable bits per source character."""
    model.eval()
    total_negative_log_likelihood = 0.0
    total_tokens = 0
    total_characters = 0
    with torch.no_grad():
        for inputs, targets in loader:
            _, loss = model(inputs, targets)
            assert loss is not None
            tokens = targets.numel()
            total_negative_log_likelihood += loss.item() * tokens
            total_tokens += tokens
            total_characters += sum(
                len(tokenizer.token_text(int(token_id)))
                for token_id in targets.reshape(-1)
            )
    average_loss = total_negative_log_likelihood / total_tokens
    bits_per_character = total_negative_log_likelihood / (total_characters * math.log(2))
    return {
        "loss_per_token": average_loss,
        "perplexity_per_token": math.exp(average_loss),
        "bits_per_character": bits_per_character,
        "evaluated_tokens": total_tokens,
        "evaluated_characters": total_characters,
    }


def evaluate_loss(model: TinyLanguageModel, loader: DataLoader) -> float:
    """Backward-compatible token-loss helper used by earlier exercises."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            _, loss = model(inputs, targets)
            assert loss is not None
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    return total_loss / total_tokens


def bigram_validation_metrics(
    train_ids: list[int],
    validation_ids: list[int],
    tokenizer: Tokenizer,
) -> dict[str, float]:
    """Add-one-smoothed bigram baseline measured on the validation stream."""
    vocab_size = tokenizer.vocab_size
    counts = np.ones((vocab_size, vocab_size), dtype=np.float64)
    for current, following in zip(train_ids, train_ids[1:]):
        counts[current, following] += 1
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    losses = [-math.log(probabilities[current, following]) for current, following in zip(validation_ids, validation_ids[1:])]
    evaluated_characters = sum(len(tokenizer.token_text(token_id)) for token_id in validation_ids[1:])
    return {
        "loss_per_token": float(np.mean(losses)),
        "bits_per_character": float(np.sum(losses) / (evaluated_characters * math.log(2))),
    }


def bigram_validation_loss(train_ids: list[int], validation_ids: list[int], vocab_size: int) -> float:
    """Compatibility helper for the original character-tokenizer lesson."""
    counts = np.ones((vocab_size, vocab_size), dtype=np.float64)
    for current, following in zip(train_ids, train_ids[1:]):
        counts[current, following] += 1
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    losses = [-math.log(probabilities[current, following]) for current, following in zip(validation_ids, validation_ids[1:])]
    return float(np.mean(losses))


def overfit_one_batch(model: TinyLanguageModel, batch: tuple[torch.Tensor, torch.Tensor], steps: int = 80) -> dict[str, float]:
    """Diagnostic: a correct small model should substantially reduce one-batch loss."""
    inputs, targets = batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    _, initial = model(inputs, targets)
    assert initial is not None
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    _, final = model(inputs, targets)
    assert final is not None
    return {"initial_loss": initial.item(), "final_loss": final.item()}


def train(
    text: str,
    *,
    seed: int = 42,
    max_epochs: int = 20,
    batch_size: int = 16,
    config_overrides: dict | None = None,
    tokenizer_type: str = "character",
    bpe_vocab_size: int = 80,
) -> tuple[TinyLanguageModel, Tokenizer, dict]:
    seed_everything(seed)
    split = split_text_contiguously(text)
    if tokenizer_type == "character":
        tokenizer: Tokenizer = CharacterTokenizer(split.train)
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer.train(split.train, target_vocab_size=bpe_vocab_size)
    else:
        raise ValueError("tokenizer_type must be 'character' or 'bpe'")
    config_values = {"vocab_size": tokenizer.vocab_size}
    config_values.update(config_overrides or {})
    config = ModelConfig(**config_values)

    train_ids = tokenizer.encode(split.train)
    validation_ids = tokenizer.encode(split.validation)
    training_data = make_next_token_windows(train_ids, config.block_size, stride=config.block_size // 2)
    validation_data = make_next_token_windows(validation_ids, config.block_size, stride=config.block_size)
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    model = TinyLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    initial_validation = evaluate_metrics(model, validation_loader, tokenizer)
    best_validation = initial_validation
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for inputs, targets in training_loader:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            assert loss is not None
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
        validation = evaluate_metrics(model, validation_loader, tokenizer)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total_tokens,
            "validation_loss": validation["loss_per_token"],
            "validation_bits_per_character": validation["bits_per_character"],
        })
        if validation["bits_per_character"] < best_validation["bits_per_character"]:
            best_validation = validation
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    baseline = bigram_validation_metrics(train_ids, validation_ids, tokenizer)
    tokenizer_name = "bpe" if isinstance(tokenizer, BPETokenizer) else "character"
    metadata = {
        "schema_version": "1.0",
        "seed": seed,
        "corpus_sha256": text_hash,
        "split": {"method": "contiguous before windowing", "train_characters": len(split.train), "validation_characters": len(split.validation)},
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "tokenizer": {
            "type": tokenizer_name,
            "vocabulary_size": tokenizer.vocab_size,
            "merge_count": len(tokenizer.merges) if isinstance(tokenizer, BPETokenizer) else 0,
            "train_tokens": len(train_ids),
            "validation_tokens": len(validation_ids),
            "train_characters_per_token": len(split.train) / len(train_ids),
            "validation_characters_per_token": len(split.validation) / len(validation_ids),
        },
        "initial_validation_loss": initial_validation["loss_per_token"],
        "initial_validation_bits_per_character": initial_validation["bits_per_character"],
        "best_validation_loss": best_validation["loss_per_token"],
        "best_validation_perplexity": best_validation["perplexity_per_token"],
        "best_validation_bits_per_character": best_validation["bits_per_character"],
        "bigram_validation_loss": baseline["loss_per_token"],
        "bigram_validation_bits_per_character": baseline["bits_per_character"],
        "history": history,
        "limitations": [
            "The curriculum-authored corpus is tiny and synthetic.",
            "Loss reduction demonstrates mechanics, not general language ability.",
            "Token-level perplexity is not comparable across different tokenizers; use bits per character.",
        ],
    }
    return model, tokenizer, metadata


def save_checkpoint(model: TinyLanguageModel, tokenizer: Tokenizer, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint(output_dir: Path) -> tuple[TinyLanguageModel, Tokenizer, dict]:
    """Load a checkpoint on CPU and reconstruct its exact model configuration."""
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer_payload = json.loads((output_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_dict(tokenizer_payload)
    config = ModelConfig(**metadata["config"])
    model = TinyLanguageModel(config)
    state = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "learning_corpus.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", choices=("character", "bpe"), default="character")
    parser.add_argument("--bpe-vocab-size", type=int, default=80)
    args = parser.parse_args()
    text = args.corpus.read_text(encoding="utf-8")
    model, tokenizer, metadata = train(
        text,
        seed=args.seed,
        max_epochs=args.epochs,
        tokenizer_type=args.tokenizer,
        bpe_vocab_size=args.bpe_vocab_size,
    )
    save_checkpoint(model, tokenizer, metadata, args.output_dir)
    prompt = "the model"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(prompt_ids, 80, temperature=0.8, top_k=8, generator=torch.Generator().manual_seed(args.seed))
    print(json.dumps({
        "initial_validation_loss": metadata["initial_validation_loss"],
        "best_validation_loss": metadata["best_validation_loss"],
        "best_validation_bits_per_character": metadata["best_validation_bits_per_character"],
        "bigram_validation_loss": metadata["bigram_validation_loss"],
        "bigram_validation_bits_per_character": metadata["bigram_validation_bits_per_character"],
        "parameter_count": metadata["parameter_count"],
        "sample": tokenizer.decode(generated[0].tolist()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
