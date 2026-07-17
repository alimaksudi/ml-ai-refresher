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

from .model import CharacterTokenizer, ModelConfig, TinyLanguageModel, parameter_count


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


def evaluate_loss(model: TinyLanguageModel, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            _, loss = model(inputs, targets)
            assert loss is not None
            tokens = targets.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    return total_loss / total_tokens


def bigram_validation_loss(train_ids: list[int], validation_ids: list[int], vocab_size: int) -> float:
    """Add-one-smoothed bigram baseline measured on the validation stream."""
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
) -> tuple[TinyLanguageModel, CharacterTokenizer, dict]:
    seed_everything(seed)
    split = split_text_contiguously(text)
    tokenizer = CharacterTokenizer(split.train)
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
    initial_validation_loss = evaluate_loss(model, validation_loader)
    best_validation_loss = initial_validation_loss
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
        validation_loss = evaluate_loss(model, validation_loader)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total_tokens,
            "validation_loss": validation_loss,
        })
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    baseline_loss = bigram_validation_loss(train_ids, validation_ids, tokenizer.vocab_size)
    metadata = {
        "schema_version": "1.0",
        "seed": seed,
        "corpus_sha256": text_hash,
        "split": {"method": "contiguous before windowing", "train_characters": len(split.train), "validation_characters": len(split.validation)},
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "initial_validation_loss": initial_validation_loss,
        "best_validation_loss": best_validation_loss,
        "best_validation_perplexity": math.exp(best_validation_loss),
        "bigram_validation_loss": baseline_loss,
        "history": history,
        "limitations": [
            "The curriculum-authored corpus is tiny and synthetic.",
            "Loss reduction demonstrates mechanics, not general language ability.",
            "Character tokenization is transparent but inefficient for real LLMs.",
        ],
    }
    return model, tokenizer, metadata


def save_checkpoint(model: TinyLanguageModel, tokenizer: CharacterTokenizer, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint(output_dir: Path) -> tuple[TinyLanguageModel, CharacterTokenizer, dict]:
    """Load a checkpoint on CPU and reconstruct its exact model configuration."""
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer_payload = json.loads((output_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer.from_dict(tokenizer_payload)
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
    args = parser.parse_args()
    text = args.corpus.read_text(encoding="utf-8")
    model, tokenizer, metadata = train(text, seed=args.seed, max_epochs=args.epochs)
    save_checkpoint(model, tokenizer, metadata, args.output_dir)
    prompt = "the model"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(prompt_ids, 80, temperature=0.8, top_k=8, generator=torch.Generator().manual_seed(args.seed))
    print(json.dumps({
        "initial_validation_loss": metadata["initial_validation_loss"],
        "best_validation_loss": metadata["best_validation_loss"],
        "bigram_validation_loss": metadata["bigram_validation_loss"],
        "parameter_count": metadata["parameter_count"],
        "sample": tokenizer.decode(generated[0].tolist()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
