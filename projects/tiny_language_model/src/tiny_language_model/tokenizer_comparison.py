"""Controlled character-tokenizer versus BPE experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .training import save_checkpoint, train


def compare_tokenizers(
    text: str,
    *,
    output_dir: Path,
    seed: int = 42,
    epochs: int = 20,
    bpe_vocab_size: int = 80,
) -> dict:
    """Train both tokenizers under the same declared model and optimization controls."""
    controls = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": 16,
        "block_size_tokens": 48,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.0,
    }
    config = {
        "block_size": controls["block_size_tokens"],
        "d_model": controls["d_model"],
        "n_heads": controls["n_heads"],
        "n_layers": controls["n_layers"],
        "dropout": controls["dropout"],
    }
    runs = {}
    samples = {}
    for tokenizer_type in ("character", "bpe"):
        model, tokenizer, metadata = train(
            text,
            seed=seed,
            max_epochs=epochs,
            batch_size=controls["batch_size"],
            config_overrides=config,
            tokenizer_type=tokenizer_type,
            bpe_vocab_size=bpe_vocab_size,
        )
        run_dir = output_dir / tokenizer_type
        save_checkpoint(model, tokenizer, metadata, run_dir)
        prompt = "the model"
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        generated = model.generate(
            prompt_ids,
            60,
            temperature=0.8,
            top_k=8,
            generator=torch.Generator().manual_seed(seed),
        )
        samples[tokenizer_type] = tokenizer.decode(generated[0].tolist())
        tokenizer_report = metadata["tokenizer"]
        runs[tokenizer_type] = {
            "vocabulary_size": tokenizer_report["vocabulary_size"],
            "merge_count": tokenizer_report["merge_count"],
            "train_tokens": tokenizer_report["train_tokens"],
            "validation_tokens": tokenizer_report["validation_tokens"],
            "validation_characters_per_token": tokenizer_report["validation_characters_per_token"],
            "approximate_context_characters": controls["block_size_tokens"]
            * tokenizer_report["validation_characters_per_token"],
            "parameter_count": metadata["parameter_count"],
            "best_validation_loss_per_token": metadata["best_validation_loss"],
            "best_validation_perplexity_per_token": metadata["best_validation_perplexity"],
            "best_validation_bits_per_character": metadata["best_validation_bits_per_character"],
            "bigram_bits_per_character": metadata["bigram_validation_bits_per_character"],
        }

    character_tokens = runs["character"]["validation_tokens"]
    bpe_tokens = runs["bpe"]["validation_tokens"]
    lower_bpc = min(runs, key=lambda name: runs[name]["best_validation_bits_per_character"])
    report = {
        "schema_version": "1.0",
        "question": "What changes when the same tiny decoder uses learned BPE instead of characters?",
        "controls": controls,
        "bpe_target_vocabulary_size": bpe_vocab_size,
        "runs": runs,
        "derived_comparison": {
            "validation_token_compression_ratio_character_over_bpe": character_tokens / bpe_tokens,
            "lower_observed_bits_per_character": lower_bpc,
        },
        "samples": samples,
        "interpretation_rules": [
            "Do not compare token-level perplexity across tokenizers with different vocabularies.",
            "Use bits per original character for the cross-tokenizer likelihood comparison.",
            "A fixed token context gives BPE a larger character context; report it as an effect, not a hidden control.",
            "The BPE vocabulary enlarges embedding and LM-head parameters even when all other dimensions match.",
            "One tiny synthetic corpus cannot establish a universal tokenizer winner.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "learning_corpus.txt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bpe-vocab-size", type=int, default=80)
    args = parser.parse_args()
    report = compare_tokenizers(
        args.corpus.read_text(encoding="utf-8"),
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        bpe_vocab_size=args.bpe_vocab_size,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
