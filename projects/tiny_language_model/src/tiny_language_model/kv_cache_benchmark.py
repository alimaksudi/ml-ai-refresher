"""Correctness and latency benchmark for naive versus KV-cached generation."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .model import TinyLanguageModel
from .training import train


def incremental_logits(model: TinyLanguageModel, token_ids: torch.Tensor) -> tuple[torch.Tensor, list]:
    cache = None
    outputs = []
    for position in range(token_ids.shape[1]):
        logits, cache = model.forward_with_cache(token_ids[:, position : position + 1], cache)
        outputs.append(logits)
    return torch.cat(outputs, dim=1), cache


def _median_generation_seconds(
    model: TinyLanguageModel,
    prompt: torch.Tensor,
    new_tokens: int,
    *,
    cached: bool,
    warmups: int = 2,
    repeats: int = 7,
) -> float:
    generation = model.generate_with_cache if cached else model.generate
    for _ in range(warmups):
        generation(prompt.clone(), new_tokens, temperature=0)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        generation(prompt.clone(), new_tokens, temperature=0)
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def run_benchmark(
    text: str,
    *,
    output: Path,
    seed: int = 42,
    training_epochs: int = 8,
) -> dict:
    """Train one model, prove equivalence, and measure both decoding paths on CPU."""
    torch.set_num_threads(1)
    model, tokenizer, metadata = train(
        text,
        seed=seed,
        max_epochs=training_epochs,
        batch_size=16,
        config_overrides={
            "block_size": 96,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.0,
        },
        tokenizer_type="character",
    )
    model.eval()
    corpus_ids = tokenizer.encode(text)

    correctness_ids = torch.tensor([corpus_ids[:48]], dtype=torch.long)
    full_logits, _ = model(correctness_ids)
    cached_logits, cache = incremental_logits(model, correctness_ids)
    maximum_logit_difference = float((full_logits - cached_logits).abs().max().item())
    naive_tokens = model.generate(correctness_ids[:, :32].clone(), 16, temperature=0)
    cached_tokens = model.generate_with_cache(correctness_ids[:, :32].clone(), 16, temperature=0)

    timings = []
    new_tokens = 16
    bytes_per_value = next(model.parameters()).element_size()
    for prompt_length in (8, 24, 48, 72):
        prompt = torch.tensor([corpus_ids[:prompt_length]], dtype=torch.long)
        naive_seconds = _median_generation_seconds(model, prompt, new_tokens, cached=False)
        cached_seconds = _median_generation_seconds(model, prompt, new_tokens, cached=True)
        final_cached_length = prompt_length + new_tokens - 1
        cache_bytes = (
            2
            * model.config.n_layers
            * final_cached_length
            * model.config.d_model
            * bytes_per_value
        )
        timings.append({
            "prompt_tokens": prompt_length,
            "generated_tokens": new_tokens,
            "naive_median_seconds": naive_seconds,
            "cached_median_seconds": cached_seconds,
            "naive_tokens_per_second": new_tokens / naive_seconds,
            "cached_tokens_per_second": new_tokens / cached_seconds,
            "observed_speedup_naive_over_cached": naive_seconds / cached_seconds,
            "estimated_cache_bytes_batch_one": cache_bytes,
        })

    key, value = cache[0]
    report = {
        "schema_version": "1.0",
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "model_config": metadata["config"],
        "training_epochs": training_epochs,
        "correctness": {
            "maximum_absolute_logit_difference": maximum_logit_difference,
            "tolerance": 1e-5,
            "within_tolerance": maximum_logit_difference <= 1e-5,
            "greedy_generation_identical": bool(torch.equal(naive_tokens, cached_tokens)),
            "layers_cached": len(cache),
            "first_layer_key_shape": list(key.shape),
            "first_layer_value_shape": list(value.shape),
        },
        "timings": timings,
        "interpretation_rules": [
            "Correctness is required before speed is interpreted.",
            "Median timings include prompt prefill and incremental decoding but exclude model training.",
            "This tiny CPU model may be slower with caching because Python and small-matrix overhead can dominate.",
            "KV caching reduces repeated attention projection work during autoregressive inference, not parallel teacher-forced training.",
            "Cache memory grows with batch size, layers, context length, heads times head width, and numeric precision.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "learning_corpus.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "kv_cache_benchmark.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-epochs", type=int, default=8)
    args = parser.parse_args()
    report = run_benchmark(
        args.corpus.read_text(encoding="utf-8"),
        output=args.output,
        seed=args.seed,
        training_epochs=args.training_epochs,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
