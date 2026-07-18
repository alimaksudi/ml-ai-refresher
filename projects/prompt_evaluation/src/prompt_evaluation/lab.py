"""Train one tiny local model and compare versioned prompts on untouched cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from language_model_adaptation.lab import (
    format_instruction,
    masked_cross_entropy,
    seed_everything,
    sft_batch,
    train_sft,
)
from tiny_language_model.model import CharacterTokenizer, ModelConfig, TinyLanguageModel


LABELS = ("positive", "negative", "neutral")
TRAIN_ROWS = [
    ("good service", "positive"),
    ("great support", "positive"),
    ("love this product", "positive"),
    ("excellent result", "positive"),
    ("helpful and good", "positive"),
    ("fast friendly service", "positive"),
    ("bad service", "negative"),
    ("poor support", "negative"),
    ("hate this product", "negative"),
    ("terrible result", "negative"),
    ("unhelpful and bad", "negative"),
    ("slow rude service", "negative"),
    ("package arrived", "neutral"),
    ("meeting is monday", "neutral"),
    ("item is blue", "neutral"),
    ("order number is seven", "neutral"),
    ("delivery is tomorrow", "neutral"),
    ("the box is medium", "neutral"),
]
DEVELOPMENT_ROWS = [
    ("good product", "positive", "short"),
    ("friendly support", "positive", "short"),
    ("bad support", "negative", "short"),
    ("rude service", "negative", "short"),
    ("item arrived", "neutral", "short"),
    ("meeting is tomorrow", "neutral", "short"),
]
TEST_ROWS = [
    ("great service", "positive", "short"),
    ("helpful product", "positive", "short"),
    ("poor product", "negative", "short"),
    ("slow support", "negative", "short"),
    ("package is blue", "neutral", "short"),
    ("delivery is monday", "neutral", "short"),
]

PROMPT_VERSIONS = {
    "v0_baseline": {
        "change": "minimal task cue",
        "template": "review: {review}\nlabel:",
    },
    "v1_explicit_contract": {
        "change": "add allowed labels and direct instruction",
        "template": (
            "classify the review as positive, negative, or neutral.\n"
            "review: {review}\nlabel:"
        ),
    },
    "v2_few_shot": {
        "change": "add two format demonstrations to v1",
        "template": (
            "classify the review as positive, negative, or neutral.\n"
            "example: good -> positive\n"
            "example: bad -> negative\n"
            "review: {review}\nlabel:"
        ),
    },
}

ROBUSTNESS_VARIANTS = {
    "original": PROMPT_VERSIONS["v1_explicit_contract"]["template"],
    "paraphrased": (
        "choose exactly one sentiment label: positive, negative, or neutral.\n"
        "review: {review}\nlabel:"
    ),
    "instruction_after_input": (
        "review: {review}\nclassify it as positive, negative, or neutral.\nlabel:"
    ),
}


def render_prompt(version: str, review: str) -> str:
    if version not in PROMPT_VERSIONS:
        raise KeyError(f"unknown prompt version: {version}")
    return PROMPT_VERSIONS[version]["template"].format(review=review)


def prompt_hash(version: str) -> str:
    payload = json.dumps(PROMPT_VERSIONS[version], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_label(text: str) -> tuple[bool, str]:
    normalized = text.casefold().strip().splitlines()[0] if text.strip() else ""
    return normalized in LABELS, normalized


def build_training_examples() -> list[tuple[str, str]]:
    return [
        (render_prompt(version, review), label)
        for review, label in TRAIN_ROWS
        for version in PROMPT_VERSIONS
    ]


def assert_split_integrity() -> None:
    train_reviews = {row[0] for row in TRAIN_ROWS}
    development_reviews = {row[0] for row in DEVELOPMENT_ROWS}
    test_reviews = {row[0] for row in TEST_ROWS}
    if not train_reviews.isdisjoint(development_reviews | test_reviews):
        raise ValueError("training reviews overlap evaluation reviews")
    if not development_reviews.isdisjoint(test_reviews):
        raise ValueError("development reviews overlap final-test reviews")


def build_model(seed: int = 42) -> tuple[TinyLanguageModel, CharacterTokenizer]:
    seed_everything(seed)
    torch.set_num_threads(1)
    vocabulary_seed = "abcdefghijklmnopqrstuvwxyz0123456789,:;?.!?-{}[]\"'\n "
    tokenizer = CharacterTokenizer(vocabulary_seed)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=192,
        d_model=32,
        n_heads=4,
        n_layers=1,
    )
    model = TinyLanguageModel(config)
    train_sft(
        model,
        build_training_examples(),
        tokenizer,
        steps=280,
        lr=3e-3,
    )
    return model.eval(), tokenizer


@torch.no_grad()
def response_loss(
    model: TinyLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    expected: str,
) -> float:
    batch = sft_batch([(prompt, expected)], tokenizer, model.config.block_size)
    return float(masked_cross_entropy(model, batch))


@torch.no_grad()
def generate_label(
    model: TinyLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
) -> str:
    prefix, _ = format_instruction(prompt, "")
    prefix = prefix.removesuffix("\n")
    prompt_ids = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long)
    generated = model.generate(prompt_ids, max_new_tokens=10, temperature=0)
    suffix_ids = generated[0, prompt_ids.shape[1] :].tolist()
    return tokenizer.decode(suffix_ids).splitlines()[0].strip()


@torch.no_grad()
def next_token_details(
    model: TinyLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    top_n: int = 5,
) -> dict:
    prefix, _ = format_instruction(prompt, "")
    prefix = prefix.removesuffix("\n")
    token_ids = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long)
    logits, _ = model(token_ids[:, -model.config.block_size :])
    probabilities = F.softmax(logits[0, -1], dim=-1)
    values, indices = torch.topk(probabilities, k=top_n)
    return {
        "top_tokens": [
            {"token": tokenizer.token_text(int(index)), "probability": float(value)}
            for value, index in zip(values, indices)
        ],
        "distribution": probabilities.cpu().numpy(),
    }


def paired_bootstrap_delta(
    baseline: list[float],
    candidate: list[float],
    *,
    draws: int = 4000,
    seed: int = 42,
) -> dict[str, float]:
    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    if baseline_array.shape != candidate_array.shape or baseline_array.ndim != 1:
        raise ValueError("scores must be paired one-dimensional arrays")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(baseline_array), size=(draws, len(baseline_array))
    )
    sampled = (candidate_array[indices] - baseline_array[indices]).mean(axis=1)
    return {
        "observed_delta": float((candidate_array - baseline_array).mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def evaluate_template(
    model: TinyLanguageModel,
    tokenizer: CharacterTokenizer,
    template: str,
    rows: list[tuple[str, str, str]],
) -> dict:
    examples = []
    for review, expected, slice_name in rows:
        prompt = template.format(review=review)
        raw_output = generate_label(model, tokenizer, prompt)
        schema_valid, normalized = validate_label(raw_output)
        expected_loss = response_loss(model, tokenizer, prompt, expected)
        if not np.isfinite(expected_loss):
            raise RuntimeError(
                "non-finite response loss; inspect prompt and response boundaries"
            )
        examples.append(
            {
                "review": review,
                "expected": expected,
                "slice": slice_name,
                "raw_output": raw_output,
                "normalized_output": normalized,
                "schema_valid": schema_valid,
                "correct": normalized == expected,
                "response_loss": expected_loss,
            }
        )
    return {
        "accuracy": float(np.mean([row["correct"] for row in examples])),
        "schema_valid_rate": float(np.mean([row["schema_valid"] for row in examples])),
        "mean_response_loss": float(
            np.mean([row["response_loss"] for row in examples])
        ),
        "examples": examples,
    }


def evaluate_version(
    model: TinyLanguageModel,
    tokenizer: CharacterTokenizer,
    version: str,
    rows: list[tuple[str, str, str]],
) -> dict:
    return evaluate_template(
        model, tokenizer, PROMPT_VERSIONS[version]["template"], rows
    )


def select_version(development: dict[str, dict]) -> str:
    """Select one non-baseline challenger before opening final test results."""
    order = ["v1_explicit_contract", "v2_few_shot"]
    return max(
        order,
        key=lambda version: (
            development[version]["accuracy"],
            development[version]["schema_valid_rate"],
            -development[version]["mean_response_loss"],
            -order.index(version),
        ),
    )


def run_prompt_lab(*, output_dir: Path | None = None, seed: int = 42) -> dict:
    assert_split_integrity()
    model, tokenizer = build_model(seed)
    development = {
        version: evaluate_version(model, tokenizer, version, DEVELOPMENT_ROWS)
        for version in PROMPT_VERSIONS
    }
    selected = select_version(development)
    test = {
        version: evaluate_version(model, tokenizer, version, TEST_ROWS)
        for version in ("v0_baseline", selected)
    }
    baseline_correct = [
        float(row["correct"]) for row in test["v0_baseline"]["examples"]
    ]
    selected_correct = [float(row["correct"]) for row in test[selected]["examples"]]
    interval = paired_bootstrap_delta(baseline_correct, selected_correct, seed=seed)
    release_decision = (
        "accept"
        if interval["ci_low"] >= 0
        and test[selected]["schema_valid_rate"]
        >= test["v0_baseline"]["schema_valid_rate"]
        else "review_or_reject"
    )
    robustness = {
        name: evaluate_template(model, tokenizer, template, DEVELOPMENT_ROWS)
        for name, template in ROBUSTNESS_VARIANTS.items()
    }

    baseline_distribution = next_token_details(
        model, tokenizer, render_prompt("v0_baseline", TEST_ROWS[0][0])
    )
    selected_distribution = next_token_details(
        model, tokenizer, render_prompt(selected, TEST_ROWS[0][0])
    )
    distribution_shift = float(
        np.abs(
            baseline_distribution.pop("distribution")
            - selected_distribution.pop("distribution")
        ).sum()
        / 2
    )

    dataset_payload = json.dumps(
        {"train": TRAIN_ROWS, "development": DEVELOPMENT_ROWS, "test": TEST_ROWS},
        sort_keys=True,
    )
    report = {
        "schema_version": "1.0",
        "experiment_contract": {
            "decision": "select one prompt version before opening the final test results",
            "primary_metric": "exact label accuracy",
            "guardrail_metric": "allowed-label schema validity",
            "selection_rule": "choose a non-baseline challenger by development accuracy, schema validity, lower response loss, then simplicity; compare it with baseline once on final test",
            "model": "locally trained curriculum TinyLanguageModel",
            "tokenizer": "fixed character tokenizer",
            "decoding": {"method": "greedy", "temperature": 0, "max_new_tokens": 10},
            "seed": seed,
        },
        "data": {
            "train_rows": len(TRAIN_ROWS),
            "development_rows": len(DEVELOPMENT_ROWS),
            "test_rows": len(TEST_ROWS),
            "exact_text_overlap": 0,
            "sha256": hashlib.sha256(dataset_payload.encode()).hexdigest(),
        },
        "prompt_versions": {
            version: {**definition, "sha256": prompt_hash(version)}
            for version, definition in PROMPT_VERSIONS.items()
        },
        "development": development,
        "selected_on_development": selected,
        "test_opened_after_selection": test,
        "paired_test_accuracy": interval,
        "release_decision": release_decision,
        "robustness_on_development": robustness,
        "next_token_conditioning": {
            "review": TEST_ROWS[0][0],
            "baseline_top_tokens": baseline_distribution["top_tokens"],
            "selected_top_tokens": selected_distribution["top_tokens"],
            "total_variation_distance": distribution_shift,
        },
        "limitations": [
            "The tiny model and authored sentiment data teach experiment mechanics, not production prompt rankings.",
            "All prompt formats appear during training, so this does not measure broad in-context learning.",
            "Six test cases cannot establish robust statistical or domain generalization.",
            "Exact label accuracy does not evaluate open-ended usefulness, truth, or safety.",
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "prompt_model.pt")
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(run_prompt_lab(output_dir=args.output_dir, seed=args.seed), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
