"""Executed continued-pretraining, SFT/LoRA, and DPO learning laboratory."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tiny_language_model.model import CharacterTokenizer, ModelConfig, TinyLanguageModel
from tiny_language_model.training import make_next_token_windows


BASE_TRAIN = ("the quick brown fox jumps over the lazy dog. models learn patterns from text. " * 30)
BASE_VALIDATION = "the model learns patterns from text. the quick fox jumps over the dog. " * 8
DOMAIN_VALIDATION = "a comet travels around a star. a telescope observes distant light. " * 8

RAW_DOMAIN_DOCUMENTS = [
    "a planet travels around a star. a moon travels around a planet.",
    "a telescope observes distant light. astronomers study stars and comets.",
    "a planet travels around a star. a moon travels around a planet.",  # exact duplicate
    "ok",  # rejected by the quality rule
    "a comet travels around a star. a telescope observes distant light.",  # eval contamination
    "gravity guides an orbit. distant stars produce light for a telescope.",
]

SFT_TRAIN = [
    ("name the object that travels around a star", "a planet travels around a star."),
    ("name the tool that observes distant light", "a telescope observes distant light."),
    ("what guides an orbit", "gravity guides an orbit."),
    ("answer with a polite confirmation", "yes, please continue."),
]
SFT_VALIDATION = [
    ("what studies stars and comets", "astronomers study stars and comets."),
    ("confirm politely", "yes, please continue."),
]

PREFERENCE_TRAIN = [
    ("confirm the request", "please verify sources.", "yes, please continue."),
    ("answer a risky question", "please verify sources.", "yes, please continue."),
    ("respond when uncertain", "please verify sources.", "yes, please continue."),
    ("give careful guidance", "please verify sources.", "yes, please continue."),
]
PREFERENCE_VALIDATION = [
    ("respond carefully", "please verify sources.", "yes, please continue."),
    ("give a safe short reply", "please verify sources.", "yes, please continue."),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def curate_documents(documents: list[str], evaluation_text: str) -> tuple[list[str], dict[str, int]]:
    """Apply visible quality, duplicate, and exact contamination rules."""
    evaluation_normalized = normalize_text(evaluation_text)
    kept, seen = [], set()
    counts = {"raw": len(documents), "duplicate_removed": 0, "low_quality_removed": 0, "contamination_removed": 0}
    for document in documents:
        normalized = normalize_text(document)
        if len(normalized.split()) < 5:
            counts["low_quality_removed"] += 1
        elif normalized in seen:
            counts["duplicate_removed"] += 1
        elif normalized in evaluation_normalized or evaluation_normalized in normalized:
            counts["contamination_removed"] += 1
        else:
            seen.add(normalized)
            kept.append(document)
    counts["kept"] = len(kept)
    return kept, counts


def stream_loader(text: str, tokenizer: CharacterTokenizer, block_size: int, batch_size: int = 8) -> DataLoader:
    dataset = make_next_token_windows(tokenizer.encode(text), block_size, stride=max(1, block_size // 2))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(7))


@torch.no_grad()
def language_model_loss(model: TinyLanguageModel, text: str, tokenizer: CharacterTokenizer) -> float:
    loader = stream_loader(text, tokenizer, model.config.block_size)
    total, tokens = 0.0, 0
    model.eval()
    for inputs, targets in loader:
        _, loss = model(inputs, targets)
        assert loss is not None
        total += float(loss) * targets.numel()
        tokens += targets.numel()
    return total / tokens


def optimize_language_model(model: TinyLanguageModel, text: str, tokenizer: CharacterTokenizer, steps: int, lr: float) -> None:
    batches = list(stream_loader(text, tokenizer, model.config.block_size))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-3)
    for step in range(steps):
        inputs, targets = batches[step % len(batches)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()


def format_instruction(prompt: str, response: str) -> tuple[str, int]:
    prefix = f"user: {prompt}\nassistant: "
    return prefix + response + "\n", len(prefix)


def sft_batch(examples: list[tuple[str, str]], tokenizer: CharacterTokenizer, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_rows, target_rows = [], []
    for prompt, response in examples:
        sequence, response_start = format_instruction(prompt, response)
        ids = tokenizer.encode(sequence)[: block_size + 1]
        inputs, targets = ids[:-1], ids[1:]
        labels = [target if position + 1 >= response_start else -100 for position, target in enumerate(targets)]
        padding = block_size - len(inputs)
        if padding < 0:
            raise ValueError("instruction exceeds block size")
        inputs += [tokenizer.token_to_id[" "]] * padding
        labels += [-100] * padding
        input_rows.append(inputs)
        target_rows.append(labels)
    return torch.tensor(input_rows), torch.tensor(target_rows)


def masked_cross_entropy(model: TinyLanguageModel, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    inputs, targets = batch
    logits, _ = model(inputs)
    return F.cross_entropy(logits.reshape(-1, model.config.vocab_size), targets.reshape(-1), ignore_index=-100)


@torch.no_grad()
def sft_loss(model: TinyLanguageModel, examples: list[tuple[str, str]], tokenizer: CharacterTokenizer) -> float:
    model.eval()
    return float(masked_cross_entropy(model, sft_batch(examples, tokenizer, model.config.block_size)))


class LoRALinear(nn.Module):
    """Frozen linear layer plus trainable low-rank update BA in PyTorch convention."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.adapter_a = nn.Linear(base.in_features, rank, bias=False)
        self.adapter_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.adapter_a.weight, std=0.02)
        nn.init.zeros_(self.adapter_b.weight)
        self.scale = alpha / rank

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.adapter_b(self.adapter_a(inputs)) * self.scale


def add_lora_to_attention(model: TinyLanguageModel, rank: int = 4) -> TinyLanguageModel:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.blocks:
        block.attention.query_key_value = LoRALinear(block.attention.query_key_value, rank=rank)
        block.attention.output_projection = LoRALinear(block.attention.output_projection, rank=rank)
    return model


def train_sft(model: TinyLanguageModel, examples: list[tuple[str, str]], tokenizer: CharacterTokenizer, steps: int, lr: float) -> None:
    batch = sft_batch(examples, tokenizer, model.config.block_size)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_cross_entropy(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def response_log_probability(model: TinyLanguageModel, tokenizer: CharacterTokenizer, prompt: str, response: str) -> torch.Tensor:
    sequence, response_start = format_instruction(prompt, response)
    ids = tokenizer.encode(sequence)[: model.config.block_size + 1]
    inputs = torch.tensor([ids[:-1]])
    targets = torch.tensor(ids[1:])
    logits, _ = model(inputs)
    log_probs = F.log_softmax(logits[0], dim=-1)
    positions = torch.arange(len(targets))
    selected = log_probs[positions, targets]
    response_mask = positions + 1 >= response_start
    return selected[response_mask].sum()


def dpo_loss(policy: TinyLanguageModel, reference: TinyLanguageModel, tokenizer: CharacterTokenizer, examples: list[tuple[str, str, str]], beta: float = 0.1) -> torch.Tensor:
    losses = []
    for prompt, chosen, rejected in examples:
        policy_margin = response_log_probability(policy, tokenizer, prompt, chosen) - response_log_probability(policy, tokenizer, prompt, rejected)
        with torch.no_grad():
            reference_margin = response_log_probability(reference, tokenizer, prompt, chosen) - response_log_probability(reference, tokenizer, prompt, rejected)
        losses.append(-F.logsigmoid(beta * (policy_margin - reference_margin)))
    return torch.stack(losses).mean()


@torch.no_grad()
def preference_accuracy(model: TinyLanguageModel, tokenizer: CharacterTokenizer, examples: list[tuple[str, str, str]]) -> float:
    wins = [response_log_probability(model, tokenizer, prompt, chosen) > response_log_probability(model, tokenizer, prompt, rejected) for prompt, chosen, rejected in examples]
    return float(torch.tensor(wins, dtype=torch.float32).mean())


@torch.no_grad()
def mean_preference_margin(model: TinyLanguageModel, tokenizer: CharacterTokenizer, examples: list[tuple[str, str, str]]) -> float:
    margins = [
        response_log_probability(model, tokenizer, prompt, chosen)
        - response_log_probability(model, tokenizer, prompt, rejected)
        for prompt, chosen, rejected in examples
    ]
    return float(torch.stack(margins).mean())


def train_dpo(policy: TinyLanguageModel, reference: TinyLanguageModel, tokenizer: CharacterTokenizer, steps: int = 120) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, weight_decay=0.0)
    initial = float(dpo_loss(policy, reference, tokenizer, PREFERENCE_TRAIN).detach())
    for _ in range(steps):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss = dpo_loss(policy, reference, tokenizer, PREFERENCE_TRAIN)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    return initial, float(dpo_loss(policy, reference, tokenizer, PREFERENCE_TRAIN).detach())


def run_adaptation_lab(*, output_dir: Path | None = None, seed: int = 42) -> dict:
    seed_everything(seed)
    torch.set_num_threads(1)
    curated, curation = curate_documents(RAW_DOMAIN_DOCUMENTS, DOMAIN_VALIDATION)
    domain_train = (" ".join(curated) + " ") * 20
    # Base corpus deliberately contains every lowercase character and punctuation used later.
    vocabulary_corpus = BASE_TRAIN + "abcdefghijklmnopqrstuvwxyz,:;?\n"
    tokenizer = CharacterTokenizer(vocabulary_corpus)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=96, d_model=32, n_heads=4, n_layers=1)

    base_model = TinyLanguageModel(config)
    optimize_language_model(base_model, BASE_TRAIN, tokenizer, steps=100, lr=2e-3)
    base_before = language_model_loss(base_model, BASE_VALIDATION, tokenizer)
    domain_before = language_model_loss(base_model, DOMAIN_VALIDATION, tokenizer)

    continued_model = copy.deepcopy(base_model)
    optimize_language_model(continued_model, domain_train, tokenizer, steps=100, lr=1e-3)
    base_after = language_model_loss(continued_model, BASE_VALIDATION, tokenizer)
    domain_after = language_model_loss(continued_model, DOMAIN_VALIDATION, tokenizer)

    full_sft = copy.deepcopy(continued_model)
    full_initial = sft_loss(full_sft, SFT_TRAIN, tokenizer)
    train_sft(full_sft, SFT_TRAIN, tokenizer, steps=140, lr=8e-4)
    full_final = sft_loss(full_sft, SFT_TRAIN, tokenizer)
    full_validation = sft_loss(full_sft, SFT_VALIDATION, tokenizer)

    lora_sft = add_lora_to_attention(copy.deepcopy(continued_model), rank=4)
    lora_initial = sft_loss(lora_sft, SFT_TRAIN, tokenizer)
    lora_zero_delta = max(
        float((continued_model(inputs)[0] - lora_sft(inputs)[0]).abs().max().detach())
        for inputs, _ in [sft_batch(SFT_TRAIN[:1], tokenizer, config.block_size)]
    )
    train_sft(lora_sft, SFT_TRAIN, tokenizer, steps=180, lr=3e-3)
    lora_final = sft_loss(lora_sft, SFT_TRAIN, tokenizer)
    lora_validation = sft_loss(lora_sft, SFT_VALIDATION, tokenizer)

    reference = copy.deepcopy(full_sft).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    policy = copy.deepcopy(full_sft)
    preference_before = preference_accuracy(policy, tokenizer, PREFERENCE_VALIDATION)
    preference_margin_before = mean_preference_margin(policy, tokenizer, PREFERENCE_VALIDATION)
    dpo_initial, dpo_final = train_dpo(policy, reference, tokenizer)
    preference_after = preference_accuracy(policy, tokenizer, PREFERENCE_VALIDATION)
    preference_margin_after = mean_preference_margin(policy, tokenizer, PREFERENCE_VALIDATION)

    report = {
        "schema_version": "1.0",
        "seed": seed,
        "data_pipeline": {
            **curation,
            "curated_sha256": hashlib.sha256(domain_train.encode()).hexdigest(),
            "rule_scope": "exact normalized duplicates and exact evaluation-substring contamination only",
        },
        "continued_pretraining": {
            "domain_loss_before": domain_before,
            "domain_loss_after": domain_after,
            "base_retention_loss_before": base_before,
            "base_retention_loss_after": base_after,
        },
        "instruction_tuning": {
            "full": {"train_loss_before": full_initial, "train_loss_after": full_final, "held_out_loss": full_validation, "trainable_parameters": trainable_parameter_count(full_sft)},
            "lora": {"train_loss_before": lora_initial, "train_loss_after": lora_final, "held_out_loss": lora_validation, "trainable_parameters": trainable_parameter_count(lora_sft), "zero_initial_logit_delta": lora_zero_delta},
            "total_base_parameters": sum(parameter.numel() for parameter in continued_model.parameters()),
        },
        "preference_alignment": {
            "objective": "DPO on response-token sequence log probabilities",
            "dpo_loss_before": dpo_initial,
            "dpo_loss_after": dpo_final,
            "held_out_preference_accuracy_before": preference_before,
            "held_out_preference_accuracy_after": preference_after,
            "held_out_preference_margin_before": preference_margin_before,
            "held_out_preference_margin_after": preference_margin_after,
            "sft_retention_loss_before": sft_loss(reference, SFT_VALIDATION, tokenizer),
            "sft_retention_loss_after": sft_loss(policy, SFT_VALIDATION, tokenizer),
        },
        "limitations": [
            "The small curated texts prove mechanics, not broad language or alignment quality.",
            "Exact contamination checks do not detect semantic or paraphrased leakage.",
            "The DPO preference set is tiny and style-focused; human agreement is not measured.",
            "PPO/RLHF is explained conceptually in the course but not presented as executed training.",
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(continued_model.state_dict(), output_dir / "continued_model.pt")
        torch.save(full_sft.state_dict(), output_dir / "full_sft_model.pt")
        torch.save(policy.state_dict(), output_dir / "dpo_policy.pt")
        (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_adaptation_lab(output_dir=args.output_dir, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
