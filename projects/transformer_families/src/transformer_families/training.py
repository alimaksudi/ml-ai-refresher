"""Real controlled training tasks for GPT-, BERT-, and T5-style models."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .models import (
    DecoderOnlyModel,
    EncoderDecoderModel,
    EncoderOnlyModel,
    FamilyConfig,
    parameter_count,
)

PAD, MASK, BOS, EOS, DIGIT_OFFSET = 0, 1, 2, 3, 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def digit_token(value: int) -> int:
    return DIGIT_OFFSET + value


def make_cycle_sequences(count: int = 80, length: int = 8) -> torch.Tensor:
    return torch.tensor(
        [[digit_token((start + position) % 10) for position in range(length)] for start in range(count)],
        dtype=torch.long,
    )


def make_classification_data(count: int = 80, length: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    rows, labels = [], []
    for index in range(count):
        start = index % 10
        ascending = index % 2 == 0
        step = 1 if ascending else -1
        rows.append([digit_token((start + step * position) % 10) for position in range(length)])
        labels.append(int(ascending))
    return torch.tensor(rows), torch.tensor(labels)


def make_reversal_data(count: int = 80, source_length: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_rows, decoder_inputs, targets = [], [], []
    for index in range(count):
        values = [(index * 3 + position * 2) % 10 for position in range(source_length)]
        source = [digit_token(value) for value in values]
        target = [digit_token(value) for value in reversed(values)] + [EOS]
        source_rows.append(source)
        decoder_inputs.append([BOS, *target[:-1]])
        targets.append(target)
    return torch.tensor(source_rows), torch.tensor(decoder_inputs), torch.tensor(targets)


def optimize(loss_function, parameters, steps: int, learning_rate: float = 3e-3) -> tuple[float, float]:
    parameter_list = list(parameters)
    optimizer = torch.optim.AdamW(parameter_list, lr=learning_rate, weight_decay=1e-3)
    initial = float(loss_function().item())
    final = initial
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameter_list, 1.0)
        optimizer.step()
        final = float(loss.item())
    return initial, final


def run_family_lab(
    *,
    output_dir: Path | None = None,
    seed: int = 42,
    steps: int = 120,
) -> dict:
    seed_everything(seed)
    torch.set_num_threads(1)
    config = FamilyConfig()

    # GPT: causal next-token prediction.
    gpt = DecoderOnlyModel(config)
    cycle = make_cycle_sequences()
    gpt_inputs, gpt_targets = cycle[:, :-1], cycle[:, 1:]
    gpt_mask = torch.ones_like(gpt_inputs, dtype=torch.bool)

    def gpt_loss():
        logits = gpt(gpt_inputs, gpt_mask)
        return F.cross_entropy(logits.reshape(-1, config.vocab_size), gpt_targets.reshape(-1))

    gpt_initial, gpt_final = optimize(gpt_loss, gpt.parameters(), steps)
    with torch.no_grad():
        gpt_accuracy = float((gpt(gpt_inputs, gpt_mask).argmax(-1) == gpt_targets).float().mean())

    # BERT MLM: one center token is hidden; both left and right context remain visible.
    bert = EncoderOnlyModel(config)
    bert_original = make_cycle_sequences(length=7)
    mask_position = 3
    bert_inputs = bert_original.clone()
    bert_targets = bert_original[:, mask_position]
    bert_inputs[:, mask_position] = MASK
    bert_mask = torch.ones_like(bert_inputs, dtype=torch.bool)

    def mlm_loss():
        logits = bert(bert_inputs, bert_mask)[:, mask_position]
        return F.cross_entropy(logits, bert_targets)

    mlm_initial, mlm_final = optimize(mlm_loss, bert.parameters(), steps)
    with torch.no_grad():
        mlm_accuracy = float((bert(bert_inputs, bert_mask)[:, mask_position].argmax(-1) == bert_targets).float().mean())

    # BERT classifier: reuse the trained encoder, add a pooled classification objective.
    class_inputs, class_targets = make_classification_data()
    class_mask = torch.ones_like(class_inputs, dtype=torch.bool)

    def classification_loss():
        return F.cross_entropy(bert.classify(class_inputs, class_mask), class_targets)

    classification_initial, classification_final = optimize(classification_loss, bert.parameters(), steps // 2)
    with torch.no_grad():
        classification_accuracy = float((bert.classify(class_inputs, class_mask).argmax(-1) == class_targets).float().mean())

    # T5: encoder reads a source sequence; causal decoder produces its reversal.
    t5 = EncoderDecoderModel(config)
    source, decoder_inputs, target = make_reversal_data()
    source_mask = torch.ones_like(source, dtype=torch.bool)
    target_mask = torch.ones_like(decoder_inputs, dtype=torch.bool)

    def t5_loss():
        logits = t5(source, decoder_inputs, source_mask, target_mask)
        return F.cross_entropy(logits.reshape(-1, config.vocab_size), target.reshape(-1))

    t5_initial, t5_final = optimize(t5_loss, t5.parameters(), steps)
    with torch.no_grad():
        t5_logits = t5(source, decoder_inputs, source_mask, target_mask)
        t5_token_accuracy = float((t5_logits.argmax(-1) == target).float().mean())
        generated_with_bos = t5.generate(
            source[:8],
            source_mask[:8],
            bos_token_id=BOS,
            eos_token_id=EOS,
            max_new_tokens=target.shape[1],
        )
        if generated_with_bos.shape[1] < target.shape[1] + 1:
            missing = target.shape[1] + 1 - generated_with_bos.shape[1]
            generated_with_bos = F.pad(generated_with_bos, (0, missing), value=EOS)
        generated = generated_with_bos[:, 1 : target.shape[1] + 1]
        t5_exact_match = float((generated == target[:8]).all(dim=1).float().mean())

    cross_shape = t5.decoder_blocks[0].cross_attention.last_score_shape
    report = {
        "schema_version": "1.0",
        "seed": seed,
        "steps_per_primary_objective": steps,
        "shared_config": config.__dict__,
        "gpt_decoder_only": {
            "objective": "causal next-token prediction",
            "initial_loss": gpt_initial,
            "final_loss": gpt_final,
            "token_accuracy": gpt_accuracy,
            "parameters": parameter_count(gpt),
        },
        "bert_encoder_only": {
            "objectives": ["masked-token prediction", "sequence classification"],
            "mlm_initial_loss": mlm_initial,
            "mlm_final_loss": mlm_final,
            "mlm_accuracy": mlm_accuracy,
            "classification_initial_loss": classification_initial,
            "classification_final_loss": classification_final,
            "classification_accuracy": classification_accuracy,
            "parameters": parameter_count(bert),
        },
        "t5_encoder_decoder": {
            "objective": "source-to-target reversal with teacher forcing",
            "initial_loss": t5_initial,
            "final_loss": t5_final,
            "teacher_forced_token_accuracy": t5_token_accuracy,
            "greedy_exact_match_first_eight": t5_exact_match,
            "cross_attention_score_shape": list(cross_shape) if cross_shape else None,
            "parameters": parameter_count(t5),
        },
        "limitations": [
            "Tasks are synthetic diagnostics, not language-quality benchmarks.",
            "Training accuracy verifies learnability but does not estimate broad generalization.",
            "Parameter counts differ because encoder-decoder models contain both stacks.",
            "The models omit many production details such as tokenizer pipelines and relative positions.",
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(gpt.state_dict(), output_dir / "gpt_decoder.pt")
        torch.save(bert.state_dict(), output_dir / "bert_encoder.pt")
        torch.save(t5.state_dict(), output_dir / "t5_encoder_decoder.pt")
        (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(run_family_lab(output_dir=args.output_dir, seed=args.seed, steps=args.steps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
