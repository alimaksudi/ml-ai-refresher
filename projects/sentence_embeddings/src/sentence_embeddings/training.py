"""Train and evaluate a tiny local sentence bi-encoder without API credentials."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from transformer_families.models import FamilyConfig

from .data import EXAMPLES, all_training_texts, assert_no_text_leakage, evaluation_documents, evaluation_queries
from .evaluation import TfidfEncoder, retrieval_details, retrieval_metrics
from .model import SentenceEncoder, multiple_negatives_ranking_loss
from .tokenizer import WordTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def encode_texts(model: SentenceEncoder, tokenizer: WordTokenizer, texts: list[str]) -> np.ndarray:
    model.eval()
    token_ids, attention_mask = tokenizer.encode_batch(texts)
    return model(token_ids, attention_mask).cpu().numpy()


def evaluate_model(model: SentenceEncoder, tokenizer: WordTokenizer) -> dict[str, float]:
    queries = encode_texts(model, tokenizer, evaluation_queries())
    documents = encode_texts(model, tokenizer, evaluation_documents())
    return retrieval_metrics(queries, documents)


def median_encoding_ms_per_text(model: SentenceEncoder, tokenizer: WordTokenizer, texts: list[str]) -> float:
    durations = []
    for _ in range(30):
        start = time.perf_counter()
        encode_texts(model, tokenizer, texts)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations) * 1000 / len(texts))


def run_experiment(*, output_dir: Path | None = None, seed: int = 42, steps: int = 320) -> dict:
    seed_everything(seed)
    torch.set_num_threads(1)
    assert_no_text_leakage()

    # Evaluation documents may define vocabulary, but held-out evaluation queries never do.
    tokenizer = WordTokenizer.fit([*all_training_texts(), *evaluation_documents()])
    config = FamilyConfig(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
        d_model=48,
        n_heads=4,
        n_layers=2,
    )
    model = SentenceEncoder(config, pooling="mean")
    untrained_metrics = evaluate_model(model, tokenizer)

    tfidf = TfidfEncoder().fit(evaluation_documents())
    tfidf_metrics = retrieval_metrics(
        tfidf.transform(evaluation_queries()),
        tfidf.transform(evaluation_documents()),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    intent_to_row = {row.intent: row for row in EXAMPLES}
    initial_loss = None
    final_loss = None
    for step in range(steps):
        anchors = [row.train_queries[step % len(row.train_queries)] for row in EXAMPLES]
        positives = [row.train_documents[(step // len(row.train_queries)) % len(row.train_documents)] for row in EXAMPLES]
        hard_negatives = [intent_to_row[row.hard_negative_intent].train_documents[0] for row in EXAMPLES]
        anchor_ids, anchor_mask = tokenizer.encode_batch(anchors)
        positive_ids, positive_mask = tokenizer.encode_batch(positives)
        negative_ids, negative_mask = tokenizer.encode_batch(hard_negatives)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        anchor_vectors = model(anchor_ids, anchor_mask)
        positive_vectors = model(positive_ids, positive_mask)
        negative_vectors = model(negative_ids, negative_mask)
        loss = multiple_negatives_ranking_loss(
            anchor_vectors,
            positive_vectors,
            temperature=0.08,
            hard_negatives=negative_vectors,
        )
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())

    trained_query_vectors = encode_texts(model, tokenizer, evaluation_queries())
    trained_document_vectors = encode_texts(model, tokenizer, evaluation_documents())
    trained_metrics = retrieval_metrics(trained_query_vectors, trained_document_vectors)
    details = retrieval_details(trained_query_vectors, trained_document_vectors)
    per_intent = {
        row.intent: {**detail, "correct_at_1": detail["rank"] == 1}
        for row, detail in zip(EXAMPLES, details)
    }
    report = {
        "schema_version": "1.0",
        "seed": seed,
        "steps": steps,
        "dataset": {
            "domain": "curated customer-support retrieval",
            "intents": len(EXAMPLES),
            "training_texts": len(all_training_texts()),
            "evaluation_queries": len(evaluation_queries()),
            "exact_text_overlap": 0,
        },
        "training": {
            "objective": "multiple-negatives ranking with explicit hard negatives",
            "temperature": 0.08,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        },
        "tfidf_baseline": tfidf_metrics,
        "untrained_transformer": untrained_metrics,
        "trained_contrastive_encoder": trained_metrics,
        "evaluation_slices": per_intent,
        "latency": {
            "device": "cpu",
            "batch_size": len(EXAMPLES),
            "median_encoding_ms_per_text": median_encoding_ms_per_text(
                model, tokenizer, evaluation_queries()
            ),
            "warning": "Environment-specific diagnostic; remeasure on deployment hardware.",
        },
        "limitations": [
            "The dataset is small and curated for diagnosis, not a production benchmark.",
            "Vocabulary is fitted on training text and evaluation documents, but never evaluation queries.",
            "A single seed and eight intents do not establish broad language generalization.",
            "TF-IDF may remain competitive when query and document vocabulary overlaps.",
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "sentence_encoder.pt")
        (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=320)
    args = parser.parse_args()
    print(json.dumps(run_experiment(output_dir=args.output_dir, seed=args.seed, steps=args.steps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
