"""Command-line entry point for the RAG-07 reranking benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from .reranking import evaluate_reranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neural-model")
    parser.add_argument("--neural-revision")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    report = evaluate_reranking(
        args.data_dir,
        args.output,
        neural_model=args.neural_model,
        neural_revision=args.neural_revision,
        local_files_only=args.local_files_only,
    )
    evaluation = report["systems"]["evaluation"]
    baseline = evaluation["baseline_metrics"]
    reranked = evaluation["local_pair_reranker_metrics"]
    print(f"report: {args.output}")
    print(f"selected development alpha: {report['local_pair_scorer']['selected_alpha']:.1f}")
    print(f"held-out MRR: {baseline['mrr']:.4f} -> {reranked['mrr']:.4f}")
    print(f"held-out nDCG@5: {baseline['ndcg_at_k']:.4f} -> {reranked['ndcg_at_k']:.4f}")
    print(f"candidate hit rate: {reranked['candidate_hit_rate']:.4f}")
    if report["neural_cross_encoder"]:
        neural = report["neural_cross_encoder"]
        print(f"neural model: {neural['model']}")
        print(f"neural held-out MRR: {neural['metrics']['mrr']:.4f}")


if __name__ == "__main__":
    main()
