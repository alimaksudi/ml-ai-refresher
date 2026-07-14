from __future__ import annotations
import argparse
import json
from pathlib import Path
from .evaluation import evaluate

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.data_dir, args.output)
    summary = {key: {metric: round(value[metric], 4) for metric in ("recall_at_k", "mrr", "ndcg_at_k", "unanswerable_abstention_rate")} for key, value in report["experiments"].items()}
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
