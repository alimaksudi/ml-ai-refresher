from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rag_evaluation import evaluate_rag_systems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_rag_systems(args.data_dir, args.output)
    summary = {
        mode: {
            **{name: round(value, 4) for name, value in result["metrics"].items()},
            "gate_passed": result["quality_gate"]["passed"],
        }
        for mode, result in report["systems"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
