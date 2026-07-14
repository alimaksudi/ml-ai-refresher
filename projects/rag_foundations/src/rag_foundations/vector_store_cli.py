from __future__ import annotations

import argparse
import json
from pathlib import Path

from .vector_store import evaluate_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and evaluate the persistent vector-store lesson")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_vector_store(args.data_dir, args.index_dir, args.output)
    print(json.dumps({
        "numpy_exact": {key: value for key, value in report["numpy_exact"].items() if key != "rows"},
        "qdrant_local_exact": {key: value for key, value in report["qdrant_local_exact"].items() if key != "rows"},
        "parity": report["parity"],
        "persistence": report["persistence"],
        "policy_filters": report["policy_filters"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

