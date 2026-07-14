from __future__ import annotations
import argparse
import json
from pathlib import Path
from .grounded import evaluate_answers, evaluate_security

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_answers(args.data_dir, args.output)
    report["security"] = evaluate_security(args.data_dir)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"metrics": report["metrics"], "failures": report["failure_counts"],
                      "security_pass_rate": report["security"]["pass_rate"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
