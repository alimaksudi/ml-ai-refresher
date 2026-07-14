"""Regenerate every notebook from its builder.

    python3 tools/build_all.py            # build all
    python3 tools/build_all.py RAG-05     # build one lesson by semantic ID

Each file in ``tools/builders/`` is a self-contained script that, when run,
writes one ``.ipynb`` into ``notebooks/``. We run them in isolated subprocesses
so a failure in one builder never half-writes another.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).parent
CURRICULUM_PATH = ROOT / "docs" / "CURRICULUM_PATH.json"


def curriculum_builders() -> list[tuple[str, Path, str]]:
    modules = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))["modules"]
    return [
        (
            module["id"],
            Path(HERE) / "builders" / Path(module["path"]).with_suffix(".py"),
            module["path"],
        )
        for module in modules
    ]


BUILDERS = curriculum_builders()


def main(argv):
    filters = [a for a in argv if not a.startswith("-")]
    selected = []
    for module_id, builder, notebook_path in BUILDERS:
        searchable = f"{module_id} {builder.name} {notebook_path}".lower()
        if not filters or any(token.lower() in searchable for token in filters):
            selected.append((module_id, builder))
    if not selected:
        print("no builders matched", filters)
        return 1

    failures = []
    for module_id, builder in selected:
        rel = builder.relative_to(ROOT)
        if not builder.exists():
            failures.append(f"{module_id}: missing {rel}")
            continue
        print(f"\n>>> building {module_id} · {rel}")
        res = subprocess.run([sys.executable, str(builder)], cwd=HERE)
        if res.returncode != 0:
            failures.append(f"{module_id}: {rel}")

    print(f"\n{'=' * 60}")
    print(f"built {len(selected) - len(failures)}/{len(selected)} notebooks")
    if failures:
        print("FAILED:", *failures, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
