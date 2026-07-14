"""Regenerate every notebook from its builder.

    python3 tools/build_all.py            # build all
    python3 tools/build_all.py 05 18      # build only builders matching these tokens

Each file in ``tools/builders/`` is a self-contained script that, when run,
writes one ``.ipynb`` into ``notebooks/``. We run them in isolated subprocesses
so a failure in one builder never half-writes another.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
def _builder_order(path):
    name = os.path.basename(path)
    return (0 if name.startswith("phase_minus1_") else 1, name)


BUILDERS = sorted(glob.glob(os.path.join(HERE, "builders", "*.py")), key=_builder_order)


def main(argv):
    filters = [a for a in argv if not a.startswith("-")]
    selected = [
        b
        for b in BUILDERS
        if os.path.basename(b) != "__init__.py"
        and (not filters or any(tok in os.path.basename(b) for tok in filters))
    ]
    if not selected:
        print("no builders matched", filters)
        return 1

    failures = []
    for b in selected:
        rel = os.path.relpath(b, os.path.dirname(HERE))
        print(f"\n>>> building {rel}")
        res = subprocess.run([sys.executable, b], cwd=HERE)
        if res.returncode != 0:
            failures.append(rel)

    print(f"\n{'=' * 60}")
    print(f"built {len(selected) - len(failures)}/{len(selected)} notebooks")
    if failures:
        print("FAILED:", *failures, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
