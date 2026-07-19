"""Validate generated notebooks and their builder sources.

Fast validation (no kernel execution):
    python3 tools/validate_curriculum.py

Execute prerequisite and Section 01 notebooks as well:
    python3 tools/validate_curriculum.py --execute foundations

Execute the applied capstone notebook:
    python3 tools/validate_curriculum.py --execute capstone

Execute the complete curriculum:
    python3 tools/validate_curriculum.py --execute all
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_PATH = ROOT / "docs" / "CURRICULUM_PATH.json"
BUILDERS = sorted((ROOT / "tools" / "builders").rglob("*.py"))
NOTEBOOKS = sorted((ROOT / "notebooks").rglob("*.ipynb"))

FOUNDATION_PREFIXES = ("00_prerequisites/", "01_ml_foundations/")
SELF_CONTAINED_LESSON_IDS = {"PRE-01", "PRE-02", "PRE-03", "PRE-04", "PRE-05", "PRE-06", "FND-01", "FND-02", "FND-03", "FND-04", "CML-01", "CML-02", "CML-03", "CML-04", "CML-05", "CML-06"}
FORMULA_CUE = re.compile(
    r"(?i)\b(symbols?|where|means?|represents?|denotes?|read(?:s)? as|in words)\b"
)


@dataclass
class Finding:
    level: str
    path: str
    message: str


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def validate_notebook(path: Path) -> list[Finding]:
    rel = path.relative_to(ROOT).as_posix()
    findings: list[Finding] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    markdown = "\n".join(source(c) for c in cells if c.get("cell_type") == "markdown")
    rel_notebook = rel.removeprefix("notebooks/")
    expected = CURRICULUM_BY_PATH.get(rel_notebook)
    actual_metadata = notebook.get("metadata", {}).get("curriculum")
    if expected is None:
        findings.append(Finding("ERROR", rel, "missing from docs/CURRICULUM_PATH.json"))
    elif actual_metadata != expected["metadata"]:
        findings.append(
            Finding("ERROR", rel, "curriculum metadata differs from canonical path")
        )

    shared_mastery_cues = (
        "Worked example",
        "Guided practice",
        "Independent practice",
        "Challenge",
        "Self-check",
        "Solution and scoring rubric",
        "Common mistakes",
        "Readiness threshold",
    )
    if actual_metadata and actual_metadata.get("id") in SELF_CONTAINED_LESSON_IDS:
        mastery_cues = shared_mastery_cues + (
            "## Ready to move on?",
            "### Quick check",
            "### Teach it back",
            "### Memory aid",
        )
    else:
        mastery_cues = shared_mastery_cues + (
            "## Student Lesson Companion",
            "### Practical problem before history",
            "### Concept, analogy, and analogy limit",
            "### Use / avoid / alternatives",
            "## Lesson Close · Summary, Student Check, and Memory Aid",
            "### Five short student checks",
            "### Plain-language summary",
            "### One-sentence memory aid",
            "## Required Core Mastery Gate",
        )
    for cue in mastery_cues:
        if cue not in markdown:
            findings.append(Finding("ERROR", rel, f"missing mastery-gate cue: {cue}"))

    for number in range(1, 15):
        if not re.search(rf"(?m)^##\s+{number}(?:\s|\.|\s*·)", markdown):
            findings.append(Finding("ERROR", rel, f"missing section {number}"))

    for index, cell in enumerate(cells, 1):
        if cell.get("cell_type") != "code":
            continue
        code = source(cell)
        try:
            ast.parse(code)
        except SyntaxError as exc:
            findings.append(
                Finding("ERROR", rel, f"code cell {index} has invalid syntax: {exc.msg}")
            )
        if cell.get("outputs"):
            findings.append(Finding("ERROR", rel, f"code cell {index} has saved output"))
        if cell.get("execution_count") is not None:
            findings.append(
                Finding("ERROR", rel, f"code cell {index} has an execution count")
            )

    notebook_rel = rel.removeprefix("notebooks/")
    if notebook_rel.startswith(FOUNDATION_PREFIXES):
        for index, cell in enumerate(cells, 1):
            if cell.get("cell_type") != "markdown":
                continue
            text = source(cell)
            check_formula = notebook_rel.startswith("00_prerequisites/") or (
                notebook_rel.startswith("01_ml_foundations/") and "## 4" in text
            )
            if (
                check_formula
                and re.search(r"\$\$|\\\[|\\begin\{", text)
                and not FORMULA_CUE.search(text)
            ):
                findings.append(
                    Finding(
                        "ERROR",
                        rel,
                        f"formula in markdown cell {index} has no symbol explanation cue",
                    )
                )
    return findings


def load_curriculum() -> tuple[list[dict], dict[str, dict]]:
    payload = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))
    modules = payload.get("modules", [])
    by_id: dict[str, dict] = {}
    by_path: dict[str, dict] = {}
    for order, module in enumerate(modules, 1):
        module_id = module.get("id")
        path = module.get("path")
        if module_id in by_id:
            raise ValueError(f"duplicate curriculum id: {module_id}")
        if path in by_path:
            raise ValueError(f"duplicate curriculum path: {path}")
        enriched = dict(module)
        enriched["metadata"] = {
            "id": module_id,
            "prerequisites": module.get("prerequisites", []),
            "gate": module.get("gate"),
            "canonical_order": order,
        }
        by_id[module_id] = enriched
        by_path[path] = enriched
    return modules, by_path


CURRICULUM_MODULES, CURRICULUM_BY_PATH = load_curriculum()


def validate_curriculum_dependencies() -> list[Finding]:
    findings: list[Finding] = []
    known_ids = {module["id"] for module in CURRICULUM_MODULES}
    seen: set[str] = set()
    notebook_paths = {path.relative_to(ROOT / "notebooks").as_posix() for path in NOTEBOOKS}
    mapped_paths = {module["path"] for module in CURRICULUM_MODULES}
    builder_paths = {
        path.relative_to(ROOT / "tools" / "builders").with_suffix(".ipynb").as_posix()
        for path in BUILDERS
    }

    for missing in sorted(notebook_paths - mapped_paths):
        findings.append(Finding("ERROR", "docs/CURRICULUM_PATH.json", f"unmapped notebook: {missing}"))
    for missing in sorted(mapped_paths - notebook_paths):
        findings.append(Finding("ERROR", "docs/CURRICULUM_PATH.json", f"missing notebook: {missing}"))
    for missing in sorted(mapped_paths - builder_paths):
        findings.append(Finding("ERROR", "docs/CURRICULUM_PATH.json", f"missing builder: {missing}"))
    for extra in sorted(builder_paths - mapped_paths):
        findings.append(Finding("ERROR", "tools/builders", f"unmapped builder: {extra}"))

    for module in CURRICULUM_MODULES:
        module_id = module["id"]
        prerequisites = module.get("prerequisites", [])
        for prerequisite in prerequisites:
            if prerequisite not in known_ids:
                findings.append(Finding("ERROR", module["path"], f"unknown prerequisite {prerequisite}"))
            elif prerequisite not in seen:
                findings.append(
                    Finding("ERROR", module["path"], f"forward prerequisite {prerequisite} for {module_id}")
                )
        seen.add(module_id)
    return findings


def rebuild_to_temp() -> list[Finding]:
    findings: list[Finding] = []
    with tempfile.TemporaryDirectory(prefix="ml-ai-refresher-build-") as temp:
        temp_root = Path(temp)
        env = os.environ.copy()
        env["NB_OUTPUT_DIR"] = str(temp_root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        for builder in BUILDERS:
            result = subprocess.run(
                [sys.executable, str(builder)],
                cwd=ROOT / "tools",
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                findings.append(
                    Finding("ERROR", builder.relative_to(ROOT).as_posix(), result.stderr.strip())
                )

        generated = sorted(temp_root.rglob("*.ipynb"))
        if len(generated) != len(BUILDERS):
            findings.append(
                Finding(
                    "ERROR",
                    "tools/builders",
                    f"{len(BUILDERS)} builders produced {len(generated)} notebooks",
                )
            )

        for generated_path in generated:
            rel = generated_path.relative_to(temp_root)
            committed_path = ROOT / "notebooks" / rel
            if not committed_path.exists():
                findings.append(Finding("ERROR", rel.as_posix(), "generated notebook is not committed"))
                continue
            generated_nb = json.loads(generated_path.read_text(encoding="utf-8"))
            committed_nb = json.loads(committed_path.read_text(encoding="utf-8"))
            generated_cells = [
                (c.get("cell_type"), source(c)) for c in generated_nb.get("cells", [])
            ]
            committed_cells = [
                (c.get("cell_type"), source(c)) for c in committed_nb.get("cells", [])
            ]
            if generated_cells != committed_cells:
                findings.append(
                    Finding("ERROR", rel.as_posix(), "committed cells differ from builder output")
                )
    return findings


def execute_notebooks(scope: str) -> list[Finding]:
    if scope == "none":
        return []
    import nbformat
    from nbclient import NotebookClient

    findings: list[Finding] = []
    selected = NOTEBOOKS
    if scope == "foundations":
        selected = [
            path
            for path in NOTEBOOKS
            if path.relative_to(ROOT / "notebooks").as_posix().startswith(FOUNDATION_PREFIXES)
        ]
    elif scope == "capstone":
        selected = [
            path
            for path in NOTEBOOKS
            if path.relative_to(ROOT / "notebooks").as_posix().startswith("11_capstone/")
        ]
    elif scope == "classical":
        modules = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))["modules"]
        cutoff = next(index for index, item in enumerate(modules) if item["id"] == "CML-06")
        classical_paths = {
            module["path"]
            for module in modules[: cutoff + 1]
        }
        selected = [
            path
            for path in NOTEBOOKS
            if path.relative_to(ROOT / "notebooks").as_posix() in classical_paths
        ]

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for index, path in enumerate(selected, 1):
        rel = path.relative_to(ROOT).as_posix()
        try:
            notebook = nbformat.read(path, as_version=4)
            NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
                allow_errors=False,
            ).execute()
            print(f"EXEC PASS {index:02d}/{len(selected):02d} {rel}")
        except Exception as exc:
            findings.append(Finding("ERROR", rel, f"execution failed: {type(exc).__name__}: {exc}"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        choices=("none", "foundations", "classical", "capstone", "all"),
        default="none",
    )
    parser.add_argument("--skip-generated-check", action="store_true")
    args = parser.parse_args()

    findings: list[Finding] = []
    findings.extend(validate_curriculum_dependencies())
    for path in NOTEBOOKS:
        findings.extend(validate_notebook(path))

    if not args.skip_generated_check:
        findings.extend(rebuild_to_temp())

    findings.extend(execute_notebooks(args.execute))

    for finding in findings:
        print(f"{finding.level}: {finding.path}: {finding.message}")
    errors = sum(f.level == "ERROR" for f in findings)
    print(
        f"Validated {len(NOTEBOOKS)} notebooks and {len(BUILDERS)} builders: "
        f"{errors} error(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
