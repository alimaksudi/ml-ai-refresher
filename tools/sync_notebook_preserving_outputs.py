"""Replace notebook source from a generated artifact while preserving run evidence.

Use this only when a learner intentionally keeps execution output in a working
notebook.  Builder output remains authoritative for cell source and ordering;
matching code cells carry their execution count, outputs, and cell metadata into
the refreshed notebook.
"""
from __future__ import annotations

import argparse
import ast
import difflib
from pathlib import Path

import nbformat


def normalized_code(source: str) -> str:
    """Ignore execution-only whitespace changes when matching code cells."""
    try:
        return ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        pass
    lines = [line.rstrip() for line in source.strip().splitlines()]
    return "\n".join(line for line in lines if line.strip())


def sync(generated_path: Path, working_path: Path) -> tuple[int, int]:
    generated = nbformat.read(generated_path, as_version=4)
    working = nbformat.read(working_path, as_version=4)

    evidence_by_source: dict[str, list] = {}
    evidence_cells = 0
    for cell in working.cells:
        if cell.cell_type != "code":
            continue
        has_evidence = cell.execution_count is not None or bool(cell.outputs)
        if not has_evidence:
            continue
        evidence_cells += 1
        evidence_by_source.setdefault(normalized_code(cell.source), []).append(cell)

    restored = 0
    restored_generated_ids: set[str] = set()
    for cell in generated.cells:
        if cell.cell_type != "code":
            continue
        matches = evidence_by_source.get(normalized_code(cell.source), [])
        if not matches:
            continue
        old_cell = matches.pop(0)
        cell.execution_count = old_cell.execution_count
        cell.outputs = old_cell.outputs
        cell.metadata.update(old_cell.metadata)
        restored += 1
        restored_generated_ids.add(cell.id)

    # A learner may add diagnostic prints while exploring.  Preserve such a cell
    # only when it has one clearly similar generated counterpart; never guess.
    remaining_old = [cell for cells in evidence_by_source.values() for cell in cells]
    remaining_new = [
        cell
        for cell in generated.cells
        if cell.cell_type == "code" and cell.id not in restored_generated_ids
    ]
    for old_cell in remaining_old:
        scored = sorted(
            (
                difflib.SequenceMatcher(
                    None, normalized_code(old_cell.source), normalized_code(new_cell.source)
                ).ratio(),
                new_cell,
            )
            for new_cell in remaining_new
        )
        best_score, best_cell = scored[-1]
        second_score = scored[-2][0] if len(scored) > 1 else 0.0
        if best_score < 0.70 or best_score - second_score < 0.10:
            continue
        best_cell.source = old_cell.source
        best_cell.execution_count = old_cell.execution_count
        best_cell.outputs = old_cell.outputs
        best_cell.metadata.update(old_cell.metadata)
        remaining_new.remove(best_cell)
        restored += 1

    if restored != evidence_cells:
        raise RuntimeError(
            f"refusing to write: restored {restored} of {evidence_cells} "
            "code cells containing execution evidence"
        )

    nbformat.write(generated, working_path)
    return restored, evidence_cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generated", type=Path)
    parser.add_argument("working", type=Path)
    args = parser.parse_args()
    restored, total = sync(args.generated, args.working)
    print(f"updated {args.working}; preserved execution evidence for {restored}/{total} cells")


if __name__ == "__main__":
    main()
