#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def clean_notebook(nb: dict[str, Any]) -> dict[str, Any]:
    cells = nb.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            if not isinstance(cell.get("metadata"), dict):
                cell["metadata"] = {}

    if not isinstance(nb.get("metadata"), dict):
        nb["metadata"] = {}

    return nb


def clean_text(text: str) -> str:
    nb = json.loads(text)
    if not isinstance(nb, dict):
        raise ValueError("notebook root must be a JSON object")
    cleaned = clean_notebook(nb)
    return json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n"


def clean_stdin_to_stdout() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        sys.stdout.write(clean_text(raw))
    except Exception as exc:  # pragma: no cover
        print(f"ipynb_clean_filter: {exc}", file=sys.stderr)
        return 1
    return 0


def clean_in_place(paths: list[Path]) -> int:
    rc = 0
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8")
            cleaned = clean_text(raw)
            path.write_text(cleaned, encoding="utf-8")
            print(f"cleaned {path}")
        except Exception as exc:  # pragma: no cover
            print(f"ipynb_clean_filter: failed to clean {path}: {exc}", file=sys.stderr)
            rc = 1
    return rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Jupyter notebooks for git by stripping code cell outputs and "
            "execution counts. Default mode reads notebook JSON from stdin and writes "
            "cleaned JSON to stdout."
        )
    )
    parser.add_argument(
        "--in-place",
        nargs="+",
        type=Path,
        help="Clean one or more notebook files in place.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.in_place:
        return clean_in_place(args.in_place)
    return clean_stdin_to_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
