from __future__ import annotations

from pathlib import Path
from typing import Iterable


def iter_files(path: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted(files)
