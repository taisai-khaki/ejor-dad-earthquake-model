from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    """Return stable absolute-path to SHA-256 mappings for existing files."""
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cannot hash missing file: {path}")
        hashes[str(path)] = sha256_file(path)
    return dict(sorted(hashes.items()))


def reproducibility_metadata(
    *,
    input_files: Iterable[str | Path],
    source_files: Iterable[str | Path],
) -> dict[str, object]:
    """Describe the exact data, code, and numerical runtime used by a run."""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "input_sha256": file_hashes(input_files),
        "source_sha256": file_hashes(source_files),
    }
