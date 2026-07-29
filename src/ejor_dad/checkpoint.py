from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CheckpointStore:
    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root))
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        safe = key.replace("/", "__").replace("\\", "__").replace(":", "_")
        return self.root / f"{safe}.json"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def load(self, key: str) -> dict[str, Any]:
        return json.loads(self.path(key).read_text(encoding="utf-8"))

    def save(self, key: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("checkpoint_key", key)
        payload["updated_at_epoch"] = time.time()
        atomic_write_text(self.path(key), json.dumps(payload, indent=2, ensure_ascii=False))

    def get_or_compute(self, key: str, compute: Callable[[], dict[str, Any]], force: bool = False) -> dict[str, Any]:
        if not force and self.exists(key):
            return self.load(key)
        payload = compute()
        self.save(key, payload)
        return payload


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def atomic_write_dataframe(dataframe, path: str | Path, kind: str = "csv", **kwargs: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    if kind == "csv":
        dataframe.to_csv(temp_path, index=False, **kwargs)
    elif kind == "latex":
        dataframe.to_latex(temp_path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported dataframe output kind: {kind}")
    os.replace(temp_path, path)
