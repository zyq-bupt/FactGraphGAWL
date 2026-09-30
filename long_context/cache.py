"""按 sample_id + mode + config_hash 缓存局部/合并图，支持断点续跑。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cache_path(cache_dir: Path, sample_id: str, mode: str, config_hash: str) -> Path:
    safe = str(sample_id).replace("/", "_")
    return cache_dir / f"{safe}__{mode}__{config_hash}.json"


def load_cached(path: Path, expected_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("config_hash") != expected_hash:
        return None
    return data


def save_cached(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)
