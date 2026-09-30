"""共享 IO、日志、哈希与配置加载。路径一律来自配置或命令行。"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

BD2T_DIR = Path(__file__).resolve().parent
REPO_ROOT = BD2T_DIR.parents[1]


def ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    here = str(BD2T_DIR)
    if here not in sys.path:
        sys.path.insert(0, here)


ensure_repo_on_path()


class MissingCandidatesError(RuntimeError):
    """数据集未提供摘要系统输出，且配置中也没有 candidates 文件。"""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置必须是 mapping: {path}")
    return data


def deep_set(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: dict = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def deep_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in dotted.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(Path(path))


def resolve_path(value: str | Path | None, *, base: Path | None = None) -> Path | None:
    if value is None or value == "":
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    root = base if base is not None else REPO_ROOT
    return (root / p).resolve()


def setup_logging(log_path: Path, *, level: int = logging.INFO) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bd2tsumm")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo: Path | None = None) -> str | None:
    root = repo or REPO_ROOT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return repr(value)
        return format(value, ".17g")
    if isinstance(value, (np.floating,)):
        return format(float(value), ".17g")
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_cell(row.get(k)) for k in fieldnames})
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_run_metadata(cfg: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "python": sys.version,
        "executable": sys.executable,
        "git_commit": git_commit(),
        "cwd": os.getcwd(),
        "timestamp_utc": utc_now(),
        "packages": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pyyaml": package_version("pyyaml"),
            "rouge-score": package_version("rouge-score"),
            "matplotlib": package_version("matplotlib"),
            "networkx": package_version("networkx"),
            "torch": package_version("torch"),
        },
        "config": json.loads(json.dumps(cfg, default=str)),
    }
    if extra:
        meta.update(dict(extra))
    return meta


def file_exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0
