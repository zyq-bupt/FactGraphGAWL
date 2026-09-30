"""滑动窗口实验配置。路径均可覆盖，不硬编码用户目录或 GPU。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LongContextConfig:
    mode: str = "prefix_1024"  # prefix_1024 | sliding_window | both
    max_model_tokens: int = 1024
    overlap_tokens: int = 256
    sentence_boundary: bool = True
    merge_entities_by_kb_id: bool = True
    infer_cross_window_relations: bool = False
    cache_local_graphs: bool = True

    domain: str = "GovReport"
    exclude_human1: bool = True
    require_success: bool = True

    tokenizer_path: str | None = None
    dreeam_dir: str = "/root/autodl-fs/zyq/unisumeval_data_gawl/dreeam_result/test"
    labels_jsonl: str = "/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2_processed.jsonl"
    output_dir: str = str(REPO_ROOT / "experiments" / "results" / "long_context_govreport")
    cache_dir: str | None = None

    seed: int = 20260818
    bootstrap_n: int = 10000
    limit: int | None = None
    score_tol: float = 1e-6

    wTT: float = 1.0
    wTM: float = 1.0
    wME: float = 0.5
    wEE: float = 0.0
    wl_t: int = 1
    use_emb_labels: bool = False
    use_node_labels: bool = True
    use_edge_labels: int = 2

    def resolved_cache_dir(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir)
        return Path(self.output_dir) / "cache"

    def config_hash(self) -> str:
        """缓存隔离键：窗口/合并/tokenizer/评分超参变化时失效。"""
        payload = {
            "max_model_tokens": self.max_model_tokens,
            "overlap_tokens": self.overlap_tokens,
            "sentence_boundary": self.sentence_boundary,
            "merge_entities_by_kb_id": self.merge_entities_by_kb_id,
            "infer_cross_window_relations": self.infer_cross_window_relations,
            "tokenizer_path": self.tokenizer_path,
            "wTT": self.wTT,
            "wTM": self.wTM,
            "wME": self.wME,
            "wEE": self.wEE,
            "wl_t": self.wl_t,
            "use_emb_labels": self.use_emb_labels,
            "use_node_labels": self.use_node_labels,
            "use_edge_labels": self.use_edge_labels,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config_hash"] = self.config_hash()
        return d


def _coerce(field_name: str, value: Any, example: Any) -> Any:
    if value is None:
        return None
    if isinstance(example, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if isinstance(example, int) and not isinstance(example, bool):
        return int(value)
    if isinstance(example, float):
        return float(value)
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 mapping: {path}")
    nested = data.get("long_context")
    if isinstance(nested, dict):
        merged = dict(data)
        merged.update(nested)
        return merged
    return data


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> LongContextConfig:
    cfg = LongContextConfig()
    if path is not None:
        raw = load_yaml(Path(path))
        known = {f.name for f in fields(LongContextConfig)}
        for k, v in raw.items():
            if k in known:
                setattr(cfg, k, _coerce(k, v, getattr(cfg, k)))
    if overrides:
        known = {f.name for f in fields(LongContextConfig)}
        for k, v in overrides.items():
            if k in known and v is not None:
                setattr(cfg, k, _coerce(k, v, getattr(cfg, k)))
    if cfg.mode not in {"prefix_1024", "sliding_window", "both"}:
        raise ValueError(f"未知 mode: {cfg.mode}")
    if cfg.max_model_tokens <= 0:
        raise ValueError("max_model_tokens 必须为正")
    if cfg.overlap_tokens < 0 or cfg.overlap_tokens >= cfg.max_model_tokens:
        raise ValueError("overlap_tokens 须满足 0 <= overlap < max_model_tokens")
    return cfg


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "long_context.yaml"))
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--long-context-mode", dest="mode", type=str, default=None,
                        choices=["prefix_1024", "sliding_window", "both"])
    parser.add_argument("--max-model-tokens", dest="max_model_tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", dest="overlap_tokens", type=int, default=None)
    parser.add_argument("--sentence-boundary", dest="sentence_boundary", action="store_true")
    parser.add_argument("--no-sentence-boundary", dest="sentence_boundary", action="store_false")
    parser.set_defaults(sentence_boundary=None)
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default=None)
    parser.add_argument("--dreeam-dir", dest="dreeam_dir", type=str, default=None)
    parser.add_argument("--labels-jsonl", dest="labels_jsonl", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--cache-dir", dest="cache_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bootstrap-n", dest="bootstrap_n", type=int, default=None)
    parser.add_argument("--wTT", type=float, default=None)
    parser.add_argument("--wTM", type=float, default=None)
    parser.add_argument("--wME", type=float, default=None)
    parser.add_argument("--wEE", type=float, default=None)
    parser.add_argument("--allow-mock-tokenizer", action="store_true",
                        help="无 HF tokenizer 时允许 mock（仅调试/测试）")
    parser.add_argument("--consistency-check", action="store_true",
                        help="在短文档上比较 prefix 与 sliding，不跑完整实验")
    parser.add_argument("--smoke", action="store_true", help="1–2 个样本冒烟")
    return parser


def config_from_args(args: argparse.Namespace) -> LongContextConfig:
    overrides = {}
    for f in fields(LongContextConfig):
        if hasattr(args, f.name) and getattr(args, f.name) is not None:
            overrides[f.name] = getattr(args, f.name)
    cfg_path = getattr(args, "config", None)
    path = Path(cfg_path) if cfg_path and Path(cfg_path).is_file() else None
    return load_config(path, overrides)
