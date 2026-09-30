"""调用现有 FK-Graph + S²-K (PT) 计算 CompKG-FactEval。"""

from __future__ import annotations

import argparse
import logging
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from util import (
    ensure_repo_on_path,
    load_config,
    read_jsonl,
    resolve_path,
    setup_logging,
    sha256_text,
    write_csv,
    write_json,
)
from long_context.cache import cache_path, load_cached, save_cached
from long_context.config import LongContextConfig

ensure_repo_on_path()

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def mock_compkg_score(source_text: str, candidate: str) -> float:
    """词集合 Jaccard，方向为越高越一致。仅用于测试/无模型冒烟。"""
    a = set(TOKEN_RE.findall((source_text or "").lower()))
    b = set(TOKEN_RE.findall((candidate or "").lower()))
    if not a or not b:
        raise ValueError("empty_source_or_candidate_tokens")
    return float(len(a & b) / len(a | b))


def apply_score_direction(raw: float, *, higher_is_better: bool, treat_as_distance: bool) -> float:
    if not math.isfinite(raw):
        raise ValueError(f"non_finite_score:{raw}")
    score = -raw if treat_as_distance else raw
    if not higher_is_better:
        score = -score
    return float(score)


def _graph_empty(graph_data: dict | None) -> bool:
    if not graph_data:
        return True
    nodes = graph_data.get("nodes") or []
    links = graph_data.get("links") or graph_data.get("edges") or []
    return len(nodes) == 0 and len(links) == 0


def _load_json(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dreeam_section(path: Path, key: str) -> dict:
    data = _load_json(path)
    if key in data and isinstance(data[key], dict):
        return data[key]
    if "text" in data and ("parse_trees" in data or "sents" in data):
        return data
    raise FileNotFoundError(f"{path} 中找不到 DREEAM 小节 {key}")


def candidate_dreeam_path(directory: Path, event_id: str, system_name: str) -> Path:
    safe_sys = str(system_name).replace("/", "_")
    named = directory / f"{event_id}__{safe_sys}.json"
    if named.is_file():
        return named
    nested = directory / safe_sys / f"{event_id}.json"
    if nested.is_file():
        return nested
    return named


def long_context_cfg_from_experiment(cfg: dict) -> LongContextConfig:
    lc = cfg.get("long_context") or {}
    ck = cfg.get("compkg") or {}
    return LongContextConfig(
        mode="sliding_window",
        max_model_tokens=int(lc.get("max_model_tokens") or 1024),
        overlap_tokens=int(lc.get("overlap_tokens") or 256),
        sentence_boundary=bool(lc.get("sentence_boundary", True)),
        merge_entities_by_kb_id=bool(lc.get("merge_entities_by_kb_id", True)),
        infer_cross_window_relations=bool(lc.get("infer_cross_window_relations", False)),
        cache_local_graphs=bool(ck.get("cache_source_graphs", True)),
        tokenizer_path=lc.get("tokenizer_path"),
        wTT=float(ck.get("wTT", 1.0)),
        wTM=float(ck.get("wTM", 1.0)),
        wME=float(ck.get("wME", 0.5)),
        wEE=float(ck.get("wEE", 0.0)),
        wl_t=int(ck.get("wl_t", 1)),
        use_emb_labels=bool(ck.get("use_emb_labels", False)),
        use_node_labels=bool(ck.get("use_node_labels", True)),
        use_edge_labels=int(ck.get("use_edge_labels", 2)),
        output_dir=".",
    )


def build_source_graph(
    article_section: dict,
    tokenizer,
    lc_cfg: LongContextConfig,
    *,
    cache_dir: Path,
    event_id: str,
    text_hash: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    from long_context.graphs import build_document_graph
    from long_context.tokenizer_util import tokenizer_fingerprint

    mode = "sliding_window"
    cfg_hash = lc_cfg.config_hash()
    sample_key = f"{event_id}_{text_hash[:12]}"
    cpath = cache_path(cache_dir / "sources", sample_key, mode, cfg_hash)
    tok_fp = tokenizer_fingerprint(tokenizer)
    cached = load_cached(cpath, cfg_hash) if lc_cfg.cache_local_graphs else None
    if cached and cached.get("tokenizer") == tok_fp and cached.get("source_sha256") == text_hash:
        logger.info("复用源图缓存 %s", cpath.name)
        return cached
    built = build_document_graph(
        article_section,
        tokenizer,
        mode=mode,
        max_model_tokens=lc_cfg.max_model_tokens,
        overlap_tokens=lc_cfg.overlap_tokens,
        sentence_boundary=lc_cfg.sentence_boundary,
        merge_entities_by_kb_id=lc_cfg.merge_entities_by_kb_id,
        infer_cross_window_relations=lc_cfg.infer_cross_window_relations,
    )
    payload = {
        "config_hash": cfg_hash,
        "tokenizer": tok_fp,
        "mode": mode,
        "sample_id": sample_key,
        "source_sha256": text_hash,
        "graph_data": built["graph_data"],
        "num_windows": built.get("num_windows"),
        "num_windows_full_split": built.get("num_windows_full_split"),
        "token_coverage_ratio": built.get("token_coverage_ratio"),
        "document_model_tokens": built.get("document_model_tokens"),
        "unique_tokens": built.get("unique_tokens"),
        "num_token_nodes": built.get("num_token_nodes"),
        "num_mention_nodes": built.get("num_mention_nodes"),
        "num_entity_nodes": built.get("num_entity_nodes"),
        "graph_build_seconds": built.get("graph_build_seconds"),
        "merge_stats": built.get("merge_stats"),
    }
    if lc_cfg.cache_local_graphs:
        save_cached(cpath, payload)
    return payload


def build_candidate_graph_cached(
    candidate_section: dict,
    *,
    cache_dir: Path,
    event_id: str,
    system_name: str,
    text_hash: str,
    cfg_hash: str,
    enabled: bool,
    logger: logging.Logger,
) -> dict:
    from long_context.pipeline import build_candidate_graph

    sample_key = f"{event_id}__{system_name}_{text_hash[:12]}"
    cpath = cache_path(cache_dir / "candidates", sample_key, "summary", cfg_hash)
    cached = load_cached(cpath, cfg_hash) if enabled else None
    if cached and cached.get("source_sha256") == text_hash:
        logger.info("复用摘要图缓存 %s", cpath.name)
        return cached
    graph_data = build_candidate_graph(candidate_section)
    payload = {
        "config_hash": cfg_hash,
        "source_sha256": text_hash,
        "graph_data": graph_data,
        "sample_id": sample_key,
    }
    if enabled:
        save_cached(cpath, payload)
    return payload


def score_graphs(article_graph: dict, candidate_graph: dict, cfg: dict, work_dir: Path) -> float:
    from long_context.scoring import score_article_candidate

    ck = cfg.get("compkg") or {}
    try:
        import torch

        torch_ctx = torch.no_grad()
    except Exception:
        from contextlib import nullcontext

        torch_ctx = nullcontext()
    with torch_ctx:
        return float(
            score_article_candidate(
                article_graph,
                candidate_graph,
                work_dir=work_dir,
                wTT=float(ck.get("wTT", 1.0)),
                wTM=float(ck.get("wTM", 1.0)),
                wME=float(ck.get("wME", 0.5)),
                wEE=float(ck.get("wEE", 0.0)),
                wl_t=int(ck.get("wl_t", 1)),
                use_emb_labels=bool(ck.get("use_emb_labels", False)),
                use_node_labels=bool(ck.get("use_node_labels", True)),
                use_edge_labels=int(ck.get("use_edge_labels", 2)),
            )
        )


def run_compkg(
    cfg: dict,
    output_dir: Path,
    logger: logging.Logger,
    *,
    mock_fn: Callable[[str, str], float] | None = None,
) -> dict[str, Any]:
    out = cfg.get("output") or {}
    ck = cfg.get("compkg") or {}
    lc = cfg.get("long_context") or {}
    backend = str(ck.get("backend") or "real")
    if mock_fn is not None:
        backend = "mock"
    variant = str(ck.get("variant") or "S2-K_PT")
    sources = {r["event_id"]: r for r in read_jsonl(output_dir / out.get("prepared_sources", "prepared_sources.jsonl"))}
    candidates = read_jsonl(output_dir / out.get("prepared_candidates", "prepared_candidates.jsonl"))
    if not candidates:
        raise RuntimeError(
            "没有候选摘要，无法计算 CompKG-FactEval。请提供系统输出，不要使用参考摘要代替。"
        )
    if str(lc.get("mode") or "sliding_window") != "sliding_window" and not lc.get("allow_prefix_truncation"):
        raise RuntimeError("禁止静默截断：long_context.mode 必须为 sliding_window")

    higher = bool(ck.get("higher_is_better", True))
    as_dist = bool(ck.get("treat_as_distance", False))
    cache_dir = output_dir / "cache"
    tmp_root = output_dir / "tmp_gawl"
    tmp_root.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    lc_cfg = None
    dreeam_src = resolve_path(ck.get("dreeam_source_dir"), base=Path(cfg["dataset"].get("data_root") or "."))
    dreeam_cand = resolve_path(ck.get("dreeam_candidate_dir"), base=Path(cfg["dataset"].get("data_root") or "."))
    if backend == "real":
        from long_context.tokenizer_util import load_tokenizer

        tokenizer = load_tokenizer(lc.get("tokenizer_path"), allow_mock=bool(lc.get("allow_mock_tokenizer")))
        lc_cfg = long_context_cfg_from_experiment(cfg)

    source_cache: dict[str, dict] = {}
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        eid = str(cand["event_id"])
        sysn = str(cand["system_name"])
        summary = cand.get("candidate_summary") or ""
        src = sources.get(eid)
        t0 = time.perf_counter()
        row = {
            "event_id": eid,
            "system_name": sysn,
            "compkg_score": None,
            "compkg_score_raw": None,
            "compkg_variant": variant,
            "source_num_tokens": src.get("num_tokens") if src else None,
            "source_num_windows": src.get("num_windows") if src else None,
            "status": "failed",
            "error_message": "",
            "runtime_seconds": None,
            "backend": backend,
        }
        try:
            if src is None or not str(src.get("source_text") or "").strip():
                raise ValueError("empty_source_text")
            if not str(summary).strip():
                raise ValueError("empty_candidate")
            if backend == "mock":
                fn = mock_fn or mock_compkg_score
                raw = float(fn(src["source_text"], summary))
            else:
                if dreeam_src is None or not dreeam_src.is_dir():
                    raise FileNotFoundError(
                        "missing_dreeam_source_dir: 现有 FK-Graph 需要 DREEAM 格式的 article 小节，"
                        "请设置 compkg.dreeam_source_dir。不会从原始推文静默截断建图。"
                    )
                src_path = dreeam_src / f"{eid}.json"
                if not src_path.is_file():
                    raise FileNotFoundError(f"missing_dreeam_source:{src_path}")
                if eid not in source_cache:
                    article = load_dreeam_section(src_path, "article")
                    source_cache[eid] = build_source_graph(
                        article,
                        tokenizer,
                        lc_cfg,
                        cache_dir=cache_dir,
                        event_id=eid,
                        text_hash=str(src.get("source_sha256") or sha256_text(src["source_text"])),
                        logger=logger,
                    )
                built_src = source_cache[eid]
                row["source_num_windows"] = built_src.get("num_windows")
                row["source_num_tokens"] = built_src.get("document_model_tokens") or src.get("num_tokens")
                if _graph_empty(built_src.get("graph_data")):
                    raise ValueError("empty_source_graph")
                if dreeam_cand is None or not dreeam_cand.is_dir():
                    raise FileNotFoundError(
                        "missing_dreeam_candidate_dir: 请为每个事件–系统提供 DREEAM candidate JSON"
                    )
                cpath = candidate_dreeam_path(dreeam_cand, eid, sysn)
                if not cpath.is_file():
                    raise FileNotFoundError(f"missing_dreeam_candidate:{cpath}")
                cand_section = load_dreeam_section(cpath, "candidate")
                cand_built = build_candidate_graph_cached(
                    cand_section,
                    cache_dir=cache_dir,
                    event_id=eid,
                    system_name=sysn,
                    text_hash=sha256_text(summary),
                    cfg_hash=lc_cfg.config_hash(),
                    enabled=bool(ck.get("cache_candidate_graphs", True)),
                    logger=logger,
                )
                if _graph_empty(cand_built.get("graph_data")):
                    raise ValueError("empty_candidate_graph")
                work = tmp_root / f"{eid}__{sysn}".replace("/", "_")
                if work.exists():
                    shutil.rmtree(work, ignore_errors=True)
                raw = score_graphs(built_src["graph_data"], cand_built["graph_data"], cfg, work)
            directed = apply_score_direction(raw, higher_is_better=higher, treat_as_distance=as_dist)
            if not math.isfinite(directed):
                raise ValueError("non_finite_directed_score")
            row["compkg_score_raw"] = raw
            row["compkg_score"] = directed
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            logger.error("CompKG 失败 event=%s system=%s: %s: %s", eid, sysn, type(exc).__name__, exc)
            row["status"] = "failed"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
        row["runtime_seconds"] = time.perf_counter() - t0
        rows.append(row)

    write_csv(
        output_dir / out.get("compkg_scores", "compkg_scores.csv"),
        rows,
        [
            "event_id",
            "system_name",
            "compkg_score",
            "compkg_score_raw",
            "compkg_variant",
            "source_num_tokens",
            "source_num_windows",
            "status",
            "error_message",
            "runtime_seconds",
            "backend",
        ],
    )
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    write_json(
        output_dir / "compkg_metadata.json",
        {
            "variant": variant,
            "backend": backend,
            "higher_is_better": higher,
            "treat_as_distance": as_dist,
            "kernel": "kernel.gawl.compute_gawl_kernel_v2 via long_context.scoring.score_article_candidate",
            "n_ok": n_ok,
            "n_failed": len(rows) - n_ok,
        },
    )
    logger.info("CompKG 完成 ok=%s failed=%s variant=%s backend=%s", n_ok, len(rows) - n_ok, variant, backend)
    return {"rows": rows}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BD2TSumm CompKG-FactEval")
    p.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--mock-compkg", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    if args.mock_compkg:
        cfg.setdefault("compkg", {})
        cfg["compkg"]["backend"] = "mock"
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / "logs" / "compute_compkg.log")
    run_compkg(cfg, output_dir, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
