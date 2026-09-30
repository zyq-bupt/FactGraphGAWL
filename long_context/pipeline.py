"""配对实验主流程：同批样本、摘要图不变、仅源文档图构建方式不同。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from build.graph_embedding.graph_builder import GraphBuilder
from common.data_loader_saver import load_data

from long_context.cache import cache_path, load_cached, save_cached
from long_context.config import LongContextConfig
from long_context.graphs import build_document_graph, nx_to_link_data
from long_context.scoring import score_article_candidate
from long_context.tokenizer_util import TokenizerLike, tokenizer_fingerprint

logger = logging.getLogger(__name__)

SAMPLE_FIELDS = [
    "sample_id",
    "domain",
    "document_model_tokens",
    "human_score",
    "baseline_score",
    "sliding_score",
    "baseline_num_windows",
    "sliding_num_windows",
    "baseline_unique_tokens",
    "sliding_unique_tokens",
    "baseline_token_coverage_ratio",
    "sliding_token_coverage_ratio",
    "baseline_num_token_nodes",
    "sliding_num_token_nodes",
    "baseline_num_mention_nodes",
    "sliding_num_mention_nodes",
    "baseline_num_entity_nodes",
    "sliding_num_entity_nodes",
    "baseline_num_E_tt",
    "sliding_num_E_tt",
    "baseline_num_E_tm",
    "sliding_num_E_tm",
    "baseline_num_E_me",
    "sliding_num_E_me",
    "baseline_num_E_ee",
    "sliding_num_E_ee",
    "baseline_graph_build_seconds",
    "sliding_graph_build_seconds",
    "baseline_total_seconds",
    "sliding_total_seconds",
    "status",
    "error_type",
]


def _empty_row(sample_id: str, domain: str, human: float | None) -> dict[str, Any]:
    row = {k: None for k in SAMPLE_FIELDS}
    row["sample_id"] = sample_id
    row["domain"] = domain
    row["human_score"] = human
    row["status"] = "pending"
    row["error_type"] = ""
    return row


def _fill_mode(row: dict, prefix: str, built: dict, score: float | None, total_s: float) -> None:
    row[f"{prefix}_score"] = score
    row[f"{prefix}_num_windows"] = built.get("num_windows")
    row[f"{prefix}_unique_tokens"] = built.get("unique_tokens")
    row[f"{prefix}_token_coverage_ratio"] = built.get("token_coverage_ratio")
    row[f"{prefix}_num_token_nodes"] = built.get("num_token_nodes")
    row[f"{prefix}_num_mention_nodes"] = built.get("num_mention_nodes")
    row[f"{prefix}_num_entity_nodes"] = built.get("num_entity_nodes")
    row[f"{prefix}_num_E_tt"] = built.get("num_E_tt")
    row[f"{prefix}_num_E_tm"] = built.get("num_E_tm")
    row[f"{prefix}_num_E_me"] = built.get("num_E_me")
    row[f"{prefix}_num_E_ee"] = built.get("num_E_ee")
    row[f"{prefix}_graph_build_seconds"] = built.get("graph_build_seconds")
    row[f"{prefix}_total_seconds"] = total_s
    row["document_model_tokens"] = built.get("document_model_tokens")


def build_candidate_graph(section: dict) -> dict:
    G = GraphBuilder().build_graph(section)
    return nx_to_link_data(G)


def run_one_mode(
    section: dict,
    tokenizer: TokenizerLike,
    cfg: LongContextConfig,
    mode: str,
) -> dict:
    return build_document_graph(
        section,
        tokenizer,
        mode=mode,
        max_model_tokens=cfg.max_model_tokens,
        overlap_tokens=cfg.overlap_tokens,
        sentence_boundary=cfg.sentence_boundary,
        merge_entities_by_kb_id=cfg.merge_entities_by_kb_id,
        infer_cross_window_relations=cfg.infer_cross_window_relations,
    )


def process_sample(
    sample: dict,
    dreeam_dir: Path,
    tokenizer: TokenizerLike,
    cfg: LongContextConfig,
    tmp_root: Path,
    *,
    modes: tuple[str, ...] = ("prefix_1024", "sliding_window"),
) -> dict[str, Any]:
    sample_id = str(sample.get("doc_id"))
    domain = sample.get("source") or sample.get("domain") or cfg.domain
    human = sample.get("faithfulness_score")
    row = _empty_row(sample_id, str(domain), float(human) if human is not None else None)
    cfg_hash = cfg.config_hash()
    cache_dir = cfg.resolved_cache_dir()

    dreeam_path = dreeam_dir / f"{sample_id}.json"
    if not dreeam_path.is_file():
        row["status"] = "failed"
        row["error_type"] = "missing_dreeam"
        return row

    try:
        dreeam = load_data(str(dreeam_path))
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error_type"] = f"load_dreeam:{type(exc).__name__}"
        return row

    article = dreeam.get("article")
    candidate = dreeam.get("candidate")
    if not isinstance(article, dict) or not isinstance(candidate, dict):
        row["status"] = "failed"
        row["error_type"] = "missing_article_or_candidate"
        return row

    try:
        cand_graph = build_candidate_graph(candidate)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error_type"] = f"candidate_graph:{type(exc).__name__}"
        return row

    tok_fp = tokenizer_fingerprint(tokenizer)
    built_by_mode: dict[str, dict] = {}
    for mode in modes:
        t0 = time.perf_counter()
        cpath = cache_path(cache_dir, sample_id, mode, cfg_hash)
        cached = load_cached(cpath, cfg_hash) if cfg.cache_local_graphs else None
        if cached and cached.get("tokenizer") == tok_fp:
            built = cached
            article_graph = cached["graph_data"]
        else:
            try:
                built = run_one_mode(article, tokenizer, cfg, mode)
            except Exception as exc:  # noqa: BLE001
                logger.exception("建图失败 sample=%s mode=%s", sample_id, mode)
                row["status"] = "failed"
                row["error_type"] = f"graph_build_{mode}:{type(exc).__name__}"
                return row
            article_graph = built["graph_data"]
            payload = {
                "config_hash": cfg_hash,
                "tokenizer": tok_fp,
                "mode": mode,
                "sample_id": sample_id,
                "graph_data": article_graph,
                "windows": built.get("windows"),
                "windows_used": built.get("windows_used"),
                "num_windows": built.get("num_windows"),
                "unique_tokens": built.get("unique_tokens"),
                "token_coverage_ratio": built.get("token_coverage_ratio"),
                "document_model_tokens": built.get("document_model_tokens"),
                "num_token_nodes": built.get("num_token_nodes"),
                "num_mention_nodes": built.get("num_mention_nodes"),
                "num_entity_nodes": built.get("num_entity_nodes"),
                "num_E_tt": built.get("num_E_tt"),
                "num_E_tm": built.get("num_E_tm"),
                "num_E_me": built.get("num_E_me"),
                "num_E_ee": built.get("num_E_ee"),
                "graph_build_seconds": built.get("graph_build_seconds"),
                "merge_stats": built.get("merge_stats"),
            }
            if cfg.cache_local_graphs:
                save_cached(cpath, payload)
            built = payload
        work = tmp_root / f"{sample_id}_{mode}"
        try:
            score = score_article_candidate(
                article_graph,
                cand_graph,
                work_dir=work,
                wTT=cfg.wTT,
                wTM=cfg.wTM,
                wME=cfg.wME,
                wEE=cfg.wEE,
                wl_t=cfg.wl_t,
                use_emb_labels=cfg.use_emb_labels,
                use_node_labels=cfg.use_node_labels,
                use_edge_labels=cfg.use_edge_labels,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("评分失败 sample=%s mode=%s", sample_id, mode)
            row["status"] = "failed"
            row["error_type"] = f"score_{mode}:{type(exc).__name__}"
            return row
        total_s = time.perf_counter() - t0
        prefix = "baseline" if mode == "prefix_1024" else "sliding"
        _fill_mode(row, prefix, built, score, total_s)
        built_by_mode[mode] = built

    row["status"] = "ok"
    row["error_type"] = ""
    row["_built"] = {k: {kk: vv for kk, vv in v.items() if kk != "graph_data"} for k, v in built_by_mode.items()}
    return row


def graphs_structurally_equal(a: dict, b: dict, *, score_tol: float = 1e-6) -> tuple[bool, str]:
    """短文档一致性：比较节点/边计数（允许内部 ID 排序差异）。"""
    from long_context.merge import graph_counts
    import networkx as nx

    ga = nx.node_link_graph(a)
    gb = nx.node_link_graph(b)
    ca, cb = graph_counts(ga), graph_counts(gb)
    for k in ca:
        if ca[k] != cb[k]:
            return False, f"{k}: {ca[k]} vs {cb[k]}"
    return True, "ok"
