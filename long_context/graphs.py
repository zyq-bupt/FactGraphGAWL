"""接入现有 GraphBuilder：prefix 复用 build_graph；sliding 为窗口切片后合并。"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any

import networkx as nx

from build.graph_embedding.graph_builder import GraphBuilder
from long_context.coverage import token_coverage_ratio
from long_context.merge import graph_counts, merge_window_graphs
from long_context.offsets import TokenSpan, align_sents_to_text, mention_char_span
from long_context.slice_dreeam import slice_section
from long_context.tokenizer_util import TokenizerLike
from long_context.windows import Window, split_windows, validate_windows

logger = logging.getLogger(__name__)


def nx_to_link_data(G: nx.Graph) -> dict:
    data = nx.node_link_data(G)
    for node in data.get("nodes") or []:
        if "node_id" not in node and "id" in node:
            node["node_id"] = node["id"]
    return data


def annotate_local_graph(G: nx.DiGraph, sliced: dict, window: Window, token_spans: list[TokenSpan]) -> None:
    for tok in sliced.get("_local_tokens") or []:
        nid = f"Token_{tok['local_idx']}"
        if nid not in G:
            continue
        G.nodes[nid].update(
            {
                "window_id": window.window_id,
                "local_char_start": tok["local_char_start"],
                "local_char_end": tok["local_char_end"],
                "global_char_start": tok["global_char_start"],
                "global_char_end": tok["global_char_end"],
                "surface": tok["surface"],
            }
        )
    for m in sliced.get("mentions") or []:
        nid = f"Mention_{m['mention_id']}"
        if nid not in G:
            continue
        G.nodes[nid].update(
            {
                "window_id": window.window_id,
                "global_char_start": m.get("global_char_start"),
                "global_char_end": m.get("global_char_end"),
                "local_char_start": m.get("local_char_start"),
                "local_char_end": m.get("local_char_end"),
                "surface": m.get("name"),
                "type": m.get("type"),
            }
        )
    for ei, entity in enumerate(sliced.get("vertexSet") or []):
        nid = f"Entity_{ei}"
        if nid not in G or not entity:
            continue
        head = entity[0]
        G.nodes[nid].update(
            {
                "window_id": window.window_id,
                "wikidata_id": head.get("wikidata_id") or G.nodes[nid].get("wikidata_id"),
                "wikipedia_id": head.get("wikipedia_id"),
                "score": head.get("score"),
                "url": head.get("url"),
                "entity_name": head.get("entity_name"),
                "surface": head.get("name") or head.get("mention"),
            }
        )
    for rel in sliced.get("ent_relation") or []:
        u = f"Entity_{rel['h_idx']}"
        v = f"Entity_{rel['t_idx']}"
        if G.has_edge(u, v) and rel.get("score") is not None:
            old = G.edges[u, v].get("score")
            if old is None or float(rel["score"]) > float(old):
                G.edges[u, v]["score"] = rel["score"]
            G.edges[u, v]["window_id"] = window.window_id


def build_local_graph(sliced: dict, window: Window, token_spans: list[TokenSpan]) -> nx.DiGraph:
    G = GraphBuilder().build_graph(sliced)
    annotate_local_graph(G, sliced, window, token_spans)
    return G


def _prepare_spans(section: dict):
    text = section.get("text") or ""
    sents = section.get("sents") or []
    if sents:
        token_spans, sent_spans = align_sents_to_text(text, sents)
    else:
        token_spans, sent_spans = [], None
    return text, token_spans, sent_spans


def make_windows_for_section(
    section: dict,
    tokenizer: TokenizerLike,
    *,
    max_model_tokens: int,
    overlap_tokens: int,
    sentence_boundary: bool,
) -> tuple[str, list[TokenSpan], list[Window]]:
    text, token_spans, sent_spans = _prepare_spans(section)
    windows = split_windows(
        text,
        tokenizer,
        max_model_tokens=max_model_tokens,
        overlap_tokens=overlap_tokens,
        sentence_boundary=sentence_boundary,
        sentences=sent_spans,
    )
    validate_windows(windows, text, tokenizer, max_model_tokens)
    return text, token_spans, windows


def build_document_graph(
    section: dict,
    tokenizer: TokenizerLike,
    *,
    mode: str,
    max_model_tokens: int = 1024,
    overlap_tokens: int = 256,
    sentence_boundary: bool = True,
    merge_entities_by_kb_id: bool = True,
    infer_cross_window_relations: bool = False,
) -> dict[str, Any]:
    """
    prefix_1024：仅第一窗口（短文档则整篇，直接 GraphBuilder）。
    sliding_window：覆盖全文的窗口图合并。
    """
    if mode not in {"prefix_1024", "sliding_window"}:
        raise ValueError(mode)
    t0 = time.perf_counter()
    text, token_spans, windows = make_windows_for_section(
        section,
        tokenizer,
        max_model_tokens=max_model_tokens,
        overlap_tokens=overlap_tokens,
        sentence_boundary=sentence_boundary,
    )
    if mode == "prefix_1024":
        used = windows[:1]
    else:
        used = windows

    local_graphs: list[nx.DiGraph] = []
    # 短文档单窗口：直接走现有 GraphBuilder 全量路径
    if mode == "prefix_1024" and len(windows) == 1:
        G = GraphBuilder().build_graph(section)
        dummy = Window(
            window_id=0,
            text=text,
            global_char_start=0,
            global_char_end=len(text),
            model_token_count=used[0].model_token_count if used else 0,
            sentence_start_idx=0,
            sentence_end_idx=None,
        )
        sliced_full = deepcopy(section)
        sliced_full["_local_tokens"] = [
            {
                "local_idx": t.token_index,
                "old_idx": t.token_index,
                "surface": t.surface,
                "global_char_start": t.char_start,
                "global_char_end": t.char_end,
                "local_char_start": t.char_start,
                "local_char_end": t.char_end,
                "window_id": 0,
            }
            for t in token_spans
        ]
        # mention 全局区间
        for m in sliced_full.get("mentions") or []:
            cs, ce = mention_char_span(m, token_spans)
            m["global_char_start"] = cs
            m["global_char_end"] = ce
            m["local_char_start"] = cs
            m["local_char_end"] = ce
        annotate_local_graph(G, sliced_full, dummy, token_spans)
        local_graphs = [G]
        used = windows
    else:
        for w in used:
            sliced = slice_section(section, w, token_spans)
            local_graphs.append(build_local_graph(sliced, w, token_spans))

    G, merge_stats = merge_window_graphs(
        local_graphs,
        used,
        merge_entities_by_kb_id=merge_entities_by_kb_id,
        infer_cross_window_relations=infer_cross_window_relations,
    )
    cov = token_coverage_ratio(text, used, tokenizer)
    elapsed = time.perf_counter() - t0
    counts = graph_counts(G)
    return {
        "graph": G,
        "graph_data": nx_to_link_data(G),
        "windows": [w.to_dict() for w in windows],
        "windows_used": [w.to_dict() for w in used],
        "num_windows": len(used),
        "num_windows_full_split": len(windows),
        "merge_stats": merge_stats,
        "graph_build_seconds": elapsed,
        **cov,
        **counts,
        "mode": mode,
    }
