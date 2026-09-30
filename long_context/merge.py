"""合并窗口局部 FK-Graph。不推断跨窗口新关系。"""

from __future__ import annotations

import logging

import networkx as nx

from long_context.offsets import normalize_surface
from long_context.windows import Window

logger = logging.getLogger(__name__)

EMPTY_KB = {"", "none", "null", "nil", "-1", "nan", "None"}


def canonical_kb_id(node: dict) -> str | None:
    wid = node.get("wikidata_id")
    if wid is not None and str(wid).strip() not in EMPTY_KB:
        return f"wd:{str(wid).strip()}"
    wiki = node.get("wikipedia_id", node.get("name"))
    if wiki is not None and str(wiki).strip() not in EMPTY_KB:
        return f"wiki:{str(wiki).strip()}"
    return None


def _node_label(nid: str, data: dict) -> str:
    return str(data.get("label") or "")


def _is_token(nid: str, data: dict) -> bool:
    return _node_label(nid, data) == "Token" or str(nid).startswith("Token_")


def _is_mention(nid: str, data: dict) -> bool:
    return _node_label(nid, data) == "Mention" or str(nid).startswith("Mention_")


def _is_entity(nid: str, data: dict) -> bool:
    return _node_label(nid, data) == "Entity" or str(nid).startswith("Entity_")


def _span_key(data: dict) -> tuple[int, int]:
    return int(data.get("global_char_start") or 0), int(data.get("global_char_end") or 0)


def graph_counts(G: nx.Graph) -> dict[str, int]:
    n_token = n_mention = n_entity = 0
    for nid, data in G.nodes(data=True):
        if _is_token(nid, data):
            n_token += 1
        elif _is_mention(nid, data):
            n_mention += 1
        elif _is_entity(nid, data):
            n_entity += 1
    e_tt = e_tm = e_me = e_ee = 0
    for u, v, data in G.edges(data=True):
        du, dv = G.nodes[u], G.nodes[v]
        ut, um, ue = _is_token(u, du), _is_mention(u, du), _is_entity(u, du)
        vt, vm, ve = _is_token(v, dv), _is_mention(v, dv), _is_entity(v, dv)
        if ut and vt:
            e_tt += 1
        elif (ut and vm) or (um and vt):
            e_tm += 1
        elif (um and ve) or (ue and vm):
            e_me += 1
        elif ue and ve:
            e_ee += 1
    return {
        "num_token_nodes": n_token,
        "num_mention_nodes": n_mention,
        "num_entity_nodes": n_entity,
        "num_E_tt": e_tt,
        "num_E_tm": e_tm,
        "num_E_me": e_me,
        "num_E_ee": e_ee,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
    }


def merge_window_graphs(
    window_graphs: list[nx.DiGraph],
    windows: list[Window] | None = None,
    *,
    merge_entities_by_kb_id: bool = True,
    infer_cross_window_relations: bool = False,
) -> tuple[nx.DiGraph, dict]:
    if infer_cross_window_relations:
        raise ValueError("初步实验禁止推断跨窗口关系")
    stats = {
        "token_span_conflicts": 0,
        "mention_boundary_conflicts": 0,
        "entity_embed_conflicts": 0,
        "n_windows": len(window_graphs),
    }
    if not window_graphs:
        return nx.DiGraph(), stats
    if len(window_graphs) == 1:
        g = window_graphs[0].copy()
        stats.update({f"merged_{k}": v for k, v in graph_counts(g).items()})
        return g, stats

    merged = nx.DiGraph()
    token_key_to_id: dict[tuple, str] = {}
    mention_key_to_id: dict[tuple, str] = {}
    entity_key_to_id: dict[str, str] = {}
    remap: list[dict[str, str]] = []

    token_n = mention_n = entity_n = 0
    mention_spans: list[tuple[int, int, str]] = []

    def add_token(data: dict, window_id: int | None) -> str:
        nonlocal token_n
        surface = normalize_surface(str(data.get("name") or data.get("surface") or ""))
        cs, ce = _span_key(data)
        key = (cs, ce, surface)
        if key in token_key_to_id:
            nid = token_key_to_id[key]
            prov = list(merged.nodes[nid].get("provenance_windows") or [])
            if window_id is not None and window_id not in prov:
                prov.append(window_id)
                merged.nodes[nid]["provenance_windows"] = prov
            return nid
        # 同一字符区间、不同 surface → 冲突
        for (ocs, oce, osurf), exist_id in token_key_to_id.items():
            if (ocs, oce) == (cs, ce) and osurf != surface:
                stats["token_span_conflicts"] += 1
                logger.warning("token 边界冲突: %s vs %s @ (%s,%s)", osurf, surface, cs, ce)
                return exist_id
        nid = f"Token_{token_n}"
        token_n += 1
        token_key_to_id[key] = nid
        attrs = dict(data)
        attrs.update(
            {
                "node_id": nid,
                "label": "Token",
                "name": data.get("name") or surface,
                "global_char_start": cs,
                "global_char_end": ce,
                "provenance_windows": [window_id] if window_id is not None else [],
            }
        )
        merged.add_node(nid, **attrs)
        return nid

    def add_mention(data: dict, window_id: int | None) -> str:
        nonlocal mention_n
        surface = normalize_surface(str(data.get("name") or ""))
        ner = data.get("type")
        cs, ce = _span_key(data)
        key = (cs, ce, surface, ner)
        if key in mention_key_to_id:
            nid = mention_key_to_id[key]
            prov = list(merged.nodes[nid].get("provenance_windows") or [])
            if window_id is not None and window_id not in prov:
                prov.append(window_id)
                merged.nodes[nid]["provenance_windows"] = prov
            return nid
        for mcs, mce, ms in mention_spans:
            overlap = not (ce <= mcs or cs >= mce)
            if overlap and (cs, ce, surface) != (mcs, mce, ms):
                stats["mention_boundary_conflicts"] += 1
        mention_spans.append((cs, ce, surface))
        nid = f"Mention_{mention_n}"
        mention_n += 1
        mention_key_to_id[key] = nid
        attrs = dict(data)
        attrs.update(
            {
                "node_id": nid,
                "label": "Mention",
                "name": data.get("name") or surface,
                "type": ner,
                "global_char_start": cs,
                "global_char_end": ce,
                "provenance_windows": [window_id] if window_id is not None else [],
            }
        )
        merged.add_node(nid, **attrs)
        return nid

    def add_entity(data: dict, window_id: int | None, local_id: str) -> str:
        nonlocal entity_n
        kb = canonical_kb_id(data) if merge_entities_by_kb_id else None
        if kb is None:
            kb = f"unlinked:{window_id}:{local_id}"
        if kb in entity_key_to_id:
            nid = entity_key_to_id[kb]
            prov = list(merged.nodes[nid].get("provenance_windows") or [])
            if window_id is not None and window_id not in prov:
                prov.append(window_id)
            merged.nodes[nid]["provenance_windows"] = prov
            mentions_prov = list(merged.nodes[nid].get("provenance_mentions") or [])
            mid = data.get("source_mention_id")
            if mid is not None:
                mentions_prov.append(mid)
                merged.nodes[nid]["provenance_mentions"] = mentions_prov
            return nid
        nid = f"Entity_{entity_n}"
        entity_n += 1
        entity_key_to_id[kb] = nid
        attrs = dict(data)
        attrs.update(
            {
                "node_id": nid,
                "label": "Entity",
                "canonical_kb_id": kb,
                "provenance_windows": [window_id] if window_id is not None else [],
                "provenance_mentions": [data.get("source_mention_id")] if data.get("source_mention_id") is not None else [],
            }
        )
        merged.add_node(nid, **attrs)
        return nid

    edge_best: dict[tuple, dict] = {}

    for gi, G in enumerate(window_graphs):
        wid = None
        if windows and gi < len(windows):
            wid = windows[gi].window_id
        local_map: dict[str, str] = {}
        for nid, data in G.nodes(data=True):
            if _is_token(nid, data):
                local_map[nid] = add_token(data, wid)
            elif _is_mention(nid, data):
                local_map[nid] = add_mention(data, wid)
            elif _is_entity(nid, data):
                local_map[nid] = add_entity(data, wid, nid)
            else:
                local_map[nid] = nid
                if nid not in merged:
                    merged.add_node(nid, **dict(data))
        remap.append(local_map)
        for u, v, edata in G.edges(data=True):
            mu, mv = local_map.get(u), local_map.get(v)
            if mu is None or mv is None:
                continue
            rel = edata.get("relationship")
            key = (mu, mv, rel)
            prev = edge_best.get(key)
            score = edata.get("score")
            if prev is None:
                rec = dict(edata)
                rec["count"] = 1
                rec["source_windows"] = [wid] if wid is not None else []
                edge_best[key] = rec
            else:
                prev["count"] = int(prev.get("count") or 1) + 1
                if wid is not None:
                    sw = list(prev.get("source_windows") or [])
                    if wid not in sw:
                        sw.append(wid)
                    prev["source_windows"] = sw
                if score is not None:
                    old = prev.get("score")
                    if old is None or float(score) > float(old):
                        prev["score"] = score

    for (u, v, rel), edata in edge_best.items():
        merged.add_edge(u, v, **edata)

    stats.update({f"merged_{k}": v for k, v in graph_counts(merged).items()})
    return merged, stats
