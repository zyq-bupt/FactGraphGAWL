"""按窗口字符区间切片已有 dreeam 记录，复用 spaCy/BLINK/DREEAM 输出。"""

from __future__ import annotations

from copy import deepcopy

from long_context.offsets import TokenSpan, mention_char_span
from long_context.windows import Window


def _token_in_window(span: TokenSpan, window: Window) -> bool:
    if span.char_end <= span.char_start:
        return window.global_char_start <= span.char_start < window.global_char_end
    return span.char_start >= window.global_char_start and span.char_end <= window.global_char_end


def slice_section(
    section: dict,
    window: Window,
    token_spans: list[TokenSpan],
) -> dict:
    """生成可供 GraphBuilder.build_graph 使用的局部 dreeam section。"""
    in_old = [i for i, sp in enumerate(token_spans) if _token_in_window(sp, window)]
    old_to_new = {old: new for new, old in enumerate(in_old)}

    parse_trees = []
    for pt in section.get("parse_trees") or []:
        h = pt.get("h_idx")
        t = pt.get("t_idx")
        if h in old_to_new and t in old_to_new:
            item = dict(pt)
            item["h_idx"] = old_to_new[h]
            item["t_idx"] = old_to_new[t]
            parse_trees.append(item)

    old_mention_to_new: dict[int, int] = {}
    mentions = []
    for m in section.get("mentions") or []:
        cs, ce = mention_char_span(m, token_spans)
        if cs < window.global_char_start or ce > window.global_char_end:
            # 不完全落在窗口内：不跨窗口撕裂 mention
            continue
        tok_idx = [old_to_new[i] for i in (m.get("token_index") or []) if i in old_to_new]
        if not tok_idx:
            continue
        nm = deepcopy(m)
        new_id = len(mentions)
        old_id = m.get("mention_id", new_id)
        old_mention_to_new[int(old_id)] = new_id
        nm["mention_id"] = new_id
        nm["token_index"] = tok_idx
        nm["global_char_start"] = cs
        nm["global_char_end"] = ce
        nm["local_char_start"] = cs - window.global_char_start
        nm["local_char_end"] = ce - window.global_char_start
        nm["window_id"] = window.window_id
        mentions.append(nm)

    vertex_set = []
    old_ent_to_new: dict[int, int] = {}
    for old_e, entity in enumerate(section.get("vertexSet") or []):
        kept = []
        for ment in entity:
            mid = ment.get("mention_id")
            if mid in old_mention_to_new:
                nm = deepcopy(ment)
                nm["mention_id"] = old_mention_to_new[mid]
                tok_idx = [old_to_new[i] for i in (ment.get("token_index") or []) if i in old_to_new]
                nm["token_index"] = tok_idx
                kept.append(nm)
        if not kept:
            continue
        old_ent_to_new[old_e] = len(vertex_set)
        vertex_set.append(kept)

    ent_relation = []
    for rel in section.get("ent_relation") or []:
        h = rel.get("h_idx")
        t = rel.get("t_idx")
        if h in old_ent_to_new and t in old_ent_to_new:
            nr = dict(rel)
            nr["h_idx"] = old_ent_to_new[h]
            nr["t_idx"] = old_ent_to_new[t]
            nr["window_id"] = window.window_id
            ent_relation.append(nr)

    local_tokens = []
    for new_i, old_i in enumerate(in_old):
        sp = token_spans[old_i]
        local_tokens.append(
            {
                "local_idx": new_i,
                "old_idx": old_i,
                "surface": sp.surface,
                "global_char_start": sp.char_start,
                "global_char_end": sp.char_end,
                "local_char_start": sp.char_start - window.global_char_start,
                "local_char_end": sp.char_end - window.global_char_start,
                "window_id": window.window_id,
            }
        )

    sliced = {
        "title": section.get("title"),
        "text": window.text,
        "sents": _rebuild_sents(token_spans, in_old),
        "mentions": mentions,
        "parse_trees": parse_trees,
        "vertexSet": vertex_set,
        "ent_relation": ent_relation,
        "_local_tokens": local_tokens,
        "_window_id": window.window_id,
        "_old_to_new_token": old_to_new,
    }
    return sliced


def _rebuild_sents(token_spans: list[TokenSpan], in_old: list[int]) -> list[list[str]]:
    if not in_old:
        return []
    sents: list[list[str]] = []
    cur_sid = token_spans[in_old[0]].sent_id
    buf: list[str] = []
    for old_i in in_old:
        sp = token_spans[old_i]
        if sp.sent_id != cur_sid:
            if buf:
                sents.append(buf)
            buf = []
            cur_sid = sp.sent_id
        buf.append(sp.surface)
    if buf:
        sents.append(buf)
    return sents
