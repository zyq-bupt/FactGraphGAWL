"""将 dreeam/spaCy 的 token、句子对齐到原文全局字符区间。不使用 str.find 猜测。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from long_context.windows import SentenceSpan

logger = logging.getLogger(__name__)


@dataclass
class TokenSpan:
    token_index: int
    sent_id: int
    char_start: int
    char_end: int
    surface: str


def _skip_space(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def _match_token(text: str, tok: str, pos: int) -> tuple[int, int] | None:
    """从 pos 起匹配一个 spaCy token，只向前看有限窗口，不在全文 find。"""
    n = len(text)
    pos = _skip_space(text, pos)
    if not tok:
        return pos, pos
    if text.startswith(tok, pos):
        return pos, pos + len(tok)
    # spaCy 常见规范化：不在全文搜索，只在当前位置附近
    window_n = max(len(tok) * 4, 24)
    region = text[pos:min(n, pos + window_n)]
    if region.startswith(tok):
        return pos, pos + len(tok)
    # 大小写
    low = region.lower()
    tlow = tok.lower()
    if low.startswith(tlow):
        return pos, pos + len(tok)
    # 当前位置起的最短前缀：若 token 是原文的前缀（带空格被吃掉）
    if tok in {"``", "''"}:
        for q in ('"', "“", "”", "'"):
            if text.startswith(q, pos):
                return pos, pos + len(q)
    # 在有限窗口内找第一次出现（仍是局部，不是全文 find）
    idx = region.find(tok)
    if idx >= 0:
        return pos + idx, pos + idx + len(tok)
    idx = low.find(tlow)
    if idx >= 0:
        return pos + idx, pos + idx + len(tok)
    return None


def align_sents_to_text(text: str, sents: list[list[str]]) -> tuple[list[TokenSpan], list[SentenceSpan]]:
    tokens: list[TokenSpan] = []
    sentences: list[SentenceSpan] = []
    pos = 0
    flat = 0
    text = text or ""
    for sid, sent in enumerate(sents or []):
        sent_start = None
        sent_end = pos
        for tok in sent:
            matched = _match_token(text, tok, pos)
            if matched is None:
                pos = _skip_space(text, pos)
                # 无法对齐时不回退到全文搜索；消耗 min(len(tok), 剩余) 并记 warning
                take = min(max(len(tok), 1), max(0, len(text) - pos))
                if take == 0 and pos >= len(text):
                    logger.warning("token 超出原文末尾: %r idx=%s", tok, flat)
                    start = end = len(text)
                else:
                    logger.warning("token 局部对齐失败，按当前位置消耗: %r idx=%s pos=%s", tok, flat, pos)
                    start, end = pos, pos + max(take, 1)
                    end = min(end, len(text))
                pos = end
            else:
                start, end = matched
                pos = end
            if sent_start is None:
                sent_start = start
            sent_end = end
            tokens.append(TokenSpan(flat, sid, start, end, tok))
            flat += 1
        if sent_start is None:
            sent_start = _skip_space(text, sent_end)
        sentences.append(SentenceSpan(sid, sent_start, sent_end, text[sent_start:sent_end]))
    # 句末空白并入当前句，便于窗口覆盖
    for i, sp in enumerate(sentences):
        nxt = sentences[i + 1].char_start if i + 1 < len(sentences) else len(text)
        end = sp.char_end
        while end < nxt and end < len(text) and text[end].isspace():
            end += 1
        sentences[i] = SentenceSpan(sp.sent_idx, sp.char_start, end, text[sp.char_start:end])
    if sentences:
        # 文首尚未对齐的非空白并入第一句
        first = sentences[0]
        lead = 0
        while lead < first.char_start and text[lead].isspace():
            lead += 1
        if lead < first.char_start:
            sentences[0] = SentenceSpan(
                first.sent_idx, lead, first.char_end, text[lead:first.char_end]
            )
        last = sentences[-1]
        if last.char_end < len(text):
            sentences[-1] = SentenceSpan(
                last.sent_idx, last.char_start, len(text), text[last.char_start:]
            )
    elif text:
        sentences = [SentenceSpan(0, 0, len(text), text)]
    return tokens, sentences


def mention_char_span(mention: dict, token_spans: list[TokenSpan]) -> tuple[int, int]:
    idxs = mention.get("token_index") or []
    if not idxs:
        return 0, 0
    spans = [token_spans[i] for i in idxs if 0 <= i < len(token_spans)]
    if not spans:
        return 0, 0
    return spans[0].char_start, spans[-1].char_end


def normalize_surface(text: str) -> str:
    """仅做空白折叠，不改变实体大小写或词形。"""
    return " ".join((text or "").split())
