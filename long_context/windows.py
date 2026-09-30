"""按模型 tokenizer 真实长度做句界感知滑动窗口。"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

from long_context.tokenizer_util import (
    TokenizerLike,
    content_token_offsets,
    count_tokens,
    effective_max_length,
)

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Window:
    window_id: int
    text: str
    global_char_start: int
    global_char_end: int
    model_token_count: int
    sentence_start_idx: int | None
    sentence_end_idx: int | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentenceSpan:
    sent_idx: int
    char_start: int
    char_end: int
    text: str


def fallback_sentence_spans(text: str) -> list[SentenceSpan]:
    if not text:
        return []
    spans: list[SentenceSpan] = []
    start = 0
    parts = _SENT_SPLIT.split(text)
    if len(parts) == 1:
        return [SentenceSpan(0, 0, len(text), text)]
    # 用指针切分，避免 str.find 在重复句子上错位
    idx = 0
    pos = 0
    for part in parts:
        if not part:
            continue
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if not text.startswith(part, pos):
            # 保底：从当前位置取 part 长度
            end = min(len(text), pos + len(part))
        else:
            end = pos + len(part)
        spans.append(SentenceSpan(idx, pos, end, text[pos:end]))
        idx += 1
        pos = end
    if pos < len(text) and any(not c.isspace() for c in text[pos:]):
        spans.append(SentenceSpan(idx, pos, len(text), text[pos:]))
    if not spans:
        spans.append(SentenceSpan(0, 0, len(text), text))
    return spans


def _window_ok(tokenizer: TokenizerLike, text: str, max_len: int) -> bool:
    return count_tokens(tokenizer, text, add_special_tokens=True) <= max_len


def _max_end_char(
    tokenizer: TokenizerLike,
    text: str,
    local_start: int,
    local_end_limit: int,
    max_len: int,
) -> int:
    """在 [local_start, local_end_limit) 内二分最大 end，使 substring 编码长度 <= max_len。"""
    if local_start >= local_end_limit:
        return local_start
    lo = local_start + 1
    hi = local_end_limit
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = text[local_start:mid]
        if _window_ok(tokenizer, chunk, max_len):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        # 单字符仍超限：强制切 1 字符并报警（避免死循环）
        logger.warning("单字符窗口仍超过 max_len=%s，强制前进 1 字符", max_len)
        return min(local_start + 1, local_end_limit)
    return best


def _token_boundary_cuts(tokenizer: TokenizerLike, snippet: str) -> list[int]:
    offs = content_token_offsets(tokenizer, snippet)
    cuts = sorted({e for _, e in offs if e > 0})
    if not cuts:
        cuts = list(range(1, len(snippet) + 1))
    return cuts


def _split_long_span(
    text: str,
    global_start: int,
    global_end: int,
    tokenizer: TokenizerLike,
    max_len: int,
    overlap_tokens: int,
    window_id_start: int,
    sentence_idx: int | None,
) -> list[Window]:
    """超长单句：按 tokenizer offset 在 token 边界安全切分，保留局部→全局映射。"""
    snippet = text[global_start:global_end]
    windows: list[Window] = []
    local_start = 0
    wid = window_id_start
    n = len(snippet)
    guard = 0
    max_steps = max(n * 2, 8)
    while local_start < n:
        while local_start < n and snippet[local_start].isspace():
            local_start += 1
        if local_start >= n:
            break
        guard += 1
        if guard > max_steps:
            raise RuntimeError("超长单句切分疑似死循环")
        remaining = snippet[local_start:]
        if _window_ok(tokenizer, remaining, max_len):
            g0 = global_start + local_start
            g1 = global_end
            wtext = text[g0:g1]
            windows.append(
                Window(
                    window_id=wid,
                    text=wtext,
                    global_char_start=g0,
                    global_char_end=g1,
                    model_token_count=count_tokens(tokenizer, wtext, True),
                    sentence_start_idx=sentence_idx,
                    sentence_end_idx=sentence_idx,
                )
            )
            break
        cuts = [c for c in _token_boundary_cuts(tokenizer, remaining) if c > 0]
        best_local_end = None
        for cut in cuts:
            cand = remaining[:cut]
            if _window_ok(tokenizer, cand, max_len):
                best_local_end = local_start + cut
            else:
                break
        if best_local_end is None:
            best_local_end = _max_end_char(tokenizer, snippet, local_start, n, max_len)
        g0 = global_start + local_start
        g1 = global_start + best_local_end
        wtext = text[g0:g1]
        ntok = count_tokens(tokenizer, wtext, True)
        if ntok > max_len:
            best_local_end = _max_end_char(tokenizer, snippet, local_start, best_local_end, max_len)
            g1 = global_start + best_local_end
            wtext = text[g0:g1]
            ntok = count_tokens(tokenizer, wtext, True)
        windows.append(
            Window(
                window_id=wid,
                text=wtext,
                global_char_start=g0,
                global_char_end=g1,
                model_token_count=ntok,
                sentence_start_idx=sentence_idx,
                sentence_end_idx=sentence_idx,
            )
        )
        wid += 1
        # 回退重叠：使下一窗口与当前尾部重叠约 overlap_tokens
        next_local = _overlap_local_start(
            tokenizer, snippet, local_start, best_local_end, overlap_tokens, max_len
        )
        if next_local <= local_start:
            next_local = best_local_end
        local_start = next_local
    return windows


def _overlap_local_start(
    tokenizer: TokenizerLike,
    snippet: str,
    start: int,
    end: int,
    overlap_tokens: int,
    max_len: int,
) -> int:
    if end <= start:
        return end
    # 从窗口尾部向前找，使 snippet[t:end] 的 token 数最接近 overlap
    best = end
    best_diff = overlap_tokens + 1
    # 在 token 边界上搜索
    piece = snippet[start:end]
    cuts = [start + c for c in _token_boundary_cuts(tokenizer, piece)]
    cuts = [c for c in cuts if start < c < end]
    if not cuts:
        return end
    for t in reversed(cuts):
        c = count_tokens(tokenizer, snippet[t:end], True)
        diff = abs(c - overlap_tokens)
        if c <= max_len and diff < best_diff:
            best = t
            best_diff = diff
        if c >= overlap_tokens and t < end:
            best = t
            break
    return best


def _pack_from_sentences(
    text: str,
    sentences: list[SentenceSpan],
    tokenizer: TokenizerLike,
    max_len: int,
    overlap_tokens: int,
) -> list[Window]:
    windows: list[Window] = []
    n = len(sentences)
    start_i = 0
    wid = 0
    guard = 0
    while start_i < n:
        guard += 1
        if guard > n * 4 + 8:
            raise RuntimeError("句界窗口划分疑似死循环")
        last_ok = None
        end_i = start_i
        while end_i < n:
            g0 = sentences[start_i].char_start
            g1 = sentences[end_i].char_end
            if _window_ok(tokenizer, text[g0:g1], max_len):
                last_ok = end_i
                end_i += 1
            else:
                break
        if last_ok is None:
            # 当前起始句自身超长
            sent = sentences[start_i]
            extra = _split_long_span(
                text,
                sent.char_start,
                sent.char_end,
                tokenizer,
                max_len,
                overlap_tokens,
                wid,
                sent.sent_idx,
            )
            if not extra:
                start_i += 1
                continue
            windows.extend(extra)
            wid = windows[-1].window_id + 1
            start_i += 1
            continue
        g0 = sentences[start_i].char_start
        g1 = sentences[last_ok].char_end
        wtext = text[g0:g1]
        windows.append(
            Window(
                window_id=wid,
                text=wtext,
                global_char_start=g0,
                global_char_end=g1,
                model_token_count=count_tokens(tokenizer, wtext, True),
                sentence_start_idx=sentences[start_i].sent_idx,
                sentence_end_idx=sentences[last_ok].sent_idx,
            )
        )
        wid += 1
        next_i = _next_sentence_start(sentences, start_i, last_ok, tokenizer, text, overlap_tokens)
        if next_i <= start_i:
            next_i = start_i + 1
        start_i = next_i
    return windows


def _next_sentence_start(
    sentences: list[SentenceSpan],
    start_i: int,
    last_ok: int,
    tokenizer: TokenizerLike,
    text: str,
    overlap_tokens: int,
) -> int:
    """从当前窗口尾部回退，使重叠接近 overlap_tokens，且起点严格前进。"""
    if last_ok <= start_i:
        return last_ok + 1
    best = last_ok
    best_diff = 10**9
    for t in range(start_i + 1, last_ok + 1):
        g0 = sentences[t].char_start
        g1 = sentences[last_ok].char_end
        c = count_tokens(tokenizer, text[g0:g1], True)
        diff = abs(c - overlap_tokens)
        if diff < best_diff or (diff == best_diff and c <= overlap_tokens):
            best = t
            best_diff = diff
    return best


def split_windows(
    text: str,
    tokenizer: TokenizerLike,
    *,
    max_model_tokens: int = 1024,
    overlap_tokens: int = 256,
    sentence_boundary: bool = True,
    sentences: list[SentenceSpan] | None = None,
) -> list[Window]:
    text = text or ""
    max_len = effective_max_length(tokenizer, max_model_tokens)
    if not text.strip():
        return [
            Window(
                window_id=0,
                text=text,
                global_char_start=0,
                global_char_end=len(text),
                model_token_count=count_tokens(tokenizer, text, True),
                sentence_start_idx=0,
                sentence_end_idx=0,
            )
        ]
    # 整篇已不超过上限：必须只生成一个窗口，且与原实现一致
    if count_tokens(tokenizer, text, True) <= max_len:
        sents = sentences if sentences is not None else fallback_sentence_spans(text)
        return [
            Window(
                window_id=0,
                text=text,
                global_char_start=0,
                global_char_end=len(text),
                model_token_count=count_tokens(tokenizer, text, True),
                sentence_start_idx=sents[0].sent_idx if sents else 0,
                sentence_end_idx=sents[-1].sent_idx if sents else 0,
            )
        ]
    if sentence_boundary:
        sents = sentences if sentences is not None else fallback_sentence_spans(text)
        if not sents:
            sents = [SentenceSpan(0, 0, len(text), text)]
        windows = _pack_from_sentences(text, sents, tokenizer, max_len, overlap_tokens)
    else:
        windows = _split_long_span(
            text, 0, len(text), tokenizer, max_len, overlap_tokens, 0, None
        )
    for i, w in enumerate(windows):
        w.window_id = i
    return windows


class WindowValidationError(ValueError):
    pass


def validate_windows(
    windows: list[Window],
    text: str,
    tokenizer: TokenizerLike,
    max_model_tokens: int,
) -> None:
    max_len = effective_max_length(tokenizer, max_model_tokens)
    if not windows:
        raise WindowValidationError("未生成任何窗口")
    for w in windows:
        ntok = count_tokens(tokenizer, w.text, True)
        if ntok > max_len:
            raise WindowValidationError(
                f"窗口 {w.window_id} token 数 {ntok} 超过上限 {max_len}"
            )
        if ntok != w.model_token_count:
            w.model_token_count = ntok
        if text[w.global_char_start:w.global_char_end] != w.text:
            raise WindowValidationError(
                f"窗口 {w.window_id} 文本与全局字符区间不一致（禁止用 str.find 猜测）"
            )
    for a, b in zip(windows, windows[1:]):
        if b.global_char_start <= a.global_char_start:
            raise WindowValidationError("相邻窗口起点未严格增加")
        if a.global_char_start > b.global_char_start:
            raise WindowValidationError("窗口未按原文顺序排列")
    covered = [False] * len(text)
    for w in windows:
        for i in range(max(0, w.global_char_start), min(len(text), w.global_char_end)):
            covered[i] = True
    for i, ch in enumerate(text):
        if not ch.isspace() and not covered[i]:
            raise WindowValidationError(f"原文非空白字符未被覆盖，位置 {i}")
    full = count_tokens(tokenizer, text, True)
    if full <= max_len and len(windows) != 1:
        raise WindowValidationError("短于限制的文档应只生成一个窗口")
