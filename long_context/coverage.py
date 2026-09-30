"""覆盖率与图规模统计。tokenizer 口径与窗口划分相同；重叠 token 只计一次。"""

from __future__ import annotations

from long_context.tokenizer_util import TokenizerLike, content_token_offsets, count_tokens
from long_context.windows import Window


def token_coverage_ratio(
    text: str,
    windows: list[Window],
    tokenizer: TokenizerLike,
) -> dict:
    """
    分母：全文 content tokens（add_special_tokens=True 后去掉 special 的 (0,0)）。
    分子：字符区间被至少一个窗口覆盖的 content tokens。
    重叠窗口不会使覆盖率 > 1。
    """
    text = text or ""
    content = content_token_offsets(tokenizer, text)
    n_doc = count_tokens(tokenizer, text, True)
    if not content:
        return {
            "document_model_tokens": n_doc,
            "unique_tokens": 0,
            "token_coverage_ratio": 1.0 if not text.strip() else 0.0,
        }
    covered_char = [False] * (len(text) + 1)
    for w in windows:
        a = max(0, w.global_char_start)
        b = min(len(text), w.global_char_end)
        for i in range(a, b):
            covered_char[i] = True
    uniq = 0
    for s, e in content:
        e = min(e, len(text))
        s = max(0, s)
        if s >= e:
            continue
        if all(covered_char[i] for i in range(s, e)):
            uniq += 1
    ratio = uniq / len(content)
    if ratio > 1.0:
        ratio = 1.0
    return {
        "document_model_tokens": n_doc,
        "unique_tokens": uniq,
        "n_content_tokens": len(content),
        "token_coverage_ratio": float(ratio),
    }
