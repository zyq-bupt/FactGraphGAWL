"""关系抽取模型 tokenizer 适配。禁止 truncation=True 静默截断。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


PLACEHOLDER_MAX = 100_000


class TokenizerLike(Protocol):
    model_max_length: int

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...


@dataclass
class MockTokenizer:
    """单元测试用：按空白切词，并计入 2 个 special tokens。不用于正式实验。"""

    model_max_length: int = 1024
    unit: str = "word"  # word | char
    name_or_path: str = "mock"

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if truncation:
            raise ValueError("禁止 truncation=True：窗口必须预先满足长度上限")
        pieces: list[tuple[int, int]] = []
        if self.unit == "char":
            for i, ch in enumerate(text):
                if not ch.isspace():
                    pieces.append((i, i + 1))
        else:
            i = 0
            n = len(text)
            while i < n:
                if text[i].isspace():
                    i += 1
                    continue
                j = i
                while j < n and not text[j].isspace():
                    j += 1
                pieces.append((i, j))
                i = j
        ids = [10 + k for k in range(len(pieces))]
        offs = list(pieces)
        if add_special_tokens:
            ids = [0] + ids + [2]
            offs = [(0, 0)] + offs + [(0, 0)]
        out: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


class HFTokenizerWrapper:
    def __init__(self, inner: Any, name_or_path: str):
        self.inner = inner
        self.name_or_path = name_or_path
        self.model_max_length = int(getattr(inner, "model_max_length", PLACEHOLDER_MAX) or PLACEHOLDER_MAX)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if truncation:
            raise ValueError("禁止 truncation=True：窗口必须预先满足长度上限")
        enc = self.inner(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
            truncation=False,
            **kwargs,
        )
        return {
            "input_ids": list(enc["input_ids"]),
            "offset_mapping": list(enc["offset_mapping"]) if return_offsets_mapping else None,
        }


def effective_max_length(tokenizer: TokenizerLike, configured: int) -> int:
    """HF 常把 model_max_length 设成极大占位值；实验硬上限以配置为准（DREEAM 为 1024）。"""
    if configured <= 0:
        raise ValueError("configured max length 必须为正")
    return int(configured)


def count_tokens(tokenizer: TokenizerLike, text: str, add_special_tokens: bool = True) -> int:
    enc = tokenizer(text or "", add_special_tokens=add_special_tokens, truncation=False)
    return len(enc["input_ids"])


def content_token_offsets(tokenizer: TokenizerLike, text: str) -> list[tuple[int, int]]:
    """全文 content token 的字符区间（special tokens 的 (0,0) 排除）。重叠覆盖率用此口径。"""
    enc = tokenizer(text or "", add_special_tokens=True, return_offsets_mapping=True, truncation=False)
    offs = enc.get("offset_mapping") or []
    out: list[tuple[int, int]] = []
    for s, e in offs:
        if s == 0 and e == 0:
            continue
        if e <= s:
            continue
        out.append((int(s), int(e)))
    return out


def tokenizer_fingerprint(tokenizer: TokenizerLike) -> str:
    return str(getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__)


def load_tokenizer(path: str | None, *, allow_mock: bool = False) -> TokenizerLike:
    if path == "mock" or (allow_mock and not path):
        return MockTokenizer()
    candidates: list[str] = []
    if path:
        candidates.append(path)
    candidates.extend(
        [
            "/root/autodl-fs/zyq/models/roberta-large",
            "/root/autodl-fs/zyq/models/t5-large",
            "/root/autodl-fs/zyq/models/flan-t5-base",
        ]
    )
    last_err: Exception | None = None
    for cand in candidates:
        p = Path(cand)
        if not p.exists():
            continue
        try:
            from transformers import AutoTokenizer

            inner = AutoTokenizer.from_pretrained(str(p), use_fast=True)
            return HFTokenizerWrapper(inner, str(p))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if allow_mock:
        return MockTokenizer()
    msg = "无法加载关系抽取 tokenizer。请用 --tokenizer-path 指向含 tokenizer 文件的目录。"
    if last_err is not None:
        msg += f" 最后错误: {last_err}"
    raise FileNotFoundError(msg)
