"""Direct prompt-based LLM factual consistency evaluators.

Methods:
  - ChatGPT-ZS / ERNIE-ZS: binary yes/no
  - ChatGPT-Star / ERNIE-Star: 1-5 star rating
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from baselines.llm_client import LLMClient, LLMConfig
from baselines.llm_direct.prompts import STAR_PROMPT, ZS_PROMPT


@dataclass
class DirectLLMResult:
    method: str
    raw_output: str
    score: Optional[float]
    binary_label: Optional[int]
    error: Optional[str] = None


def parse_zs(text: str) -> tuple[Optional[float], Optional[int]]:
    """Parse yes/no answer -> score in {0,1} and binary label."""
    cleaned = text.strip().lower()
    m = re.search(r"\b(yes|no)\b", cleaned)
    if not m:
        return None, None
    ans = m.group(1)
    label = 1 if ans == "yes" else 0
    return float(label), label


def parse_star(text: str) -> tuple[Optional[float], Optional[int]]:
    """Parse 1-5 star rating. Score is stars/5 in [0.2, 1.0]."""
    cleaned = text.strip().lower()
    m = re.search(r"([1-5])\s*(?:stars?|星)?", cleaned)
    if not m:
        m = re.search(r"\b([1-5])\b", cleaned)
    if not m:
        return None, None
    stars = int(m.group(1))
    score = stars / 5.0
    binary = 1 if stars >= 4 else 0
    return score, binary


class DirectLLMEvaluator:
    """Unified evaluator for ChatGPT/ERNIE ZS and Star methods."""

    METHOD_PROMPTS = {
        "ChatGPT-ZS": ("zs", ZS_PROMPT),
        "ChatGPT-Star": ("star", STAR_PROMPT),
        "ERNIE-ZS": ("zs", ZS_PROMPT),
        "ERNIE-Star": ("star", STAR_PROMPT),
    }

    def __init__(self, method: str, client: LLMClient):
        if method not in self.METHOD_PROMPTS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHOD_PROMPTS)}")
        self.method = method
        self.mode, self.template = self.METHOD_PROMPTS[method]
        self.client = client

    def score(self, article: str, summary: str) -> DirectLLMResult:
        if not summary or not summary.strip():
            return DirectLLMResult(
                method=self.method,
                raw_output="",
                score=None,
                binary_label=None,
                error="empty_summary",
            )
        prompt = self.template.format(article=article, summary=summary)
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.client.chat(messages)
        except Exception as exc:  # noqa: BLE001
            return DirectLLMResult(
                method=self.method,
                raw_output="",
                score=None,
                binary_label=None,
                error=str(exc),
            )

        if self.mode == "zs":
            score, binary = parse_zs(raw)
        else:
            score, binary = parse_star(raw)

        err = None if score is not None else "parse_failed"
        return DirectLLMResult(
            method=self.method,
            raw_output=raw,
            score=score,
            binary_label=binary,
            error=err,
        )


def build_client_for_method(method: str, cache_dir: Optional[str] = None) -> LLMClient:
    """Create a default client for each named method."""
    if method.startswith("ChatGPT"):
        cfg = LLMConfig(
            provider="openai",
            model=os.environ.get("CHATGPT_MODEL", "gpt-3.5-turbo"),
            api_key_env="OPENAI_API_KEY",
            # 默认走智增增 OpenAI 兼容网关；可用 OPENAI_BASE_URL 覆盖
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1"),
            temperature=0.0,
            max_tokens=64 if method.endswith("ZS") else 32,
        )
    elif method.startswith("ERNIE"):
        # 默认走智增增；模型默认 ernie-3.5-128k
        cfg = LLMConfig(
            provider="openai",
            model=os.environ.get("ERNIE_MODEL", "ernie-3.5-128k"),
            # 与 ChatGPT 共用智增增 key；也可单独设 ERNIE_API_KEY
            api_key_env="ERNIE_API_KEY" if os.environ.get("ERNIE_API_KEY") else "OPENAI_API_KEY",
            base_url=os.environ.get(
                "ERNIE_BASE_URL",
                os.environ.get("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1"),
            ),
            temperature=0.0,
            max_tokens=64 if method.endswith("ZS") else 32,
        )
    else:
        raise ValueError(method)
    return LLMClient(cfg, cache_dir=cache_dir)
