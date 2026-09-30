"""Shared LLM client utilities for Direct-LLM and KP-LLM baselines."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class LLMConfig:
    provider: str  # openai | ernie | openai_compatible
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: Optional[int] = 42
    max_retries: int = 4
    timeout: float = 120.0


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat APIs and Baidu ERNIE.

    Environment variables (do not hardcode secrets):
      OPENAI_API_KEY / OPENAI_BASE_URL
      ERNIE_API_KEY / ERNIE_SECRET_KEY / ERNIE_BASE_URL
    """

    def __init__(self, config: LLMConfig, cache_dir: Optional[str | Path] = None):
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ernie_token: Optional[str] = None
        self._ernie_token_expire: float = 0.0

    def _cache_key(self, messages: list[dict[str, str]], extra: dict[str, Any]) -> str:
        payload = {
            "provider": self.config.provider,
            "model": self.config.model,
            "messages": messages,
            "extra": extra,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, key: str, text: str) -> None:
        if not self.cache_dir:
            return
        (self.cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")

    def chat(self, messages: list[dict[str, str]], *, force_json: bool = False) -> str:
        """Send chat messages and return assistant text content."""
        extra = {"force_json": force_json}
        key = self._cache_key(messages, extra)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                if self.config.provider in {"openai", "openai_compatible"}:
                    text = self._chat_openai(messages, force_json=force_json)
                elif self.config.provider == "ernie":
                    text = self._chat_ernie(messages, force_json=force_json)
                else:
                    raise ValueError(f"Unknown provider: {self.config.provider}")
                self._write_cache(key, text)
                return text
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def _chat_openai(self, messages: list[dict[str, str]], *, force_json: bool) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Please install openai: pip install openai") from exc

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing API key env: {self.config.api_key_env}")

        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.config.timeout}
        base_url = self.config.base_url or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def _get_ernie_access_token(self) -> str:
        now = time.time()
        if self._ernie_token and now < self._ernie_token_expire - 60:
            return self._ernie_token

        import urllib.parse
        import urllib.request

        api_key = os.environ.get("ERNIE_API_KEY") or os.environ.get(self.config.api_key_env)
        secret_key = os.environ.get("ERNIE_SECRET_KEY")
        if not api_key or not secret_key:
            raise EnvironmentError("ERNIE requires ERNIE_API_KEY and ERNIE_SECRET_KEY")

        url = (
            "https://aip.baidubce.com/oauth/2.0/token?"
            + urllib.parse.urlencode(
                {
                    "grant_type": "client_credentials",
                    "client_id": api_key,
                    "client_secret": secret_key,
                }
            )
        )
        with urllib.request.urlopen(url, timeout=self.config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"Failed to get ERNIE token: {data}")
        self._ernie_token = token
        self._ernie_token_expire = now + float(data.get("expires_in", 2592000))
        return token

    def _chat_ernie(self, messages: list[dict[str, str]], *, force_json: bool) -> str:
        """Call Baidu ERNIE via OpenAI-compatible gateway or official HTTP API."""
        import urllib.request

        base_url = self.config.base_url or os.environ.get("ERNIE_BASE_URL")
        if base_url:
            api_key = os.environ.get(self.config.api_key_env) or os.environ.get("ERNIE_API_KEY")
            if not api_key:
                raise EnvironmentError(f"Missing API key env: {self.config.api_key_env}")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Please install openai: pip install openai") from exc
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.timeout)
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            }
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        token = self._get_ernie_access_token()
        chat_path = os.environ.get(
            "ERNIE_CHAT_PATH",
            f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.config.model}",
        )
        url = f"{chat_path}?access_token={token}"

        system = ""
        ernie_msgs = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system = content
            else:
                ernie_msgs.append(
                    {"role": "user" if role == "user" else "assistant", "content": content}
                )

        body: dict[str, Any] = {
            "messages": ernie_msgs,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_tokens,
        }
        if system:
            body["system"] = system

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "result" not in data:
            raise RuntimeError(f"ERNIE error: {data}")
        return str(data["result"]).strip()
