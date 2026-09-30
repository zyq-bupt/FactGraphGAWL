"""T5 / FLAN-T5 keyphrase extractor for KP-LLM."""

from __future__ import annotations

import re
from typing import Optional

from baselines.kp_llm.prompts import DIST5_EXTRACTOR_PROMPT, EXTRACTOR_PROMPT


DEFAULT_EXTRACTOR_MODEL = "/root/autodl-fs/zyq/models/DisT5"


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "as", "at",
}


def normalize_keyphrases(raw: str, max_keyphrases: int = 8) -> list[str]:
    """Split, clean, deduplicate keyphrases; keep original casing of first occurrence."""
    parts = re.split(r"[|\n;,]+", raw)
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        phrase = part.strip()
        phrase = re.sub(r"^\d+[\).\]]\s*", "", phrase)
        phrase = phrase.strip(" \"'`")
        if not phrase:
            continue
        tokens = re.findall(r"[A-Za-z0-9]+", phrase.lower())
        if not tokens or all(t in _STOPWORDS for t in tokens):
            continue
        key = phrase.lower()
        if key in seen:
            continue
        # Prefer longer overlapping phrase: drop shorter contained ones later.
        seen.add(key)
        cleaned.append(phrase)

    # Drop phrases that are strict substrings of a longer retained phrase.
    kept: list[str] = []
    lower_sorted = sorted(cleaned, key=lambda x: len(x), reverse=True)
    accepted_lower: list[str] = []
    for phrase in lower_sorted:
        pl = phrase.lower()
        if any(pl != other and pl in other for other in accepted_lower):
            continue
        accepted_lower.append(pl)
        kept.append(phrase)
    # Restore roughly original generation order among kept.
    order = {p.lower(): i for i, p in enumerate(cleaned)}
    kept.sort(key=lambda p: order.get(p.lower(), 10**9))
    return kept[:max_keyphrases]


class KeyphraseExtractor:
    """Extract keyphrases from a summary using a seq2seq T5-style model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EXTRACTOR_MODEL,
        max_keyphrases: int = 8,
        max_input_tokens: int = 512,
        max_new_tokens: int = 96,
        num_beams: int = 4,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.max_keyphrases = max_keyphrases
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.seed = seed
        self.device = device
        self._tokenizer = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        import transformers

        ver = getattr(transformers, "__version__", "0.0")
        # FLAN-T5 needs reasonably modern transformers; 2.x loads but outputs garbage.
        try:
            major = int(str(ver).split(".")[0])
        except ValueError:
            major = 0
        if major < 4 and "flan" in self.model_name.lower():
            raise RuntimeError(
                f"Current transformers=={ver} is too old for FLAN-T5 "
                f"({self.model_name}). Use an env with transformers>=4.x "
                f"(e.g. conda base), or: pip install -U 'transformers>=4.30'."
            )

        # Compatible with old transformers (e.g. 2.11 in factgraph) and newer versions.
        try:
            from transformers import AutoModelForSeq2SeqLM as ModelCls
            from transformers import AutoTokenizer as TokCls

            tok_kwargs = {"use_fast": False}
        except ImportError:
            from transformers import T5ForConditionalGeneration as ModelCls
            from transformers import T5Tokenizer as TokCls

            tok_kwargs = {}

        try:
            self._tokenizer = TokCls.from_pretrained(self.model_name, **tok_kwargs)
        except TypeError:
            # Older tokenizers may not accept use_fast=
            self._tokenizer = TokCls.from_pretrained(self.model_name)
        self._model = ModelCls.from_pretrained(self.model_name)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()

    def extract(self, summary: str) -> list[str]:
        """Return cleaned keyphrases from summary text."""
        if not summary or not summary.strip():
            return []
        self._lazy_load()
        assert self._tokenizer is not None and self._model is not None

        import torch

        name_l = self.model_name.lower()
        use_dist5_prompt = ("dist5" in name_l) or ("flan" not in name_l)
        if use_dist5_prompt:
            prompt = DIST5_EXTRACTOR_PROMPT.format(generated_summary=summary)
        else:
            prompt = EXTRACTOR_PROMPT.format(
                max_keyphrases=self.max_keyphrases,
                generated_summary=summary,
            )

        # transformers>=3 / encode_plus style; keep fallback for very old APIs
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )
        except TypeError:
            inputs = self._tokenizer.encode_plus(
                prompt,
                return_tensors="pt",
                max_length=self.max_input_tokens,
                truncation=True,
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        torch.manual_seed(self.seed)
        gen_kwargs = {
            "num_beams": self.num_beams,
            "do_sample": False,
        }
        input_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            try:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    **gen_kwargs,
                )
            except TypeError:
                # transformers 2.x: no max_new_tokens
                outputs = self._model.generate(
                    **inputs,
                    max_length=input_len + self.max_new_tokens,
                    **gen_kwargs,
                )

        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        phrases = normalize_keyphrases(text, max_keyphrases=self.max_keyphrases)
        if phrases:
            return phrases
        # One deterministic retry with same settings (already deterministic).
        phrases = normalize_keyphrases(text, max_keyphrases=self.max_keyphrases)
        return phrases
