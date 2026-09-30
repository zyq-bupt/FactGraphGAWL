"""LLM factuality checker for keyphrase+sentence units."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from baselines.kp_llm.prompts import CHECKER_SYSTEM_PROMPT, CHECKER_USER_PROMPT
from baselines.kp_llm.schemas import ALLOWED_LABELS, CheckerOutput
from baselines.llm_client import LLMClient


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


class FactualityChecker:
    """Check whether a keyphrase's local fact is supported by the source document."""

    def __init__(self, client: LLMClient, max_source_chars: int = 12000):
        self.client = client
        self.max_source_chars = max_source_chars

    def _prepare_source(self, source: str, sentence: str, keyphrase: str) -> tuple[str, str]:
        if len(source) <= self.max_source_chars:
            return source, "full"
        # Simple relevance truncation: keep paragraphs containing overlapping tokens.
        query_toks = set(re.findall(r"[A-Za-z0-9]+", (keyphrase + " " + sentence).lower()))
        paras = re.split(r"\n\s*\n", source)
        scored = []
        for p in paras:
            pt = set(re.findall(r"[A-Za-z0-9]+", p.lower()))
            score = len(query_toks & pt)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [p for s, p in scored if s > 0][:5] or [p for _, p in scored[:3]]
        # Preserve approximate original order
        chosen_set = set(chosen)
        ordered = [p for p in paras if p in chosen_set]
        clipped = "\n\n".join(ordered)
        if len(clipped) > self.max_source_chars:
            clipped = clipped[: self.max_source_chars]
        return clipped, "retrieved"

    def check(self, source: str, sentence: str, keyphrase: str) -> CheckerOutput:
        prepared, _mode = self._prepare_source(source, sentence, keyphrase)
        user = CHECKER_USER_PROMPT.format(
            source_document=prepared,
            containing_summary_sentence=sentence,
            keyphrase=keyphrase,
        )
        messages = [
            {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

        raw = ""
        data = None
        for _ in range(2):
            try:
                raw = self.client.chat(messages, force_json=True)
            except Exception:
                raw = self.client.chat(messages, force_json=False)
            data = _extract_json(raw)
            if data and str(data.get("label", "")).upper() in ALLOWED_LABELS:
                break

        if not data:
            return CheckerOutput(
                label="NOT_ENOUGH_INFORMATION",
                evidence="",
                reason="parse_failed",
                invalid_evidence=False,
                error="json_parse_failed",
            )

        label = str(data.get("label", "")).upper().strip()
        if label not in ALLOWED_LABELS:
            return CheckerOutput(
                label="NOT_ENOUGH_INFORMATION",
                evidence="",
                reason=str(data.get("reason", "")),
                invalid_evidence=False,
                error=f"invalid_label:{label}",
            )

        evidence = str(data.get("evidence", "") or "")
        invalid_evidence = False
        if label == "SUPPORTED" and evidence:
            # Evidence should be locatable in source (soft check).
            if evidence.lower() not in prepared.lower():
                # one re-ask
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your evidence was not found in the source. Re-answer with valid JSON and evidence copied from the source, or empty evidence.",
                    }
                )
                try:
                    raw2 = self.client.chat(messages, force_json=True)
                except Exception:
                    raw2 = self.client.chat(messages, force_json=False)
                data2 = _extract_json(raw2)
                if data2 and str(data2.get("label", "")).upper() in ALLOWED_LABELS:
                    label = str(data2.get("label", "")).upper().strip()
                    evidence = str(data2.get("evidence", "") or "")
                    data = data2
                if evidence and evidence.lower() not in prepared.lower():
                    invalid_evidence = True

        return CheckerOutput(
            label=label,  # type: ignore[typeddict-item]
            evidence=evidence,
            reason=str(data.get("reason", "")),
            invalid_evidence=invalid_evidence,
            error=None,
        )
