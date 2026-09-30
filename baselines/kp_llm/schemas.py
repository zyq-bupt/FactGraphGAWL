"""Schemas and label constants for KP-LLM."""

from __future__ import annotations

from typing import Optional

try:
    from typing import Literal, TypedDict
except ImportError:  # Python < 3.8
    from typing_extensions import Literal, TypedDict  # type: ignore

ALLOWED_LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFORMATION")
LabelType = Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFORMATION"]


class KeyphraseAlignment(TypedDict, total=False):
    keyphrase: str
    summary_sentence: str
    sentence_index: int
    alignment_type: str  # exact | casefold | overlap | semantic
    candidate_indices: list[int]


class CheckerOutput(TypedDict, total=False):
    label: LabelType
    evidence: str
    reason: str
    invalid_evidence: bool
    error: Optional[str]
