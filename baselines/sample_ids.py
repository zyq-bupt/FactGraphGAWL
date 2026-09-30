"""Shared helpers for UniSumEval / DeFacto JSONL baseline runners."""

from __future__ import annotations

from typing import Any, Optional


FIELD_MAP = {
    "source": ["source_document", "input_context", "article", "doc", "document"],
    "summary": ["generated_summary", "summary", "candidate"],
    "id": ["id", "uid", "doc_id"],
    "dataset": ["dataset", "source", "domain"],
}


def pick(obj: dict, keys: list[str], default=None):
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def make_sample_id(row: dict, explicit: Optional[str] = None) -> str:
    """Build a stable unique id.

    UniSumEval reuses the same uid across summarizer models, so uid alone is
    not unique. Prefer uid__model (or id if already composite).
    """
    if explicit:
        return str(explicit)

    # If caller already set a composite id, keep it.
    if row.get("id") is not None and "__" in str(row["id"]):
        return str(row["id"])

    uid = pick(row, ["uid", "id", "doc_id"], default="")
    model = row.get("model")
    if uid and model:
        # sanitize model for filesystem-ish ids
        model_s = str(model).replace("/", "_").replace(" ", "_")
        return f"{uid}__{model_s}"
    if uid:
        return str(uid)
    return ""


def extract_text_field(value: Any) -> str:
    """Flatten nested DeFacto-style {text: ...} blocks if present."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("text", "input_context", "document", "doc"):
            if isinstance(value.get(k), str):
                return value[k]
    return str(value)
