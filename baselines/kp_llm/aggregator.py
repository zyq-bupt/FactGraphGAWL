"""Aggregate keyphrase-level labels into a sample score."""

from __future__ import annotations

from typing import Any

from baselines.kp_llm.schemas import ALLOWED_LABELS


def aggregate_labels(labels: list[str]) -> dict[str, Any]:
    """Compute support-rate score and auxiliary ratios."""
    if not labels:
        return {
            "score": None,
            "supported_count": 0,
            "contradicted_count": 0,
            "not_enough_information_count": 0,
            "supported_ratio": None,
            "contradicted_ratio": None,
            "not_enough_information_ratio": None,
            "num_keyphrases": 0,
        }

    counts = {k: 0 for k in ALLOWED_LABELS}
    for lab in labels:
        if lab in counts:
            counts[lab] += 1
    m = len(labels)
    supported = counts["SUPPORTED"]
    return {
        "score": supported / m,
        "supported_count": supported,
        "contradicted_count": counts["CONTRADICTED"],
        "not_enough_information_count": counts["NOT_ENOUGH_INFORMATION"],
        "supported_ratio": supported / m,
        "contradicted_ratio": counts["CONTRADICTED"] / m,
        "not_enough_information_ratio": counts["NOT_ENOUGH_INFORMATION"] / m,
        "num_keyphrases": m,
    }
