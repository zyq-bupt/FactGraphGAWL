"""UniSumEval 读取与 GovReport 过滤，规则对齐 evaluators_benchmark / el_perturbation。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def load_unisum_labels(jsonl_path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def domain_match(item: dict, domain: str) -> bool:
    want = (domain or "").strip().lower()
    if not want:
        return True
    for key in ("source", "dataset", "domain"):
        val = item.get(key)
        if val is not None and str(val).strip().lower() == want:
            return True
    return False


def passes_eval_filter(item: dict, *, require_success: bool = True, exclude_human1: bool = True) -> bool:
    if require_success and item.get("summary_success_state") != "success":
        return False
    if exclude_human1 and item.get("faithfulness_score") == 1:
        return False
    return True


def iter_experiment_samples(
    labels: list[dict],
    *,
    domain: str,
    require_success: bool = True,
    exclude_human1: bool = True,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    original = [x for x in labels if domain_match(x, domain)]
    kept = []
    excluded = []
    for item in original:
        if not passes_eval_filter(item, require_success=require_success, exclude_human1=exclude_human1):
            reason = "not_success" if item.get("summary_success_state") != "success" else "faithfulness_score_eq_1"
            excluded.append({"sample_id": str(item.get("doc_id")), "reason": reason})
            continue
        kept.append(item)
    if limit is not None:
        kept = kept[: max(0, limit)]
    meta = {
        "n_original_domain": len(original),
        "n_after_filter": len(kept) if limit is None else None,
        "n_selected": len(kept),
        "excluded": excluded,
    }
    return kept, meta
