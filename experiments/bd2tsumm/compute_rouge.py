"""候选摘要 vs 人工参考摘要的 ROUGE-1/2/L F1。"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from util import (
    ensure_repo_on_path,
    load_config,
    package_version,
    read_jsonl,
    setup_logging,
    write_csv,
    write_json,
)

ensure_repo_on_path()


def make_scorer(use_stemmer: bool = True):
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)


def score_pair(scorer, candidate: str, reference: str) -> dict[str, float]:
    scores = scorer.score(reference or "", candidate or "")
    return {
        "rouge1_f1": float(scores["rouge1"].fmeasure),
        "rouge2_f1": float(scores["rouge2"].fmeasure),
        "rougeL_f1": float(scores["rougeL"].fmeasure),
        "rouge1_precision": float(scores["rouge1"].precision),
        "rouge1_recall": float(scores["rouge1"].recall),
        "rouge2_precision": float(scores["rouge2"].precision),
        "rouge2_recall": float(scores["rouge2"].recall),
        "rougeL_precision": float(scores["rougeL"].precision),
        "rougeL_recall": float(scores["rougeL"].recall),
    }


def aggregate_reference_f1(rows: list[dict[str, float]], method: str = "mean") -> dict[str, float]:
    if not rows:
        raise ValueError("没有参考级 ROUGE 可聚合")
    method = str(method).lower()
    if method not in {"mean"}:
        raise ValueError(f"不支持的多参考聚合: {method}（当前仅 mean，避免混用 max/mean）")
    keys = ["rouge1_f1", "rouge2_f1", "rougeL_f1"]
    out = {}
    n = float(len(rows))
    for k in keys:
        out[k] = sum(float(r[k]) for r in rows) / n
    out["num_references"] = len(rows)
    return out


def run_rouge(cfg: dict, output_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    out = cfg.get("output") or {}
    cand_path = output_dir / out.get("prepared_candidates", "prepared_candidates.jsonl")
    ref_path = output_dir / out.get("prepared_references", "prepared_references.jsonl")
    candidates = read_jsonl(cand_path)
    references = read_jsonl(ref_path)
    if not candidates:
        raise RuntimeError(
            "没有候选摘要。BD2TSumm 原始包不含系统输出，请提供 dataset.candidates_file，"
            "不要把人工参考摘要当作候选摘要。"
        )
    refs_by_event: dict[str, list[dict]] = defaultdict(list)
    for r in references:
        if str(r.get("reference_summary") or "").strip():
            refs_by_event[str(r["event_id"])].append(r)

    rouge_cfg = cfg.get("rouge") or {}
    use_stemmer = bool(rouge_cfg.get("use_stemmer", True))
    agg_method = str(rouge_cfg.get("multi_reference_aggregation") or "mean")
    scorer = make_scorer(use_stemmer=use_stemmer)

    ref_level: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for cand in candidates:
        eid = str(cand["event_id"])
        sysn = str(cand["system_name"])
        summary = cand.get("candidate_summary") or ""
        refs = refs_by_event.get(eid) or []
        if not str(summary).strip():
            sample_rows.append(
                {
                    "event_id": eid,
                    "system_name": sysn,
                    "rouge1_f1": None,
                    "rouge2_f1": None,
                    "rougeL_f1": None,
                    "num_references": len(refs),
                    "status": "failed",
                    "error_message": "empty_candidate",
                }
            )
            continue
        if not refs:
            sample_rows.append(
                {
                    "event_id": eid,
                    "system_name": sysn,
                    "rouge1_f1": None,
                    "rouge2_f1": None,
                    "rougeL_f1": None,
                    "num_references": 0,
                    "status": "failed",
                    "error_message": "missing_reference",
                }
            )
            continue
        pair_scores = []
        for ref in refs:
            sc = score_pair(scorer, summary, ref["reference_summary"])
            row = {
                "event_id": eid,
                "system_name": sysn,
                "reference_id": ref.get("reference_id"),
                **sc,
            }
            ref_level.append(row)
            pair_scores.append(sc)
        agg = aggregate_reference_f1(pair_scores, agg_method)
        sample_rows.append(
            {
                "event_id": eid,
                "system_name": sysn,
                "rouge1_f1": agg["rouge1_f1"],
                "rouge2_f1": agg["rouge2_f1"],
                "rougeL_f1": agg["rougeL_f1"],
                "num_references": agg["num_references"],
                "status": "ok",
                "error_message": "",
            }
        )

    meta = {
        "package": "rouge-score",
        "package_version": package_version("rouge-score"),
        "use_stemmer": use_stemmer,
        "multi_reference_aggregation": agg_method,
        "official_protocol": False,
        "note": "BD2TSumm 数据目录中未发现官方 ROUGE 脚本；多参考使用算术平均 F1。",
        "n_candidate_rows": len(candidates),
        "n_ok": sum(1 for r in sample_rows if r["status"] == "ok"),
        "n_failed": sum(1 for r in sample_rows if r["status"] != "ok"),
    }
    write_csv(output_dir / out.get("rouge_reference_level", "rouge_reference_level.csv"), ref_level)
    write_csv(
        output_dir / out.get("rouge_scores", "rouge_scores.csv"),
        sample_rows,
        [
            "event_id",
            "system_name",
            "rouge1_f1",
            "rouge2_f1",
            "rougeL_f1",
            "num_references",
            "status",
            "error_message",
        ],
    )
    write_json(output_dir / "rouge_metadata.json", meta)
    logger.info("ROUGE 完成 ok=%s failed=%s agg=%s", meta["n_ok"], meta["n_failed"], agg_method)
    return {"rows": sample_rows, "reference_level": ref_level, "meta": meta}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BD2TSumm ROUGE")
    p.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--output-dir", type=str, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / "logs" / "compute_rouge.log")
    run_rouge(cfg, output_dir, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
