#!/usr/bin/env python3
"""Evaluate Direct-LLM / KP-LLM predictions against human labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.sample_ids import make_sample_id


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--labels",
        required=True,
        help="JSONL with faithfulness_score / human_score / human_label",
    )
    parser.add_argument("--pred-key", default="score")
    parser.add_argument("--label-key", default=None, help="auto-detect if omitted")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional metrics JSON path",
    )
    parser.add_argument(
        "--exclude-human1",
        action="store_true",
        help="Match evaluators_benchmark: drop faithfulness_score==1",
    )
    args = parser.parse_args()

    preds = {}
    for r in load_jsonl(Path(args.predictions)):
        sid = make_sample_id(r, explicit=r.get("id"))
        preds[sid] = r

    labels = load_jsonl(Path(args.labels))

    ys, xs, missing = [], [], 0
    for row in labels:
        if (
            "summary_success_state" in row
            and row.get("summary_success_state") != "success"
        ):
            continue
        sid = make_sample_id(row)
        if sid not in preds or preds[sid].get(args.pred_key) is None or preds[sid].get("error"):
            missing += 1
            continue
        if args.label_key:
            y = row.get(args.label_key)
        else:
            y = (
                row.get("human_score")
                if row.get("human_score") is not None
                else row.get("faithfulness_score")
                if row.get("faithfulness_score") is not None
                else row.get("human_label")
                if row.get("human_label") is not None
                else row.get("label")
            )
        if y is None:
            missing += 1
            continue
        if args.exclude_human1 and float(y) == 1.0:
            continue
        ys.append(float(y))
        xs.append(float(preds[sid][args.pred_key]))

    metrics = {"n": len(xs), "missing_or_invalid": missing}
    if len(xs) >= 2:
        import numpy as np
        from scipy.stats import pearsonr, spearmanr, kendalltau

        if len(set(xs)) > 1 and len(set(ys)) > 1:
            metrics["pearson"] = float(pearsonr(xs, ys)[0])
            metrics["spearman"] = float(spearmanr(xs, ys)[0])
            metrics["kendall"] = float(kendalltau(xs, ys)[0])
        else:
            metrics["pearson"] = None
            metrics["spearman"] = None
            metrics["kendall"] = None
            metrics["note"] = "correlation undefined (constant input)"

        if set(np.unique(ys)).issubset({0.0, 1.0}):
            from sklearn.metrics import (
                average_precision_score,
                balanced_accuracy_score,
                f1_score,
                roc_auc_score,
            )

            y_bin = [int(v) for v in ys]
            thr = 0.5
            y_hat = [1 if v >= thr else 0 for v in xs]
            try:
                metrics["auroc"] = float(roc_auc_score(y_bin, xs))
            except ValueError:
                metrics["auroc"] = None
            try:
                metrics["auprc"] = float(average_precision_score(y_bin, xs))
            except ValueError:
                metrics["auprc"] = None
            metrics["f1"] = float(f1_score(y_bin, y_hat, zero_division=0))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_bin, y_hat))

    print(json.dumps(metrics, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
