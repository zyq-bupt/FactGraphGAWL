#!/usr/bin/env python3
"""Evaluate FACTSCORE outputs on DeFacto and UniSumEval.

DeFacto protocol (same as eval.el_perturbation.summarize_defacto):
  pair human vs candidate by document; correct iff score_human > score_candidate
  (strict). Only exclusive-intrinsic or exclusive-extrinsic samples count.

UniSumEval protocol (same as evaluators_benchmark / el_perturbation):
  keep summary_success_state==success and faithfulness_score!=1;
  this FACTSCORE file is the non-dialogue subset (CNNDM/GovReport/Pubmed/SQuALITY/wikihow).
  Match rows by uid + summary text (uid alone is not unique across models).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.el_perturbation import summarize_defacto

NON_DIALOGUE_SOURCES = {"CNNDM", "GovReport", "Pubmed", "SQuALITY", "wikihow"}
CAND_SUFFIX = "_candidate"
HUMAN_SUFFIX = "_humman_summary"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def parse_defacto_id(sample_id: str) -> tuple[str, str] | None:
    if sample_id.endswith(CAND_SUFFIX):
        return sample_id[: -len(CAND_SUFFIX)], "candidate"
    if sample_id.endswith(HUMAN_SUFFIX):
        return sample_id[: -len(HUMAN_SUFFIX)], "human"
    return None


def load_defacto_labels(sample_dir: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        path = sample_dir / f"defacto_{split}_candidate.jsonl"
        for row in load_jsonl(path):
            key = f"{split}_{row['doc_id']}"
            labels[key] = row
    return labels


def evaluate_defacto(
    cand_rows: list[dict],
    human_rows: list[dict],
    labels: dict[str, dict],
) -> dict:
    cand = {}
    human = {}
    parse_fail = []
    for row in cand_rows:
        parsed = parse_defacto_id(row["id"])
        if parsed is None:
            parse_fail.append(row["id"])
            continue
        cand[parsed[0]] = row
    for row in human_rows:
        parsed = parse_defacto_id(row["id"])
        if parsed is None:
            parse_fail.append(row["id"])
            continue
        human[parsed[0]] = row

    keys = sorted(set(cand) | set(human))
    rows = []
    missing_pair = []
    missing_label = []
    missing_score = []
    skipped_both = []
    per_split = defaultdict(list)

    for key in keys:
        if key not in cand or key not in human:
            missing_pair.append(key)
            continue
        lab = labels.get(key)
        if lab is None:
            missing_label.append(key)
            continue
        k_c = cand[key].get("factscore")
        k_h = human[key].get("factscore")
        # Same as eval.evaluation.factgraph: missing/NaN similarity is treated as 0
        # so the official 1570 exclusive-error pairs all remain evaluable.
        if k_c is None or k_h is None:
            missing_score.append(
                {
                    "key": key,
                    "candidate_score": k_c,
                    "human_score": k_h,
                    "candidate_num_facts": cand[key].get("num_facts"),
                    "human_num_facts": human[key].get("num_facts"),
                    "intrinsic": lab.get("intrinsic_error"),
                    "extrinsic": lab.get("extrinsic_error"),
                    "imputed_as": 0.0,
                }
            )
        rec = {
            "key": key,
            "intrinsic": lab.get("intrinsic_error"),
            "extrinsic": lab.get("extrinsic_error"),
            "k_cand": 0.0 if k_c is None else float(k_c),
            "k_human": 0.0 if k_h is None else float(k_h),
        }
        ie, ee = rec["intrinsic"], rec["extrinsic"]
        if ie is True and ee is True:
            skipped_both.append(key)
            continue
        if (ie is True and ee is False) or (ie is False and ee is True):
            rows.append(rec)
            per_split[key.split("_", 1)[0]].append(rec)

    summary = summarize_defacto(rows)
    split_summaries = {sp: summarize_defacto(rs) for sp, rs in per_split.items()}

    def mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "n_candidate_rows": len(cand_rows),
        "n_human_rows": len(human_rows),
        "n_paired_keys": len(set(cand) & set(human)),
        "n_eval": summary["n_eval"],
        "n_missing_pair": len(missing_pair),
        "n_missing_label": len(missing_label),
        "n_missing_score": len(missing_score),
        "n_skipped_both_error_types": len(skipped_both),
        "parse_fail_ids": parse_fail,
        "missing_score": missing_score,
        "mean_candidate": mean([r["k_cand"] for r in rows]),
        "mean_human": mean([r["k_human"] for r in rows]),
        "mean_candidate_intrinsic": mean(
            [r["k_cand"] for r in rows if r["intrinsic"] is True]
        ),
        "mean_human_intrinsic": mean(
            [r["k_human"] for r in rows if r["intrinsic"] is True]
        ),
        "mean_candidate_extrinsic": mean(
            [r["k_cand"] for r in rows if r["extrinsic"] is True]
        ),
        "mean_human_extrinsic": mean(
            [r["k_human"] for r in rows if r["extrinsic"] is True]
        ),
        **summary,
        "by_split": split_summaries,
    }


def match_unisum_rows(fs_rows: list[dict], labels: list[dict]) -> tuple[list[dict], list[dict]]:
    filtered = [
        r
        for r in labels
        if r.get("summary_success_state") == "success"
        and r.get("faithfulness_score") != 1
        and r.get("source") in NON_DIALOGUE_SOURCES
    ]
    by_uid: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(filtered):
        by_uid[str(row.get("uid"))].append(i)

    used: set[int] = set()
    pairs: list[dict] = []
    unmatched: list[dict] = []

    # Greedy 1-1: highest summary similarity within the same uid.
    pending = []
    for fs in fs_rows:
        uid = str(fs.get("id"))
        cands = [(j, filtered[j]) for j in by_uid.get(uid, [])]
        if not cands:
            unmatched.append({"id": uid, "reason": "uid_not_in_filtered_labels", "factscore": fs.get("factscore")})
            continue
        scored = []
        fs_sum = norm_text(fs.get("summary"))
        for j, lab in cands:
            ratio = SequenceMatcher(None, fs_sum, norm_text(lab.get("summary"))).ratio()
            scored.append((ratio, j, lab))
        scored.sort(reverse=True)
        pending.append((scored[0][0], fs, scored))

    pending.sort(reverse=True, key=lambda x: x[0])
    for best_ratio, fs, scored in pending:
        chosen = None
        chosen_ratio = None
        for ratio, j, lab in scored:
            if j not in used:
                chosen = (j, lab)
                chosen_ratio = ratio
                break
        if chosen is None or chosen_ratio is None or chosen_ratio < 0.95:
            unmatched.append(
                {
                    "id": fs.get("id"),
                    "reason": "no_unused_high_sim_summary",
                    "best_ratio": chosen_ratio,
                    "factscore": fs.get("factscore"),
                }
            )
            continue
        j, lab = chosen
        used.add(j)
        pairs.append(
            {
                "id": fs.get("id"),
                "doc_id": str(lab.get("doc_id")),
                "uid": lab.get("uid"),
                "model": lab.get("model"),
                "source": lab.get("source"),
                "factscore": fs.get("factscore"),
                "faithfulness_score": lab.get("faithfulness_score"),
                "summary_sim": chosen_ratio,
                "num_facts": fs.get("num_facts"),
            }
        )
    return pairs, unmatched


def _corr(xs: list[float], ys: list[float]) -> dict:
    out = {
        "n": len(xs),
        "pearson": None,
        "pearson_pvalue": None,
        "spearman": None,
        "spearman_pvalue": None,
        "kendall": None,
        "kendall_pvalue": None,
    }
    if len(xs) < 3 or len(set(xs)) <= 1 or len(set(ys)) <= 1:
        out["note"] = "correlation undefined (n<3 or constant input)"
        return out
    from scipy.stats import kendalltau, pearsonr, spearmanr

    pr, pp = pearsonr(ys, xs)
    sr, sp = spearmanr(ys, xs)
    kr, kp = kendalltau(ys, xs)
    out.update(
        {
            "pearson": float(pr),
            "pearson_pvalue": float(pp),
            "spearman": float(sr),
            "spearman_pvalue": float(sp),
            "kendall": float(kr),
            "kendall_pvalue": float(kp),
        }
    )
    return out


def system_rank_corr(pairs: list[dict]) -> dict:
    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in pairs:
        by_model[str(p["model"])].append((p["faithfulness_score"], p["factscore"]))
    models, human_means, pred_means = [], [], []
    for model, vals in sorted(by_model.items()):
        models.append(model)
        human_means.append(sum(v[0] for v in vals) / len(vals))
        pred_means.append(sum(v[1] for v in vals) / len(vals))
    from scipy.stats import rankdata, spearmanr

    if len(models) < 3 or len(set(human_means)) <= 1 or len(set(pred_means)) <= 1:
        return {
            "n_models": len(models),
            "rank_spearman": None,
            "rank_spearman_pvalue": None,
            "models": models,
            "note": "insufficient or constant system means",
        }
    r, p = spearmanr(rankdata(human_means), rankdata(pred_means))
    return {
        "n_models": len(models),
        "rank_spearman": float(r),
        "rank_spearman_pvalue": float(p),
        "per_model": [
            {
                "model": m,
                "n": len(by_model[m]),
                "mean_human": human_means[i],
                "mean_factscore": pred_means[i],
            }
            for i, m in enumerate(models)
        ],
    }


def evaluate_unisum(fs_rows: list[dict], labels: list[dict]) -> dict:
    pairs, unmatched = match_unisum_rows(fs_rows, labels)
    valid = [p for p in pairs if p.get("factscore") is not None and p.get("faithfulness_score") is not None]
    missing_score = [p for p in pairs if p.get("factscore") is None]
    xs = [float(p["factscore"]) for p in valid]
    ys = [float(p["faithfulness_score"]) for p in valid]
    overall = _corr(xs, ys)
    by_source = {}
    for src in sorted({p["source"] for p in valid}):
        sub = [p for p in valid if p["source"] == src]
        by_source[src] = _corr(
            [float(p["factscore"]) for p in sub],
            [float(p["faithfulness_score"]) for p in sub],
        )
    return {
        "n_factscore_rows": len(fs_rows),
        "n_matched": len(pairs),
        "n_used": len(valid),
        "n_missing_score": len(missing_score),
        "n_unmatched": len(unmatched),
        "unmatched": unmatched,
        "missing_score": missing_score,
        "mean_factscore": (sum(xs) / len(xs)) if xs else None,
        "mean_human": (sum(ys) / len(ys)) if ys else None,
        "min_summary_sim": min((p["summary_sim"] for p in pairs), default=None),
        **overall,
        "by_source": by_source,
        "system_level": system_rank_corr(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FACTSCORE on DeFacto / UniSumEval")
    parser.add_argument(
        "--factscore-dir",
        default=str(ROOT / "experiments/Factscore_output"),
    )
    parser.add_argument(
        "--defacto-label-dir",
        default=str(ROOT / "data/samples"),
    )
    parser.add_argument(
        "--unisum-labels",
        default="/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2_processed.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "experiments/results/factscore/metrics.json"),
    )
    args = parser.parse_args()

    fs_dir = Path(args.factscore_dir)
    cand = load_jsonl(fs_dir / "blink_candidate_factscore_per_sample.jsonl")
    human = load_jsonl(fs_dir / "blink_human_factscore_per_sample.jsonl")
    unisum_fs = load_jsonl(fs_dir / "unisumeval_factscore_per_sample.jsonl")
    defacto_labels = load_defacto_labels(Path(args.defacto_label_dir))
    unisum_labels = load_jsonl(Path(args.unisum_labels))

    defacto = evaluate_defacto(cand, human, defacto_labels)
    unisum = evaluate_unisum(unisum_fs, unisum_labels)
    result = {"method": "FACTSCORE", "defacto": defacto, "unisumeval": unisum}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
