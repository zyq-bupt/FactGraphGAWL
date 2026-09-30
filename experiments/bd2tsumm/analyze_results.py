"""合并 CompKG 与 ROUGE，计算相关性和系统排名。"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import kendalltau, linregress, pearsonr, rankdata, spearmanr

from util import (
    ensure_repo_on_path,
    load_config,
    parse_optional_float,
    read_csv,
    setup_logging,
    write_csv,
    write_json,
)

ensure_repo_on_path()

UNDEFINED = "undefined_constant_input"
INSUFFICIENT = "insufficient_data"
ROUGE_METRICS = ("rouge1_f1", "rouge2_f1", "rougeL_f1")


def _finite_pairs(xs: Sequence[float], ys: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        fx, fy = float(x), float(y)
        if math.isfinite(fx) and math.isfinite(fy):
            a.append(fx)
            b.append(fy)
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def correlation_record(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    analysis_level: str,
    rouge_metric: str,
    method: str,
    min_n: int,
    extra: dict | None = None,
) -> dict[str, Any]:
    rec = {
        "analysis_level": analysis_level,
        "rouge_metric": rouge_metric,
        "correlation_method": method,
        "coefficient": None,
        "p_value": None,
        "ci95_low": None,
        "ci95_high": None,
        "n": 0,
        "status": "ok",
        "note": "",
    }
    if extra:
        rec.update(extra)
    a, b = _finite_pairs(xs, ys)
    rec["n"] = int(a.size)
    if a.size < min_n:
        rec["status"] = INSUFFICIENT
        rec["note"] = f"n<{min_n}"
        return rec
    if np.std(a) == 0 or np.std(b) == 0 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        rec["status"] = UNDEFINED
        rec["note"] = UNDEFINED
        rec["coefficient"] = None
        rec["p_value"] = None
        return rec
    if method == "pearson":
        r, p = pearsonr(a, b)
    elif method == "spearman":
        r, p = spearmanr(a, b)
    elif method in {"kendall", "kendall_tau_b"}:
        try:
            r, p = kendalltau(a, b, variant="b")
        except TypeError:
            r, p = kendalltau(a, b)
    else:
        raise ValueError(method)
    rec["coefficient"] = float(r) if r is not None and math.isfinite(r) else None
    rec["p_value"] = float(p) if p is not None and math.isfinite(p) else None
    if rec["coefficient"] is None:
        rec["status"] = UNDEFINED
        rec["note"] = "non_finite_coefficient"
    return rec


def average_rank_desc(values: Sequence[float]) -> list[float]:
    arr = np.asarray(list(values), dtype=float)
    return list(rankdata(-arr, method="average"))


def pair_key(row: MappingLike) -> tuple[str, str]:
    return str(row["event_id"]), str(row["system_name"])


MappingLike = dict[str, Any]


def detect_duplicate_keys(rows: list[dict], label: str) -> list[tuple[str, str]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        counts[pair_key(r)] += 1
    return sorted(k for k, n in counts.items() if n > 1)


def merge_scores(
    compkg_rows: list[dict],
    rouge_rows: list[dict],
    logger: logging.Logger,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    dup_c = detect_duplicate_keys(compkg_rows, "compkg")
    dup_r = detect_duplicate_keys(rouge_rows, "rouge")
    if dup_c or dup_r:
        logger.error("检测到重复主键 compkg=%s rouge=%s", dup_c, dup_r)
    cmap: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rmap: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in compkg_rows:
        cmap[pair_key(r)].append(r)
    for r in rouge_rows:
        rmap[pair_key(r)].append(r)
    all_keys = sorted(set(cmap) | set(rmap))
    merged: list[dict] = []
    failures: list[dict] = []
    only_compkg = sorted(set(cmap) - set(rmap))
    only_rouge = sorted(set(rmap) - set(cmap))
    n_many = 0
    for k in all_keys:
        cl, rl = cmap.get(k, []), rmap.get(k, [])
        many = len(cl) > 1 or len(rl) > 1
        if many:
            n_many += 1
        c = cl[0] if cl else {}
        r = rl[0] if rl else {}
        c_ok = bool(cl) and str(c.get("status") or "") == "ok" and parse_optional_float(c.get("compkg_score")) is not None
        r_ok = bool(rl) and str(r.get("status") or "") == "ok"
        rouge_ok = r_ok and all(parse_optional_float(r.get(m)) is not None for m in ROUGE_METRICS)
        status = "ok" if c_ok and rouge_ok and not many else "excluded"
        row = {
            "event_id": k[0],
            "system_name": k[1],
            "compkg_score": parse_optional_float(c.get("compkg_score")),
            "compkg_score_raw": parse_optional_float(c.get("compkg_score_raw")),
            "compkg_variant": c.get("compkg_variant"),
            "source_num_tokens": parse_optional_float(c.get("source_num_tokens")),
            "source_num_windows": parse_optional_float(c.get("source_num_windows")),
            "rouge1_f1": parse_optional_float(r.get("rouge1_f1")),
            "rouge2_f1": parse_optional_float(r.get("rouge2_f1")),
            "rougeL_f1": parse_optional_float(r.get("rougeL_f1")),
            "num_references": r.get("num_references"),
            "compkg_status": c.get("status") or ("missing" if not cl else ""),
            "rouge_status": r.get("status") or ("missing" if not rl else ""),
            "compkg_error": c.get("error_message") or "",
            "rouge_error": r.get("error_message") or "",
            "join_status": status,
            "many_to_many": many,
        }
        merged.append(row)
        if status != "ok":
            reason = []
            if not cl:
                reason.append("missing_compkg")
            if not rl:
                reason.append("missing_rouge")
            if cl and not c_ok:
                reason.append(str(c.get("error_message") or c.get("status") or "compkg_failed"))
            if rl and not rouge_ok:
                reason.append(str(r.get("error_message") or r.get("status") or "rouge_failed"))
            if many:
                reason.append("many_to_many")
            failures.append(
                {
                    "event_id": k[0],
                    "system_name": k[1],
                    "reason": "|".join(reason),
                    "compkg_status": row["compkg_status"],
                    "rouge_status": row["rouge_status"],
                    "compkg_error": row["compkg_error"],
                    "rouge_error": row["rouge_error"],
                }
            )
    valid = [r for r in merged if r["join_status"] == "ok"]
    summary = {
        "n_compkg_rows": len(compkg_rows),
        "n_rouge_rows": len(rouge_rows),
        "n_keys_union": len(all_keys),
        "n_valid": len(valid),
        "n_excluded": len(failures),
        "n_duplicate_compkg_keys": len(dup_c),
        "n_duplicate_rouge_keys": len(dup_r),
        "n_many_to_many": n_many,
        "only_compkg": [list(x) for x in only_compkg],
        "only_rouge": [list(x) for x in only_rouge],
        "duplicate_compkg_keys": [list(x) for x in dup_c],
        "duplicate_rouge_keys": [list(x) for x in dup_r],
    }
    return merged, failures, summary


def sample_correlations(valid: list[dict], min_n: int) -> list[dict]:
    rows = []
    xs = [r["compkg_score"] for r in valid]
    for metric in ROUGE_METRICS:
        ys = [r[metric] for r in valid]
        for method in ("pearson", "spearman", "kendall_tau_b"):
            rows.append(correlation_record(xs, ys, analysis_level="sample", rouge_metric=metric, method=method, min_n=min_n))
    return rows


def event_centered_correlations(valid: list[dict], min_n: int) -> list[dict]:
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_event[r["event_id"]].append(r)
    out = []
    for metric in ROUGE_METRICS:
        cx, cy = [], []
        for _eid, rows in by_event.items():
            mc = float(np.mean([r["compkg_score"] for r in rows]))
            mr = float(np.mean([r[metric] for r in rows]))
            for r in rows:
                cx.append(r["compkg_score"] - mc)
                cy.append(r[metric] - mr)
        for method in ("pearson", "spearman", "kendall_tau_b"):
            rec = correlation_record(
                cx, cy, analysis_level="event_centered", rouge_metric=metric, method=method, min_n=min_n
            )
            out.append(rec)
    return out


def per_event_correlations(valid: list[dict], min_systems: int) -> list[dict]:
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_event[r["event_id"]].append(r)
    rows = []
    for eid, items in sorted(by_event.items()):
        n = len(items)
        xs = [r["compkg_score"] for r in items]
        for metric in ROUGE_METRICS:
            ys = [r[metric] for r in items]
            for method in ("spearman", "kendall_tau_b"):
                rec = correlation_record(
                    xs,
                    ys,
                    analysis_level="within_event",
                    rouge_metric=metric,
                    method=method,
                    min_n=min_systems,
                    extra={"event_id": eid, "n_systems": n},
                )
                if n < min_systems:
                    rec["status"] = INSUFFICIENT
                    rec["note"] = f"n_systems<{min_systems}"
                    rec["coefficient"] = None
                    rec["p_value"] = None
                rows.append(rec)
    return rows


def summarize_within_event(per_event: list[dict]) -> list[dict]:
    out = []
    for metric in ROUGE_METRICS:
        for method in ("spearman", "kendall_tau_b"):
            vals = [
                r["coefficient"]
                for r in per_event
                if r["rouge_metric"] == metric
                and r["correlation_method"] == method
                and r["status"] == "ok"
                and r["coefficient"] is not None
            ]
            n_undef = sum(
                1
                for r in per_event
                if r["rouge_metric"] == metric and r["correlation_method"] == method and r["status"] == UNDEFINED
            )
            n_ins = sum(
                1
                for r in per_event
                if r["rouge_metric"] == metric and r["correlation_method"] == method and r["status"] == INSUFFICIENT
            )
            rec = {
                "analysis_level": "within_event_summary",
                "rouge_metric": metric,
                "correlation_method": method,
                "coefficient": float(np.mean(vals)) if vals else None,
                "p_value": None,
                "ci95_low": None,
                "ci95_high": None,
                "n": len(vals),
                "status": "ok" if vals else INSUFFICIENT,
                "note": f"mean_of_defined_events; median={(float(np.median(vals)) if vals else None)}; "
                f"std={(float(np.std(vals, ddof=1)) if len(vals) > 1 else None)}; "
                f"undefined={n_undef}; insufficient={n_ins}",
                "mean": float(np.mean(vals)) if vals else None,
                "median": float(np.median(vals)) if vals else None,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            }
            out.append(rec)
    return out


def system_means(valid: list[dict]) -> dict[str, dict[str, float]]:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_sys[r["system_name"]].append(r)
    out: dict[str, dict[str, float]] = {}
    for sysn, rows in by_sys.items():
        out[sysn] = {
            "n_events": float(len(rows)),
            "compkg_score": float(np.mean([r["compkg_score"] for r in rows])),
            **{m: float(np.mean([r[m] for r in rows])) for m in ROUGE_METRICS},
        }
    return out


def system_level_correlations(means: dict[str, dict[str, float]], min_n: int) -> list[dict]:
    systems = sorted(means)
    xs = [means[s]["compkg_score"] for s in systems]
    rows = []
    for metric in ROUGE_METRICS:
        ys = [means[s][metric] for s in systems]
        for method in ("pearson", "spearman", "kendall_tau_b"):
            rec = correlation_record(
                xs, ys, analysis_level="system", rouge_metric=metric, method=method, min_n=min_n
            )
            rec["n"] = len(systems)
            rows.append(rec)
    return rows


def bootstrap_system_corr(
    valid: list[dict],
    *,
    n_boot: int,
    seed: int,
    min_n: int,
) -> list[dict]:
    events = sorted({r["event_id"] for r in valid})
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_event[r["event_id"]].append(r)
    rng = np.random.default_rng(seed)
    out = []
    if len(events) < min_n:
        for metric in ROUGE_METRICS:
            for method in ("pearson", "spearman", "kendall_tau_b"):
                rec = correlation_record(
                    [], [], analysis_level="system_bootstrap", rouge_metric=metric, method=method, min_n=min_n
                )
                rec["status"] = INSUFFICIENT
                rec["note"] = "too_few_events_for_bootstrap"
                out.append(rec)
        return out
    for metric in ROUGE_METRICS:
        for method in ("pearson", "spearman", "kendall_tau_b"):
            coeffs = []
            skipped = 0
            for _ in range(int(n_boot)):
                chosen = rng.choice(events, size=len(events), replace=True)
                boot_rows = []
                for eid in chosen:
                    boot_rows.extend(by_event[str(eid)])
                means = system_means(boot_rows)
                systems = sorted(means)
                xs = [means[s]["compkg_score"] for s in systems]
                ys = [means[s][metric] for s in systems]
                rec = correlation_record(xs, ys, analysis_level="boot", rouge_metric=metric, method=method, min_n=min_n)
                if rec["status"] == "ok" and rec["coefficient"] is not None:
                    coeffs.append(rec["coefficient"])
                else:
                    skipped += 1
            row = {
                "analysis_level": "system_bootstrap",
                "rouge_metric": metric,
                "correlation_method": method,
                "coefficient": float(np.mean(coeffs)) if coeffs else None,
                "p_value": None,
                "ci95_low": float(np.percentile(coeffs, 2.5)) if coeffs else None,
                "ci95_high": float(np.percentile(coeffs, 97.5)) if coeffs else None,
                "n": len(events),
                "status": "ok" if coeffs else INSUFFICIENT,
                "note": f"n_boot_valid={len(coeffs)}; skipped={skipped}",
            }
            out.append(row)
    return out


def build_ranking(
    valid: list[dict],
    *,
    min_coverage: float,
) -> tuple[list[dict], list[dict]]:
    events = sorted({r["event_id"] for r in valid})
    n_events = max(1, len(events))
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_sys[r["system_name"]].append(r)
    table = []
    for sysn, rows in sorted(by_sys.items()):
        covered = {r["event_id"] for r in rows}
        coverage = len(covered) / n_events
        complete = len(covered) == n_events
        in_main = complete or coverage + 1e-12 >= min_coverage
        # 覆盖不一致的系统不进入主排名：避免用更容易的子集均值比较
        if min_coverage >= 1.0:
            in_main = complete
        table.append(
            {
                "system_name": sysn,
                "n_events": len(covered),
                "n_events_universe": n_events,
                "coverage": coverage,
                "complete_coverage": complete,
                "in_main_ranking": in_main,
                "compkg_mean": float(np.mean([r["compkg_score"] for r in rows])),
                "rouge1_mean": float(np.mean([r["rouge1_f1"] for r in rows])),
                "rouge2_mean": float(np.mean([r["rouge2_f1"] for r in rows])),
                "rougeL_mean": float(np.mean([r["rougeL_f1"] for r in rows])),
            }
        )
    main = [r for r in table if r["in_main_ranking"]]
    if main:
        for col, rank_col in (
            ("compkg_mean", "compkg_rank"),
            ("rouge1_mean", "rouge1_rank"),
            ("rouge2_mean", "rouge2_rank"),
            ("rougeL_mean", "rougeL_rank"),
        ):
            ranks = average_rank_desc([r[col] for r in main])
            for row, rk in zip(main, ranks):
                row[rank_col] = float(rk)
    for row in table:
        if not row["in_main_ranking"]:
            row["compkg_rank"] = None
            row["rouge1_rank"] = None
            row["rouge2_rank"] = None
            row["rougeL_rank"] = None
        else:
            # copy ranks already set on main objects (same dicts)
            pass
    agree = ranking_agreement(main)
    return table, agree


def ranking_agreement(main: list[dict]) -> list[dict]:
    rows = []
    if len(main) < 2:
        return [
            {
                "rouge_metric": m,
                "spearman_rank": None,
                "kendall_tau_b_rank": None,
                "mean_abs_rank_diff": None,
                "top1_match": None,
                "top3_overlap": None,
                "top3_jaccard": None,
                "n_systems": len(main),
                "status": INSUFFICIENT,
            }
            for m in ("rouge1", "rouge2", "rougeL")
        ]
    cr = [r["compkg_rank"] for r in main]
    names = [r["system_name"] for r in main]
    top1_c = names[int(np.argmin(cr))]
    for metric, rank_key in (("rouge1", "rouge1_rank"), ("rouge2", "rouge2_rank"), ("rougeL", "rougeL_rank")):
        rr = [r[rank_key] for r in main]
        sp = correlation_record(cr, rr, analysis_level="rank", rouge_metric=metric, method="spearman", min_n=2)
        kd = correlation_record(cr, rr, analysis_level="rank", rouge_metric=metric, method="kendall_tau_b", min_n=2)
        diffs = [abs(float(a) - float(b)) for a, b in zip(cr, rr)]
        top1_r = names[int(np.argmin(rr))]
        n = len(main)
        if n >= 3:
            order_c = [names[i] for i in np.argsort(cr, kind="stable")][:3]
            order_r = [names[i] for i in np.argsort(rr, kind="stable")][:3]
            set_c, set_r = set(order_c), set(order_r)
            overlap = len(set_c & set_r)
            jacc = overlap / len(set_c | set_r) if (set_c | set_r) else None
        else:
            overlap, jacc = None, None
        rows.append(
            {
                "rouge_metric": metric,
                "spearman_rank": sp["coefficient"],
                "spearman_p": sp["p_value"],
                "kendall_tau_b_rank": kd["coefficient"],
                "kendall_p": kd["p_value"],
                "mean_abs_rank_diff": float(np.mean(diffs)),
                "top1_compkg": top1_c,
                "top1_rouge": top1_r,
                "top1_match": top1_c == top1_r,
                "top3_overlap": overlap,
                "top3_jaccard": jacc,
                "n_systems": n,
                "status": "ok",
            }
        )
    return rows


def _plot_scatter(valid: list[dict], out_dir: Path, dpi: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    xs = np.asarray([r["compkg_score"] for r in valid], dtype=float)
    for ax, metric, title in zip(axes, ROUGE_METRICS, ("ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1")):
        ys = np.asarray([r[metric] for r in valid], dtype=float)
        ax.scatter(xs, ys, s=22, alpha=0.75, c="#1f4e79")
        if xs.size >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
            sl, intercept, r, _p, _se = linregress(xs, ys)
            xline = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xline, sl * xline + intercept, color="#c0392b", lw=1.2, label=f"fit r={r:.3f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("CompKG-FactEval")
        ax.set_ylabel(title)
        ax.set_title(f"CompKG vs {title}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "scatter_compkg_vs_rouge.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    paths.append(str(p.relative_to(out_dir) if p.is_relative_to(out_dir) else p.name))
    return paths


def _plot_ranking(main: list[dict], out_dir: Path, dpi: int) -> str | None:
    if not main:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["system_name"] for r in main]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(names) + 2), 4.2))
    ax.bar(x - 1.5 * width, [r["compkg_rank"] for r in main], width, label="CompKG")
    ax.bar(x - 0.5 * width, [r["rouge1_rank"] for r in main], width, label="ROUGE-1")
    ax.bar(x + 0.5 * width, [r["rouge2_rank"] for r in main], width, label="ROUGE-2")
    ax.bar(x + 1.5 * width, [r["rougeL_rank"] for r in main], width, label="ROUGE-L")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("System ranking: CompKG vs ROUGE")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "ranking_comparison.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p.name


def _plot_event_corr(per_event: list[dict], out_dir: Path, dpi: int) -> str | None:
    vals = [
        r["coefficient"]
        for r in per_event
        if r["correlation_method"] == "spearman" and r["rouge_metric"] == "rouge1_f1" and r["status"] == "ok"
    ]
    if len(vals) < 2:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(vals, bins=min(10, len(vals)), color="#2e86ab", edgecolor="white")
    ax.set_xlabel("Within-event Spearman (CompKG vs ROUGE-1)")
    ax.set_ylabel("Events")
    ax.set_title("Distribution of within-event correlations")
    fig.tight_layout()
    p = out_dir / "per_event_corr_dist.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p.name


def _plot_coverage(ranking: list[dict], out_dir: Path, dpi: int) -> str | None:
    if not ranking:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["system_name"] for r in ranking]
    cov = [r["coverage"] for r in ranking]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.5 * len(names) + 2), 3.8))
    ax.bar(names, cov, color="#1f7a8c")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Event coverage")
    ax.set_title("Per-system event coverage (complete-case universe)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = out_dir / "system_coverage.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p.name


def _fmt(x: Any, nd=4) -> str:
    if x is None or x == "":
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_report(
    path: Path,
    *,
    cfg: dict,
    validation: dict | None,
    merge_summary: dict,
    corr_rows: list[dict],
    ranking: list[dict],
    agree: list[dict],
    figures: list[str],
    systems: list[str],
    n_valid: int,
) -> None:
    ck = cfg.get("compkg") or {}
    rg = cfg.get("rouge") or {}
    cl = cfg.get("cleaning") or {}
    lc = cfg.get("long_context") or {}
    lines = [
        "# BD2TSumm: CompKG-FactEval vs ROUGE",
        "",
        "## Data",
        f"- Events / systems / valid event–system pairs: {validation.get('n_events') if validation else 'NA'} / "
        f"{len(systems)} / {n_valid}",
        f"- Systems evaluated: {', '.join(systems) if systems else '(none)'}",
        f"- Candidate summaries: {validation.get('n_candidates') if validation else 'NA'}",
        f"- Reference summaries: {validation.get('n_references') if validation else 'NA'}",
        "",
        "## Tweet aggregation",
        f"- Dedup exact duplicates: `{cl.get('dedup_exact', True)}`; URL/mention placeholders: `{cl.get('placeholder_style')}`",
        f"- Hashtags kept as text: `{cl.get('keep_hashtag_text', True)}`; strip RT prefix: `{cl.get('strip_rt_prefix', False)}`",
        f"- Long context mode: `{lc.get('mode')}` (max={lc.get('max_model_tokens')}, overlap={lc.get('overlap_tokens')})",
        "- Silent prefix truncation is disabled unless `allow_prefix_truncation` is true.",
        "",
        "## CompKG-FactEval",
        f"- Variant: `{ck.get('variant')}` (S²-K (PT) = `kernel.gawl.compute_gawl_kernel_v2`)",
        f"- Edge weights wTT/wTM/wME/wEE = {ck.get('wTT')}/{ck.get('wTM')}/{ck.get('wME')}/{ck.get('wEE')}; T={ck.get('wl_t')}",
        f"- use_emb_labels={ck.get('use_emb_labels')}; higher_is_better={ck.get('higher_is_better')}; "
        f"treat_as_distance={ck.get('treat_as_distance')}",
        f"- Backend: `{ck.get('backend')}`",
        "",
        "## ROUGE",
        f"- Package: `{rg.get('package')}`; stemming={rg.get('use_stemmer')}",
        f"- Multi-reference aggregation: `{rg.get('multi_reference_aggregation')}` (no official BD2TSumm ROUGE script found)",
        "- Reported values are F1, not precision/recall.",
        "",
        "## Success / failure",
        f"- Union keys: {merge_summary.get('n_keys_union')}; valid complete-case: {merge_summary.get('n_valid')}; "
        f"excluded: {merge_summary.get('n_excluded')}",
        f"- CompKG-only keys: {merge_summary.get('only_compkg')}",
        f"- ROUGE-only keys: {merge_summary.get('only_rouge')}",
        f"- Many-to-many joins: {merge_summary.get('n_many_to_many')}",
        "",
        "## Correlations",
        "",
        "| level | ROUGE | method | r | p | n | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in corr_rows:
        if r.get("analysis_level") == "within_event":
            continue
        lines.append(
            f"| {r.get('analysis_level')} | {r.get('rouge_metric')} | {r.get('correlation_method')} | "
            f"{_fmt(r.get('coefficient'))} | {_fmt(r.get('p_value'))} | {r.get('n')} | {r.get('status')} |"
        )
    lines += [
        "",
        "## System means and ranks (main ranking = consistent event coverage)",
        "",
        "| system | n_events | coverage | CompKG mean | CompKG rank | R-1 mean | R-1 rank | R-2 mean | R-2 rank | R-L mean | R-L rank | in_main |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in ranking:
        lines.append(
            f"| {r['system_name']} | {int(r['n_events'])} | {_fmt(r['coverage'], 3)} | "
            f"{_fmt(r['compkg_mean'])} | {_fmt(r.get('compkg_rank'), 2)} | "
            f"{_fmt(r['rouge1_mean'])} | {_fmt(r.get('rouge1_rank'), 2)} | "
            f"{_fmt(r['rouge2_mean'])} | {_fmt(r.get('rouge2_rank'), 2)} | "
            f"{_fmt(r['rougeL_mean'])} | {_fmt(r.get('rougeL_rank'), 2)} | {r['in_main_ranking']} |"
        )
    lines += ["", "## Ranking agreement", ""]
    for r in agree:
        lines.append(
            f"- vs {r.get('rouge_metric')}: Spearman={_fmt(r.get('spearman_rank'))}, "
            f"Kendall τb={_fmt(r.get('kendall_tau_b_rank'))}, "
            f"mean |Δrank|={_fmt(r.get('mean_abs_rank_diff'))}, "
            f"Top-1 match={r.get('top1_match')}, Top-3 Jaccard={_fmt(r.get('top3_jaccard'))}"
        )
    lines += ["", "## Figures"]
    if figures:
        for fig in figures:
            lines.append(f"- `{fig}`")
    else:
        lines.append("- (no figures)")
    lines += [
        "",
        "## Conclusion",
        "",
        "ROUGE measures n-gram overlap between a candidate summary and the human reference summary. "
        "CompKG-FactEval measures factual consistency between the candidate summary and the source tweet set "
        "via FK-Graph similarity (S²-K / GAWL). Agreement in correlation or system ranking is supplementary "
        "evidence that the two metrics sometimes order systems similarly; it does not by itself prove that they "
        "measure the same capability. Scores used in this report are original (not min-max normalized).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analyze(cfg: dict, output_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    out = cfg.get("output") or {}
    an = cfg.get("analysis") or {}
    min_n = int(an.get("min_n_for_corr") or 3)
    min_sys = int(an.get("min_systems_for_event_corr") or 3)
    min_cov = float(an.get("min_event_coverage") or 1.0)
    dpi = int(an.get("dpi") or 300)
    seed = int(cfg.get("seed") or 42)

    compkg = read_csv(output_dir / out.get("compkg_scores", "compkg_scores.csv"))
    rouge = read_csv(output_dir / out.get("rouge_scores", "rouge_scores.csv"))
    merged, failures, merge_summary = merge_scores(compkg, rouge, logger)
    valid = [r for r in merged if r["join_status"] == "ok"]
    logger.info("合并完成 valid=%s excluded=%s", len(valid), len(failures))

    # 失败原因 × 系统
    fail_by_sys: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in failures:
        fail_by_sys[str(f["system_name"])][str(f["reason"])] += 1

    corr = []
    corr.extend(sample_correlations(valid, min_n))
    corr.extend(event_centered_correlations(valid, min_n))
    per_event = per_event_correlations(valid, min_sys)
    corr.extend(summarize_within_event(per_event))
    means = system_means(valid) if valid else {}
    corr.extend(system_level_correlations(means, min_n))
    corr.extend(
        bootstrap_system_corr(
            valid,
            n_boot=int(an.get("bootstrap_n") or 10000),
            seed=seed,
            min_n=int(an.get("bootstrap_min_n") or 3),
        )
    )
    ranking, agree = build_ranking(valid, min_coverage=min_cov)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    if len(valid) >= 2:
        figures.extend(_plot_scatter(valid, fig_dir, dpi))
    rp = _plot_ranking([r for r in ranking if r["in_main_ranking"]], fig_dir, dpi)
    if rp:
        figures.append(f"figures/{rp}")
    ep = _plot_event_corr(per_event, fig_dir, dpi)
    if ep:
        figures.append(f"figures/{ep}")
    cp = _plot_coverage(ranking, fig_dir, dpi)
    if cp:
        figures.append(f"figures/{cp}")
    # scatter paths are filenames under figures/
    figures = [f if f.startswith("figures/") else f"figures/{Path(f).name}" for f in figures]

    write_csv(output_dir / out.get("all_sample_scores", "all_sample_scores.csv"), merged)
    write_csv(output_dir / out.get("failure_report", "failure_report.csv"), failures)
    write_csv(
        output_dir / out.get("correlation_results", "correlation_results.csv"),
        corr,
        [
            "analysis_level",
            "rouge_metric",
            "correlation_method",
            "coefficient",
            "p_value",
            "ci95_low",
            "ci95_high",
            "n",
            "status",
            "note",
            "event_id",
            "n_systems",
            "mean",
            "median",
            "std",
        ],
    )
    write_csv(output_dir / out.get("per_event_correlations", "per_event_correlations.csv"), per_event)
    write_csv(
        output_dir / out.get("system_ranking", "system_ranking.csv"),
        ranking,
        [
            "system_name",
            "n_events",
            "n_events_universe",
            "coverage",
            "complete_coverage",
            "in_main_ranking",
            "compkg_mean",
            "compkg_rank",
            "rouge1_mean",
            "rouge1_rank",
            "rouge2_mean",
            "rouge2_rank",
            "rougeL_mean",
            "rougeL_rank",
        ],
    )
    write_csv(output_dir / out.get("ranking_agreement", "ranking_agreement.csv"), agree)

    validation = None
    vpath = output_dir / out.get("data_validation", "data_validation.json")
    if vpath.is_file():
        import json

        validation = json.loads(vpath.read_text(encoding="utf-8"))
    systems = sorted({r["system_name"] for r in valid} | {r["system_name"] for r in ranking})
    write_report(
        output_dir / out.get("summary_report", "summary_report.md"),
        cfg=cfg,
        validation=validation,
        merge_summary=merge_summary,
        corr_rows=corr,
        ranking=ranking,
        agree=agree,
        figures=figures,
        systems=systems,
        n_valid=len(valid),
    )
    write_json(output_dir / "failure_by_system.json", {s: dict(v) for s, v in fail_by_sys.items()})
    logger.info("分析完成 figures=%s", figures)
    return {"valid": valid, "ranking": ranking, "correlations": corr, "merge_summary": merge_summary}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BD2TSumm 相关性与排名")
    p.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--output-dir", type=str, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / "logs" / "analyze_results.log")
    run_analyze(cfg, output_dir, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
