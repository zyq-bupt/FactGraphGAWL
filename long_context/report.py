"""根据真实结果生成 CSV / JSON / 论文表 / report.md。不得编造数字。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from long_context.stats import mean_std, median, paired_bootstrap_delta_r, pearson_with_p, spearman_optional

LIMITATIONS = """
- Cross-window entity relations are not inferred; only relations observed inside a window are kept.
- Conflicting or inconsistent entity links across windows may remain after exact KB-id merge.
- Sliding windows increase graph size and runtime.
- This is a preliminary study on UniSumEval GovReport only.
- Legal and scientific long documents still require separate evaluation.
""".strip()


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _col(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v is not None:
            out.append(float(v))
    return out


def choose_wording(coverage_up: bool, delta_r: float | None, ci_low: float | None, ci_high: float | None) -> tuple[str, str]:
    if delta_r is None:
        text = (
            "Correlation statistics could not be computed on the paired sample "
            "(insufficient variance or too few samples). Coverage and cost figures below are still based on actual runs."
        )
        return "insufficient_stats", text
    ci_has_zero = ci_low is not None and ci_high is not None and ci_low <= 0.0 <= ci_high
    pearson_up = delta_r > 0
    modest = pearson_up and (delta_r < 0.05 or ci_has_zero)
    if coverage_up and pearson_up and not ci_has_zero and not modest:
        return "A", (
            "The preliminary results indicate that extending FK-Graph construction beyond the first 1,024 tokens "
            "improves document coverage and correlation with human judgments on GovReport. This suggests that "
            "input truncation is one factor contributing to the relatively lower performance on long documents "
            "and supports the feasibility of sliding-window graph construction."
        )
    if coverage_up and modest:
        return "B", (
            "The sliding-window strategy substantially increases graph coverage and yields a modest correlation "
            "improvement. However, the uncertainty of the observed gain indicates that larger-scale evaluation "
            "and explicit cross-window relation modeling are still needed."
        )
    return "C", (
        "Although sliding windows increase document coverage, simple merging of independently constructed "
        "local graphs does not improve correlation on GovReport. This result suggests that coverage alone "
        "is insufficient and that cross-window relations, entity-linking consistency, and hierarchical "
        "structure should be modeled explicitly."
    )


def write_outputs(
    output_dir: Path,
    *,
    cfg: dict,
    implementation_notes: str,
    sample_rows: list[dict],
    n_original: int,
    failures: list[dict],
    tokenizer_note: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = [r for r in sample_rows if r.get("status") == "ok"
              and r.get("baseline_score") is not None
              and r.get("sliding_score") is not None]
    human = [float(r["human_score"]) for r in paired]
    base_s = [float(r["baseline_score"]) for r in paired]
    slid_s = [float(r["sliding_score"]) for r in paired]
    r_b, p_b = pearson_with_p(human, base_s)
    r_s, p_s = pearson_with_p(human, slid_s)
    delta_r = None if r_b is None or r_s is None else float(r_s - r_b)
    boot = paired_bootstrap_delta_r(
        human, base_s, slid_s,
        n_boot=int(cfg.get("bootstrap_n") or 10000),
        seed=int(cfg.get("seed") or 20260818),
    ) if paired else {
        "n_boot_requested": cfg.get("bootstrap_n"),
        "n_boot_valid": 0,
        "n_boot_skipped": 0,
        "delta_r_ci95_low": None,
        "delta_r_ci95_high": None,
        "delta_r_boot_mean": None,
        "deltas": [],
    }

    def agg(prefix: str) -> dict:
        cov = _col(paired, f"{prefix}_token_coverage_ratio")
        ent = _col(paired, f"{prefix}_num_entity_nodes")
        ee = _col(paired, f"{prefix}_num_E_ee")
        tok_n = _col(paired, f"{prefix}_num_token_nodes")
        men = _col(paired, f"{prefix}_num_mention_nodes")
        ett = _col(paired, f"{prefix}_num_E_tt")
        etm = _col(paired, f"{prefix}_num_E_tm")
        eme = _col(paired, f"{prefix}_num_E_me")
        gb = _col(paired, f"{prefix}_graph_build_seconds")
        tot = _col(paired, f"{prefix}_total_seconds")
        mu_cov, sd_cov = mean_std(cov)
        mu_ent, sd_ent = mean_std(ent)
        mu_ee, sd_ee = mean_std(ee)
        mu_gb, sd_gb = mean_std(gb)
        mu_tot, sd_tot = mean_std(tot)
        return {
            "pearson_r": r_b if prefix == "baseline" else r_s,
            "p_value": p_b if prefix == "baseline" else p_s,
            "mean_token_coverage": mu_cov,
            "std_token_coverage": sd_cov,
            "median_token_coverage": median(cov),
            "mean_entity_nodes": mu_ent,
            "std_entity_nodes": sd_ent,
            "median_entity_nodes": median(ent),
            "mean_relation_edges": mu_ee,
            "std_relation_edges": sd_ee,
            "median_relation_edges": median(ee),
            "mean_token_nodes": mean_std(tok_n)[0],
            "mean_mention_nodes": mean_std(men)[0],
            "mean_E_tt": mean_std(ett)[0],
            "mean_E_tm": mean_std(etm)[0],
            "mean_E_me": mean_std(eme)[0],
            "mean_graph_build_seconds": mu_gb,
            "std_graph_build_seconds": sd_gb,
            "median_graph_build_seconds": median(gb),
            "mean_total_seconds": mu_tot,
            "std_total_seconds": sd_tot,
            "median_total_seconds": median(tot),
        }

    baseline = agg("baseline")
    sliding = agg("sliding")
    runtime_mult = None
    if baseline.get("mean_total_seconds") and sliding.get("mean_total_seconds"):
        if baseline["mean_total_seconds"]:
            runtime_mult = sliding["mean_total_seconds"] / baseline["mean_total_seconds"]

    cov_b = baseline.get("mean_token_coverage") or 0.0
    cov_s = sliding.get("mean_token_coverage") or 0.0
    coverage_up = cov_s > cov_b + 0.01
    case, paragraph = choose_wording(
        coverage_up, delta_r, boot.get("delta_r_ci95_low"), boot.get("delta_r_ci95_high")
    )

    d_cov = [ (r.get("sliding_token_coverage_ratio") or 0) - (r.get("baseline_token_coverage_ratio") or 0) for r in paired ]
    d_score = [ (r.get("sliding_score") or 0) - (r.get("baseline_score") or 0) for r in paired ]
    expl_r, expl_p = spearman_optional(d_cov, d_score) if paired else (None, None)

    aggregate = {
        "n_original": n_original,
        "n_paired": len(paired),
        "n_failed": len(failures),
        "tokenizer_note": tokenizer_note,
        "baseline": baseline,
        "sliding_window": sliding,
        "comparison": {
            "delta_r": delta_r,
            "delta_r_ci95_low": boot.get("delta_r_ci95_low"),
            "delta_r_ci95_high": boot.get("delta_r_ci95_high"),
            "runtime_multiplier": runtime_mult,
            "wording_case": case,
            "exploratory_spearman_coverage_vs_score_delta": expl_r,
            "exploratory_spearman_p": expl_p,
            "bootstrap_valid": boot.get("n_boot_valid"),
            "bootstrap_skipped": boot.get("n_boot_skipped"),
        },
        "stat_convention": {
            "entity_nodes": "mean (also report median in report.md)",
            "entity_relation_edges": "mean of E_ee",
            "time_per_sample": "mean of total_seconds",
            "token_coverage": "mean of unique content-token coverage; overlap counted once; ratio capped at 1",
        },
    }

    # files
    (output_dir / "config.yaml").write_text(
        __import__("yaml").safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "implementation_notes.md").write_text(implementation_notes, encoding="utf-8")
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fieldnames = [
        "sample_id", "domain", "document_model_tokens", "human_score",
        "baseline_score", "sliding_score",
        "baseline_num_windows", "sliding_num_windows",
        "baseline_unique_tokens", "sliding_unique_tokens",
        "baseline_token_coverage_ratio", "sliding_token_coverage_ratio",
        "baseline_num_token_nodes", "sliding_num_token_nodes",
        "baseline_num_mention_nodes", "sliding_num_mention_nodes",
        "baseline_num_entity_nodes", "sliding_num_entity_nodes",
        "baseline_num_E_tt", "sliding_num_E_tt",
        "baseline_num_E_tm", "sliding_num_E_tm",
        "baseline_num_E_me", "sliding_num_E_me",
        "baseline_num_E_ee", "sliding_num_E_ee",
        "baseline_graph_build_seconds", "sliding_graph_build_seconds",
        "baseline_total_seconds", "sliding_total_seconds",
        "status", "error_type",
    ]
    with (output_dir / "sample_level_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sample_rows:
            w.writerow(r)

    with (output_dir / "failures.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "error_type", "status"])
        w.writeheader()
        for r in failures:
            w.writerow({"sample_id": r.get("sample_id"), "error_type": r.get("error_type"), "status": r.get("status")})

    with (output_dir / "bootstrap_delta_r.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["replicate", "delta_r"])
        for i, d in enumerate(boot.get("deltas") or []):
            w.writerow([i, d])

    table_rows = [
        {
            "Graph construction": "Prefix truncation",
            "Token coverage (%)": None if baseline["mean_token_coverage"] is None else 100 * baseline["mean_token_coverage"],
            "Entity nodes": baseline["mean_entity_nodes"],
            "Entity-relation edges": baseline["mean_relation_edges"],
            "Pearson r": baseline["pearson_r"],
            "p-value": baseline["p_value"],
            "Time/sample": baseline["mean_total_seconds"],
        },
        {
            "Graph construction": "Sliding window",
            "Token coverage (%)": None if sliding["mean_token_coverage"] is None else 100 * sliding["mean_token_coverage"],
            "Entity nodes": sliding["mean_entity_nodes"],
            "Entity-relation edges": sliding["mean_relation_edges"],
            "Pearson r": sliding["pearson_r"],
            "p-value": sliding["p_value"],
            "Time/sample": sliding["mean_total_seconds"],
        },
    ]
    with (output_dir / "long_context_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    report = _render_report(
        aggregate, tokenizer_note, paragraph, case, n_original, paired, failures, cfg
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return aggregate


def _render_report(agg, tokenizer_note, paragraph, case, n_original, paired, failures, cfg) -> str:
    b = agg["baseline"]
    s = agg["sliding_window"]
    c = agg["comparison"]
    md = []
    md.append("# Sliding-window FK-Graph on UniSumEval GovReport\n")
    md.append("All numbers below are computed from the actual run. None are placeholders.\n")
    md.append("## Setup\n")
    md.append(f"- Domain filter: `{cfg.get('domain')}` with UniSumEval rules "
              f"`summary_success_state==success` and `faithfulness_score!=1`.\n")
    md.append(f"- Original domain samples: **{n_original}**. Paired successful samples: **{agg['n_paired']}**. "
              f"Failures: **{agg['n_failed']}**.\n")
    md.append(f"- Tokenizer: {tokenizer_note}\n")
    md.append(f"- Window: max_model_tokens={cfg.get('max_model_tokens')}, overlap_tokens={cfg.get('overlap_tokens')}, "
              f"sentence_boundary={cfg.get('sentence_boundary')}.\n")
    md.append("- Summary-side FK-Graph is identical across conditions (built from the same dreeam `candidate`).\n")
    md.append("- S²-K (PT) is `kernel.gawl.compute_gawl_kernel_v2` via `eval.el_perturbation.score_sample_gawl`.\n")
    md.append("- Entity nodes / entity-relation edges / time in the table are **means** over paired samples; "
              "medians are listed in aggregate_metrics.json.\n")
    md.append("- Token coverage uses the same tokenizer as windowing; special tokens excluded from the ratio; "
              "overlapping windows count each content token at most once; ratio is capped at 1.\n")
    md.append("- Entity-relation edges (E_ee) are taken from precomputed DREEAM `ent_relation`; "
              "on this GovReport slice they are all zero, so sliding windows do not add new relations "
              "without re-running DREEAM.\n")
    md.append("\n## Paper table\n")
    md.append("| Graph construction | Token coverage (%) | Entity nodes | Entity-relation edges | Pearson r | p-value | Time/sample |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    md.append(
        f"| Prefix truncation | {_fmt(None if b['mean_token_coverage'] is None else 100*b['mean_token_coverage'], 2)} | "
        f"{_fmt(b['mean_entity_nodes'], 2)} | {_fmt(b['mean_relation_edges'], 2)} | "
        f"{_fmt(b['pearson_r'])} | {_fmt(b['p_value'])} | {_fmt(b['mean_total_seconds'], 3)} |\n"
    )
    md.append(
        f"| Sliding window | {_fmt(None if s['mean_token_coverage'] is None else 100*s['mean_token_coverage'], 2)} | "
        f"{_fmt(s['mean_entity_nodes'], 2)} | {_fmt(s['mean_relation_edges'], 2)} | "
        f"{_fmt(s['pearson_r'])} | {_fmt(s['p_value'])} | {_fmt(s['mean_total_seconds'], 3)} |\n"
    )
    md.append("\n## Correlation comparison\n")
    md.append(f"- delta_r = r_sliding - r_baseline = **{_fmt(c['delta_r'])}**\n")
    md.append(f"- Paired bootstrap 95% percentile CI: "
              f"[{_fmt(c['delta_r_ci95_low'])}, {_fmt(c['delta_r_ci95_high'])}] "
              f"(valid resamples={c.get('bootstrap_valid')}, skipped={c.get('bootstrap_skipped')})\n")
    md.append(f"- Runtime multiplier (mean total time): {_fmt(c['runtime_multiplier'], 3)}\n")
    md.append("\n## Wording (case "
              + case
              + ")\n\n")
    md.append(paragraph + "\n")
    md.append("\n## Limitations\n\n")
    md.append(LIMITATIONS + "\n")
    md.append("\n## Failures\n")
    if not failures:
        md.append("None.\n")
    else:
        for f in failures:
            md.append(f"- `{f.get('sample_id')}`: {f.get('error_type')}\n")
    md.append("\nDo not interpret a significant sliding-window Pearson and a non-significant baseline Pearson "
              "as a significant difference between methods; use delta_r and its CI.\n")
    return "".join(md)
