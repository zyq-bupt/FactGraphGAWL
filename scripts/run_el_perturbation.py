#!/usr/bin/env python3
"""实体链接扰动实验 CLI。

在已有 factgraph JSON 上随机删除成功 EL，再跑 GAWL，不重训、不换链接器。

示例（先用 --limit 冒烟）：

  PYTHONPATH=. python scripts/run_el_perturbation.py \\
      --datasets DeFacto UniSumEval --repeats 3 --limit 8

完整实验（与 kernel.gawl 默认权重一致；可按主结果改权重）：

  PYTHONPATH=. python scripts/run_el_perturbation.py \\
      --datasets DeFacto UniSumEval --repeats 5 \\
      --ratios 0 0.1 0.2 0.3 0.5 \\
      --wTT 1.0 --wTM 1.0 --wME 0.5 --wEE 0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel.gawl import default_edge_major_type_weights  # noqa: E402
from eval.el_perturbation import (  # noqa: E402
    REPO_ROOT,
    aggregate_repeats,
    evaluate_once,
    load_unisum_labels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entity-linking drop perturbation for GAWL")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["DeFacto", "UniSumEval"],
        choices=["DeFacto", "UniSumEval"],
    )
    p.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.5],
        help="删除成功实体链接的比例",
    )
    p.add_argument("--repeats", type=int, default=5, help="每个非零比例的重复次数（建议 3–5）")
    p.add_argument("--seed", type=int, default=42, help="第 1 次 repeat 的种子，之后 +1,+2,...")
    p.add_argument("--limit", type=int, default=None, help="每个数据集最多用多少个 json（冒烟用）")
    p.add_argument("--T", type=int, default=1, help="WL 迭代轮数，与 kernel.gawl 一致")
    p.add_argument("--wTT", type=float, default=1.0)
    p.add_argument("--wTM", type=float, default=1.0)
    p.add_argument("--wME", type=float, default=0.5)
    p.add_argument("--wEE", type=float, default=0.0)
    p.add_argument(
        "--defacto-root",
        type=Path,
        default=Path("/root/autodl-fs/zyq/defacto_data_gawl/factgraph_result_withemb"),
    )
    p.add_argument(
        "--unisum-root",
        type=Path,
        default=Path("/root/autodl-fs/zyq/unisumeval_data_gawl/factgraph_result"),
    )
    p.add_argument(
        "--unisum-jsonl",
        type=Path,
        default=Path("/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2_processed.jsonl"),
        help="含 faithfulness_score 的 UniSumEval jsonl",
    )
    p.add_argument("--defacto-splits", nargs="+", default=["test", "val", "train"])
    p.add_argument("--unisum-splits", nargs="+", default=["test"])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "el_perturbation",
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--use-emb-unisum",
        action="store_true",
        help="UniSumEval 也用节点 embedding（需 factgraph 含 embedding 字段）",
    )
    p.add_argument(
        "--unisum-all-files",
        action="store_true",
        help="UniSumEval 对全部 json 跑相似度（默认只跑 Pearson 评价子集：success 且 score!=1）",
    )
    return p.parse_args()


def _fmt(mu, sd, digits=4):
    if mu is None:
        return "NA"
    if sd is None or sd == 0:
        return f"{mu:.{digits}g}"
    return f"{mu:.{digits}g} ± {sd:.{digits}g}"


def save_plot(summary_rows: list[dict], out_png: Path, out_eps: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    defacto = [r for r in summary_rows if r["dataset"].lower() == "defacto"]
    unisum = [r for r in summary_rows if r["dataset"].lower() != "defacto"]
    n_panels = int(bool(defacto)) + int(bool(unisum))
    if n_panels == 0:
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.2), squeeze=False)
    ax_i = 0
    if defacto:
        ax = axes[0, ax_i]
        ax_i += 1
        xs = [r["drop_ratio"] * 100 for r in defacto]
        ys = [r["n_correct_mean"] for r in defacto]
        yerr = [r["n_correct_std"] or 0.0 for r in defacto]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
        ax.set_xlabel("Dropped EL (%)")
        ax.set_ylabel("DeFacto (n correct)")
        ax.set_xticks(xs)
    if unisum:
        ax = axes[0, ax_i]
        xs = [r["drop_ratio"] * 100 for r in unisum]
        ys = [r["pearson_mean"] for r in unisum]
        yerr = [r["pearson_std"] or 0.0 for r in unisum]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, color="C1")
        ax.set_xlabel("Dropped EL (%)")
        ax.set_ylabel("UniSumEval Pearson")
        ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_eps)
    plt.close(fig)
    _ = np  # keep import used if axes empty


def print_table(summary_rows: list[dict]) -> None:
    print("\n========== Entity-linking perturbation ==========")
    for r in summary_rows:
        ratio = f"{r['drop_ratio']:.0%}"
        if r["dataset"].lower() == "defacto":
            print(
                f"DeFacto  drop={ratio:>4}  "
                f"n_correct={_fmt(r.get('n_correct_mean'), r.get('n_correct_std'), 4)}  "
                f"acc={_fmt(r.get('accuracy_mean'), r.get('accuracy_std'), 4)}  "
                f"FNR={_fmt(r.get('fn_rate_mean'), r.get('fn_rate_std'), 4)}  "
                f"n_eval={_fmt(r.get('n_eval_mean'), r.get('n_eval_std'), 6)}"
            )
        else:
            print(
                f"UniSum   drop={ratio:>4}  "
                f"Pearson={_fmt(r.get('pearson_mean'), r.get('pearson_std'), 4)}  "
                f"Spearman={_fmt(r.get('spearman_mean'), r.get('spearman_std'), 4)}  "
                f"n={_fmt(r.get('n_used_mean'), r.get('n_used_std'), 6)}"
            )
    print("=================================================\n")


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = out_dir / "tmp_gawl"
    tmp_root.mkdir(parents=True, exist_ok=True)

    weights = default_edge_major_type_weights(args.wTT, args.wTM, args.wME, args.wEE)
    unisum_labels = None
    if "UniSumEval" in args.datasets:
        unisum_labels = load_unisum_labels(args.unisum_jsonl)

    all_runs = []
    summary_rows = []

    for dataset in args.datasets:
        if dataset == "DeFacto":
            graph_root = args.defacto_root
            splits = args.defacto_splits
            keys = ["article", "candidate", "humman_summary"]
            use_emb = True
        else:
            graph_root = args.unisum_root
            splits = args.unisum_splits
            keys = ["article", "candidate"]
            use_emb = bool(args.use_emb_unisum)

        for ratio in args.ratios:
            n_rep = 1 if abs(ratio) < 1e-12 else args.repeats
            runs = []
            for i in range(n_rep):
                seed = args.seed + i
                run = evaluate_once(
                    dataset=dataset,
                    graph_root=graph_root,
                    splits=splits,
                    drop_ratio=ratio,
                    seed=seed,
                    graph_keys=keys,
                    use_emb_labels=use_emb,
                    wl_t=args.T,
                    major_weights=weights,
                    unisum_labels=unisum_labels,
                    limit=args.limit,
                    tmp_root=tmp_root,
                    unisum_all_files=args.unisum_all_files,
                )
                runs.append(run)
                all_runs.append(run)
                print(json.dumps(run, ensure_ascii=False, default=str))
            summary_rows.append(aggregate_repeats(runs))

    raw_path = out_dir / "runs.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for run in all_runs:
            f.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "summary.csv"
    try:
        import pandas as pd

        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 未能写 CSV: {exc}")

    meta = {
        "weights": weights,
        "ratios": args.ratios,
        "repeats": args.repeats,
        "seed": args.seed,
        "limit": args.limit,
        "datasets": args.datasets,
        "T": args.T,
        "defacto_root": str(args.defacto_root),
        "unisum_root": str(args.unisum_root),
        "unisum_jsonl": str(args.unisum_jsonl),
        "note": (
            "0% 只跑 1 次（确定结果）。FNR = 1 - accuracy，"
            "把并列也算作未检出错误。UniSumEval Pearson 与 evaluators_benchmark 一致："
            "summary_success_state==success 且 faithfulness_score!=1。"
        ),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print_table(summary_rows)
    print(f"runs   -> {raw_path}")
    print(f"summary-> {summary_path}")
    print(f"csv    -> {csv_path}")

    if not args.no_plot:
        try:
            save_plot(
                summary_rows,
                out_dir / "el_perturbation.png",
                out_dir / "el_perturbation.eps",
            )
            print(f"plot   -> {out_dir / 'el_perturbation.png'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 画图失败: {exc}")


if __name__ == "__main__":
    main()
