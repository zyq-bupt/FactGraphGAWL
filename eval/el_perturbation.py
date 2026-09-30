"""实体链接扰动：在已建好的事实图上随机删除成功 EL，再跑 GAWL，无需重训或更换链接器。

成功实体链接 = 图中带非空 wikidata_id 的 Entity 节点。
删除一个实体时同时去掉：
  - 该实体节点
  - 连到它的 mention–entity 边（BELONGS_TO）
  - 涉及它的 entity–entity 关系
Mention / Token 及 token–mention 边保留，模拟“识别到提及但链接失败”。
每个子图（article / candidate / human_summary）独立按同一比例抽样。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

EMPTY_WIKIDATA = {"", "none", "null", "nil", "-1", "nan"}


def is_entity_node(node: dict) -> bool:
    nid = str(node.get("node_id") or node.get("id") or "")
    return node.get("label") == "Entity" or nid.startswith("Entity_")


def is_successful_entity(node: dict) -> bool:
    """已成功链到 KB 的实体（有可用 wikidata_id）。"""
    if not is_entity_node(node):
        return False
    wid = node.get("wikidata_id")
    if wid is None:
        return False
    return str(wid).strip().lower() not in EMPTY_WIKIDATA


def _node_id(node: dict) -> str:
    return str(node.get("node_id") or node.get("id") or "")


def _is_entity_ref(ref: Any) -> bool:
    return str(ref).startswith("Entity_")


def perturb_graph(
    graph: dict,
    drop_ratio: float,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """按比例删除成功实体链接，返回 (新图, 统计)。不修改输入 graph。"""
    if drop_ratio < 0 or drop_ratio > 1:
        raise ValueError(f"drop_ratio 必须在 [0, 1]，得到 {drop_ratio}")

    nodes = list(graph.get("nodes") or [])
    links = list(graph.get("links") or [])
    success_ids = [_node_id(n) for n in nodes if is_successful_entity(n) and _node_id(n)]
    n_success = len(success_ids)
    # 四舍五入（0.5 向上），保证 50%×奇数个实体至少删 1 个
    n_drop = int(np.floor(n_success * drop_ratio + 0.5)) if drop_ratio > 0 else 0
    n_drop = min(max(n_drop, 0), n_success)

    drop_ids: set[str] = set()
    if n_drop > 0:
        chosen = rng.choice(success_ids, size=n_drop, replace=False)
        drop_ids = {str(x) for x in chosen}

    n_me_removed = 0
    n_ee_removed = 0
    kept_links = []
    for link in links:
        src, tgt = link.get("source"), link.get("target")
        if str(src) not in drop_ids and str(tgt) not in drop_ids:
            kept_links.append(link)
            continue
        if _is_entity_ref(src) and _is_entity_ref(tgt):
            n_ee_removed += 1
        else:
            n_me_removed += 1

    kept_nodes = [n for n in nodes if _node_id(n) not in drop_ids]
    new_graph = dict(graph)
    new_graph["nodes"] = kept_nodes
    new_graph["links"] = kept_links
    stats = {
        "n_success_entities": n_success,
        "n_dropped_entities": n_drop,
        "n_mention_entity_removed": n_me_removed,
        "n_entity_entity_removed": n_ee_removed,
    }
    return new_graph, stats


def perturb_sample(
    data: dict,
    graph_keys: Iterable[str],
    drop_ratio: float,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """浅拷贝样本并对各子图独立扰动。节点 dict 共享引用（不改节点内容）。"""
    new_data = dict(data)
    agg = {
        "n_success_entities": 0,
        "n_dropped_entities": 0,
        "n_mention_entity_removed": 0,
        "n_entity_entity_removed": 0,
        "per_graph": {},
    }
    for key in graph_keys:
        if key not in data or not isinstance(data[key], dict):
            continue
        if "graph_without_emb" not in data[key]:
            continue
        section = dict(data[key])
        new_graph, st = perturb_graph(section["graph_without_emb"], drop_ratio, rng)
        section["graph_without_emb"] = new_graph
        new_data[key] = section
        agg["per_graph"][key] = st
        for k in (
            "n_success_entities",
            "n_dropped_entities",
            "n_mention_entity_removed",
            "n_entity_entity_removed",
        ):
            agg[k] += st[k]
    return new_data, agg


def is_graph_data_empty(data: dict, graph_keys: list[str]) -> bool:
    """与 kernel.gawl.is_graph_file_empty 一致：任一子图完全空则跳过。"""
    for key in graph_keys:
        if key in data and "graph_without_emb" in data[key]:
            g = data[key]["graph_without_emb"] or {}
            if len(g.get("nodes") or []) == 0 and len(g.get("links") or []) == 0:
                return True
    return False


def unisum_eval_doc_ids(labels: dict[str, dict], exclude_human1: bool = True) -> set[str]:
    """与 evaluators_benchmark 相同的评价子集。"""
    keep = set()
    for doc_id, item in labels.items():
        if item.get("summary_success_state") != "success":
            continue
        if exclude_human1 and item.get("faithfulness_score") == 1:
            continue
        keep.add(str(doc_id))
    return keep


def load_unisum_labels(jsonl_path: Path) -> dict[str, dict]:
    """filename stem == merged jsonl 的 doc_id。"""
    labels = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            labels[str(item["doc_id"])] = item
    return labels


def unisum_pearson(
    scores: dict[str, float],
    labels: dict[str, dict],
    exclude_human1: bool = True,
) -> dict:
    from scipy.stats import pearsonr, spearmanr

    ys, xs = [], []
    n_skip_state = 0
    n_skip_score1 = 0
    n_missing = 0
    for doc_id, item in labels.items():
        if item.get("summary_success_state") != "success":
            n_skip_state += 1
            continue
        if exclude_human1 and item.get("faithfulness_score") == 1:
            n_skip_score1 += 1
            continue
        if doc_id not in scores or scores[doc_id] is None:
            n_missing += 1
            continue
        y = item.get("faithfulness_score")
        if y is None:
            n_missing += 1
            continue
        ys.append(float(y))
        xs.append(float(scores[doc_id]))

    out = {
        "n_used": len(xs),
        "n_skip_state": n_skip_state,
        "n_skip_score1": n_skip_score1,
        "n_missing_sim": n_missing,
        "pearson": None,
        "pearson_pvalue": None,
        "spearman": None,
        "spearman_pvalue": None,
    }
    if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
        pr, pp = pearsonr(ys, xs)
        sr, sp = spearmanr(ys, xs)
        out["pearson"] = float(pr)
        out["pearson_pvalue"] = float(pp)
        out["spearman"] = float(sr)
        out["spearman_pvalue"] = float(sp)
    return out


def summarize_defacto(rows: list[dict]) -> dict:
    """rows: 每个可评估样本的 {intrinsic, extrinsic, k_cand, k_human}。"""
    right_in = equ_in = all_in = 0
    right_ex = equ_ex = all_ex = 0
    for r in rows:
        k_h, k_c = r["k_human"], r["k_cand"]
        if r["intrinsic"] is True and r["extrinsic"] is False:
            all_in += 1
            if k_h > k_c:
                right_in += 1
            elif k_h == k_c:
                equ_in += 1
        elif r["intrinsic"] is False and r["extrinsic"] is True:
            all_ex += 1
            if k_h > k_c:
                right_ex += 1
            elif k_h == k_c:
                equ_ex += 1

    n_eval = all_in + all_ex
    n_correct = right_in + right_ex
    n_tie = equ_in + equ_ex
    n_wrong = n_eval - n_correct  # 含并列：未能严格把错误摘要排低
    acc = (n_correct / n_eval) if n_eval else None
    fnr = (n_wrong / n_eval) if n_eval else None
    return {
        "n_eval": n_eval,
        "n_correct": n_correct,
        "n_tie": n_tie,
        "accuracy": acc,
        "fn_rate": fnr,
        "right_intrinsic": right_in,
        "tie_intrinsic": equ_in,
        "n_intrinsic": all_in,
        "right_extrinsic": right_ex,
        "tie_extrinsic": equ_ex,
        "n_extrinsic": all_ex,
    }


def score_sample_gawl(
    data: dict,
    graph_keys: list[str],
    *,
    use_node_labels: bool,
    use_emb_labels: bool,
    use_edge_labels: int,
    wl_t: int,
    major_weights: dict,
    work_dir: Path,
) -> list[float]:
    """对一个样本的各子图跑 GAWL，返回与 graph_keys[1:] 对应的 article 相似度。"""
    from kernel.dfc2 import convert_json_graph
    from kernel.gawl import (
        compute_gawl_kernel,
        compute_gawl_kernel_v2,
        get_wl_labels,
        load_graph_data,
    )

    convert_json_graph(
        graph_keys=graph_keys,
        output_prefix="",
        undirected=False,
        use_emb_labels=use_emb_labels,
        output_dir=work_dir,
        data=data,
    )
    gs, _y, edge_type_weights = load_graph_data(
        str(work_dir) + "/",
        use_node_labels,
        use_emb_labels,
        use_edge_labels,
        major_weights=major_weights,
    )
    sims = []
    if len(graph_keys) >= 3:
        for gidx in range(1, min(3, len(gs))):
            compare = [gs[0], gs[gidx]]
            node_labels = get_wl_labels(compare, wl_t, use_node_labels)
            if use_edge_labels == 0:
                k = compute_gawl_kernel(compare, wl_t, node_labels)
            else:
                k = compute_gawl_kernel_v2(
                    compare, wl_t, node_labels, edge_type_weights, use_emb_labels
                )
            sims.append(float(k[0, 1]))
    else:
        compare = gs
        node_labels = get_wl_labels(compare, wl_t, use_node_labels)
        if use_edge_labels == 0:
            k = compute_gawl_kernel(compare, wl_t, node_labels)
        else:
            k = compute_gawl_kernel_v2(
                compare, wl_t, node_labels, edge_type_weights, use_emb_labels
            )
        sims.append(float(k[0, 1]))
    return sims


def iter_json_files(
    graph_root: Path,
    splits: list[str],
    limit: int | None,
    allowed_stems: set[str] | None = None,
) -> list[Path]:
    files: list[Path] = []
    for split in splits:
        split_dir = graph_root / split
        if not split_dir.is_dir():
            continue
        files.extend(sorted(split_dir.glob("*.json")))
    files = [p for p in files if p.is_file()]
    if allowed_stems is not None:
        files = [p for p in files if p.stem in allowed_stems]
    if limit is not None:
        files = files[: max(0, limit)]
    return files


def evaluate_once(
    *,
    dataset: str,
    graph_root: Path,
    splits: list[str],
    drop_ratio: float,
    seed: int,
    graph_keys: list[str],
    use_emb_labels: bool,
    wl_t: int,
    major_weights: dict,
    unisum_labels: dict[str, dict] | None,
    limit: int | None,
    tmp_root: Path,
    use_node_labels: bool = True,
    use_edge_labels: int = 2,
    unisum_all_files: bool = False,
) -> dict:
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    allowed = None
    if dataset.lower() != "defacto" and unisum_labels is not None and not unisum_all_files:
        allowed = unisum_eval_doc_ids(unisum_labels, exclude_human1=True)
    files = iter_json_files(graph_root, splits, limit, allowed_stems=allowed)
    defacto_rows: list[dict] = []
    unisum_scores: dict[str, float] = {}
    n_empty = 0
    n_error = 0
    drop_stats_sum = {
        "n_success_entities": 0,
        "n_dropped_entities": 0,
        "n_mention_entity_removed": 0,
        "n_entity_entity_removed": 0,
    }

    tmp_root.mkdir(parents=True, exist_ok=True)
    for path in tqdm(files, desc=f"{dataset} drop={drop_ratio:.0%} seed={seed}"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            n_error += 1
            continue
        if is_graph_data_empty(data, graph_keys):
            n_empty += 1
            continue

        perturbed, st = perturb_sample(data, graph_keys, drop_ratio, rng)
        for k in drop_stats_sum:
            drop_stats_sum[k] += st[k]

        work_dir = tmp_root / f"{path.stem}__pid{os.getpid()}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            sims = score_sample_gawl(
                perturbed,
                graph_keys,
                use_node_labels=use_node_labels,
                use_emb_labels=use_emb_labels,
                use_edge_labels=use_edge_labels,
                wl_t=wl_t,
                major_weights=major_weights,
                work_dir=work_dir,
            )
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            print(f"[warn] {path}: {exc}")
            continue
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        if dataset.lower() == "defacto":
            if len(sims) < 2:
                n_error += 1
                continue
            defacto_rows.append(
                {
                    "file": path.name,
                    "intrinsic": data.get("intrinsic_error"),
                    "extrinsic": data.get("extrinsic_error"),
                    "k_cand": sims[0],
                    "k_human": sims[1],
                }
            )
        else:
            unisum_scores[path.stem] = sims[0]

    result = {
        "dataset": dataset,
        "drop_ratio": drop_ratio,
        "seed": seed,
        "n_files": len(files),
        "n_empty": n_empty,
        "n_error": n_error,
        "drop_stats": drop_stats_sum,
        "defacto": None,
        "unisum": None,
    }
    if dataset.lower() == "defacto":
        result["defacto"] = summarize_defacto(defacto_rows)
    else:
        if unisum_labels is None:
            raise ValueError("UniSumEval 需要 labels jsonl")
        result["unisum"] = unisum_pearson(unisum_scores, unisum_labels, exclude_human1=True)
    return result


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None, None
    arr = np.array(nums, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate_repeats(runs: list[dict]) -> dict:
    """同一 drop_ratio 下多次 repeat 的均值±标准差。"""
    if not runs:
        return {}
    dataset = runs[0]["dataset"]
    ratio = runs[0]["drop_ratio"]
    out: dict[str, Any] = {
        "dataset": dataset,
        "drop_ratio": ratio,
        "n_repeats": len(runs),
        "seeds": [r["seed"] for r in runs],
    }
    if dataset.lower() == "defacto":
        keys = ["n_correct", "accuracy", "fn_rate", "n_eval", "n_tie"]
        for key in keys:
            vals = [r["defacto"][key] if r.get("defacto") else None for r in runs]
            mu, sd = mean_std(vals)
            out[f"{key}_mean"] = mu
            out[f"{key}_std"] = sd
        # 分项
        for key in ("right_intrinsic", "right_extrinsic", "n_intrinsic", "n_extrinsic"):
            vals = [r["defacto"][key] if r.get("defacto") else None for r in runs]
            mu, sd = mean_std(vals)
            out[f"{key}_mean"] = mu
            out[f"{key}_std"] = sd
    else:
        for key in ("pearson", "spearman", "n_used"):
            vals = [r["unisum"][key] if r.get("unisum") else None for r in runs]
            mu, sd = mean_std(vals)
            out[f"{key}_mean"] = mu
            out[f"{key}_std"] = sd
    drop_keys = (
        "n_success_entities",
        "n_dropped_entities",
        "n_mention_entity_removed",
        "n_entity_entity_removed",
    )
    for key in drop_keys:
        vals = [r["drop_stats"][key] for r in runs]
        mu, sd = mean_std(vals)
        out[f"{key}_mean"] = mu
        out[f"{key}_std"] = sd
    return out
