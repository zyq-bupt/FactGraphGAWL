"""复用 eval.el_perturbation.score_sample_gawl（S²-K / GAWL-PT）。"""

from __future__ import annotations

import shutil
from pathlib import Path

from eval.el_perturbation import score_sample_gawl
from kernel.gawl import default_edge_major_type_weights


def score_article_candidate(
    article_graph: dict,
    candidate_graph: dict,
    *,
    work_dir: Path,
    wTT: float,
    wTM: float,
    wME: float,
    wEE: float,
    wl_t: int,
    use_emb_labels: bool,
    use_node_labels: bool,
    use_edge_labels: int,
) -> float:
    data = {
        "article": {"graph_without_emb": article_graph},
        "candidate": {"graph_without_emb": candidate_graph},
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        sims = score_sample_gawl(
            data,
            ["article", "candidate"],
            use_node_labels=use_node_labels,
            use_emb_labels=use_emb_labels,
            use_edge_labels=use_edge_labels,
            wl_t=wl_t,
            major_weights=default_edge_major_type_weights(wTT, wTM, wME, wEE),
            work_dir=work_dir,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    if not sims:
        raise RuntimeError("GAWL 未返回分数")
    return float(sims[0])
