"""Unit tests for entity-linking perturbation (no GAWL / no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.el_perturbation import (
    is_successful_entity,
    perturb_graph,
    perturb_sample,
    summarize_defacto,
)


def _toy_graph():
    return {
        "nodes": [
            {"node_id": "Token_0", "label": "Token", "name": "Celtic"},
            {"node_id": "Mention_0", "label": "Mention", "name": "Celtic"},
            {"node_id": "Mention_1", "label": "Mention", "name": "Scotland"},
            {
                "node_id": "Entity_0",
                "label": "Entity",
                "name": 1,
                "wikidata_id": "Q19593",
            },
            {
                "node_id": "Entity_1",
                "label": "Entity",
                "name": 2,
                "wikidata_id": "Q22",
            },
            {
                "node_id": "Entity_2",
                "label": "Entity",
                "name": 3,
                "wikidata_id": "",
            },
        ],
        "links": [
            {"relationship": "RELATED_TO", "source": "Mention_0", "target": "Token_0"},
            {"relationship": "BELONGS_TO", "source": "Mention_0", "target": "Entity_0"},
            {"relationship": "BELONGS_TO", "source": "Mention_1", "target": "Entity_1"},
            {"relationship": "P17", "source": "Entity_0", "target": "Entity_1"},
            {"relationship": "nsubj", "source": "Token_0", "target": "Token_0"},
        ],
    }


def test_successful_entity_requires_wikidata():
    assert is_successful_entity({"label": "Entity", "node_id": "Entity_0", "wikidata_id": "Q1"})
    assert not is_successful_entity({"label": "Entity", "node_id": "Entity_2", "wikidata_id": ""})
    assert not is_successful_entity({"label": "Mention", "node_id": "Mention_0", "wikidata_id": "Q1"})


def test_drop_zero_keeps_graph():
    g = _toy_graph()
    new_g, st = perturb_graph(g, 0.0, np.random.default_rng(0))
    assert st["n_dropped_entities"] == 0
    assert len(new_g["nodes"]) == len(g["nodes"])
    assert len(new_g["links"]) == len(g["links"])


def test_drop_all_successful_keeps_mentions_and_failed_el():
    g = _toy_graph()
    new_g, st = perturb_graph(g, 1.0, np.random.default_rng(0))
    assert st["n_success_entities"] == 2
    assert st["n_dropped_entities"] == 2
    ids = {n["node_id"] for n in new_g["nodes"]}
    assert "Entity_0" not in ids and "Entity_1" not in ids
    assert "Entity_2" in ids  # 失败 EL 不进删除池
    assert "Mention_0" in ids and "Token_0" in ids
    rels = {(e["relationship"], e["source"], e["target"]) for e in new_g["links"]}
    assert ("BELONGS_TO", "Mention_0", "Entity_0") not in rels
    assert ("P17", "Entity_0", "Entity_1") not in rels
    assert ("RELATED_TO", "Mention_0", "Token_0") in rels
    assert ("nsubj", "Token_0", "Token_0") in rels
    assert st["n_mention_entity_removed"] == 2
    assert st["n_entity_entity_removed"] == 1


def test_drop_half_is_reproducible():
    g = _toy_graph()
    a, sa = perturb_graph(g, 0.5, np.random.default_rng(7))
    b, sb = perturb_graph(g, 0.5, np.random.default_rng(7))
    assert sa == sb
    assert {n["node_id"] for n in a["nodes"]} == {n["node_id"] for n in b["nodes"]}


def test_perturb_sample_independent_subgraphs():
    data = {
        "article": {"graph_without_emb": _toy_graph()},
        "candidate": {"graph_without_emb": _toy_graph()},
    }
    new_data, st = perturb_sample(data, ["article", "candidate"], 1.0, np.random.default_rng(1))
    assert st["n_dropped_entities"] == 4
    art_ids = {n["node_id"] for n in new_data["article"]["graph_without_emb"]["nodes"]}
    assert "Entity_0" not in art_ids
    # 原文图未被原地修改
    orig_ids = {n["node_id"] for n in data["article"]["graph_without_emb"]["nodes"]}
    assert "Entity_0" in orig_ids


def test_summarize_defacto_fnr():
    rows = [
        {"intrinsic": True, "extrinsic": False, "k_cand": 1.0, "k_human": 2.0},
        {"intrinsic": True, "extrinsic": False, "k_cand": 3.0, "k_human": 2.0},
        {"intrinsic": False, "extrinsic": True, "k_cand": 1.0, "k_human": 1.0},
        {"intrinsic": True, "extrinsic": True, "k_cand": 0.0, "k_human": 9.0},  # 跳过
    ]
    s = summarize_defacto(rows)
    assert s["n_eval"] == 3
    assert s["n_correct"] == 1
    assert s["n_tie"] == 1
    assert abs(s["fn_rate"] - 2 / 3) < 1e-9


if __name__ == "__main__":
    test_successful_entity_requires_wikidata()
    test_drop_zero_keeps_graph()
    test_drop_all_successful_keeps_mentions_and_failed_el()
    test_drop_half_is_reproducible()
    test_perturb_sample_independent_subgraphs()
    test_summarize_defacto_fnr()
    print("ok")
