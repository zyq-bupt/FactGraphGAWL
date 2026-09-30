"""滑动窗口 FK-Graph：单元测试（mock tokenizer / mock 图，不下载 BLINK/DREEAM）。"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from long_context.cache import cache_path
from long_context.config import LongContextConfig
from long_context.coverage import token_coverage_ratio
from long_context.graphs import build_document_graph, nx_to_link_data
from long_context.merge import graph_counts, merge_window_graphs
from long_context.offsets import align_sents_to_text
from long_context.report import write_outputs
from long_context.stats import paired_bootstrap_delta_r, pearson_with_p
from long_context.tokenizer_util import MockTokenizer, count_tokens, effective_max_length
from long_context.windows import Window, split_windows, validate_windows


def _tok(max_len=1024, unit="word"):
    return MockTokenizer(model_max_length=max_len, unit=unit)


def test_short_doc_single_window():
    tok = _tok(16)
    text = "Hello world this is short."
    windows = split_windows(text, tok, max_model_tokens=16, overlap_tokens=4)
    validate_windows(windows, text, tok, 16)
    assert len(windows) == 1
    assert windows[0].global_char_start == 0
    assert windows[0].global_char_end == len(text)
    assert windows[0].model_token_count <= 16


def test_long_doc_full_coverage():
    tok = _tok(8)
    sents = [f"Sentence number {i} is here." for i in range(12)]
    text = " ".join(sents)
    windows = split_windows(text, tok, max_model_tokens=8, overlap_tokens=3)
    validate_windows(windows, text, tok, 8)
    assert len(windows) > 1
    assert windows[-1].global_char_end == len(text) or all(
        ch.isspace() or any(w.global_char_start <= i < w.global_char_end for w in windows)
        for i, ch in enumerate(text)
    )


def test_each_window_respects_max_len():
    tok = _tok(10)
    text = " ".join(["alpha"] * 40)
    windows = split_windows(text, tok, max_model_tokens=10, overlap_tokens=3, sentence_boundary=False)
    for w in windows:
        assert count_tokens(tok, w.text, True) <= 10
        assert count_tokens(tok, w.text, True) == w.model_token_count


def test_adjacent_windows_overlap_and_progress():
    tok = _tok(10)
    text = " ".join([f"word{i}" for i in range(30)])
    windows = split_windows(text, tok, max_model_tokens=10, overlap_tokens=3, sentence_boundary=False)
    assert len(windows) >= 2
    for a, b in zip(windows, windows[1:]):
        assert b.global_char_start > a.global_char_start
        assert b.global_char_start < a.global_char_end  # overlap


def test_overlong_sentence_no_deadlock_or_silent_truncation():
    tok = _tok(8, unit="char")
    text = "x" * 50
    windows = split_windows(text, tok, max_model_tokens=8, overlap_tokens=2, sentence_boundary=True)
    validate_windows(windows, text, tok, 8)
    assert len(windows) > 1
    joined_nonspace = set()
    for w in windows:
        assert count_tokens(tok, w.text, True) <= 8
        for i, ch in enumerate(w.text):
            if not ch.isspace():
                joined_nonspace.add(w.global_char_start + i)
    assert all((i in joined_nonspace) for i, ch in enumerate(text) if not ch.isspace())


def _toy_token_graph(spans, window_id=0):
    G = nx.DiGraph()
    prev = None
    for i, (s, e, name) in enumerate(spans):
        nid = f"Token_{i}"
        G.add_node(
            nid,
            node_id=nid,
            label="Token",
            name=name,
            global_char_start=s,
            global_char_end=e,
            window_id=window_id,
        )
        if prev is not None:
            G.add_edge(prev, nid, relationship="nsubj")
        prev = nid
    return G


def test_overlap_tokens_dedup_by_global_offset():
    g0 = _toy_token_graph([(0, 4, "This"), (5, 7, "is"), (8, 12, "text")], 0)
    g1 = _toy_token_graph([(5, 7, "is"), (8, 12, "text"), (13, 17, "here")], 1)
    merged, stats = merge_window_graphs([g0, g1], merge_entities_by_kb_id=True)
    counts = graph_counts(merged)
    assert counts["num_token_nodes"] == 4
    assert stats["token_span_conflicts"] == 0


def test_overlap_mentions_dedup():
    def add_mention(G, i, s, e, name, ner="ORG"):
        nid = f"Mention_{i}"
        G.add_node(nid, node_id=nid, label="Mention", name=name, type=ner,
                   global_char_start=s, global_char_end=e)
        return nid

    g0 = nx.DiGraph()
    add_mention(g0, 0, 0, 3, "UN")
    g1 = nx.DiGraph()
    add_mention(g1, 0, 0, 3, "UN")
    add_mention(g1, 1, 10, 15, "NATO")
    merged, _ = merge_window_graphs([g0, g1])
    assert graph_counts(merged)["num_mention_nodes"] == 2


def test_entity_merge_same_kb_id():
    def ent(G, i, kb, name):
        nid = f"Entity_{i}"
        G.add_node(nid, node_id=nid, label="Entity", name=name, wikidata_id=kb)
        return nid

    g0 = nx.DiGraph()
    e0 = ent(g0, 0, "Q30", "USA")
    g0.add_node("Mention_0", node_id="Mention_0", label="Mention", name="US",
                type="GPE", global_char_start=0, global_char_end=2)
    g0.add_edge("Mention_0", e0, relationship="BELONGS_TO")
    g1 = nx.DiGraph()
    e1 = ent(g1, 0, "Q30", "United States")
    g1.add_node("Mention_0", node_id="Mention_0", label="Mention", name="America",
                type="GPE", global_char_start=20, global_char_end=27)
    g1.add_edge("Mention_0", e1, relationship="BELONGS_TO")
    merged, _ = merge_window_graphs([g0, g1], merge_entities_by_kb_id=True)
    c = graph_counts(merged)
    assert c["num_entity_nodes"] == 1
    assert c["num_mention_nodes"] == 2
    assert c["num_E_me"] == 2


def test_entities_not_merged_by_surface_if_kb_differs():
    g0 = nx.DiGraph()
    g0.add_node("Entity_0", node_id="Entity_0", label="Entity", name="Apple", wikidata_id="Q312")
    g1 = nx.DiGraph()
    g1.add_node("Entity_0", node_id="Entity_0", label="Entity", name="Apple", wikidata_id="Q89")
    merged, _ = merge_window_graphs([g0, g1], merge_entities_by_kb_id=True)
    assert graph_counts(merged)["num_entity_nodes"] == 2


def test_edge_dedup_and_keep_distinct():
    g0 = nx.DiGraph()
    g0.add_node("Entity_0", node_id="Entity_0", label="Entity", wikidata_id="Q1", name=1)
    g0.add_node("Entity_1", node_id="Entity_1", label="Entity", wikidata_id="Q2", name=2)
    g0.add_edge("Entity_0", "Entity_1", relationship="P17", score=0.4)
    g1 = nx.DiGraph()
    g1.add_node("Entity_0", node_id="Entity_0", label="Entity", wikidata_id="Q1", name=1)
    g1.add_node("Entity_1", node_id="Entity_1", label="Entity", wikidata_id="Q2", name=2)
    g1.add_node("Entity_2", node_id="Entity_2", label="Entity", wikidata_id="Q3", name=3)
    g1.add_edge("Entity_0", "Entity_1", relationship="P17", score=0.9)
    g1.add_edge("Entity_1", "Entity_2", relationship="P27", score=0.5)
    merged, _ = merge_window_graphs([g0, g1])
    c = graph_counts(merged)
    assert c["num_entity_nodes"] == 3
    assert c["num_E_ee"] == 2
    e0 = [n for n, d in merged.nodes(data=True) if d.get("wikidata_id") == "Q1"][0]
    e1 = [n for n, d in merged.nodes(data=True) if d.get("wikidata_id") == "Q2"][0]
    assert merged.edges[e0, e1]["score"] == 0.9
    assert merged.edges[e0, e1]["count"] == 2


def test_coverage_never_exceeds_one():
    tok = _tok(8)
    text = " ".join(["token"] * 25)
    windows = split_windows(text, tok, max_model_tokens=8, overlap_tokens=3, sentence_boundary=False)
    cov = token_coverage_ratio(text, windows, tok)
    assert cov["token_coverage_ratio"] <= 1.0
    assert cov["token_coverage_ratio"] > 0.99


def test_cache_key_changes_with_window_params():
    a = LongContextConfig(overlap_tokens=256, max_model_tokens=1024, tokenizer_path="/a")
    b = LongContextConfig(overlap_tokens=128, max_model_tokens=1024, tokenizer_path="/a")
    c = LongContextConfig(overlap_tokens=256, max_model_tokens=1024, tokenizer_path="/b")
    assert a.config_hash() != b.config_hash()
    assert a.config_hash() != c.config_hash()
    p1 = cache_path(Path("/tmp"), "67", "prefix_1024", a.config_hash())
    p2 = cache_path(Path("/tmp"), "67", "sliding_window", a.config_hash())
    assert p1 != p2


def _mini_dreeam(text: str, sents: list[list[str]], mentions=None, trees=None, vertex=None, rels=None):
    if trees is None:
        trees = []
        idx = 0
        for sent in sents:
            for j, w in enumerate(sent):
                trees.append({"head": w, "h_idx": idx, "child": w, "t_idx": idx, "r": "ROOT" if j == 0 else "punct"})
                if j > 0:
                    trees.append({"head": sent[0], "h_idx": idx - j, "child": w, "t_idx": idx, "r": "compound"})
                idx += 1
    return {
        "text": text,
        "sents": sents,
        "mentions": mentions or [],
        "parse_trees": trees,
        "vertexSet": vertex or [],
        "ent_relation": rels or [],
    }


def test_short_text_prefix_matches_sliding_graph_stats():
    tok = _tok(64)
    text = "Alice met Bob in Paris."
    sents = [["Alice", "met", "Bob", "in", "Paris", "."]]
    mentions = [
        {"mention_id": 0, "name": "Alice", "type": "PERSON", "token_index": [0]},
        {"mention_id": 1, "name": "Bob", "type": "PERSON", "token_index": [2]},
        {"mention_id": 2, "name": "Paris", "type": "GPE", "token_index": [4]},
    ]
    vertex = [
        [{"mention_id": 0, "name": "Alice", "wikipedia_id": 1, "wikidata_id": "Q1", "token_index": [0]}],
        [{"mention_id": 1, "name": "Bob", "wikipedia_id": 2, "wikidata_id": "Q2", "token_index": [2]}],
        [{"mention_id": 2, "name": "Paris", "wikipedia_id": 3, "wikidata_id": "Q90", "token_index": [4]}],
    ]
    rels = [{"h_idx": 0, "t_idx": 2, "r": "P19", "score": 0.8}]
    section = _mini_dreeam(text, sents, mentions, vertex=vertex, rels=rels)
    assert count_tokens(tok, text, True) <= 64
    base = build_document_graph(section, tok, mode="prefix_1024", max_model_tokens=64, overlap_tokens=8)
    slid = build_document_graph(section, tok, mode="sliding_window", max_model_tokens=64, overlap_tokens=8)
    assert base["num_windows"] == 1
    assert slid["num_windows"] == 1
    for k in ("num_token_nodes", "num_mention_nodes", "num_entity_nodes", "num_E_tt", "num_E_tm", "num_E_me", "num_E_ee"):
        assert base[k] == slid[k], (k, base[k], slid[k])


def test_paired_stats_only_use_joint_success():
    rows = [
        {"status": "ok", "human_score": 0.2, "baseline_score": 0.1, "sliding_score": 0.2,
         "baseline_token_coverage_ratio": 0.4, "sliding_token_coverage_ratio": 0.9,
         "baseline_num_entity_nodes": 1, "sliding_num_entity_nodes": 2,
         "baseline_num_E_ee": 0, "sliding_num_E_ee": 0,
         "baseline_graph_build_seconds": 1, "sliding_graph_build_seconds": 2,
         "baseline_total_seconds": 1, "sliding_total_seconds": 2,
         "sample_id": "1", "domain": "GovReport", "error_type": ""},
        {"status": "failed", "human_score": 0.9, "baseline_score": 0.8, "sliding_score": None,
         "sample_id": "2", "domain": "GovReport", "error_type": "score_sliding_window:RuntimeError"},
        {"status": "ok", "human_score": 0.5, "baseline_score": 0.4, "sliding_score": 0.3,
         "baseline_token_coverage_ratio": 0.5, "sliding_token_coverage_ratio": 1.0,
         "baseline_num_entity_nodes": 2, "sliding_num_entity_nodes": 3,
         "baseline_num_E_ee": 0, "sliding_num_E_ee": 0,
         "baseline_graph_build_seconds": 1, "sliding_graph_build_seconds": 2,
         "baseline_total_seconds": 1, "sliding_total_seconds": 2,
         "sample_id": "3", "domain": "GovReport", "error_type": ""},
        {"status": "ok", "human_score": 0.8, "baseline_score": 0.6, "sliding_score": 0.7,
         "baseline_token_coverage_ratio": 0.3, "sliding_token_coverage_ratio": 0.95,
         "baseline_num_entity_nodes": 1, "sliding_num_entity_nodes": 4,
         "baseline_num_E_ee": 0, "sliding_num_E_ee": 1,
         "baseline_graph_build_seconds": 1, "sliding_graph_build_seconds": 3,
         "baseline_total_seconds": 1, "sliding_total_seconds": 3,
         "sample_id": "4", "domain": "GovReport", "error_type": ""},
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        agg = write_outputs(
            Path(td),
            cfg={"bootstrap_n": 50, "seed": 20260818, "domain": "GovReport", "max_model_tokens": 1024,
                 "overlap_tokens": 256, "sentence_boundary": True},
            implementation_notes="test",
            sample_rows=rows,
            n_original=10,
            failures=[rows[1]],
            tokenizer_note="mock",
        )
    assert agg["n_paired"] == 3
    assert agg["n_original"] == 10
    r, p = pearson_with_p([0.2, 0.5, 0.8], [0.1, 0.4, 0.6])
    assert r is not None
    assert abs(agg["baseline"]["pearson_r"] - r) < 1e-9


def test_effective_max_length_caps_placeholder():
    tok = MockTokenizer(model_max_length=10**18)
    assert effective_max_length(tok, 1024) == 1024


def test_no_cross_window_relation_invention():
    g0 = nx.DiGraph()
    g0.add_node("Entity_0", node_id="Entity_0", label="Entity", wikidata_id="Q1", name=1)
    g1 = nx.DiGraph()
    g1.add_node("Entity_0", node_id="Entity_0", label="Entity", wikidata_id="Q2", name=2)
    merged, _ = merge_window_graphs([g0, g1], infer_cross_window_relations=False)
    assert graph_counts(merged)["num_E_ee"] == 0


def test_align_sents_uses_pointer_not_global_find():
    text = "Paris is nice. Paris is old."
    sents = [["Paris", "is", "nice", "."], ["Paris", "is", "old", "."]]
    tokens, spans = align_sents_to_text(text, sents)
    paris = [t for t in tokens if t.surface == "Paris"]
    assert paris[0].char_start == 0
    assert paris[1].char_start == text.rfind("Paris")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, type(exc).__name__, exc)
    print(f"{len(tests)-failed}/{len(tests)} passed")
    raise SystemExit(failed)
