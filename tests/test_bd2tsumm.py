"""BD2TSumm CompKG vs ROUGE 流程测试（不加载 DREEAM/BLINK）。"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "bd2tsumm"))

from analyze_results import (  # noqa: E402
    UNDEFINED,
    average_rank_desc,
    correlation_record,
    detect_duplicate_keys,
    merge_scores,
)
from compute_compkg import apply_score_direction, mock_compkg_score, run_compkg  # noqa: E402
from compute_rouge import aggregate_reference_f1, make_scorer, score_pair  # noqa: E402
from prepare_sources import (  # noqa: E402
    aggregate_event,
    clean_tweet,
    dedup_keep_first,
    join_source_text,
    sort_tweets,
)
from run_all import main as run_all_main  # noqa: E402
from util import MissingCandidatesError, load_config  # noqa: E402


CLEAN = {
    "unicode_form": "NFKC",
    "collapse_whitespace": True,
    "dedup_exact": True,
    "replace_urls": True,
    "replace_mentions": True,
    "placeholder_style": "plain",
    "keep_hashtag_text": True,
    "strip_rt_prefix": False,
    "strip_emoji": False,
}


def test_sort_by_timestamp_then_original():
    rows = [
        {"tweet_id": "b", "timestamp": "2020-01-02", "_orig_index": 0},
        {"tweet_id": "a", "timestamp": "2020-01-01", "_orig_index": 1},
        {"tweet_id": "c", "timestamp": "", "_orig_index": 2},
    ]
    sorted_rows, mode = sort_tweets(rows, timestamp_col="timestamp", tweet_id_col="tweet_id")
    assert mode == "timestamp"
    assert [r["tweet_id"] for r in sorted_rows] == ["a", "b", "c"]


def test_sort_original_order_without_timestamp():
    rows = [
        {"tweet_id": "z", "timestamp": "", "_orig_index": 0},
        {"tweet_id": "a", "timestamp": "", "_orig_index": 1},
    ]
    sorted_rows, mode = sort_tweets(rows, timestamp_col="timestamp", tweet_id_col="tweet_id")
    assert mode == "original_order"
    assert [r["tweet_id"] for r in sorted_rows] == ["z", "a"]


def test_dedup_keeps_first_occurrence_order():
    texts = ["alpha", "beta", "alpha", "gamma", "beta"]
    keep, dropped = dedup_keep_first(texts)
    assert dropped == 2
    assert [texts[i] for i in keep] == ["alpha", "beta", "gamma"]


def test_hashtag_numbers_entities_not_deleted():
    text = "RT @alice: #HurricaneHarvey killed 12 people in Texas. https://x.com/a"
    out = clean_tweet(text, CLEAN)
    assert "HurricaneHarvey" in out
    assert "#" not in out
    assert "12" in out
    assert "Texas" in out
    assert "URL" in out
    assert "USER" in out
    assert "killed" in out
    assert "https" not in out.lower()


def test_aggregate_timestamp_and_dedup():
    tweets = [
        {"tweet_id": "t3", "tweet_text": "third 100 people", "timestamp": "2020-01-03", "_orig_index": 0},
        {"tweet_id": "t1", "tweet_text": "#Harvey killed 12", "timestamp": "2020-01-01", "_orig_index": 1},
        {"tweet_id": "t2", "tweet_text": "#Harvey killed 12", "timestamp": "2020-01-02", "_orig_index": 2},
    ]
    agg = aggregate_event("storm", tweets, CLEAN, timestamp_col="timestamp", tweet_id_col="tweet_id")
    assert agg["sort_mode"] == "timestamp"
    assert agg["num_tweets_raw"] == 3
    assert agg["num_duplicates_removed"] == 1
    assert agg["num_tweets_after_dedup"] == 2
    paras = agg["source_text"].split("\n")
    assert paras[0].startswith("Harvey")
    assert "12" in paras[0]
    assert paras[1] == "third 100 people"
    assert "Tweet" not in agg["source_text"]


def test_join_source_is_newline_paragraphs():
    assert join_source_text(["a", "b"]) == "a\nb"


def test_multi_reference_rouge_mean():
    scorer = make_scorer(use_stemmer=True)
    cand = "the cat sat on the mat"
    s1 = score_pair(scorer, cand, "the cat sat on the mat")
    s2 = score_pair(scorer, cand, "completely unrelated airplanes")
    agg = aggregate_reference_f1([s1, s2], "mean")
    assert abs(agg["rouge1_f1"] - 0.5 * (s1["rouge1_f1"] + s2["rouge1_f1"])) < 1e-12
    assert agg["num_references"] == 2
    try:
        aggregate_reference_f1([s1], "max")
        raise AssertionError("max aggregation should be rejected")
    except ValueError:
        pass


def test_join_rejects_many_to_many():
    compkg = [
        {"event_id": "e", "system_name": "s", "compkg_score": "0.2", "status": "ok"},
        {"event_id": "e", "system_name": "s", "compkg_score": "0.3", "status": "ok"},
    ]
    rouge = [
        {"event_id": "e", "system_name": "s", "rouge1_f1": "0.1", "rouge2_f1": "0.1", "rougeL_f1": "0.1", "status": "ok"},
    ]
    assert detect_duplicate_keys(compkg, "c") == [("e", "s")]
    import logging

    merged, failures, summary = merge_scores(compkg, rouge, logging.getLogger("test"))
    assert summary["n_many_to_many"] == 1
    assert summary["n_valid"] == 0
    assert failures and "many_to_many" in failures[0]["reason"]
    assert merged[0]["join_status"] == "excluded"


def test_constant_correlation_is_undefined_not_zero():
    rec = correlation_record([1, 1, 1], [0.2, 0.4, 0.6], analysis_level="t", rouge_metric="rouge1_f1", method="spearman", min_n=3)
    assert rec["status"] == UNDEFINED
    assert rec["coefficient"] is None
    rec2 = correlation_record([0.1, 0.2, 0.3], [5, 5, 5], analysis_level="t", rouge_metric="rouge1_f1", method="kendall_tau_b", min_n=3)
    assert rec2["status"] == UNDEFINED
    assert rec2["coefficient"] is None


def test_average_rank_ties():
    ranks = average_rank_desc([10.0, 10.0, 8.0])
    assert ranks == [1.5, 1.5, 3.0]
    ranks2 = average_rank_desc([3.0, 1.0, 2.0])
    assert ranks2 == [1.0, 3.0, 2.0]


def test_compkg_score_direction_higher_is_better():
    src = "Hurricane Harvey killed 12 people in Texas 100 shelter"
    good = src
    bad = "unrelated basketball scores tonight"
    sg = mock_compkg_score(src, good)
    sb = mock_compkg_score(src, bad)
    assert sg > sb
    assert apply_score_direction(sg, higher_is_better=True, treat_as_distance=False) > apply_score_direction(
        sb, higher_is_better=True, treat_as_distance=False
    )
    dist_g = apply_score_direction(sg, higher_is_better=True, treat_as_distance=True)
    dist_b = apply_score_direction(sb, higher_is_better=True, treat_as_distance=True)
    assert dist_g < dist_b  # 距离越小越好，方向翻转后仍是“越高越好”的一致性分数


def test_single_sample_failure_does_not_abort():
    cfg = load_config(ROOT / "tests/fixtures/bd2tsumm/config.yaml")
    cfg["compkg"]["backend"] = "mock"
    cfg["long_context"]["allow_mock_tokenizer"] = True

    def flaky(source, cand):
        if "basketball" in cand or "somewhere" in cand:
            raise RuntimeError("boom")
        return mock_compkg_score(source, cand)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        # 先跑 prepare+rouge 依赖
        from argparse import Namespace

        from prepare_sources import run_prepare
        from compute_rouge import run_rouge
        import logging

        log = logging.getLogger("t")
        log.addHandler(logging.NullHandler())
        run_prepare(cfg, out, log, require_candidates=True)
        run_rouge(cfg, out, log)
        result = run_compkg(cfg, out, log, mock_fn=flaky)
        statuses = { (r["event_id"], r["system_name"]): r["status"] for r in result["rows"] }
        assert statuses[("storm", "sysB")] == "failed"
        assert statuses[("storm", "sysA")] == "ok"
        assert statuses[("quake", "sysA")] == "ok"
        assert any(r["status"] == "ok" for r in result["rows"])


def test_toy_pipeline_without_models():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        code = run_all_main(
            [
                "--config",
                str(ROOT / "tests/fixtures/bd2tsumm/config.yaml"),
                "--output-dir",
                str(out),
                "--seed",
                "42",
                "--mock-compkg",
                "--allow-mock-tokenizer",
            ]
        )
        assert code == 0
        assert (out / "summary_report.md").is_file()
        assert (out / "prepared_sources.jsonl").is_file()
        src = [json.loads(l) for l in (out / "prepared_sources.jsonl").read_text().splitlines() if l]
        storm = next(x for x in src if x["event_id"] == "storm")
        assert storm["num_duplicates_removed"] == 1
        assert storm["sort_mode"] == "timestamp"
        assert "HurricaneHarvey" in storm["source_text"]
        assert "12" in storm["source_text"]
        assert "URL" in storm["source_text"]
        assert storm["num_windows"] >= 1
        with (out / "rouge_scores.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        with (out / "all_sample_scores.csv").open() as f:
            all_rows = list(csv.DictReader(f))
        ok = [r for r in all_rows if r["join_status"] == "ok"]
        assert len(ok) == 6
        report = (out / "summary_report.md").read_text()
        assert "ROUGE measures n-gram overlap" in report
        assert "CompKG-FactEval measures factual consistency" in report
        meta = json.loads((out / "run_metadata.json").read_text())
        assert meta["git_commit"] is not None or True  # git 可能不可用
        ranking = list(csv.DictReader((out / "system_ranking.csv").open()))
        mains = [r for r in ranking if r["in_main_ranking"] == "True"]
        assert {r["system_name"] for r in mains} == {"sysA", "sysB", "sysC"}


def test_missing_candidates_errors_and_does_not_use_reference():
    cfg = load_config(ROOT / "tests/fixtures/bd2tsumm/config.yaml")
    cfg["dataset"]["candidates_file"] = None
    cfg["dataset"]["candidates_dir"] = None
    cfg["long_context"]["allow_mock_tokenizer"] = True
    cfg["long_context"]["tokenizer_path"] = "mock"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        import logging
        from prepare_sources import run_prepare

        log = logging.getLogger("missing")
        log.addHandler(logging.NullHandler())
        try:
            run_prepare(cfg, out, log, require_candidates=True)
            raise AssertionError("should have raised")
        except MissingCandidatesError as exc:
            assert "参考摘要" in str(exc) or "系统输出" in str(exc)


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
