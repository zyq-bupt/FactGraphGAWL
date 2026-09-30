"""Unit tests for Direct-LLM parsers and KP-LLM helpers (no API / no T5 required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.kp_llm.aggregator import aggregate_labels
from baselines.kp_llm.aligner import KeyphraseAligner, split_sentences
from baselines.kp_llm.extractor import normalize_keyphrases
from baselines.llm_direct.evaluator import parse_star, parse_zs


def test_parse_zs_yes_no():
    assert parse_zs("Yes")[1] == 1
    assert parse_zs("Answer: no")[1] == 0
    assert parse_zs("maybe")[0] is None


def test_parse_star():
    score, binary = parse_star("4 stars")
    assert score == 0.8
    assert binary == 1
    score, binary = parse_star("Stars: 2")
    assert score == 0.4
    assert binary == 0


def test_normalize_keyphrases():
    raw = "New York | new york | New York City officials | the | !!! | Ten people"
    out = normalize_keyphrases(raw, max_keyphrases=8)
    assert "New York" not in out or "New York City officials" in out
    assert "the" not in [x.lower() for x in out]
    assert any("ten people" == x.lower() for x in out)


def test_align_exact_and_overlap():
    summary = "Ten people were injured in the earthquake. Officials arrived later."
    aligner = KeyphraseAligner()
    aligned = aligner.align(summary, ["Ten people", "officials"])
    assert aligned[0]["alignment_type"] in {"exact", "casefold"}
    assert "injured" in aligned[0]["summary_sentence"]
    assert aligned[1]["sentence_index"] == 1


def test_aggregate_support_rate():
    out = aggregate_labels(["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFORMATION", "SUPPORTED"])
    assert out["score"] == 0.5
    assert out["supported_count"] == 2
    assert out["num_keyphrases"] == 4


def test_split_sentences_nonempty():
    sents = split_sentences("A happens. B happens!")
    assert len(sents) >= 2
