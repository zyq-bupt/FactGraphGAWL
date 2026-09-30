#!/usr/bin/env python3
"""Run KP-LLM baseline on a JSONL dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from baselines.kp_llm import build_default_kp_llm_scorer
from baselines.llm_client import LLMClient, LLMConfig
from baselines.sample_ids import FIELD_MAP, extract_text_field, make_sample_id, pick

logger = logging.getLogger("kp_llm")

DEFAULT_INPUTS = [
    str(ROOT / "data/samples/blink_all_candidate.jsonl.scores.jsonl"),
    str(ROOT / "data/samples/blink_all_humman_summary.jsonl.scores.jsonl"),
    str(ROOT / "data/samples/unisumeval_nondialogue_success_faith_ne1_factgraph.jsonl.scores.jsonl"),
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def already_done(out_path: Path) -> set[str]:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(obj.get("id") or "")
            if sid:
                done.add(sid)
    return done


def unique_sample_id(row: dict, input_stem: str, line_idx: int) -> str:
    """Stable unique id across multi-file inputs (handles duplicate raw ids)."""
    base = make_sample_id(row) or f"row{line_idx}"
    return f"{input_stem}__{base}__L{line_idx}"


def default_output_for(in_path: Path) -> Path:
    return ROOT / "experiments/results/kp_llm" / f"predictions_{in_path.stem}_dist5.jsonl"


def run_one(
    scorer,
    in_path: Path,
    out_path: Path,
    limit: int | None,
) -> int:
    rows = load_jsonl(in_path)
    if limit is not None:
        rows = rows[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)
    stem = in_path.stem
    n_skip = sum(
        1
        for line_idx, row in enumerate(rows)
        if unique_sample_id(row, stem, line_idx) in done
    )
    n_todo = len(rows) - n_skip
    logger.info(
        "start file=%s total=%d skip_done=%d todo=%d -> %s",
        in_path.name,
        len(rows),
        n_skip,
        n_todo,
        out_path,
    )

    n_new = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fout:
        pbar = tqdm(rows, desc=f"KP-LLM {in_path.name}", unit="sample")
        for line_idx, row in enumerate(pbar):
            sid = unique_sample_id(row, stem, line_idx)
            if sid in done:
                continue
            source = extract_text_field(pick(row, FIELD_MAP["source"], default=""))
            summary = extract_text_field(pick(row, FIELD_MAP["summary"], default=""))
            try:
                result = scorer.score(
                    source,
                    summary,
                    sample_id=sid,
                    dataset=pick(row, FIELD_MAP["dataset"]),
                )
            except Exception:
                logger.exception("failed sample_id=%s line_idx=%d", sid, line_idx)
                raise
            result["uid"] = row.get("uid")
            result["model"] = row.get("model")
            result["doc_id"] = row.get("doc_id")
            result["source_id"] = row.get("id")
            result["source_file"] = in_path.name
            result["line_idx"] = line_idx
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            n_new += 1
            done.add(sid)
            elapsed = time.time() - t0
            rate = n_new / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix(new=n_new, rate=f"{rate:.2f}/s", refresh=False)
            if n_new % 20 == 0:
                logger.info(
                    "progress file=%s new=%d/%d elapsed=%.1fs rate=%.2f/s last_id=%s",
                    in_path.name,
                    n_new,
                    n_todo,
                    elapsed,
                    rate,
                    sid,
                )

    elapsed = time.time() - t0
    logger.info(
        "done file=%s wrote=%d elapsed=%.1fs -> %s",
        in_path.name,
        n_new,
        elapsed,
        out_path,
    )
    return n_new


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KP-LLM baseline")
    parser.add_argument(
        "--input",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="One or more input JSONL files (default: blink candidate/human + unisumeval filtered)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL. If omitted, write one file per input under experiments/results/kp_llm/",
    )
    parser.add_argument(
        "--extractor-model",
        default="/root/autodl-fs/zyq/models/DisT5",
        help="Local or HF seq2seq model for keyphrase extraction",
    )
    parser.add_argument("--max-keyphrases", type=int, default=8)
    parser.add_argument("--checker-provider", default="openai", choices=["openai", "ernie"])
    parser.add_argument("--checker-model", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Per-input row limit")
    parser.add_argument("--cache-dir", default="experiments/results/kp_llm/cache")
    parser.add_argument(
        "--log-file",
        default="experiments/results/kp_llm/run_kp_llm.log",
        help="Append log file path (also prints to stdout)",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logger.info("log_file=%s", log_path)

    if args.checker_provider == "openai":
        cfg = LLMConfig(
            provider="openai",
            model=args.checker_model or os.environ.get("CHATGPT_MODEL", "gpt-3.5-turbo"),
            api_key_env="OPENAI_API_KEY",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1"),
            temperature=0.0,
            max_tokens=160,
            seed=42,
        )
    else:
        cfg = LLMConfig(
            provider="openai",
            model=args.checker_model or os.environ.get("ERNIE_MODEL", "ernie-3.5-128k"),
            api_key_env="ERNIE_API_KEY" if os.environ.get("ERNIE_API_KEY") else "OPENAI_API_KEY",
            base_url=os.environ.get(
                "ERNIE_BASE_URL",
                os.environ.get("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1"),
            ),
            temperature=0.0,
            max_tokens=160,
            seed=42,
        )

    client = LLMClient(cfg, cache_dir=args.cache_dir)
    scorer = build_default_kp_llm_scorer(
        client,
        extractor_model=args.extractor_model,
        max_keyphrases=args.max_keyphrases,
    )

    inputs = [Path(p) for p in args.input]
    total_new = 0
    for in_path in inputs:
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")
        if args.output and len(inputs) == 1:
            out_path = Path(args.output)
        elif args.output and len(inputs) > 1:
            # Shared output path: all inputs append into one file with unique ids.
            out_path = Path(args.output)
        else:
            out_path = default_output_for(in_path)
        total_new += run_one(scorer, in_path, out_path, args.limit)

    logger.info("all done. total new rows=%d", total_new)


if __name__ == "__main__":
    main()
