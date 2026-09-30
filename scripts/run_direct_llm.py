#!/usr/bin/env python3
"""Run Direct-LLM baselines: ChatGPT-ZS/Star and ERNIE-ZS/Star."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.llm_direct import DirectLLMEvaluator, build_client_for_method
from baselines.sample_ids import FIELD_MAP, extract_text_field, make_sample_id, pick


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Direct-LLM factuality baselines")
    parser.add_argument(
        "--method",
        required=True,
        choices=["ChatGPT-ZS", "ChatGPT-Star", "ERNIE-ZS", "ERNIE-Star"],
    )
    parser.add_argument(
        "--input",
        default="/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2_processed.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: predictions_<method>_unisumeval.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--cache-dir", default="experiments/results/llm_baselines/cache")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else Path(
        f"experiments/results/llm_baselines/predictions_{args.method}_unisumeval.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = build_client_for_method(args.method, cache_dir=args.cache_dir)
    evaluator = DirectLLMEvaluator(args.method, client)

    rows = load_jsonl(in_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    done = already_done(out_path)
    n_new = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for row in rows:
            sid = make_sample_id(row)
            if sid and sid in done:
                continue
            article = extract_text_field(pick(row, FIELD_MAP["source"], default=""))
            summary = extract_text_field(pick(row, FIELD_MAP["summary"], default=""))
            result = evaluator.score(str(article), str(summary))
            out = {
                "id": sid,
                "uid": row.get("uid"),
                "model": row.get("model"),
                "doc_id": row.get("doc_id"),
                "dataset": pick(row, FIELD_MAP["dataset"]),
                "method": result.method,
                "score": result.score,
                "binary_label": result.binary_label,
                "raw_output": result.raw_output,
                "error": result.error,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            n_new += 1
            if sid:
                done.add(sid)

    print(f"[{args.method}] wrote {n_new} new rows -> {out_path}")


if __name__ == "__main__":
    main()
