#!/usr/bin/env python3
"""Convert DeFacto per-document JSON files into flat JSONL for LLM baselines.

DeFacto files (e.g. dreeam_result/test/*.json) look like:
  {
    "doc_id": 3,
    "article": {"text": "...", ...},
    "candidate": {"text": "...", ...},
    "humman_summary": {"text": "...", ...},
    "abstract": {"text": "...", ...},
    "intrinsic_error": false,
    "extrinsic_error": true,
    "has_error": true
  }

Output JSONL rows (one summary type per line by default: candidate):
  {
    "id": "3",
    "dataset": "DeFacto",
    "split": "test",
    "source_document": "...",
    "generated_summary": "...",
    "summary_type": "candidate",
    "human_label": 0,
    "intrinsic_error": false,
    "extrinsic_error": true,
    "has_error": true
  }

Example:
  python scripts/convert_defacto_to_jsonl.py \\
    --input-dir /root/autodl-fs/zyq/defacto_data_gawl/dreeam_result/test \\
    --output data/samples/defacto_test_candidate.jsonl \\
    --summary-types candidate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_SUMMARY_TYPES = ("candidate",)
ALL_SUMMARY_TYPES = ("candidate", "humman_summary", "abstract")


def extract_text(block: Any) -> str:
    """Pull plain text from a DeFacto nested section."""
    if block is None:
        return ""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        for key in ("text", "input_context", "document", "doc"):
            if isinstance(block.get(key), str):
                return block[key]
        # fallback: join sentences if present
        sents = block.get("sents")
        if isinstance(sents, list) and sents:
            if all(isinstance(s, str) for s in sents):
                return " ".join(sents)
            # nested token lists
            parts = []
            for sent in sents:
                if isinstance(sent, list):
                    parts.append(" ".join(str(t) for t in sent))
                else:
                    parts.append(str(sent))
            return " ".join(parts)
    return str(block)


def human_label_from_item(item: dict) -> Optional[int]:
    """1 = factually consistent, 0 = has factual error."""
    if "has_error" in item and item["has_error"] is not None:
        return 0 if bool(item["has_error"]) else 1
    # derive from intrinsic/extrinsic if needed
    intrinsic = item.get("intrinsic_error")
    extrinsic = item.get("extrinsic_error")
    if intrinsic is None and extrinsic is None:
        return None
    return 0 if bool(intrinsic) or bool(extrinsic) else 1


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    files = sorted(input_dir.glob("*.json"), key=lambda p: (
        int(p.stem) if p.stem.isdigit() else p.stem
    ))
    for path in files:
        yield path


def convert_file(
    path: Path,
    *,
    summary_types: tuple[str, ...],
    split: Optional[str],
    dataset: str,
) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        item = json.load(f)

    doc_id = item.get("doc_id", path.stem)
    source = extract_text(item.get("article"))
    label = human_label_from_item(item)

    rows = []
    for stype in summary_types:
        if stype not in item:
            continue
        summary = extract_text(item.get(stype))
        row = {
            "id": f"{doc_id}" if len(summary_types) == 1 else f"{doc_id}_{stype}",
            "doc_id": str(doc_id),
            "dataset": dataset,
            "split": split,
            "source_document": source,
            "generated_summary": summary,
            "summary_type": stype,
            "human_label": label,
            "intrinsic_error": item.get("intrinsic_error"),
            "extrinsic_error": item.get("extrinsic_error"),
            "has_error": item.get("has_error"),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeFacto JSON dir to flat JSONL")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory of DeFacto *.json files (e.g. .../dreeam_result/test)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--summary-types",
        nargs="+",
        default=list(DEFAULT_SUMMARY_TYPES),
        choices=list(ALL_SUMMARY_TYPES),
        help="Which nested summaries to emit (default: candidate only)",
    )
    parser.add_argument(
        "--dataset",
        default="DeFacto",
        help="Dataset name written into each row",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split name; defaults to input-dir folder name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of source JSON files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    split = args.split if args.split is not None else input_dir.name
    summary_types = tuple(args.summary_types)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_rows = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for path in iter_json_files(input_dir):
            if args.limit is not None and n_files >= args.limit:
                break
            try:
                rows = convert_file(
                    path,
                    summary_types=summary_types,
                    split=split,
                    dataset=args.dataset,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] skip {path.name}: {exc}", file=sys.stderr)
                continue
            for row in rows:
                # skip empty texts
                if not row["source_document"].strip() or not row["generated_summary"].strip():
                    print(f"[WARN] empty text in {path.name} ({row['summary_type']})", file=sys.stderr)
                    continue
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
            n_files += 1

    print(f"Converted {n_files} files -> {n_rows} rows: {out_path}")


if __name__ == "__main__":
    main()
