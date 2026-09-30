"""用本地 t5-large 对 BD2TSumm 源文本做层次摘要，写出 candidates CSV。"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from util import ensure_repo_on_path, read_jsonl, setup_logging, write_csv, write_json, write_jsonl

ensure_repo_on_path()

DEFAULT_MODEL = "/root/autodl-fs/zyq/models/t5-large"


def infer_model_defaults(model_path: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_path)
    model_type = str(getattr(cfg, "model_type", "") or "")
    max_pos = int(getattr(cfg, "max_position_embeddings", 0) or getattr(cfg, "n_positions", 512) or 512)
    if model_type == "t5":
        prefix = "summarize: "
        max_encode = min(max_pos, 512)
        length_penalty = 2.0
        num_beams = 4
        max_input = 450
    elif model_type == "pegasus":
        prefix = ""
        max_encode = min(max_pos, 1024)
        length_penalty = 0.8
        num_beams = 8
        max_input = min(900, max_encode - 8)
    else:
        prefix = ""
        max_encode = min(max_pos, 1024) if max_pos else 512
        length_penalty = 1.0
        num_beams = 4
        max_input = min(450, max_encode - 8)
    return {
        "model_type": model_type,
        "prefix": prefix,
        "max_encode": max_encode,
        "length_penalty": length_penalty,
        "num_beams": num_beams,
        "max_input": max_input,
    }


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def chunk_paragraphs(paragraphs: list[str], tokenizer, max_content_tokens: int, prefix: str) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        trial = "\n".join(buf + [para]) if buf else para
        if count_tokens(tokenizer, prefix + trial) <= max_content_tokens:
            buf.append(para)
            continue
        if buf:
            chunks.append("\n".join(buf))
            buf = []
        if count_tokens(tokenizer, prefix + para) <= max_content_tokens:
            buf = [para]
        else:
            enc = tokenizer(prefix + para, add_special_tokens=True, truncation=True, max_length=max_content_tokens)
            decoded = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
            if prefix and decoded.startswith(prefix):
                decoded = decoded[len(prefix) :]
            chunks.append(decoded.strip())
    if buf:
        chunks.append("\n".join(buf))
    return chunks


@torch.inference_mode()
def summarize_text(
    model,
    tokenizer,
    text: str,
    *,
    prefix: str,
    max_encode: int,
    max_new: int,
    min_new: int,
    num_beams: int,
    length_penalty: float,
    device: torch.device,
) -> str:
    prompt = prefix + (text or "").strip()
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_encode)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_length=max_new,
        min_length=min_new,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def hierarchical_summarize(
    source_text: str,
    model,
    tokenizer,
    *,
    prefix: str,
    max_encode: int,
    max_content_tokens: int,
    max_new: int,
    min_new: int,
    num_beams: int,
    length_penalty: float,
    device: torch.device,
    logger: logging.Logger,
    event_id: str,
) -> tuple[str, dict]:
    paragraphs = [p for p in (source_text or "").split("\n") if p.strip()]
    level = 0
    current = paragraphs
    meta = {"event_id": event_id, "n_paragraphs": len(paragraphs), "levels": []}
    gen_kw = dict(
        prefix=prefix,
        max_encode=max_encode,
        max_new=max_new,
        min_new=min_new,
        num_beams=num_beams,
        length_penalty=length_penalty,
        device=device,
    )
    while True:
        chunks = chunk_paragraphs(current, tokenizer, max_content_tokens, prefix)
        meta["levels"].append({"level": level, "n_chunks": len(chunks)})
        logger.info("event=%s level=%s chunks=%s", event_id, level, len(chunks))
        if not chunks:
            return "", meta
        summaries = [summarize_text(model, tokenizer, ch, **gen_kw) for ch in chunks]
        summaries = [s for s in summaries if s]
        if len(summaries) <= 1:
            final = summaries[0] if summaries else ""
            meta["n_final_chars"] = len(final)
            return final, meta
        joined = " ".join(summaries)
        if count_tokens(tokenizer, prefix + joined) <= max_content_tokens:
            final = summarize_text(model, tokenizer, joined, **gen_kw)
            meta["n_final_chars"] = len(final)
            meta["levels"].append({"level": level + 1, "n_chunks": 1, "note": "final_merge"})
            return final, meta
        current = summaries
        level += 1
        if level > 8:
            logger.warning("event=%s 层次过深，截取前 max_content_tokens 再摘要", event_id)
            final = summarize_text(model, tokenizer, joined, **gen_kw)
            meta["n_final_chars"] = len(final)
            return final, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="T5-large hierarchical summaries for BD2TSumm")
    p.add_argument("--sources", type=str, default="experiments/results/bd2tsumm/prepared_sources.jsonl")
    p.add_argument("--output", type=str, default="experiments/results/bd2tsumm/candidates_t5_large.csv")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--system-name", type=str, default="t5-large")
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--min-new-tokens", type=int, default=30)
    p.add_argument("--num-beams", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    defaults = infer_model_defaults(args.model)
    max_input = int(args.max_input_tokens or defaults["max_input"])
    num_beams = int(args.num_beams or defaults["num_beams"])
    prefix = defaults["prefix"]
    max_encode = int(defaults["max_encode"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_path.parent / "logs" / "generate_summaries.log")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "device=%s model=%s type=%s prefix=%r max_encode=%s max_input=%s beams=%s",
        device,
        args.model,
        defaults["model_type"],
        prefix,
        max_encode,
        max_input,
        num_beams,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    sources = read_jsonl(Path(args.sources))
    rows = []
    traces = []
    t0 = time.perf_counter()
    for src in sources:
        eid = str(src["event_id"])
        logger.info("开始 event=%s chars=%s tweets=%s", eid, src.get("num_chars"), src.get("num_tweets_after_dedup"))
        text, meta = hierarchical_summarize(
            src.get("source_text") or "",
            model,
            tokenizer,
            prefix=prefix,
            max_encode=max_encode,
            max_content_tokens=max_input,
            max_new=args.max_new_tokens,
            min_new=args.min_new_tokens,
            num_beams=num_beams,
            length_penalty=float(defaults["length_penalty"]),
            device=device,
            logger=logger,
            event_id=eid,
        )
        rows.append({"event_id": eid, "system_name": args.system_name, "candidate_summary": text})
        traces.append(meta)
        logger.info("完成 event=%s summary_chars=%s", eid, len(text))
    write_csv(out_path, rows, ["event_id", "system_name", "candidate_summary"])
    write_jsonl(out_path.with_suffix(".trace.jsonl"), traces)
    write_json(
        out_path.with_suffix(".meta.json"),
        {
            "model": args.model,
            "model_type": defaults["model_type"],
            "system_name": args.system_name,
            "prefix": prefix,
            "max_encode": max_encode,
            "max_input_tokens": max_input,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "num_beams": num_beams,
            "length_penalty": defaults["length_penalty"],
            "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - t0,
            "n_events": len(rows),
        },
    )
    logger.info("写出 %s  events=%s elapsed=%.1fs", out_path, len(rows), time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
