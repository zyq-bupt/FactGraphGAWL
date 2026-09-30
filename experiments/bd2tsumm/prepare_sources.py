"""将同一事件的多条推文聚合成 FK-Graph 源文本。"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from util import (
    MissingCandidatesError,
    REPO_ROOT,
    deep_get,
    ensure_repo_on_path,
    load_config,
    read_csv,
    resolve_path,
    setup_logging,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)

ensure_repo_on_path()

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_PREFIX_RE = re.compile(r"^RT\s+", re.IGNORECASE)
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


def placeholder_tokens(style: str) -> tuple[str, str]:
    if str(style).lower() == "angle":
        return "<URL>", "<USER>"
    return "URL", "USER"


def clean_tweet(text: str, cleaning: dict[str, Any]) -> str:
    if text is None:
        return ""
    raw = str(text)
    form = cleaning.get("unicode_form") or "NFKC"
    raw = unicodedata.normalize(str(form), raw)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if cleaning.get("collapse_whitespace", True):
        raw = re.sub(r"[\t\n]+", " ", raw)
        raw = re.sub(r" +", " ", raw)
    raw = raw.strip()
    url_ph, user_ph = placeholder_tokens(cleaning.get("placeholder_style") or "plain")
    if cleaning.get("replace_urls", True):
        raw = URL_RE.sub(url_ph, raw)
    if cleaning.get("replace_mentions", True):
        raw = MENTION_RE.sub(user_ph, raw)
    if cleaning.get("strip_rt_prefix", False):
        raw = RT_PREFIX_RE.sub("", raw).lstrip()
    if cleaning.get("keep_hashtag_text", True):
        raw = HASHTAG_RE.sub(r"\1", raw)
    if cleaning.get("strip_emoji", False):
        raw = EMOJI_RE.sub("", raw)
        raw = re.sub(r" +", " ", raw).strip()
    return raw


def _parse_ts(value: str) -> tuple[int, str]:
    s = (value or "").strip()
    if not s:
        return (1, "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return (0, datetime.strptime(s, fmt).isoformat())
        except ValueError:
            continue
    try:
        return (0, datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat())
    except ValueError:
        return (0, s)


def choose_sort_mode(rows: list[dict], timestamp_col: str | None) -> str:
    if timestamp_col:
        n_ts = sum(1 for r in rows if str(r.get(timestamp_col) or "").strip())
        if n_ts > 0:
            return "timestamp"
    return "original_order"


def sort_tweets(
    rows: list[dict[str, Any]],
    *,
    timestamp_col: str | None,
    tweet_id_col: str,
    sort_mode: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    mode = sort_mode or choose_sort_mode(rows, timestamp_col)
    if mode == "timestamp" and timestamp_col:
        def key_ts(r: dict) -> tuple:
            flag, ts = _parse_ts(str(r.get(timestamp_col) or ""))
            return (flag, ts, int(r.get("_orig_index", 0)))

        return sorted(rows, key=key_ts), "timestamp"
    if mode == "tweet_id":
        return sorted(rows, key=lambda r: (str(r.get(tweet_id_col) or ""), int(r.get("_orig_index", 0)))), "tweet_id"
    return sorted(rows, key=lambda r: int(r.get("_orig_index", 0))), "original_order"


def dedup_keep_first(texts: list[str]) -> tuple[list[int], int]:
    seen: set[str] = set()
    keep: list[int] = []
    dropped = 0
    for i, t in enumerate(texts):
        if t in seen:
            dropped += 1
            continue
        seen.add(t)
        keep.append(i)
    return keep, dropped


def join_source_text(paragraphs: Iterable[str]) -> str:
    return "\n".join(p for p in paragraphs if p)


def _read_csv_rows(path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def load_event_tweets(
    path: Path,
    *,
    event_id: str,
    text_col: str,
    tweet_id_col: str,
    timestamp_col: str | None,
    event_id_col: str | None = None,
) -> list[dict[str, Any]]:
    fields, raw = _read_csv_rows(path)
    if not fields or text_col not in fields:
        # 兼容无表头或列名不同：若只有一列则当作 text
        if fields and text_col not in fields and len(fields) == 1:
            text_col = fields[0]
        elif text_col not in (fields or []):
            raise ValueError(f"{path} 缺少推文列 {text_col}；实际列={fields}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        eid = event_id
        if event_id_col and row.get(event_id_col):
            eid = str(row[event_id_col]).strip() or event_id
        tid = str(row.get(tweet_id_col) or "").strip() or f"{eid}:{i}"
        out.append(
            {
                "event_id": eid,
                "tweet_id": tid,
                "tweet_text": row.get(text_col) or "",
                "timestamp": row.get(timestamp_col) if timestamp_col else "",
                "_orig_index": i,
                "_source_file": str(path),
            }
        )
    return out


def load_reference_txt(path: Path, event_id: str) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "event_id": event_id,
        "reference_id": path.stem,
        "reference_summary": text,
        "reference_file": str(path),
    }


def load_tabular_references(path: Path, cfg: dict) -> list[dict[str, str]]:
    ds = cfg["dataset"]
    rows = read_csv(path)
    out = []
    for i, row in enumerate(rows):
        eid = str(row.get(ds["event_id_col"]) or "").strip()
        if not eid:
            continue
        rid = str(row.get(ds.get("reference_id_col") or "") or "").strip() or f"{eid}:{i}"
        out.append(
            {
                "event_id": eid,
                "reference_id": rid,
                "reference_summary": row.get(ds["reference_col"]) or "",
            }
        )
    return out


def load_candidates(cfg: dict, data_root: Path, logger: logging.Logger) -> list[dict[str, str]]:
    ds = cfg["dataset"]
    candidates: list[dict[str, str]] = []
    tried: list[str] = []
    file_candidates = []
    for base in (REPO_ROOT, data_root, Path.cwd()):
        p = resolve_path(ds.get("candidates_file"), base=base)
        if p is not None:
            tried.append(str(p))
            if p.is_file():
                file_candidates.append(p)
                break
    if file_candidates:
        path = file_candidates[0]
        logger.info("加载候选摘要文件 %s", path)
        if path.suffix.lower() in {".jsonl", ".json"}:
            from util import read_jsonl

            raw_rows = read_jsonl(path) if path.suffix.lower() == ".jsonl" else __import__("json").loads(path.read_text(encoding="utf-8"))
            if isinstance(raw_rows, dict):
                raw_rows = raw_rows.get("candidates") or []
        else:
            raw_rows = read_csv(path)
        for row in raw_rows:
            eid = str(row.get(ds["event_id_col"]) or "").strip()
            sysn = str(row.get(ds["system_col"]) or "").strip()
            text = str(row.get(ds["candidate_col"]) or "")
            if not eid or not sysn:
                continue
            candidates.append(
                {
                    "event_id": eid,
                    "system_name": sysn,
                    "candidate_summary": text,
                }
            )
        return candidates

    cand_dir_raw = ds.get("candidates_dir")
    dir_hits: list[Path] = []
    for base in (REPO_ROOT, data_root, Path.cwd()):
        p = resolve_path(cand_dir_raw, base=base)
        if p is not None and p.is_dir():
            dir_hits.append(p)
            break
    if dir_hits:
        cdir = dir_hits[0]
        logger.info("从目录加载候选摘要 %s （{system}/{event_id}.txt）", cdir)
        for sys_dir in sorted(p for p in cdir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for fp in sorted(sys_dir.glob("*.txt")):
                if fp.name.startswith("._"):
                    continue
                candidates.append(
                    {
                        "event_id": fp.stem,
                        "system_name": sys_dir.name,
                        "candidate_summary": fp.read_text(encoding="utf-8", errors="replace").strip(),
                    }
                )
        return candidates

    logger.info("未找到候选摘要文件/目录（已尝试 %s）", tried)
    return []


def count_windows(source_text: str, cfg: dict, logger: logging.Logger) -> dict[str, Any]:
    lc = cfg.get("long_context") or {}
    mode = str(lc.get("mode") or "sliding_window")
    if mode != "sliding_window" and not lc.get("allow_prefix_truncation"):
        raise RuntimeError(
            f"长文本模式为 {mode}，但 allow_prefix_truncation=false。"
            "禁止静默截断为前 1024 tokens，请使用 sliding_window。"
        )
    from long_context.coverage import token_coverage_ratio
    from long_context.tokenizer_util import count_tokens, load_tokenizer
    from long_context.windows import split_windows, validate_windows

    tok = load_tokenizer(
        lc.get("tokenizer_path"),
        allow_mock=bool(lc.get("allow_mock_tokenizer")),
    )
    max_len = int(lc.get("max_model_tokens") or 1024)
    overlap = int(lc.get("overlap_tokens") or 256)
    sentence_boundary = bool(lc.get("sentence_boundary", True))
    windows = split_windows(
        source_text,
        tok,
        max_model_tokens=max_len,
        overlap_tokens=overlap,
        sentence_boundary=sentence_boundary,
    )
    validate_windows(windows, source_text, tok, max_len)
    if mode == "prefix_1024":
        used = windows[:1]
        logger.warning("allow_prefix_truncation=true，仅使用第一窗口（共 %s 个窗口）", len(windows))
    else:
        used = windows
    cov = token_coverage_ratio(source_text, used, tok)
    n_tok = count_tokens(tok, source_text, True)
    return {
        "num_tokens": int(n_tok),
        "num_windows": int(len(used)),
        "num_windows_full_split": int(len(windows)),
        "token_coverage_ratio": cov.get("token_coverage_ratio"),
        "document_model_tokens": cov.get("document_model_tokens"),
        "unique_tokens": cov.get("unique_tokens"),
        "tokenizer": str(getattr(tok, "name_or_path", type(tok).__name__)),
        "window_mode": mode,
        "sort_note": None,
    }


def aggregate_event(
    event_id: str,
    tweets: list[dict[str, Any]],
    cleaning: dict[str, Any],
    *,
    timestamp_col: str,
    tweet_id_col: str,
) -> dict[str, Any]:
    n_raw = len(tweets)
    sorted_rows, sort_mode = sort_tweets(tweets, timestamp_col=timestamp_col, tweet_id_col=tweet_id_col)
    cleaned = []
    for r in sorted_rows:
        c = clean_tweet(r.get("tweet_text") or "", cleaning)
        item = dict(r)
        item["tweet_text_clean"] = c
        cleaned.append(item)
    if cleaning.get("drop_empty_after_clean", True):
        cleaned = [r for r in cleaned if r["tweet_text_clean"]]
    n_empty_dropped = n_raw - len(cleaned)
    texts = [r["tweet_text_clean"] for r in cleaned]
    if cleaning.get("dedup_exact", True):
        keep_idx, n_dup = dedup_keep_first(texts)
        kept = [cleaned[i] for i in keep_idx]
    else:
        n_dup = 0
        kept = cleaned
    source_text = join_source_text(r["tweet_text_clean"] for r in kept)
    return {
        "event_id": event_id,
        "source_text": source_text,
        "num_tweets_raw": n_raw,
        "num_tweets_after_empty_drop": len(cleaned),
        "num_tweets_after_dedup": len(kept),
        "num_duplicates_removed": n_dup,
        "num_empty_removed": n_empty_dropped,
        "sort_mode": sort_mode,
        "source_sha256": sha256_text(source_text),
        "num_chars": len(source_text),
        "kept_tweet_ids": [r["tweet_id"] for r in kept],
    }


def _data_root(cfg: dict) -> Path:
    ds = cfg["dataset"]
    p = resolve_path(ds.get("data_root"), base=REPO_ROOT)
    return p if p is not None else REPO_ROOT


def load_all_tweets_and_refs(cfg: dict, logger: logging.Logger) -> tuple[dict[str, list[dict]], list[dict[str, str]]]:
    ds = cfg["dataset"]
    root = _data_root(cfg)
    layout = str(ds.get("layout") or "bd2tsumm_dirs")
    tweets_by_event: dict[str, list[dict]] = {}
    refs: list[dict[str, str]] = []
    text_col = ds.get("tweet_text_col") or "text"
    tid_col = ds.get("tweet_id_col") or "tweet_id"
    ts_col = ds.get("timestamp_col") or "timestamp"
    eid_col = ds.get("event_id_col") or "event_id"

    if layout == "tabular":
        tweets_path = None
        for base in (REPO_ROOT, root, Path.cwd()):
            p = resolve_path(ds.get("tweets_file"), base=base)
            if p is not None and p.is_file():
                tweets_path = p
                break
        if tweets_path is None:
            raise FileNotFoundError("tabular 布局需要 dataset.tweets_file")
        logger.info("tabular 推文文件 %s", tweets_path)
        _, raw = _read_csv_rows(tweets_path)
        for i, row in enumerate(raw):
            eid = str(row.get(eid_col) or "").strip()
            if not eid:
                continue
            rec = {
                "event_id": eid,
                "tweet_id": str(row.get(tid_col) or "").strip() or f"{eid}:{i}",
                "tweet_text": row.get(text_col) or "",
                "timestamp": row.get(ts_col) or "",
                "_orig_index": i,
                "_source_file": str(tweets_path),
            }
            tweets_by_event.setdefault(eid, []).append(rec)
        refs_path = None
        for base in (REPO_ROOT, root, Path.cwd()):
            p = resolve_path(ds.get("references_file"), base=base)
            if p is not None and p.is_file():
                refs_path = p
                break
        if refs_path is None:
            raise FileNotFoundError("tabular 布局需要 dataset.references_file")
        refs = load_tabular_references(refs_path, cfg)
        return tweets_by_event, refs

    events = ds.get("events") or []
    if not events:
        raise ValueError("bd2tsumm_dirs 布局需要 dataset.events 列表")
    for ev in events:
        eid = str(ev["event_id"])
        tpath = root / ev["tweets_file"]
        rpath = root / ev["reference_file"]
        if not tpath.is_file():
            logger.error("缺少推文文件 %s", tpath)
            continue
        tweets_by_event[eid] = load_event_tweets(
            tpath,
            event_id=eid,
            text_col=text_col,
            tweet_id_col=tid_col,
            timestamp_col=ts_col,
        )
        if rpath.is_file():
            refs.append(load_reference_txt(rpath, eid))
        else:
            logger.error("缺少参考摘要 %s", rpath)
    return tweets_by_event, refs


def validate_bundle(
    sources: list[dict],
    candidates: list[dict],
    references: list[dict],
    *,
    require_candidates: bool,
) -> dict[str, Any]:
    event_ids = {s["event_id"] for s in sources}
    events_with_tweets = {s["event_id"] for s in sources if int(s.get("num_tweets_after_dedup") or 0) > 0}
    refs_by_event: dict[str, int] = {}
    for r in references:
        if (r.get("reference_summary") or "").strip():
            refs_by_event[r["event_id"]] = refs_by_event.get(r["event_id"], 0) + 1
    pair_keys = [(c["event_id"], c["system_name"]) for c in candidates]
    dup_pairs = sorted({k for k in pair_keys if pair_keys.count(k) > 1})
    systems = sorted({c["system_name"] for c in candidates})
    cand_events = {c["event_id"] for c in candidates}
    missing_source = sorted(eid for eid in cand_events if eid not in events_with_tweets)
    missing_ref = sorted(eid for eid in event_ids if eid not in refs_by_event)
    events_no_tweets = sorted(eid for eid in event_ids if eid not in events_with_tweets)
    report = {
        "n_events": len(event_ids),
        "n_events_with_tweets": len(events_with_tweets),
        "n_systems": len(systems),
        "systems": systems,
        "n_candidates": len(candidates),
        "n_references": len(references),
        "n_events_with_reference": len(refs_by_event),
        "duplicate_event_system_pairs": [list(x) for x in dup_pairs],
        "candidates_missing_source": missing_source,
        "events_missing_reference": missing_ref,
        "events_without_tweets": events_no_tweets,
        "ok": True,
        "errors": [],
    }
    errors: list[str] = []
    if events_no_tweets:
        errors.append(f"事件无推文: {events_no_tweets}")
    if missing_ref:
        errors.append(f"事件无参考摘要: {missing_ref}")
    if dup_pairs:
        errors.append(f"event_id+system_name 重复: {dup_pairs}")
    if missing_source:
        errors.append(f"候选摘要事件在源推文中不存在: {missing_source}")
    if require_candidates and not candidates:
        errors.append(
            "未提供摘要系统输出。请设置 dataset.candidates_file 或 dataset.candidates_dir；"
            "不会把人工参考摘要当作候选摘要。"
        )
    report["errors"] = errors
    report["ok"] = not errors
    if require_candidates and not candidates:
        raise MissingCandidatesError(errors[-1])
    return report


def run_prepare(cfg: dict, output_dir: Path, logger: logging.Logger, *, require_candidates: bool = False) -> dict[str, Any]:
    ds = cfg["dataset"]
    cleaning = cfg.get("cleaning") or {}
    tweets_by_event, refs = load_all_tweets_and_refs(cfg, logger)
    sources = []
    ts_col = ds.get("timestamp_col") or "timestamp"
    tid_col = ds.get("tweet_id_col") or "tweet_id"
    for eid, tweets in sorted(tweets_by_event.items()):
        agg = aggregate_event(eid, tweets, cleaning, timestamp_col=ts_col, tweet_id_col=tid_col)
        logger.info(
            "事件 %s: 排序=%s raw=%s dedup=%s dup_removed=%s chars=%s",
            eid,
            agg["sort_mode"],
            agg["num_tweets_raw"],
            agg["num_tweets_after_dedup"],
            agg["num_duplicates_removed"],
            agg["num_chars"],
        )
        try:
            win = count_windows(agg["source_text"], cfg, logger)
            agg.update(win)
            logger.info(
                "事件 %s: tokens=%s windows=%s coverage=%s tokenizer=%s",
                eid,
                win.get("num_tokens"),
                win.get("num_windows"),
                win.get("token_coverage_ratio"),
                win.get("tokenizer"),
            )
        except Exception as exc:
            logger.exception("窗口统计失败 event=%s", eid)
            agg["num_tokens"] = None
            agg["num_windows"] = None
            agg["window_error"] = f"{type(exc).__name__}: {exc}"
        sources.append(agg)

    root = _data_root(cfg)
    candidates = load_candidates(cfg, root, logger)
    if candidates:
        logger.info("系统列表: %s", sorted({c['system_name'] for c in candidates}))
    validation = validate_bundle(sources, candidates, refs, require_candidates=require_candidates)

    input_hashes = {}
    if str(ds.get("layout")) == "bd2tsumm_dirs":
        for ev in ds.get("events") or []:
            for key in ("tweets_file", "reference_file"):
                p = root / ev[key]
                input_hashes[str(p)] = sha256_file(p)
    validation["input_sha256"] = input_hashes
    validation["placeholder_style"] = (cfg.get("cleaning") or {}).get("placeholder_style")
    validation["url_user_placeholders"] = list(placeholder_tokens((cfg.get("cleaning") or {}).get("placeholder_style") or "plain"))

    out = cfg.get("output") or {}
    write_jsonl(output_dir / out.get("prepared_sources", "prepared_sources.jsonl"), sources)
    write_jsonl(output_dir / out.get("prepared_candidates", "prepared_candidates.jsonl"), candidates)
    write_jsonl(output_dir / out.get("prepared_references", "prepared_references.jsonl"), refs)
    write_json(output_dir / out.get("data_validation", "data_validation.json"), validation)
    logger.info("校验: %s", validation)
    return {"sources": sources, "candidates": candidates, "references": refs, "validation": validation}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BD2TSumm 推文聚合")
    p.add_argument("--config", type=str, default=str(BD2T_DIR_CONFIG()))
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--require-candidates", action="store_true")
    return p


def BD2T_DIR_CONFIG() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "logs" / "prepare_sources.log")
    run_prepare(cfg, output_dir, logger, require_candidates=args.require_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
