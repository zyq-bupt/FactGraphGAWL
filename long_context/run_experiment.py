"""滑动窗口 FK-Graph 长文本实验入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from long_context.config import LongContextConfig, add_cli_args, config_from_args
from long_context.data import iter_experiment_samples, load_unisum_labels
from long_context.graphs import build_document_graph
from long_context.merge import graph_counts
from long_context.pipeline import graphs_structurally_equal, process_sample
from long_context.report import write_outputs
from long_context.tokenizer_util import load_tokenizer, tokenizer_fingerprint

logger = logging.getLogger("long_context")

IMPLEMENTATION_NOTES = """# Implementation notes

## Repository mapping (this repo is FactGraphGAWL, not CompKG-FactEval)

| Guide concept | Actual location |
|---|---|
| FK-Graph construction | `build/graph_embedding/graph_builder.py` :: `GraphBuilder.build_graph` |
| Token / Mention / Entity nodes | `create_tokenNode` / `create_mentionNode` / `create_entityNode` |
| E_tt / E_tm / E_me / E_ee | parse_trees edges; `RELATED_TO`; `BELONGS_TO`; `ent_relation` |
| spaCy / BLINK / DREEAM | **Not in this repo**. Precomputed in `dreeam_result/*.json` (`sents`, `mentions`, `parse_trees`, `vertexSet`, `ent_relation`) |
| Graph serialization | `nx.node_link_data` via `build/construct_graph.py`; IO in `common/data_loader_saver.py` |
| S²-K (PT) | `kernel/gawl.py` :: `compute_gawl_kernel_v2`; experiment scoring reuses `eval/el_perturbation.py` :: `score_sample_gawl` |
| UniSumEval labels | `merged_file2_processed.jsonl`; human score = `faithfulness_score`; domain = `source` |
| GovReport filter | same as `eval/evaluators_benchmark.py`: `summary_success_state==success` and `faithfulness_score != 1`, then `source==GovReport` |
| Pearson | `scipy.stats.pearsonr` (two-sided p-value), same family as `eval/evaluators_benchmark.py` |

## Independent variable

- `prefix_1024`: first window only. If the document already fits in one window, this **directly** calls `GraphBuilder.build_graph` on the existing dreeam article record.
- `sliding_window`: all windows; local graphs from sliced dreeam records; merge by global char offsets and canonical KB id (`wikidata_id`, else `wikipedia_id`).
- Summary (`candidate`) graph is rebuilt once from the same dreeam candidate and shared.

Note: `build/construct_graph.py` historically feeds **all** spaCy tokens in the dreeam JSON to GraphBuilder. This experiment's `prefix_1024` instead restricts the **source** graph to the first model-token window, so that coverage change can be attributed to truncation vs sliding windows. The default construct_graph entrypoint is unchanged.

## What this experiment does **not** do

- Does not re-run spaCy, BLINK, or DREEAM (those weights/code are not in the environment). Window graphs are sliced from precomputed dreeam JSON.
- Does not invent cross-window relations (`infer_cross_window_relations=false`).
- Does not change S²-K hyperparameters between conditions.
- Existing GovReport dreeam files contain full-document spaCy/BLINK but **zero** `ent_relation` rows; sliding windows therefore recover token/mention/entity coverage beyond 1,024 model tokens, not new DREEAM edges, unless a future live RE backend is plugged in.
- PEGASUS / RotatE paths in `config.py` are missing in this environment, so scoring uses `use_emb_labels=false` (same as `eval.el_perturbation` UniSumEval default). Absolute GAWL magnitudes can be numerically large; Pearson is still computed on the paired scores.

## Cache

`{cache_dir}/{sample_id}__{mode}__{config_hash}.json`. Hash includes window params, merge flags, tokenizer path, and GAWL weights.

## Tokenizer caveat

DREEAM typically tokenizes with RoBERTa-large; that tokenizer is not present locally. The run used the configured HF tokenizer (`t5-large` if RoBERTa is absent). Length control is still exact `len(tokenizer(window, add_special_tokens=True)) <= 1024` with **no** `truncation=True`. Replace `--tokenizer-path` with the real DREEAM tokenizer to match the original RE model exactly.
"""


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        ],
    )


def save_progress(path: Path, rows: list[dict]) -> None:
    slim = [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows]
    path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")


def run_consistency_check(cfg: LongContextConfig, tokenizer, n: int = 3) -> int:
    """短文档：sliding 应只有 1 个窗口，图计数应与 prefix / 原 GraphBuilder 一致。"""
    from long_context.tokenizer_util import count_tokens, effective_max_length

    labels = load_unisum_labels(Path(cfg.labels_jsonl))
    dreeam_dir = Path(cfg.dreeam_dir)
    max_len = effective_max_length(tokenizer, cfg.max_model_tokens)
    checked = 0
    failures = 0
    for item in labels:
        sid = str(item.get("doc_id"))
        path = dreeam_dir / f"{sid}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        article = data.get("article") or {}
        text = article.get("text") or ""
        if count_tokens(tokenizer, text, True) > max_len:
            continue
        base = build_document_graph(
            article, tokenizer, mode="prefix_1024",
            max_model_tokens=cfg.max_model_tokens, overlap_tokens=cfg.overlap_tokens,
            sentence_boundary=cfg.sentence_boundary,
            merge_entities_by_kb_id=cfg.merge_entities_by_kb_id,
            infer_cross_window_relations=False,
        )
        slid = build_document_graph(
            article, tokenizer, mode="sliding_window",
            max_model_tokens=cfg.max_model_tokens, overlap_tokens=cfg.overlap_tokens,
            sentence_boundary=cfg.sentence_boundary,
            merge_entities_by_kb_id=cfg.merge_entities_by_kb_id,
            infer_cross_window_relations=False,
        )
        ok = True
        msg = "ok"
        if base["num_windows"] != 1 or slid["num_windows"] != 1:
            ok, msg = False, f"windows base={base['num_windows']} slid={slid['num_windows']}"
        else:
            ok, msg = graphs_structurally_equal(base["graph_data"], slid["graph_data"], score_tol=cfg.score_tol)
        status = "PASS" if ok else "FAIL"
        print(f"[consistency] {sid} {status} {msg} tokens={graph_counts.__name__ if False else base.get('num_token_nodes')}")
        if not ok:
            failures += 1
        checked += 1
        if checked >= n:
            break
    if checked == 0:
        print("[consistency] 未找到短于 max_model_tokens 的真实样本；请依赖单元测试中的合成短文档。")
        return 0
    print(f"[consistency] checked={checked} failures={failures}")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sliding-window FK-Graph GovReport experiment")
    add_cli_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if getattr(args, "smoke", False) and cfg.limit is None:
        cfg.limit = 2
    out = Path(cfg.output_dir)
    _setup_logging(out)

    allow_mock = bool(getattr(args, "allow_mock_tokenizer", False))
    tokenizer = load_tokenizer(cfg.tokenizer_path, allow_mock=allow_mock)
    tok_note = (
        f"{tokenizer_fingerprint(tokenizer)}; "
        f"model_max_length={getattr(tokenizer, 'model_max_length', None)}; "
        f"effective cap={cfg.max_model_tokens}"
    )
    logger.info("tokenizer: %s", tok_note)
    if "t5" in tok_note.lower() or "mock" in tok_note.lower():
        logger.warning(
            "当前 tokenizer 不是 DREEAM/RoBERTa。窗口长度仍按真实 tokenizer 计算，"
            "但与原始关系抽取模型口径可能不一致。请在报告中保留此说明。"
        )

    if getattr(args, "consistency_check", False):
        return run_consistency_check(cfg, tokenizer)

    labels = load_unisum_labels(Path(cfg.labels_jsonl))
    samples, meta = iter_experiment_samples(
        labels,
        domain=cfg.domain,
        require_success=cfg.require_success,
        exclude_human1=cfg.exclude_human1,
        limit=cfg.limit,
    )
    n_original = meta["n_original_domain"]
    logger.info("domain=%s original=%s selected=%s", cfg.domain, n_original, len(samples))

    tmp_root = out / "tmp_gawl"
    tmp_root.mkdir(parents=True, exist_ok=True)
    progress_path = out / "sample_level_results.json"
    rows: list[dict] = []
    if progress_path.is_file():
        try:
            prev = json.loads(progress_path.read_text(encoding="utf-8"))
            done_ok = {str(r["sample_id"]) for r in prev if r.get("status") == "ok"}
            rows = [r for r in prev if r.get("status") == "ok"]
            samples = [s for s in samples if str(s.get("doc_id")) not in done_ok]
            logger.info("resume: keep %s finished, remaining %s", len(rows), len(samples))
        except json.JSONDecodeError:
            pass

    modes = ("prefix_1024", "sliding_window")
    if cfg.mode in {"prefix_1024", "sliding_window"}:
        # 配对实验仍跑两种模式；单独 mode 仅用于调试时可扩展
        modes = ("prefix_1024", "sliding_window")

    dreeam_dir = Path(cfg.dreeam_dir)
    for i, sample in enumerate(samples, start=1):
        logger.info("(%s/%s) sample %s", i, len(samples), sample.get("doc_id"))
        row = process_sample(sample, dreeam_dir, tokenizer, cfg, tmp_root, modes=modes)
        rows.append({k: v for k, v in row.items() if not str(k).startswith("_")})
        if i % 1 == 0:
            save_progress(progress_path, rows)

    failures = [r for r in rows if r.get("status") != "ok"]
    cfg_dump = cfg.to_dict()
    cfg_dump["tokenizer_note"] = tok_note
    write_outputs(
        out,
        cfg=cfg_dump,
        implementation_notes=IMPLEMENTATION_NOTES,
        sample_rows=rows,
        n_original=n_original,
        failures=failures,
        tokenizer_note=tok_note,
    )
    logger.info("wrote results to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
