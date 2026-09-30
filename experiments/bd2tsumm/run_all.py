"""BD2TSumm 全流程入口：prepare → rouge → compkg → analyze，支持断点续跑。"""

from __future__ import annotations

import argparse
import logging
import time
from copy import deepcopy
from pathlib import Path

from util import (
    MissingCandidatesError,
    REPO_ROOT,
    collect_run_metadata,
    deep_set,
    file_exists_nonempty,
    load_config,
    set_seed,
    setup_logging,
    sha256_file,
    write_json,
)

from analyze_results import run_analyze
from compute_compkg import run_compkg
from compute_rouge import run_rouge
from prepare_sources import run_prepare

STAGE_FILES = {
    "prepare": ["prepared_sources.jsonl", "data_validation.json"],
    "rouge": ["rouge_scores.csv", "rouge_reference_level.csv"],
    "compkg": ["compkg_scores.csv"],
    "analyze": ["all_sample_scores.csv", "summary_report.md"],
}


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = deepcopy(cfg)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.candidates_file:
        deep_set(cfg, "dataset.candidates_file", args.candidates_file)
    if args.candidates_dir:
        deep_set(cfg, "dataset.candidates_dir", args.candidates_dir)
    if args.data_root:
        deep_set(cfg, "dataset.data_root", args.data_root)
    if args.mock_compkg:
        deep_set(cfg, "compkg.backend", "mock")
    if args.allow_mock_tokenizer:
        deep_set(cfg, "long_context.allow_mock_tokenizer", True)
    if args.dreeam_source_dir:
        deep_set(cfg, "compkg.dreeam_source_dir", args.dreeam_source_dir)
    if args.dreeam_candidate_dir:
        deep_set(cfg, "compkg.dreeam_candidate_dir", args.dreeam_candidate_dir)
    if args.tokenizer_path:
        deep_set(cfg, "long_context.tokenizer_path", args.tokenizer_path)
    return cfg


def stage_done(output_dir: Path, stage: str, names: dict) -> bool:
    mapping = {
        "prepare": [names.get("prepared_sources", "prepared_sources.jsonl"), names.get("data_validation", "data_validation.json")],
        "rouge": [names.get("rouge_scores", "rouge_scores.csv")],
        "compkg": [names.get("compkg_scores", "compkg_scores.csv")],
        "analyze": [names.get("summary_report", "summary_report.md"), names.get("all_sample_scores", "all_sample_scores.csv")],
    }
    return all(file_exists_nonempty(output_dir / f) for f in mapping[stage])


def run(cfg: dict, output_dir: Path, logger: logging.Logger, *, stages: list[str], resume: bool) -> None:
    names = cfg.get("output") or {}
    t0 = time.perf_counter()
    if "prepare" in stages:
        if resume and stage_done(output_dir, "prepare", names):
            logger.info("跳过 prepare（已有输出）")
        else:
            need_cand = any(s in stages for s in ("rouge", "compkg", "analyze"))
            run_prepare(cfg, output_dir, logger, require_candidates=need_cand)
    if "rouge" in stages:
        if resume and stage_done(output_dir, "rouge", names):
            logger.info("跳过 rouge（已有输出）")
        else:
            run_rouge(cfg, output_dir, logger)
    if "compkg" in stages:
        if resume and stage_done(output_dir, "compkg", names):
            logger.info("跳过 compkg（已有输出）")
        else:
            run_compkg(cfg, output_dir, logger)
    if "analyze" in stages:
        if resume and stage_done(output_dir, "analyze", names):
            logger.info("跳过 analyze（已有输出）")
        else:
            run_analyze(cfg, output_dir, logger)
    elapsed = time.perf_counter() - t0
    meta = collect_run_metadata(
        cfg,
        extra={
            "output_dir": str(output_dir),
            "stages": stages,
            "resume": resume,
            "elapsed_seconds": elapsed,
            "config_sha256": None,
        },
    )
    write_json(output_dir / names.get("run_metadata", "run_metadata.json"), meta)
    write_json(output_dir / "resolved_config.yaml.json", cfg)
    logger.info("全部完成 elapsed=%.2fs stages=%s", elapsed, stages)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BD2TSumm CompKG vs ROUGE 全流程")
    p.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "experiments" / "results" / "bd2tsumm"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stages", type=str, default="prepare,rouge,compkg,analyze")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--candidates-file", type=str, default=None)
    p.add_argument("--candidates-dir", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--mock-compkg", action="store_true")
    p.add_argument("--allow-mock-tokenizer", action="store_true")
    p.add_argument("--dreeam-source-dir", type=str, default=None)
    p.add_argument("--dreeam-candidate-dir", type=str, default=None)
    p.add_argument("--tokenizer-path", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "logs" / "run_all.log")
    seed = int(cfg.get("seed") or 42)
    set_seed(seed)
    logger.info("seed=%s config=%s output=%s", seed, args.config, output_dir)
    cfg_hash = sha256_file(Path(args.config))
    logger.info("config_sha256=%s", cfg_hash)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGE_FILES)
    if unknown:
        raise ValueError(f"未知阶段: {unknown}")
    try:
        run(cfg, output_dir, logger, stages=stages, resume=args.resume)
    except MissingCandidatesError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
