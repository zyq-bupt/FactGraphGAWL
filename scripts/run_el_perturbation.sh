#!/usr/bin/env bash
# 实体链接扰动实验。默认先建议 --limit 冒烟，再去掉 --limit 跑全量。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python scripts/run_el_perturbation.py "$@"
