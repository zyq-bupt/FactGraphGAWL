#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 冒烟（1–2 个 GovReport 样本）
# python scripts/run_long_context.py --smoke --limit 2

# 短文档一致性
# python scripts/run_long_context.py --consistency-check

python scripts/run_long_context.py \
  --domain GovReport \
  --long-context-mode both \
  --max-model-tokens 1024 \
  --overlap-tokens 256 \
  --output-dir experiments/results/long_context_govreport
