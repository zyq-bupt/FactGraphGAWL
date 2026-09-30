#!/usr/bin/env bash
# 顺序跑权重组合（result-tag=2）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

build_cmd() {
  local wTT="$1" wTM="$2" wME="$3" wEE="$4"
  echo "python -m kernel.gawl --wTT $wTT --wTM $wTM --wME $wME --wEE $wEE --result-tag 2"
}

OUT_DIR="experiments/logs/run2"
OUT_CSV="experiments/results/results2.csv"
mkdir -p "$OUT_DIR" "$(dirname "$OUT_CSV")"

if [[ ! -f "$OUT_CSV" ]]; then
  echo "time,wTT,wTM,wME,wEE,raw_output_file" > "$OUT_CSV"
fi

run_once () {
  local wTT="$1" wTM="$2" wME="$3" wEE="$4"
  local tag="TT${wTT}_TM${wTM}_ME${wME}_EE${wEE}"
  local logf="${OUT_DIR}/${tag}.log"

  local cmd
  cmd="$(build_cmd "$wTT" "$wTM" "$wME" "$wEE")"
  echo "----------------------------------------"
  echo "[RUN] $cmd"
  echo "----------------------------------------"

  if bash -lc "cd '$ROOT' && export PYTHONPATH='$ROOT'\${PYTHONPATH:+:\$PYTHONPATH} && $cmd" | tee "$logf" ; then
    :
  else
    echo "[WARN] 运行失败，但继续后续组合。"
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S'),$wTT,$wTM,$wME,$wEE,$logf" >> "$OUT_CSV"
}


run_once  1.5   2.5   0   0.25
run_once  1.5   2.5   0   0.5
run_once  1.5   2.5   0.25   0.25
run_once  1.5   2.5   0.25   0.5
run_once  1.75   2   0   0.25
run_once  1.75   2   0   0.5
run_once  1.75   2   0.25   0.25
run_once  1.75   2   0.25   0.5
echo
echo "全部跑完。结果索引：$OUT_CSV"
echo "   原始输出：$OUT_DIR/*.log"
