#!/usr/bin/env bash
# 顺序跑权重组合： 
# 1) (1,1,1,1)
# 2) (1,1,0,1)
# 3) (1,1,0,0)
# 4) (2,1,0,0)
# 5) (3,1,0,0)
# 6) (4,1,0,0)
# 7) (3,0.5,0,0)
# 8) (3,1.5,0,0)

set -euo pipefail

# =============== 你可能需要改这里 ===============
# 如果 gawl.py 接口不同，请改 RUN_CMD 这一行的构造。
# 例如你可以改成创建 weights.json 再让 gawl.py 读取。
build_cmd() {
  local wTT="$1" wTM="$2" wME="$3" wEE="$4"
  # 例：gawl.py 支持 --wTT/--wTM/--wME/--wEE
  echo "python gawl.py --wTT $wTT --wTM $wTM --wME $wME --wEE $wEE"
}
#python gawl.py --wTT 1 --wTM 1 --wME 1 --wEE 1
# =============== 以上可按需修改 ==================

OUT_DIR="runs_logs"
OUT_CSV="results.csv"
mkdir -p "$OUT_DIR"

# 结果表头
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

  # 执行并保存原始输出
  if bash -lc "$cmd" | tee "$logf" ; then
    :
  else
    echo "[WARN] 运行失败，但继续后续组合。"
  fi

  # 记录一行到 CSV
  echo "$(date '+%Y-%m-%d %H:%M:%S'),$wTT,$wTM,$wME,$wEE,$logf" >> "$OUT_CSV"
}

# 8 组顺序跑
run_once 1   0   1   1
run_once 0   1   1   1
run_once 1   1   0   0
run_once 5   1   0   0
run_once 0.25   1   1   1
run_once 0.5   1   1   1
run_once 0.75   1   1   1
run_once 1.25   1   1   1
run_once 1.5   1   1   1
run_once 1.75  1   1   1
run_once 2   1   1   1
run_once 1   0.25   1   1
run_once 1   0.5   1   1
run_once 1   0.75   1   1
run_once 1   1.25   1   1
run_once 1   1.5   1   1
run_once 1   1.75   1   1
run_once 1   2   1   1
run_once 1   1   0.25   1
run_once 1   1   0.5   1
run_once 1   1   0.75   1
run_once 1   1   1.25   1
run_once 1   1   1.5   1
run_once 1   1   1.75   1
run_once 1   1   2   1
run_once 1   1   1   0.25
run_once 1   1   1   0.75
run_once 1   1   1   0.5
run_once 1   1   1   1.25
run_once 1   1   1   1.5
run_once 1   1   1   1.75
run_once 1   1   1   2

echo
echo "✅ 全部跑完。结果索引：$OUT_CSV"
echo "   原始输出：$OUT_DIR/*.log"
