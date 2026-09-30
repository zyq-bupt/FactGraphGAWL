# FactGraphGAWL

事实图构建与 GAWL 方法改进，用于摘要事实性评估。

## 目录结构

```text
FactGraphGAWL/
├── config.py                 # 数据与模型路径配置
├── common/                   # 共享 IO / 工具
├── build/                    # 建图与节点 embedding
│   └── graph_embedding/
├── kernel/                   # GAWL 核、边权重、图格式转换
├── eval/                     # 分数合并与相关性评估
├── baselines/                # LLM 基线（Direct-LLM / KP-LLM）
│   ├── llm_direct/           # ChatGPT-ZS/Star, ERNIE-ZS/Star
│   └── kp_llm/               # T5 关键词 + LLM 逐项核验
├── configs/                  # direct_llm.yaml / kp_llm.yaml
├── scripts/                  # 运行与参数扫描脚本
├── notebooks/                # 分析与画图
├── experiments/
│   ├── logs/                 # 扫描运行日志（gitignore）
│   ├── results/              # results*.csv, PTresult*.txt, llm/kp 输出
│   └── figures/
├── data/
│   ├── samples/
│   └── debug/
├── tests/
└── archive/                  # 旧副本与临时脚本（不参与主流程）
```

## 流水线概览

1. **建图** — `python -m build.construct_graph`  
   输入：`dreeam_result/` → 输出：`factgraph_result/`（路径见 `config.py`）

2. **节点 embedding** — `python -m build.get_token_EMB`  
   使用 Pegasus-XSum，为节点写入上下文 embedding。

3. **图相似度（GAWL）** — `python -m kernel.gawl`  
   比较原文图与摘要图，输出相似度；DeFacto 上统计内在/外在错误判别。

4. **合并与评测** — `python -m eval.edit2`、`python -m eval.evaluators_benchmark`

一键串联（norm 变体）：

```bash
bash scripts/run.sh
```

## GAWL 改进要点

相对原始 GAWL（[Nikolentzos & Vazirgiannis, 2023](https://github.com/giannisnik/gawl)）：

1. **节点标签**：用单词替代度数（`compute_gawl_kernel`）
2. **小图标签频率**：直接使用小图频率
3. **边类型加权**（关键，`compute_gawl_kernel_v2`）

| 大类 | 含义 | 典型关系 |
|------|------|----------|
| token-token | 依存 | nsubj, obj |
| token-mention | token–提及 | RELATED_TO |
| mention-entity | 提及–实体 | BELONGS_TO |
| entity-entity | 实体关系 | P17, P27 |

命令行权重与结果标签：

```bash
python -m kernel.gawl --wTT 1.5 --wTM 2.25 --wME 0.25 --wEE 0.5 --result-tag 1
# 结果追加到 experiments/results/PTresult_1.txt
```

参数说明：

```python
use_node_labels = True   # 单词作为节点 label
use_edge_labels = 2      # 0: 无边权; 1: 小类; 2: 大类
```

## 参数扫描

```bash
bash scripts/run_params.sh    # → experiments/logs/run1, results.csv, --result-tag 1
bash scripts/run_params_2.sh  # → run2 / results2.csv / tag 2
bash scripts/run_params_3.sh  # → run3 / results3.csv / tag 3
```

脚本会自动将仓库根加入 `PYTHONPATH`，请在任意目录执行均可。

## LLM 基线（Direct-LLM + KP-LLM）

新增五个事实一致性评价基线，**不改动**现有 GAWL / FactGraph 流程。

| 方法 | 说明 |
|------|------|
| ChatGPT-ZS | 二分类 yes/no（Luo et al., 2023 prompt） |
| ChatGPT-Star | 1–5 星评分（Wang et al., 2023 prompt） |
| ERNIE-ZS / ERNIE-Star | 与上相同 prompt，后端换 ERNIE |
| KP-LLM | FLAN-T5 抽关键词 → LLM 逐项核验 → 支持率聚合（见 `KP-LLM_baseline_implementation.md`） |

环境变量：

```bash
# ChatGPT / KP-LLM 检查器：默认走智增增 https://api.zhizengzeng.com/v1
export OPENAI_API_KEY=你的智增增key
# 可选覆盖：export OPENAI_BASE_URL=https://api.zhizengzeng.com/v1
# 可选：export CHATGPT_MODEL=gpt-3.5-turbo

# ERNIE 也默认走智增增，模型默认 ernie-3.5-128k
# 可选：export ERNIE_MODEL=ernie-3.5-128k
# 若单独设了 ERNIE_API_KEY 则优先用它，否则复用 OPENAI_API_KEY
```

先用少量样本冒烟（推荐 `--limit 2`）：

```bash
python scripts/run_direct_llm.py \
  --method ChatGPT-ZS \
  --input data/samples/updated_processed_with_factgraph_filled.jsonl \
  --limit 2

python scripts/run_direct_llm.py --method ChatGPT-Star --input ... --limit 2
python scripts/run_direct_llm.py --method ERNIE-ZS --input ... --limit 2
python scripts/run_direct_llm.py --method ERNIE-Star --input ... --limit 2

python scripts/run_kp_llm.py \
  --input data/samples/updated_processed_with_factgraph_filled.jsonl \
  --output experiments/results/kp_llm/predictions_sample.jsonl \
  --limit 2
```

评价（与人工分相关 / 二分类指标）：

```bash
python scripts/evaluate_llm_baselines.py \
  --predictions experiments/results/llm_baselines/predictions_ChatGPT-ZS.jsonl \
  --labels data/samples/updated_processed_with_factgraph_filled.jsonl \
  --output experiments/results/llm_baselines/metrics_ChatGPT-ZS.json
```

无 API 的单元测试：

```bash
python -c "from tests.test_llm_baselines import *; test_parse_zs_yes_no(); test_parse_star(); test_normalize_keyphrases(); test_align_exact_and_overlap(); test_aggregate_support_rate(); test_split_sentences_nonempty(); print('ok')"
```

配置见 `configs/direct_llm.yaml`、`configs/kp_llm.yaml`。

## 数据

- **Defacto / UniSumEval 主数据**：路径在 `config.py`（默认 `/root/autodl-fs/zyq/...`）
- 百度网盘备份：https://pan.baidu.com/s/1j7Z4mjbtc3Me1BKXxxSekw 提取码 `2tns`
- 仓库内样例：`data/samples/updated_processed_with_factgraph_filled.jsonl`

注意不同数据集键名：

```python
keys = ["article", "candidate", "humman_summary"]  # DeFacto
# UniSum 键名不同，见代码
```

