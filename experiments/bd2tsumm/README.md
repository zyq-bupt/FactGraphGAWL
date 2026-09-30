# BD2TSumm：CompKG-FactEval vs ROUGE

独立评估流程：把同一灾害事件的推文聚合成 FK-Graph 源文本，计算各摘要系统的 CompKG-FactEval（S²-K PT）分数，并用人工参考摘要计算 ROUGE，再比较相关性和系统排名。

CompKG 使用源推文 \(D\) 与候选摘要 \(S\)，**不使用**参考摘要 \(G\)。ROUGE 使用 \(S\) 与 \(G\)。

## 现有数据里有什么

解压后的 BD2TSumm（默认 `dataset.data_root`）目前只有：

- `Input_datasets/*_input_data.csv`：推文，正文列名为 `text`
- `Gold_summary/Summary_*.txt`：每个事件 **一条** 人工参考摘要

**没有**摘要系统输出。未提供 `candidates_file` 时，程序会报错退出，**不会**把人工摘要当作候选。

候选摘要 CSV 示例：

```text
event_id,system_name,candidate_summary
CyclonePam,MySystem,"24 dead, 3300 displaced in Vanuatu..."
```

`event_id` 必须与 `config.yaml` 里 `dataset.events` 的 ID 一致（如 `CyclonePam`、`USFlood`）。

## 依赖

- 已有：numpy、scipy、pyyaml、matplotlib、networkx
- ROUGE：`pip install rouge-score`
- 真实 CompKG：需要 DREEAM 格式的源/摘要 JSON（与现有 `factgraph` 输入相同），以及 tokenizer 目录

## 命令

仓库根目录：

```bash
# toy（不加载 DREEAM/BLINK；CompKG 用词 Jaccard mock）
python experiments/bd2tsumm/run_all.py \
  --config tests/fixtures/bd2tsumm/config.yaml \
  --output-dir experiments/results/bd2tsumm_toy \
  --seed 42 \
  --mock-compkg \
  --allow-mock-tokenizer

# 只聚合推文（不要求系统输出）
python experiments/bd2tsumm/prepare_sources.py \
  --config experiments/bd2tsumm/config.yaml \
  --output-dir experiments/results/bd2tsumm

# 完整真实实验（需 candidates + DREEAM）
python experiments/bd2tsumm/run_all.py \
  --config experiments/bd2tsumm/config.yaml \
  --output-dir experiments/results/bd2tsumm \
  --seed 42 \
  --candidates-file /path/to/system_outputs.csv \
  --dreeam-source-dir /path/to/dreeam/sources \
  --dreeam-candidate-dir /path/to/dreeam/candidates

# 分阶段 / 续跑
python experiments/bd2tsumm/run_all.py --stages prepare --output-dir experiments/results/bd2tsumm
python experiments/bd2tsumm/run_all.py --stages rouge,analyze --resume --output-dir experiments/results/bd2tsumm \
  --candidates-file /path/to/system_outputs.csv --mock-compkg
```

`compkg.dreeam_source_dir`：`{event_id}.json`，内含 `article`（parse_trees / mentions / vertexSet 等）。  
`compkg.dreeam_candidate_dir`：`{event_id}__{system_name}.json`，内含 `candidate`。

源图按事件缓存，同一事件的多个系统会复用。

## 默认方法

- CompKG：S²-K (PT)，`long_context.scoring.score_article_candidate` → `kernel.gawl.compute_gawl_kernel_v2`
- 边权：wTT=1.0, wTM=1.0, wME=0.5, wEE=0.0，T=1（与 `python -m kernel.gawl` 默认一致）
- 长文本：滑动窗口 1024 / overlap 256，合并已有实现；默认禁止截断为前 1024 tokens
- ROUGE：`rouge-score`，英语 stemming，报告 F1；无官方多参考脚本时对参考级 F1 取平均

## 测试

```bash
python tests/test_bd2tsumm.py
```

## 主要输出

| 文件 | 含义 |
|---|---|
| `prepared_sources.jsonl` | 事件级源文本与窗口统计 |
| `compkg_scores.csv` | 事件–系统 CompKG |
| `rouge_scores.csv` | 事件–系统 ROUGE F1 |
| `all_sample_scores.csv` | 一对一合并 |
| `correlation_results.csv` | 样本级 / 事件中心化 / 系统级相关 |
| `system_ranking.csv` | 系统均分与排名 |
| `summary_report.md` | 自动报告 |
| `run_metadata.json` | 种子、git、依赖版本 |
| `cache/` | 源图 / 摘要图缓存 |
