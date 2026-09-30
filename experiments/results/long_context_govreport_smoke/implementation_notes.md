# Implementation notes

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

## What this experiment does **not** do

- Does not re-run spaCy, BLINK, or DREEAM (those weights/code are not in the environment).
- Does not invent cross-window relations (`infer_cross_window_relations=false`).
- Does not change S²-K hyperparameters between conditions.
- Existing GovReport dreeam files contain full-document spaCy/BLINK but **zero** `ent_relation` rows; sliding windows therefore recover token/mention/entity coverage beyond 1,024 model tokens, not new DREEAM edges, unless a future live RE backend is plugged in.

## Cache

`{cache_dir}/{sample_id}__{mode}__{config_hash}.json`. Hash includes window params, merge flags, tokenizer path, and GAWL weights.
