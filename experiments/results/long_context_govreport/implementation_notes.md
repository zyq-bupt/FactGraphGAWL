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
