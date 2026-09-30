# Sliding-window FK-Graph on UniSumEval GovReport
All numbers below are computed from the actual run. None are placeholders.
## Setup
- Domain filter: `GovReport` with UniSumEval rules `summary_success_state==success` and `faithfulness_score!=1`.
- Original domain samples: **225**. Paired successful samples: **22**. Failures: **0**.
- Tokenizer: /root/autodl-fs/zyq/models/t5-large; model_max_length=1000000000000000019884624838656; effective cap=1024
- Window: max_model_tokens=1024, overlap_tokens=256, sentence_boundary=True.
- Summary-side FK-Graph is identical across conditions (built from the same dreeam `candidate`).
- S²-K (PT) is `kernel.gawl.compute_gawl_kernel_v2` via `eval.el_perturbation.score_sample_gawl`.
- Entity nodes / entity-relation edges / time in the table are **means** over paired samples; medians are listed in aggregate_metrics.json.
- Token coverage uses the same tokenizer as windowing; special tokens excluded from the ratio; overlapping windows count each content token at most once; ratio is capped at 1.
- Entity-relation edges (E_ee) are taken from precomputed DREEAM `ent_relation`; on this GovReport slice they are all zero, so sliding windows do not add new relations without re-running DREEAM.

## Paper table
| Graph construction | Token coverage (%) | Entity nodes | Entity-relation edges | Pearson r | p-value | Time/sample |
|---|---:|---:|---:|---:|---:|---:|
| Prefix truncation | 15.12 | 11.45 | 0.00 | 0.0348 | 0.8778 | 0.870 |
| Sliding window | 100.00 | 49.59 | 0.00 | 0.2017 | 0.3680 | 3.701 |

## Correlation comparison
- delta_r = r_sliding - r_baseline = **0.1669**
- Paired bootstrap 95% percentile CI: [-0.3264, 0.5349] (valid resamples=10000, skipped=0)
- Runtime multiplier (mean total time): 4.255

## Wording (case B)

The sliding-window strategy substantially increases graph coverage and yields a modest correlation improvement. However, the uncertainty of the observed gain indicates that larger-scale evaluation and explicit cross-window relation modeling are still needed.

## Limitations

- Cross-window entity relations are not inferred; only relations observed inside a window are kept.
- Conflicting or inconsistent entity links across windows may remain after exact KB-id merge.
- Sliding windows increase graph size and runtime.
- This is a preliminary study on UniSumEval GovReport only.
- Legal and scientific long documents still require separate evaluation.

## Failures
None.

Do not interpret a significant sliding-window Pearson and a non-significant baseline Pearson as a significant difference between methods; use delta_r and its CI.
