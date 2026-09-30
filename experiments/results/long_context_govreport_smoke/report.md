# Sliding-window FK-Graph on UniSumEval GovReport
All numbers below are computed from the actual run. None are placeholders.
## Setup
- Domain filter: `GovReport` with UniSumEval rules `summary_success_state==success` and `faithfulness_score!=1`.
- Original domain samples: **225**. Paired successful samples: **1**. Failures: **0**.
- Tokenizer: /root/autodl-fs/zyq/models/t5-large; model_max_length=1000000000000000019884624838656; effective cap=1024
- Window: max_model_tokens=1024, overlap_tokens=256, sentence_boundary=True.
- Summary-side FK-Graph is identical across conditions (built from the same dreeam `candidate`).
- S²-K (PT) is `kernel.gawl.compute_gawl_kernel_v2` via `eval.el_perturbation.score_sample_gawl`.
- Entity nodes / entity-relation edges / time in the table are **means** over paired samples; medians are listed in aggregate_metrics.json.
- Token coverage uses the same tokenizer as windowing; special tokens excluded from the ratio; overlapping windows count each content token at most once; ratio is capped at 1.

## Paper table
| Graph construction | Token coverage (%) | Entity nodes | Entity-relation edges | Pearson r | p-value | Time/sample |
|---|---:|---:|---:|---:|---:|---:|
| Prefix truncation | 18.59 | 13.00 | 0.00 | NA | NA | 0.587 |
| Sliding window | 100.00 | 45.00 | 0.00 | NA | NA | 1.940 |

## Correlation comparison
- delta_r = r_sliding - r_baseline = **NA**
- Paired bootstrap 95% percentile CI: [NA, NA] (valid resamples=0, skipped=10000)
- Runtime multiplier (mean total time): 3.304

## Wording (case insufficient_stats)

Correlation statistics could not be computed on the paired sample (insufficient variance or too few samples). Coverage and cost figures below are still based on actual runs.

## Limitations

- Cross-window entity relations are not inferred; only relations observed inside a window are kept.
- Conflicting or inconsistent entity links across windows may remain after exact KB-id merge.
- Sliding windows increase graph size and runtime.
- This is a preliminary study on UniSumEval GovReport only.
- Legal and scientific long documents still require separate evaluation.

## Failures
None.

Do not interpret a significant sliding-window Pearson and a non-significant baseline Pearson as a significant difference between methods; use delta_r and its CI.
