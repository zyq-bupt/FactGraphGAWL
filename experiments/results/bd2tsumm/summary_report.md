# BD2TSumm: CompKG-FactEval vs ROUGE

## Data
- Events / systems / valid event–system pairs: 13 / 2 / 26
- Systems evaluated: pegasus-large, t5-large
- Candidate summaries: 26
- Reference summaries: 13

## Tweet aggregation
- Dedup exact duplicates: `True`; URL/mention placeholders: `plain`
- Hashtags kept as text: `True`; strip RT prefix: `False`
- Long context mode: `sliding_window` (max=1024, overlap=256)
- Silent prefix truncation is disabled unless `allow_prefix_truncation` is true.

## CompKG-FactEval
- Variant: `S2-K_PT` (S²-K (PT) = `kernel.gawl.compute_gawl_kernel_v2`)
- Edge weights wTT/wTM/wME/wEE = 1.0/1.0/0.5/0.0; T=1
- use_emb_labels=False; higher_is_better=True; treat_as_distance=False
- Backend: `real`

## ROUGE
- Package: `rouge-score`; stemming=True
- Multi-reference aggregation: `mean` (no official BD2TSumm ROUGE script found)
- Reported values are F1, not precision/recall.

## Success / failure
- Union keys: 26; valid complete-case: 26; excluded: 0
- CompKG-only keys: []
- ROUGE-only keys: []
- Many-to-many joins: 0

## Correlations

| level | ROUGE | method | r | p | n | status |
|---|---|---|---:|---:|---:|---|
| sample | rouge1_f1 | pearson | 0.5777 | 0.0020 | 26 | ok |
| sample | rouge1_f1 | spearman | 0.5897 | 0.0015 | 26 | ok |
| sample | rouge1_f1 | kendall_tau_b | 0.4092 | 0.0030 | 26 | ok |
| sample | rouge2_f1 | pearson | 0.3910 | 0.0482 | 26 | ok |
| sample | rouge2_f1 | spearman | 0.3826 | 0.0537 | 26 | ok |
| sample | rouge2_f1 | kendall_tau_b | 0.2554 | 0.0704 | 26 | ok |
| sample | rougeL_f1 | pearson | 0.4777 | 0.0136 | 26 | ok |
| sample | rougeL_f1 | spearman | 0.5118 | 0.0075 | 26 | ok |
| sample | rougeL_f1 | kendall_tau_b | 0.3538 | 0.0109 | 26 | ok |
| event_centered | rouge1_f1 | pearson | 0.6871 | 0.0001 | 26 | ok |
| event_centered | rouge1_f1 | spearman | 0.7340 | 0.0000 | 26 | ok |
| event_centered | rouge1_f1 | kendall_tau_b | 0.5385 | 0.0001 | 26 | ok |
| event_centered | rouge2_f1 | pearson | 0.5592 | 0.0030 | 26 | ok |
| event_centered | rouge2_f1 | spearman | 0.6568 | 0.0003 | 26 | ok |
| event_centered | rouge2_f1 | kendall_tau_b | 0.4462 | 0.0011 | 26 | ok |
| event_centered | rougeL_f1 | pearson | 0.6813 | 0.0001 | 26 | ok |
| event_centered | rougeL_f1 | spearman | 0.7983 | 0.0000 | 26 | ok |
| event_centered | rougeL_f1 | kendall_tau_b | 0.5877 | 0.0000 | 26 | ok |
| within_event_summary | rouge1_f1 | spearman | NA | NA | 0 | insufficient_data |
| within_event_summary | rouge1_f1 | kendall_tau_b | NA | NA | 0 | insufficient_data |
| within_event_summary | rouge2_f1 | spearman | NA | NA | 0 | insufficient_data |
| within_event_summary | rouge2_f1 | kendall_tau_b | NA | NA | 0 | insufficient_data |
| within_event_summary | rougeL_f1 | spearman | NA | NA | 0 | insufficient_data |
| within_event_summary | rougeL_f1 | kendall_tau_b | NA | NA | 0 | insufficient_data |
| system | rouge1_f1 | pearson | NA | NA | 2 | insufficient_data |
| system | rouge1_f1 | spearman | NA | NA | 2 | insufficient_data |
| system | rouge1_f1 | kendall_tau_b | NA | NA | 2 | insufficient_data |
| system | rouge2_f1 | pearson | NA | NA | 2 | insufficient_data |
| system | rouge2_f1 | spearman | NA | NA | 2 | insufficient_data |
| system | rouge2_f1 | kendall_tau_b | NA | NA | 2 | insufficient_data |
| system | rougeL_f1 | pearson | NA | NA | 2 | insufficient_data |
| system | rougeL_f1 | spearman | NA | NA | 2 | insufficient_data |
| system | rougeL_f1 | kendall_tau_b | NA | NA | 2 | insufficient_data |
| system_bootstrap | rouge1_f1 | pearson | NA | NA | 13 | insufficient_data |
| system_bootstrap | rouge1_f1 | spearman | NA | NA | 13 | insufficient_data |
| system_bootstrap | rouge1_f1 | kendall_tau_b | NA | NA | 13 | insufficient_data |
| system_bootstrap | rouge2_f1 | pearson | NA | NA | 13 | insufficient_data |
| system_bootstrap | rouge2_f1 | spearman | NA | NA | 13 | insufficient_data |
| system_bootstrap | rouge2_f1 | kendall_tau_b | NA | NA | 13 | insufficient_data |
| system_bootstrap | rougeL_f1 | pearson | NA | NA | 13 | insufficient_data |
| system_bootstrap | rougeL_f1 | spearman | NA | NA | 13 | insufficient_data |
| system_bootstrap | rougeL_f1 | kendall_tau_b | NA | NA | 13 | insufficient_data |

## System means and ranks (main ranking = consistent event coverage)

| system | n_events | coverage | CompKG mean | CompKG rank | R-1 mean | R-1 rank | R-2 mean | R-2 rank | R-L mean | R-L rank | in_main |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| pegasus-large | 13 | 1.000 | 0.0103 | 1.00 | 0.2495 | 1.00 | 0.0731 | 1.00 | 0.1440 | 1.00 | True |
| t5-large | 13 | 1.000 | 0.0050 | 2.00 | 0.1673 | 2.00 | 0.0465 | 2.00 | 0.1042 | 2.00 | True |

## Ranking agreement

- vs rouge1: Spearman=1.0000, Kendall τb=1.0000, mean |Δrank|=0.0000, Top-1 match=True, Top-3 Jaccard=NA
- vs rouge2: Spearman=1.0000, Kendall τb=1.0000, mean |Δrank|=0.0000, Top-1 match=True, Top-3 Jaccard=NA
- vs rougeL: Spearman=1.0000, Kendall τb=1.0000, mean |Δrank|=0.0000, Top-1 match=True, Top-3 Jaccard=NA

## Figures
- `figures/scatter_compkg_vs_rouge.png`
- `figures/ranking_comparison.png`
- `figures/system_coverage.png`

## Conclusion

ROUGE measures n-gram overlap between a candidate summary and the human reference summary. CompKG-FactEval measures factual consistency between the candidate summary and the source tweet set via FK-Graph similarity (S²-K / GAWL). Agreement in correlation or system ranking is supplementary evidence that the two metrics sometimes order systems similarly; it does not by itself prove that they measure the same capability. Scores used in this report are original (not min-max normalized).
