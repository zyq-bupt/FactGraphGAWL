# Entity-Linking Perturbation Experiment Report (FactGraph + GAWL)

Please analyze this ablation experiment for a paper on **fact-graph based summary factuality evaluation**. I need help with: (1) how to interpret the numbers, (2) what claim this experiment can/cannot support, (3) how to write it in a paper (Results + Discussion), (4) limitations and reviewer-style critiques.

Do **not** invent numbers. Use only what is below.

---

## 1. Research context

We evaluate summary factuality by building a **fact graph** from source and summary, then computing **GAWL graph similarity**.

Graph node types: Token, Mention, Entity.  
Edge major types:

- token–token (dependency)
- token–mention
- mention–entity (BELONGS_TO; from entity linking)
- entity–entity (Wikidata relations)

**Motivation for this experiment:** a reviewer / analysis question is whether performance depends on a strong entity linker. We do **not** retrain any model and do **not** replace the linker. We only randomly drop already-successful entity links on the **already-built graphs**, then re-run GAWL scoring.

This is a **controlled EL-recall perturbation**, not an end-to-end re-EL experiment.

A related previous ablation (different protocol) removed **entire edge types**:

| Removed type | DeFacto n_correct |
|---|---|
| None (full graph) | 1197 |
| token–token | 1006 |
| token–mention | 1125 |
| mention–entity | 1216 |
| entity–entity | 1192 |

Note: that 1197 baseline used **different GAWL edge weights** than the present experiment (see §3). Do not mix the two tables as if they were the same run.

---

## 2. Perturbation protocol

**Successful entity link** = an Entity node with a non-empty `wikidata_id`.

For drop ratio \(r \in \{0, 0.1, 0.2, 0.3, 0.5\}\):

1. In each subgraph independently (article / candidate / human summary), collect successful Entity nodes.
2. Drop \(k = \mathrm{round\_half\_up}(n \cdot r)\) of them uniformly at random (no replacement).
3. Also delete:
   - the entity node
   - incident mention–entity edges
   - incident entity–entity edges
4. **Keep** Mention nodes, Token nodes, and token–mention edges.  
   Intended semantics: NER still found the mention, but linking failed.

Randomness:

- 0%: 1 run (deterministic)
- other ratios: 5 repeats, seeds 42–46
- report mean ± std (sample std, ddof=1)

No empty-graph skips and no evaluation errors in the recorded runs.

---

## 3. Scoring setup (GAWL)

GAWL Weisfeiler–Lehman iterations: **T = 1**  
Edge-type weights used in **this** experiment:

- w(token–token) = 1.0
- w(token–mention) = 1.0
- w(mention–entity) = 0.5
- w(entity–entity) = **0.0**

Important: entity–entity edges have weight 0 in the kernel, but dropping entities can still change mention WL labels / graph structure, so the perturbation is not a no-op.

DeFacto graphs: with node embeddings.  
UniSumEval graphs: **without** node embeddings (this is how that dataset was stored).

---

## 4. Metrics

### DeFacto (pairwise ranking / error detection)

- 1814 graph files (test+val+train)
- Evaluated only on **pure intrinsic XOR pure extrinsic** errors: **n_eval = 1570**
  - intrinsic-only: 315
  - extrinsic-only: 1255
- For each document: compare \(K(\text{article}, \text{human})\) vs \(K(\text{article}, \text{candidate})\)
- **Correct** iff \(K_{\text{human}} > K_{\text{candidate}}\) (strict)
- **Tie** iff equal
- **Accuracy** = n_correct / 1570
- **FNR** = 1 − accuracy  
  Interpretation used in the code: failing to strictly rank the erroneous candidate below the human summary. **Ties count as false negatives.**

### UniSumEval (correlation)

- Pearson / Spearman vs human `faithfulness_score`
- Same filter as the project’s `evaluators_benchmark`:
  - `summary_success_state == success`
  - `faithfulness_score != 1`
- **n_used = 441** in every run
- Among the 2025 jsonl rows: 197 skipped for non-success, 1387 skipped for score==1

---

## 5. Main results (mean ± std)

### 5.1 DeFacto

| Drop r | n_correct | Accuracy | FNR | n_tie | intrinsic correct / 315 | extrinsic correct / 1255 |
|---|---|---|---|---|---|---|
| 0% | 1213 | 0.7726 | 0.2274 | 50 | 197 | 1016 |
| 10% | 1208.6 ± 2.88 | 0.7698 ± 0.0018 | 0.2302 ± 0.0018 | 49.0 ± 0.0 | 197.6 ± 1.14 | 1011.0 ± 2.45 |
| 20% | 1188.8 ± 4.32 | 0.7572 ± 0.0028 | 0.2428 ± 0.0028 | 48.4 ± 0.55 | 195.0 ± 3.94 | 993.8 ± 5.40 |
| 30% | 1169.0 ± 5.15 | 0.7446 ± 0.0033 | 0.2554 ± 0.0033 | 46.6 ± 1.82 | 191.2 ± 5.26 | 977.8 ± 1.79 |
| 50% | 1159.8 ± 12.07 | 0.7387 ± 0.0077 | 0.2613 ± 0.0077 | 47.6 ± 0.89 | 190.6 ± 6.43 | 969.2 ± 8.53 |

Relative to 0% baseline (1213 correct):

- 10%: −4.4 (−0.36%)
- 20%: −24.2 (−2.00%)
- 30%: −44.0 (−3.63%)
- 50%: −53.2 (−4.39%)

Accuracy drop from 0% to 50%: **3.39 percentage points**.

### 5.2 UniSumEval

| Drop r | Pearson | Pearson p (0% run) | Spearman | Spearman p (0% run) |
|---|---|---|---|---|
| 0% | 0.0549 | 0.250 | 0.1231 | 0.0097 |
| 10% | 0.0521 ± 0.0045 | — | 0.1209 ± 0.0034 | — |
| 20% | 0.0466 ± 0.0053 | — | 0.1169 ± 0.0046 | — |
| 30% | 0.0484 ± 0.0076 | — | 0.1149 ± 0.0068 | — |
| 50% | 0.0463 ± 0.0057 | — | 0.1174 ± 0.0027 | — |

Pearson at 0% is **not significant** (p=0.25). Spearman at 0% **is** significant (p=0.0097). Absolute Pearson change from 0%→50% is only about −0.0086.

### 5.3 How much graph structure was actually removed

DeFacto, over all 1814 files (article+candidate+human graphs pooled):

| Drop r | successful entities | dropped entities | mention–entity edges removed | entity–entity edges removed |
|---|---|---|---|---|
| 0% | 29684 | 0 | 0 | 0 |
| 10% | 29684 | 2498 | 3822.4 ± 43.2 | 1167.0 ± 87.9 |
| 20% | 29684 | 5593 | 8052.8 ± 112.6 | 2424.4 ± 48.7 |
| 30% | 29684 | 9131 | 12862.6 ± 128.6 | 3545.8 ± 57.8 |
| 50% | 29684 | 16238 | 22571.0 ± 57.6 | 5228.4 ± 24.0 |

UniSumEval, over the 441 eval files:

| Drop r | successful entities | dropped entities | mention–entity removed | entity–entity removed |
|---|---|---|---|---|
| 0% | 9143 | 0 | 0 | 0 |
| 10% | 9143 | 901 | 2029.6 ± 108.7 | 64.4 ± 7.4 |
| 20% | 9143 | 1805 | 4228.8 ± 140.4 | 121.8 ± 9.2 |
| 30% | 9143 | 2781 | 6381.6 ± 184.2 | 179.2 ± 8.5 |
| 50% | 9143 | 4768 | 10986.8 ± 218.5 | 269.2 ± 8.0 |

---

## 6. Per-repeat raw scores (for variance / significance)

DeFacto n_correct:

- r=0.1: 1205, 1213, 1209, 1208, 1208
- r=0.2: 1193, 1182, 1188, 1189, 1192
- r=0.3: 1170, 1177, 1168, 1163, 1167
- r=0.5: 1167, 1145, 1158, 1153, 1176

UniSumEval Pearson:

- r=0.1: 0.0593, 0.0471, 0.0506, 0.0505, 0.0530
- r=0.2: 0.0480, 0.0501, 0.0376, 0.0507, 0.0467
- r=0.3: 0.0598, 0.0501, 0.0490, 0.0416, 0.0413
- r=0.5: 0.0477, 0.0379, 0.0477, 0.0448, 0.0535

---

## 7. Facts that must not be overclaimed

1. This is **random deletion of successful links**, not replacing BLINK/EL with a weaker linker, and not injecting linking *errors* (wrong entity). It mainly models **EL recall drop**, not precision errors.
2. w(entity–entity)=0 in this run, so EE edges do not contribute to the kernel score even when present.
3. Previous type-removal ablation suggested mention–entity edges were **not** the main driver (removing all ME even slightly *increased* DeFacto 1197→1216 under another weight setting). The present gradual drop is smaller and should be discussed together with that result, not instead of it.
4. UniSumEval Pearson is already near 0 at r=0 under **this weight/embedding setting**. Do not treat the UniSum Pearson curve as strong evidence about EL sensitivity unless you explain the weak baseline correlation.
5. DeFacto 0% here is **1213**, not 1197. Different weights / possibly embedding usage; do not call 1213 the same “full-graph baseline” as the edge-type ablation figure.

---

## 8. What I want you to produce

Please write:

1. **A 8–12 sentence Results paragraph** suitable for a paper (neutral, quantitative).
2. **A Discussion paragraph** answering: does FactGraph+GAWL heavily depend on entity linking quality?
3. **A compact table** I can paste into LaTeX (mean±std).
4. **Reviewer attacks** this experiment is vulnerable to, and short rebuttals if any.
5. Whether I should emphasize DeFacto more than UniSumEval, given the weak Pearson baseline.
6. Optional: a one-sentence takeaway for a rebuttal / appendix.

Be conservative. Prefer “mild degradation / remaining signal in token–mention structure” over “EL is unimportant” if the evidence is mixed.
