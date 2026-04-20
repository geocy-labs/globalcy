# Paper II Fragile-Sector Join Report

This report builds the first **degeneration-aware support shadow** layer for the GlobalCY II extension by joining the GeoCYData fragility proxy export to the GlobalCY pointwise diagnostics on `case_id`, `seed`, and `point_id`.

The analysis stays deliberately modest. The resulting summaries are **class-shadow style summaries** and **equation-facing residual concentration** summaries; they are not theorem-level singular invariants.

## Join status
- joined rows: 8400
- sweep rows: 6000
- ablation rows: 2400
- join issues: 0
- fragile clustering used: no; only the documented GeoCY `fragile_flag` is used

## Core frozen outputs
- `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_fragile_vs_regular_summary.csv`
- `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_fragile_vs_regular_summary.json`
- `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_fragile_vs_regular_seedwise.csv`
- `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_ablation_fragile_vs_regular_summary.csv`
- `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_fragile_sector_table.csv`
- `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_residual_alignment_table.csv`

## Baseline sweep findings
### 1. Do fragile sectors carry higher negative-frequency than regular sectors?
- `global`: fragile sectors have higher negative frequency in 4/5 sweep cases.
- `local`: fragile sectors have higher negative frequency in 4/5 sweep cases.
- strongest `local` fragile-vs-regular negative-frequency gap: `cefalu_lambda_0_75` with fragile minus regular = 0.083892.
- strongest `global` fragile-vs-regular negative-frequency gap: `cefalu_lambda_1_10` with fragile minus regular = 0.045417.

### 2. Do fragile sectors carry worse lower-tail behavior than regular sectors?
- `global`: fragile sectors have worse lower-tail behavior (lower `spectral_tail_mean`) in 4/5 sweep cases.
- `local`: fragile sectors have worse lower-tail behavior (lower `spectral_tail_mean`) in 5/5 sweep cases.
- strongest `local` fragile-vs-regular tail separation: `cefalu_lambda_0_75` with fragile minus regular spectral tail = -0.060312.
- strongest `global` fragile-vs-regular tail separation: `cefalu_lambda_0_75` with fragile minus regular spectral tail = -0.033282.

### 3. Do fragile sectors carry larger residual mass than regular sectors?
- `global`: fragile sectors have higher `logdet_residual_q90` in 1/5 sweep cases.
- `local`: fragile sectors have higher `logdet_residual_q90` in 3/5 sweep cases.
- strongest `local` equation-facing residual concentration gap: `cefalu_lambda_0_75` with fragile minus regular residual q90 = 0.510186.
- strongest `global` equation-facing residual concentration gap: `cefalu_lambda_0_50` with fragile minus regular residual q90 = 0.117119.

### 4. Do fragile sectors carry larger euler-density-proxy or determinant-facing concentration than regular sectors?
- `global`: fragile sectors have higher weighted `euler_density_proxy` in 1/5 sweep cases.
- `local`: fragile sectors have higher weighted `euler_density_proxy` in 0/5 sweep cases.
- strongest `local` class-shadow style Euler-proxy concentration gap: `cefalu_lambda_1_10` with fragile minus regular weighted Euler proxy = -0.012388.
- strongest `global` class-shadow style Euler-proxy concentration gap: `cefalu_lambda_0_50` with fragile minus regular weighted Euler proxy = 0.209609.

### 5. Are the fragile-vs-regular contrasts stronger for local or global models?
- `global` average fragile-minus-regular contrasts: negative frequency = 0.020057, spectral tail = -0.009002, residual q90 = -0.129332, weighted Euler proxy = -0.668789.
- `local` average fragile-minus-regular contrasts: negative frequency = 0.035459, spectral tail = -0.027881, residual q90 = 0.103611, weighted Euler proxy = -0.757323.
- localized instability in negative frequency is stronger on average for `local`.
- lower-tail fragile-sector degradation is stronger on average for `local`.
- equation-facing residual concentration is stronger on average for `local`.

### 6. In the `cefalu_lambda_1_0` ablation subset, do objective variants reduce fragile-sector concentration in any channel?
- `baseline`: fragile-minus-regular negative frequency = 0.028793, spectral tail = -0.022692, residual q90 = -0.131491.
- `baseline_plus_both`: fragile-minus-regular negative frequency = 0.028793, spectral tail = -0.020045, residual q90 = -0.131159.
- `baseline_plus_negativity`: fragile-minus-regular negative frequency = 0.028793, spectral tail = -0.020045, residual q90 = -0.131159.
- `baseline_plus_projective`: fragile-minus-regular negative frequency = 0.028793, spectral tail = -0.022692, residual q90 = -0.131491.
- smallest fragile-sector residual concentration is achieved by `baseline`.
- least fragile-sector lower-tail suppression is achieved by `baseline_plus_negativity`.
- smallest fragile-sector negative-frequency gap is achieved by `baseline`.

## Interpretation notes
- The fragile-vs-regular decomposition should be read as a **degeneration-aware support shadow** rather than a literal singular support decomposition.
- The Euler- and determinant-facing channels reported here are **class-shadow style summaries** built from grouped local moments and weighted means of exported proxies.
- The residual channels are **equation-facing residual concentration** summaries derived from `logdet_residual_proxy`; they qualify where the learned metric departs more strongly from the target volume-form side of the benchmark.
- No coupling, non-freeness, clustering, or spectrum claims are made here.
