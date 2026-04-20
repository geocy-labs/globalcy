# Paper II Pointwise Diagnostics Export

## Purpose

GlobalCY now exports a narrow per-run pointwise diagnostics file:

- `pointwise_diagnostics.parquet`

This export is designed to support degeneration-aware analysis in GlobalCY II without changing the current aggregate artifact layout.

It is an export-plumbing layer, not a broad new experiment campaign.

## Exported fields

The current pointwise export contains:

- `point_id`
- `case_id`
- `model_name`
- `seed`
- `prediction`
- `target`
- `min_eigenvalue`
- `determinant_g`
- `logdet_g`
- `quadrature_weight`
- `logdet_target`
- `logdet_residual_proxy`
- `negative_flag`
- `euler_density_proxy`

## Meaning of the fields

- `prediction`: model output at the sampled point
- `target`: the current training target already used by GlobalCY
- `min_eigenvalue`: pointwise minimum eigenvalue of the learned Kähler metric
- `determinant_g`: determinant of the learned Kähler metric
- `logdet_g`: `log(max(determinant_g, 1e-8))`
- `quadrature_weight`: current point weight from the bundle's `combined_weight` when available
- `logdet_target`: the current determinant-side target already used in training
- `logdet_residual_proxy`: `logdet_g - logdet_target`
- `negative_flag`: whether `min_eigenvalue <= 0`
- `euler_density_proxy`: currently identical to `logdet_g`, exposed as a lightweight local determinant-side ingredient

## Tiny validation subset

This export was validated on the requested tiny subset:

- `cefalu_lambda_1_0`
- `local` baseline
- `global` baseline
- one seed each

Representative output paths:

- `outputs/paper2_pointwise_diag_local_validation/pointwise_diagnostics.parquet`
- `outputs/paper2_pointwise_diag_global_validation/pointwise_diagnostics.parquet`

## Validation answers

### 1. Are point_ids stable and joinable to the fragility parquet from GeoCYData?

Yes, with the join key:

- `case_id`
- `seed`
- `point_id`

That is the safe join key because `point_id` is only unique within a case/seed bundle.

### 2. Are all required fields present and numerically sane?

Yes for the current narrow export.

Required practical fields now present:

- stable join ids
- predictions and targets
- pointwise minimum eigenvalue
- determinant-side quantity via `logdet_g`
- quadrature weight via `quadrature_weight`
- determinant-side residual proxy ingredients via `logdet_g`, `logdet_target`, and `logdet_residual_proxy`

The values are numerically sane on the validation subset:

- `min_eigenvalue` is finite
- `logdet_g` is finite because the determinant is clipped at `1e-8`
- weights are finite and nonnegative

### 3. Is the export size practical for the 42-run safe minimal path?

Yes.

The current file is narrow and row-wise simple. For a 200-point run, the pointwise parquet remains small enough that the 42-run safe minimal path is practical.

The practical storage scaling is approximately:

- `O(number_of_runs x number_of_points)`

On the real `cefalu_lambda_1_0`, seed `7` validation subset, each pointwise parquet was about `17.6 KB` for `200` points.

So the current 42-run safe minimal path is still modest in storage terms, roughly on the order of under `1 MB` for the narrow parquet itself, before any future wider diagnostic additions.

### 4. What exact residual proxy can be formed from the exported fields?

The exact currently supported local residual proxy is:

- `logdet_residual_proxy = logdet_g - logdet_target`

This is the most direct currently exported determinant-side / Monge-Ampere-style local discrepancy proxy.

Weighted local summaries can be formed immediately using:

- `quadrature_weight * abs(logdet_residual_proxy)`
- weighted means or weighted tail summaries of `logdet_residual_proxy`

### 5. What exact local class-shadow quantities can be formed immediately, and what would require more code?

Immediately formable after joining to GeoCYData fragility outputs:

- chartwise weighted moments of `min_eigenvalue`
- chartwise weighted moments of `logdet_residual_proxy`
- fragility-conditioned local tail summaries
- fragile-vs-nonfragile weighted determinant-side contrast
- region summaries grouped by:
  - `chart_id` from GeoCYData point tables
  - `fragile_flag` from GeoCYData fragility parquet
  - orbit metadata after a join back to GeoCYData orbit exports

Would require more code:

- explicit class-shadow definitions beyond those grouped summaries
- local coupling / non-freeness diagnostics
- local Euler-density or characteristic-form quantities beyond the current determinant-side proxy
- any scalar Laplace operator or spectral export

## Readiness for the 42-run safe minimal path

The export plumbing is ready for the full 42-run safe minimal path.

What this means:

- the run layer can now emit a joinable pointwise diagnostics parquet
- a full rerun can now generate pointwise learned metric-side quantities alongside the current aggregate artifacts

No broad rerun was launched in this step. This task stopped after the tiny validation subset as intended.
