# Paper II Degeneration Extension Audit

## Scope

This audit inspects the currently materialized GeoCYData and GlobalCY artifacts for the Cefalu hard-regime sweep and the hardest-case objective-ablation runs. It does **not** launch a new broad experiment campaign. The goal is to determine what can already be recovered from existing outputs and what requires targeted reruns or new exports to support:

1. degeneration-sensitive family profiles
2. local fragile-sector decomposition
3. regionwise class-shadow quantities
4. residual proxies
5. coupling / non-freeness analysis
6. optionally, a pilot scalar Laplace spectrum

## Executive answer

The current stack is already strong on **pointwise geometry-side data** and **aggregate learned-metric diagnostics**, but it is still thin on **pointwise learned metric-side exports**.

What already exists:

- GeoCYData bundles provide per-point sampled geometry, invariant features, chart ids, symmetry metadata, and weights.
- GlobalCY run directories provide per-point model predictions and targets.
- GlobalCY frozen artifacts provide run-level, casewise, sweep-level, and ablation-level aggregated diagnostics.

What is missing for the requested degeneration extension:

- pointwise minimum-eigenvalue traces
- pointwise determinant / volume-form traces
- pointwise Euler-style proxy density
- pointwise residual-style learned-metric proxies
- any regionwise decomposition artifacts derived from learned metric-side quantities
- any coupling / non-freeness or scalar-spectrum exports

Because the current run outputs do **not** persist reloadable checkpoints or pointwise metric-side diagnostics, the minimum safe path for learned metric-side degeneration analysis is a **targeted rerun with narrow additional exports**.

## What exists already

### GeoCYData pointwise bundle artifacts

Representative bundle:

- `C:\Users\fearl\OneDrive\Documents\ar\geo-cy-data\runs\cefalu_hard_regime_sweep_v1\bundles\cefalu_lambda_1_0\seed_7`

Confirmed pointwise files present in each bundle:

- `points.parquet`
- `invariants.parquet`
- `sample_weights.parquet`
- `canonical_representatives.parquet`
- `canonical_invariants.parquet`
- `orbits.parquet`

Representative metadata files:

- `manifest.json`
- `case_metadata.json`
- `evaluation_summary.json`
- `validation_report.json`
- `symmetry_report.json`

This means the geometry-side benchmark substrate is already rich enough for:

- family ordering by `lambda`
- chart-aware and region-aware slicing
- symmetry-orbit grouping
- point reweighting by `combined_weight`
- offline recomputation of hypersurface residuals from sampled homogeneous points

### GlobalCY run-level artifacts

Representative regime-sweep run:

- `C:\Users\fearl\OneDrive\Documents\ar\globalcy\outputs\cefalu_hard_regime_sweep_v1_regime_sweep\runs\cefalu_lambda_1_0\seed_7\global`

Representative ablation run:

- `C:\Users\fearl\OneDrive\Documents\ar\globalcy\outputs\cefalu_hard_regime_ablation_v1_ablation\runs\cefalu_lambda_1_0\seed_7\baseline`

Confirmed run-level files present:

- `config.json`
- `metrics.json`
- `params.npz`
- `predictions.parquet`
- `summary.md`

Current `predictions.parquet` columns:

- `point_id`
- `prediction`
- `target`

Current `metrics.json` contains aggregate learned diagnostics such as:

- `negative_fraction`
- `min_eigenvalue_mean`
- `spectral_tail_mean`
- `projective_invariance_drift`
- `chart_consistency`
- `determinant_mean`
- `euler_proxy`
- `runtime_seconds`

### Frozen aggregate artifacts

Regime sweep:

- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/summaries/per_run_results.csv`
- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/summaries/casewise_results.csv`
- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/summaries/sweep_results.csv`
- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/frozen/paper2_casewise_results.csv`
- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/frozen/paper2_sweep_results.csv`

Ablations:

- `outputs/cefalu_hard_regime_ablation_v1_ablation/summaries/per_run_results.csv`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/summaries/per_case_results.csv`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/summaries/per_objective_results.csv`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/frozen/paper2_ablation_results.csv`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/frozen/per_objective_results.csv`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/frozen/hardest_case_ablation.csv`

## Required questions

### 1. Do we already have per-point sampled geometry data for each sweep case?

Yes.

The GeoCYData hard-regime preset already materializes per-point sampled geometry for each sweep case and seed in:

- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/points.parquet`

and associated pointwise representations in:

- `invariants.parquet`
- `sample_weights.parquet`
- `canonical_representatives.parquet`
- `canonical_invariants.parquet`
- `orbits.parquet`

This is sufficient for pointwise geometry-side slicing and offline geometry-only postprocessing.

### 2. Do we already have per-point metric-side outputs or only aggregated summaries?

Partially.

We already have per-point **model outputs**:

- `predictions.parquet` with `point_id`, `prediction`, and `target`

But we do **not** currently persist per-point **metric-side** quantities such as:

- minimum eigenvalue per point
- full eigenvalue tail trace
- determinant per point
- Euler-style density per point
- chartwise metric discrepancies per point
- any regionwise decomposition derived from learned metrics

So the current stack is:

- pointwise geometry-side: yes
- pointwise prediction-side: yes
- pointwise learned metric-side: no
- aggregate learned diagnostics: yes

### 3. Can we reconstruct local minimum-eigenvalue behavior pointwise, or do we need a rerun/export?

We need a rerun or a new explicit export path.

Reason:

- `evaluate_model()` computes pointwise Kähler metrics transiently in memory.
- `positivity_summary()` then collapses them to aggregate scalars:
  - `min_eigenvalue_mean`
  - `min_eigenvalue_min`
  - `negative_fraction`
  - `spectral_tail_mean`
- none of the pointwise eigenvalues are written to disk

The current `params.npz` is also **not** a safe reusable checkpoint format for post hoc reevaluation, because it stores only raw tree leaves via:

- `np.savez(run_dir / "params.npz", *jax.tree_util.tree_leaves(result.params))`

without an accompanying tree structure or restore helper.

So pointwise local minimum-eigenvalue behavior is **not recoverable safely from frozen outputs alone**.

### 4. Do we already have determinant / volume-form ingredients available pointwise?

Not on the learned metric side.

What exists:

- GlobalCY saves only aggregate learned determinant summaries:
  - `determinant_mean`
  - `determinant_std`
- GeoCYData evaluation summaries include only baseline geometry-side aggregate scalar summaries, not pointwise determinant-density traces

What does not exist:

- pointwise learned determinant values
- pointwise learned log-determinant residuals
- pointwise learned volume-form ratios

So determinant / volume-form ingredients for the learned model require either:

- a narrow new export during rerun, or
- a future checkpoint-reload path that does not currently exist

### 5. Is there already an Euler proxy density or only a global Euler proxy summary?

Only a global summary.

Current GlobalCY output:

- `euler_proxy` is a single run-level scalar, defined from the mean of a determinant-based log proxy

Current GeoCYData output:

- `characteristic_form_euler_summary` provides only aggregate baseline scalar summaries and a quartic K3 Euler reference

There is no pointwise Euler-proxy density currently written to disk.

### 6. What is the smallest set of reruns needed, if any?

For geometry-only residual proxies: **no rerun is required**.

For learned metric-side degeneration analysis: **a targeted rerun is required**, because the pointwise metric-side quantities were computed transiently but not exported, and the current checkpoint format is not safely reloadable.

## What can be done without rerunning

### Already possible offline from existing artifacts

Using existing GeoCYData bundles and GlobalCY predictions, we can already build:

- family profiles indexed by case and seed
- chartwise / orbitwise / weight-aware partitions
- geometry-side residual proxies from `points.parquet`
- symmetry-orbit partitions for regionwise postprocessing
- prediction-target residual summaries from `predictions.parquet`

In particular, hypersurface residuals can be recomputed directly from sampled points using the existing GeoCYData geometry helpers:

- `src/geocydata/geometry/cefalu.py`
- `src/geocydata/geometry/hypersurfaces.py`

So a residual-proxy pilot based on geometry-side residuals does **not** require any new bundle export.

### Not already possible offline

The following are not safely recoverable from current frozen artifacts:

- pointwise learned minimum-eigenvalue traces
- pointwise learned determinant traces
- pointwise learned Euler-style density traces
- pointwise learned fragile-sector decomposition
- regionwise class-shadow quantities tied to learned metric response
- coupling / non-freeness analysis tied to learned local metric modes
- scalar Laplace pilot quantities

## Exact files and paths to use

### GeoCYData benchmark preset root

- `C:\Users\fearl\OneDrive\Documents\ar\geo-cy-data\runs\cefalu_hard_regime_sweep_v1`

### GeoCYData pointwise bundle inputs

For each case and seed:

- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/points.parquet`
- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/invariants.parquet`
- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/sample_weights.parquet`
- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/canonical_representatives.parquet`
- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/canonical_invariants.parquet`
- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/orbits.parquet`
- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/manifest.json`
- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/validation_report.json`
- `runs/cefalu_hard_regime_sweep_v1/bundles/<case_id>/seed_<seed>/evaluation_summary.json`

### GlobalCY sweep run inputs

For each case, seed, and model:

- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/runs/<case_id>/seed_<seed>/<model>/predictions.parquet`
- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/runs/<case_id>/seed_<seed>/<model>/metrics.json`
- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/runs/<case_id>/seed_<seed>/<model>/config.json`

### GlobalCY ablation run inputs

For each case, seed, and objective variant:

- `outputs/cefalu_hard_regime_ablation_v1_ablation/runs/<case_id>/seed_<seed>/<objective_variant>/predictions.parquet`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/runs/<case_id>/seed_<seed>/<objective_variant>/metrics.json`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/runs/<case_id>/seed_<seed>/<objective_variant>/config.json`

### Aggregate references

- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/summaries/per_run_results.csv`
- `outputs/cefalu_hard_regime_sweep_v1_regime_sweep/frozen/paper2_casewise_results.csv`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/summaries/per_run_results.csv`
- `outputs/cefalu_hard_regime_ablation_v1_ablation/frozen/paper2_ablation_results.csv`

## Minimum additional computation by target

### 1. Degeneration-sensitive family profiles

Status now:

- aggregate family profiles already exist
- pointwise family profiles of fragile behavior do not

Minimum additional computation:

- rerun the sweep with a narrow new pointwise diagnostic export containing at least:
  - `point_id`
  - `min_eigenvalue`
  - `determinant`
  - `log_determinant`
  - optional `hermitian_residual`

### 2. Local fragile-sector decomposition

Status now:

- possible geometry-side sectoring by chart, orbit, and weights
- not possible for learned fragile sectors because no pointwise metric-side outputs exist

Minimum additional computation:

- same narrow sweep rerun with per-point minimum-eigenvalue and determinant exports
- no broader retraining design change needed

### 3. Regionwise class-shadow quantities

Status now:

- region definitions can be built from:
  - `chart_id`
  - orbit metadata
  - affine coordinates
  - weight structure
- class-shadow quantities themselves are not present

Minimum additional computation:

- same pointwise learned metric export as above
- then derive regionwise statistics in postprocessing

### 4. Residual proxies

Status now:

- geometry-side hypersurface residuals are already recomputable offline from existing point samples
- learned residual proxies are not exported pointwise

Minimum additional computation:

- geometry-only residual pilot: no rerun
- learned residual proxy: rerun with pointwise determinant / logdet / eigenvalue export

### 5. Coupling / non-freeness analysis

Status now:

- no direct coupling or non-freeness diagnostic exists in the current artifact layer

Minimum additional computation:

- requires a new derived diagnostic definition
- likely also requires pointwise learned metric-side export
- probably needs a targeted hardest-case-first rerun rather than a full sweep on first pass

### 6. Pilot scalar Laplace spectrum

Status now:

- no scalar Laplace operator, basis, matrix, or spectrum export exists

Minimum additional computation:

- new operator implementation
- likely a dedicated postprocessing path over one or two selected cases
- not a simple extension of current frozen outputs

## Recommended minimum rerun matrix

### Safe minimal path

Goal:

- obtain enough pointwise learned metric-side data to support degeneration profiles, fragile-sector decomposition, regionwise class-shadow prototypes, and learned residual proxies
- avoid a new broad research campaign

Recommended reruns:

1. Full Paper II sweep matrix with a narrow new pointwise export hook

- cases:
  - `cefalu_lambda_0_50`
  - `cefalu_lambda_0_75`
  - `cefalu_lambda_0_90`
  - `cefalu_lambda_1_0`
  - `cefalu_lambda_1_10`
- models:
  - `local`
  - `global`
- seeds:
  - `7`
  - `11`
  - `19`

Total safe-minimal reruns:

- `5 cases x 2 models x 3 seeds = 30 runs`

Why this is the safe minimum:

- degeneration-sensitive family profiles need the full family
- local-versus-global fragile-sector decomposition needs both model families
- current checkpoints are not safely reusable offline

2. Hardest-case ablation subset with the same new pointwise export hook

Recommended cases:

- `cefalu_lambda_1_0`

Recommended objective variants:

- `baseline`
- `baseline_plus_negativity`
- `baseline_plus_projective`
- `baseline_plus_both`

Seeds:

- `7`
- `11`
- `19`

Total additional safe-minimal ablation reruns:

- `1 case x 4 variants x 3 seeds = 12 runs`

Safe-minimal total:

- `42 runs`

### Stronger path

Goal:

- support stronger hardest-case and near-hardest-case objective analysis
- prepare for coupling / non-freeness prototypes

Recommended reruns:

1. Same 30-run full sweep matrix as above
2. Expanded ablation matrix:

- cases:
  - `cefalu_lambda_0_90`
  - `cefalu_lambda_1_0`
  - `cefalu_lambda_1_10`
- variants:
  - all four objective variants
- seeds:
  - `7`
  - `11`
  - `19`

Total stronger-path ablation reruns:

- `3 cases x 4 variants x 3 seeds = 36 runs`

Stronger-path total:

- `66 runs`

Optional stronger-path pilot scalar spectrum:

- do **not** attach to the whole sweep initially
- instead restrict to:
  - `cefalu_lambda_1_0`
  - optionally `cefalu_lambda_1_10`
- model/objective candidates:
  - `local`
  - `global`
  - `baseline_plus_projective` or `baseline_plus_both`

## Safe minimal path

The safest minimal next step is:

1. add a narrow pointwise learned-metric export to the existing run path
2. rerun the full 30-run sweep matrix
3. rerun only the `cefalu_lambda_1_0` four-variant ablation matrix
4. do geometry-side residual postprocessing offline from existing bundle points

Recommended new pointwise export file per run:

- `pointwise_diagnostics.parquet`

Minimum columns:

- `point_id`
- `min_eigenvalue`
- `determinant`
- `log_determinant`
- `is_negative`
- optional `hermitian_residual`

This is the smallest change that unlocks the first five requested analysis directions without redesigning the framework.

## Stronger path

The stronger path is:

1. same pointwise export hook
2. full sweep rerun
3. full 3-case ablation rerun
4. add region labels in postprocessing:
   - chart-based
   - orbit-based
   - weight-quantile-based
   - lower-tail-defined fragile sectors
5. prototype coupling / non-freeness summaries from those regionwise decompositions
6. only then consider a one-case or two-case scalar Laplace pilot

## Bottom line

The current artifact stack is already strong enough to support:

- geometry-side residual analysis
- orbit-aware and chart-aware partitioning
- family-ordered aggregate comparisons

The single main missing layer is:

- pointwise learned metric-side export

That means the minimum scientifically useful extension is **not** a broad retraining campaign. It is:

- one narrow export addition
- a targeted rerun of the existing sweep matrix
- a targeted rerun of a small hardest-case ablation subset

That is the recommended execution path for the degeneration extension.
