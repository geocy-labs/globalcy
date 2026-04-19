# GlobalCY

GlobalCY is a JAX framework for globally defined and symmetry-aware neural Kahler potentials.

It uses GeoCYData bundles as the geometry, symmetry, and protocol substrate for Paper-1 and Paper-II experiments.

## Current scope

The current repo supports:

- loading GeoCYData bundles cleanly
- training three inspectable model families on the same underlying bundle:
  - `LocalPhiMLP`
  - `GlobalInvariantPhi`
  - `SymmetryAwareGlobalPhi`
- constructing Hermitian Kahler metrics via `g = g_FS + ddbar phi`
- running smoke and multi-seed ablations for the first reproducible Paper-1 comparison layer
- running the Paper-II Cefalu hard-regime sweep over GeoCY benchmark presets
- running geometry-aware objective ablations for the global model on the hardest Paper-II Cefalu cases
- generating paper-facing Paper II figures and tables from frozen outputs only

The current scientific focus is:

- Paper 1:
  - `lambda = 0.75`
  - `lambda = 1.0`
- Paper II hard-regime sweep:
  - `cefalu_lambda_0_50`
  - `cefalu_lambda_0_75`
  - `cefalu_lambda_0_90`
  - `cefalu_lambda_1_0`
  - `cefalu_lambda_1_10`

GeoCYData remains the source of geometry bundles and benchmark metadata. GlobalCY consumes those bundles rather than rebuilding the geometry stack.

## Model families

- `LocalPhiMLP`
- `GlobalInvariantPhi`
- `SymmetryAwareGlobalPhi`

## Current workflows

### Smoke workflow

Use a smoke config to compare `local`, `global`, and `symmetry_aware` on a single seed:

```bash
python -m pip install -e .[dev]
python -m globalcy.experiments.run_ablation --config configs/cefalu_lambda_0p75_smoke.yaml
```

### Multi-seed workflow

Use the Milestone 3 configs for the first reproducible Paper-1 comparison benchmark:

```bash
python -m globalcy.experiments.run_ablation --config configs/cefalu_lambda_0p75_multiseed.yaml
python -m globalcy.experiments.run_ablation --config configs/cefalu_lambda_1_multiseed.yaml
```

The multi-seed runs use:

- `seed = 7`
- `seed = 11`
- `seed = 19`

### Paper-II regime sweep workflow

First materialize the GeoCY benchmark preset into a stable directory, for example in the sibling `geo-cy-data` repo:

```bash
geocydata experiments sweep --preset cefalu_hard_regime_sweep_v1 --out ../geo-cy-data/runs/cefalu_hard_regime_sweep_v1
```

Then point GlobalCY at that benchmark directory and run the regime sweep:

```bash
python -m globalcy.experiments.run_regime_sweep --config configs/cefalu_hard_regime_sweep_v1.yaml
```

By default this sweep consumes:

- `benchmark_preset_manifest.json`
- per-bundle `manifest.json`
- model-facing bundle artifacts under `bundles/<case_id>/seed_<seed>/`

and evaluates:

- `local`
- `global`

with configurable seeds defaulting to:

- `7`
- `11`
- `19`

### Paper-II objective-ablation workflow

Use the focused global-model ablation runner on the hardest Cefalu cases:

```bash
python -m globalcy.experiments.run_objective_ablation --config configs/cefalu_hard_regime_ablation_v1.yaml
```

The default variants are:

- `baseline`
- `baseline_plus_negativity`
- `baseline_plus_projective`
- `baseline_plus_both`

The negativity penalty is the mean positive part of the negative minimum eigenvalue, so it discourages runs whose learned correction drives the metric into the negative-eigenvalue regime.

The projective-consistency penalty is the mean absolute prediction drift under a fixed projective rescaling, using the same rescaling idea as the evaluation-time projective-invariance diagnostic.

### Paper-II paper-artifact workflow

Once the regime-sweep and objective-ablation frozen outputs exist, generate manuscript-facing figures and tables without rerunning training:

```bash
python -m globalcy.experiments.generate_paper2_artifacts ^
  --regime-frozen-dir outputs/cefalu_hard_regime_sweep_v1_regime_sweep/frozen ^
  --ablation-frozen-dir outputs/cefalu_hard_regime_ablation_v1_ablation/frozen ^
  --out outputs/cefalu_hard_regime_paper_artifacts
```

This workflow consumes frozen outputs only and writes figures, tables, an artifact manifest, and a short summary memo.

## Current outputs

Each run writes per-run reproducibility artifacts including:

- config
- seed
- git commit hash
- metrics JSON
- short markdown summary

The comparison workflow writes:

- `comparison.csv`
- `comparison.json`
- `comparison.md`
- `comparison_aggregated.csv`
- `comparison_aggregated.md`

The Paper-II regime sweep writes a stable manuscript-facing structure:

- `outputs/<run_name>_regime_sweep/runs/`
- `outputs/<run_name>_regime_sweep/summaries/per_run_results.csv`
- `outputs/<run_name>_regime_sweep/summaries/casewise_results.csv`
- `outputs/<run_name>_regime_sweep/summaries/sweep_results.csv`
- `outputs/<run_name>_regime_sweep/frozen/paper2_casewise_results.csv`
- `outputs/<run_name>_regime_sweep/frozen/paper2_sweep_results.csv`
- `outputs/<run_name>_regime_sweep/frozen/paper2_results.json`
- `outputs/<run_name>_regime_sweep/frozen/paper2_summary.md`

These frozen artifacts summarize how geometric fidelity changes across the hard Cefalu sweep and are designed to feed later Paper-II tables and plots.
They now include `spectral_tail_mean` in the per-run records, casewise aggregates, sweep-level aggregates, and summary memo.

The objective-ablation workflow writes a parallel manuscript-facing structure:

- `outputs/<run_name>_ablation/config.json`
- `outputs/<run_name>_ablation/runs/`
- `outputs/<run_name>_ablation/summaries/per_run_results.csv`
- `outputs/<run_name>_ablation/summaries/per_case_results.csv`
- `outputs/<run_name>_ablation/summaries/per_objective_results.csv`
- `outputs/<run_name>_ablation/summaries/hardest_case_ablation.csv`
- `outputs/<run_name>_ablation/frozen/paper2_ablation_results.csv`
- `outputs/<run_name>_ablation/frozen/paper2_ablation_results.json`
- `outputs/<run_name>_ablation/frozen/paper2_ablation_summary.md`

The Paper II paper-artifact workflow writes:

- `outputs/<run_name>_paper_artifacts/figures/fig_paper2_diagnostic_trajectories.png`
- `outputs/<run_name>_paper_artifacts/figures/fig_paper2_degradation_profiles.png`
- `outputs/<run_name>_paper_artifacts/figures/fig_paper2_objective_ablation.png`
- `outputs/<run_name>_paper_artifacts/figures/fig_paper2_hardest_case.png`
- `outputs/<run_name>_paper_artifacts/tables/table_paper2_core_sweep_results.csv`
- `outputs/<run_name>_paper_artifacts/tables/table_paper2_core_sweep_results.md`
- `outputs/<run_name>_paper_artifacts/tables/table_paper2_ablation_summary.csv`
- `outputs/<run_name>_paper_artifacts/tables/table_paper2_ablation_summary.md`
- `outputs/<run_name>_paper_artifacts/tables/table_paper2_degradation_summary.csv`
- `outputs/<run_name>_paper_artifacts/tables/table_paper2_degradation_summary.md`
- `outputs/<run_name>_paper_artifacts/paper2_artifact_manifest.json`
- `outputs/<run_name>_paper_artifacts/paper2_artifact_summary.md`

## Paper-1 result freeze

Use the Milestone 4 freeze command to turn the two multi-seed Cefalu comparison outputs into paper-facing tables, figures, and a short memo:

```bash
python -m globalcy.experiments.freeze_results ^
  --comparison-dir outputs/cefalu_lambda_0p75_multiseed_ablation/comparison ^
  --comparison-dir outputs/cefalu_lambda_1_multiseed_ablation/comparison ^
  --out outputs/paper1_frozen_results
```

This writes:

- `paper1_core_results.csv`
- `paper1_core_results.md`
- `paper1_robustness.csv`
- `paper1_robustness.md`
- `paper1_summary.md`
- `fig_core_comparison.png`
- `fig_hardest_case.png`

## Current diagnostics

The current comparison layer surfaces:

- negativity / eigenvalue sanity
- chart consistency
- projective invariance drift
- symmetry consistency
- determinant / Euler-proxy summaries

The Paper-II regime sweep freezes, per run and in aggregate:

- `negative_fraction`
- `min_eigenvalue_mean`
- `spectral_tail_mean`
- `projective_invariance_drift`
- `chart_consistency`
- `determinant_mean`
- `euler_proxy`
- `train_loss`
- `runtime_seconds`

`spectral_tail_mean` is the mean of the lowest-decile slice of the per-point minimum-eigenvalue distribution. `min_eigenvalue_mean` averages the minimum eigenvalue over the full sample, while `spectral_tail_mean` is more sensitive to hardest-case degradation in the lower tail.

The objective-ablation artifacts freeze the same diagnostics, but organize them by objective variant for the global model rather than by model family across the full sweep.

The degradation summary is derived from the frozen casewise results with lambda ordered increasingly:

- for lower-is-better metrics (`negative_fraction`, `projective_invariance_drift`), deterioration ratio = `end_value / start_value`
- for higher-is-better metrics (`spectral_tail_mean`), deterioration ratio = `start_value / end_value`

So values above `1.0` indicate deterioration across the sweep in a directionally consistent way.

## Current limitations

- the current reproducible comparison layer still focuses on the two key hard Cefalu cases (`lambda = 0.75` and `lambda = 1.0`) for Paper 1 and the five-case hard-regime sweep for Paper II
- the symmetry-aware model is modest and inspectable rather than fully equivariant
- the repo does not yet implement symbolic distillation
- the repo does not yet implement moduli-dependent modeling
- the repo does not yet implement singularity-aware asymptotic modeling
- the current Paper-1 result-freeze layer is intentionally compact and is not yet a full release / validation framework
- the current Paper-II regime sweep freezes `spectral_tail_mean` and leaves clear hooks for later lower-tail diagnostics such as quantiles or tail-width summaries
- the first objective-ablation layer is intentionally limited to the global model and simple geometry-aware penalties rather than stronger topology-aware objectives
- losses are intentionally simple and diagnostic-driven rather than final research losses

## Next step

The next realistic milestone is to extend the objective layer with stronger lower-tail, topology-aware, or hardest-case objectives while keeping the current ablation artifact contract stable.
