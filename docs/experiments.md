# Experiments

GlobalCY currently supports three experiment layers built on GeoCYData bundles.

## Paper 1 smoke and multiseed ablations

Use the existing ablation commands for the first reproducible Paper-1 comparisons:

- `python -m globalcy.experiments.run_ablation --config configs/cefalu_lambda_0p75_smoke.yaml`
- `python -m globalcy.experiments.run_ablation --config configs/cefalu_lambda_0p75_multiseed.yaml`
- `python -m globalcy.experiments.run_ablation --config configs/cefalu_lambda_1_multiseed.yaml`

## Paper II hard-regime sweep

GeoCYData now exposes the benchmark preset `cefalu_hard_regime_sweep_v1`. Materialize that preset first, for example:

- `geocydata experiments sweep --preset cefalu_hard_regime_sweep_v1 --out ../geo-cy-data/runs/cefalu_hard_regime_sweep_v1`

Then run the GlobalCY regime sweep:

- `python -m globalcy.experiments.run_regime_sweep --config configs/cefalu_hard_regime_sweep_v1.yaml`

The runner consumes:

- `benchmark_preset_manifest.json`
- per-bundle `manifest.json`
- model-facing bundle artifacts from `bundles/<case_id>/seed_<seed>/`

and writes:

- per-run records in `outputs/<run_name>_regime_sweep/summaries/per_run_results.csv`
- per-case aggregates in `outputs/<run_name>_regime_sweep/summaries/casewise_results.csv`
- sweep-level aggregates in `outputs/<run_name>_regime_sweep/summaries/sweep_results.csv`
- frozen manuscript-facing artifacts in `outputs/<run_name>_regime_sweep/frozen/`

The frozen Paper-II artifacts are:

- `paper2_casewise_results.csv`
- `paper2_sweep_results.csv`
- `paper2_results.json`
- `paper2_summary.md`

The frozen outputs include `spectral_tail_mean` alongside `min_eigenvalue_mean`.

- `min_eigenvalue_mean` averages the per-point minimum eigenvalue over the full sampled bundle
- `spectral_tail_mean` averages only the lowest-decile slice of that minimum-eigenvalue distribution

This makes `spectral_tail_mean` the more sensitive summary when the hard-regime sweep develops a small but important lower tail before the full mean degrades strongly.

## Paper II geometry-aware objective ablations

Use the global-model ablation runner for the hardest Cefalu cases:

- `python -m globalcy.experiments.run_objective_ablation --config configs/cefalu_hard_regime_ablation_v1.yaml`

The default objective variants are:

- `baseline`
- `baseline_plus_negativity`
- `baseline_plus_projective`
- `baseline_plus_both`

Penalty definitions:

- negativity penalty:
  - the mean positive part of the negative minimum eigenvalue, used to discourage negative-eigenvalue behavior in the learned metric correction
- projective-consistency penalty:
  - the mean absolute prediction drift under a fixed projective rescaling, using the same rescaling notion as the evaluation-time projective-invariance diagnostic

The ablation workflow writes:

- per-run records in `outputs/<run_name>_ablation/summaries/per_run_results.csv`
- per-case aggregates in `outputs/<run_name>_ablation/summaries/per_case_results.csv`
- per-objective aggregates in `outputs/<run_name>_ablation/summaries/per_objective_results.csv`
- hardest-case summaries in `outputs/<run_name>_ablation/summaries/hardest_case_ablation.csv`
- frozen paper-facing artifacts in `outputs/<run_name>_ablation/frozen/`

The key frozen ablation artifacts are:

- `paper2_ablation_results.csv`
- `paper2_ablation_results.json`
- `paper2_ablation_summary.md`

These artifacts use the same diagnostic set as the regime sweep, but group results by objective variant rather than by model family.
