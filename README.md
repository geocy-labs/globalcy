# GlobalCY

GlobalCY is a JAX framework for globally defined and symmetry-aware neural Kahler potentials.

It uses GeoCYData bundles as the geometry, symmetry, and protocol substrate for Paper-1 experiments.

## Current scope

The current repo supports:

- loading GeoCYData bundles cleanly
- training three inspectable model families on the same underlying bundle:
  - `LocalPhiMLP`
  - `GlobalInvariantPhi`
  - `SymmetryAwareGlobalPhi`
- constructing Hermitian Kahler metrics via `g = g_FS + \partial \bar{\partial}\phi`
- running smoke and multi-seed ablations for the first reproducible Paper-1 comparison layer

The current scientific focus is the two key hard Cefalu cases:

- `lambda = 0.75`
- `lambda = 1.0`

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

## Current limitations

- the current reproducible comparison layer still focuses on the two key hard Cefalu cases (`lambda = 0.75` and `lambda = 1.0`) rather than the full Paper-1 case matrix
- the symmetry-aware model is modest and inspectable rather than fully equivariant
- the repo does not yet implement symbolic distillation
- the repo does not yet implement moduli-dependent modeling
- the repo does not yet implement singularity-aware asymptotic modeling
- the current Paper-1 result-freeze layer is intentionally compact and is not yet a full release / validation framework
- losses are intentionally simple and diagnostic-driven rather than final research losses

## Next step

The next realistic milestone is a lightweight validation and release pass for the frozen Paper-1 outputs so tables, figures, and summary artifacts can be checked and reused more systematically.
