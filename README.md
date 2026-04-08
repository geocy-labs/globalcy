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

## Current diagnostics

The current comparison layer surfaces:

- negativity / eigenvalue sanity
- chart consistency
- projective invariance drift
- symmetry consistency
- determinant / Euler-proxy summaries

## Current limitations

- Milestone 3 focuses on the two key hard Cefalu cases (`lambda = 0.75` and `lambda = 1.0`) rather than the full Paper-1 case matrix
- the symmetry-aware model is modest and inspectable rather than fully equivariant
- the repo does not yet implement symbolic distillation
- the repo does not yet implement moduli-dependent modeling
- the repo does not yet implement singularity-aware asymptotic modeling
- the repo does not yet provide a full paper-grade result-freeze / release layer
- losses are intentionally simple and diagnostic-driven rather than final research losses

## Next step

The next realistic milestone is a lightweight Paper-1 result-freeze and table-export layer built on top of the current multi-seed Cefalu benchmark outputs.
