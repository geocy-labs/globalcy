# GlobalCY

GlobalCY is a JAX-native research framework for globally defined and symmetry-aware neural Kahler potentials using GeoCYData bundles.

GlobalCY currently supports:

- loading GeoCYData bundles cleanly
- training inspectable local and global scalar models
- training a symmetry-aware global scalar model on canonical/orbit-aware Cefalu bundle views
- constructing Hermitian Kahler metrics via `g = g_FS + \partial \bar{\partial}\phi`
- running smoke and multi-seed Cefalu ablations with reproducible outputs

GeoCYData remains the source of geometry bundles and benchmark metadata. GlobalCY consumes those bundles rather than rebuilding the geometry stack.

## Quick start

```bash
python -m pip install -e .[dev]
python -m globalcy.experiments.run_train --config configs/quartic_fermat.yaml
python -m globalcy.experiments.run_ablation --config configs/cefalu_lambda_0p75_multiseed.yaml
python -m globalcy.experiments.run_ablation --config configs/cefalu_lambda_1_multiseed.yaml
```

## Initial model families

- `LocalPhiMLP`
- `GlobalInvariantPhi`
- `SymmetryAwareGlobalPhi`

## Current limitations

- Milestone 3 focuses on the two key hard Cefalu cases (`lambda = 0.75` and `lambda = 1.0`) rather than the full Paper-1 case matrix
- multi-seed comparison is now available, but it is still a small reproducible benchmark rather than the final paper evaluation suite
- losses are intentionally simple and diagnostic-driven rather than final research losses
