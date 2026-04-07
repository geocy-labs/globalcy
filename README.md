# GlobalCY

GlobalCY is a JAX-native research framework for globally defined and symmetry-aware neural Kahler potentials using GeoCYData bundles.

Milestone 1 focuses on:

- loading GeoCYData bundles cleanly
- training inspectable local and global scalar models
- constructing Hermitian Kahler metrics via `g = g_FS + \partial \bar{\partial}\phi`
- running one Fermat-quartic smoke experiment with reproducible outputs

GeoCYData remains the source of geometry bundles and benchmark metadata. GlobalCY consumes those bundles rather than rebuilding the geometry stack.

## Quick start

```bash
python -m pip install -e .[dev]
python -m globalcy.experiments.run_train --config configs/quartic_fermat.yaml
```

## Initial model families

- `LocalPhiMLP`
- `GlobalInvariantPhi`
- `SymmetryAwareGlobalPhi`

## Current limitations

- Milestone 1 only runs a Fermat-quartic smoke experiment out of the box
- symmetry-aware training is scaffolded but not yet stressed on the hard Cefalu regimes
- losses are intentionally simple and diagnostic-driven rather than final research losses
