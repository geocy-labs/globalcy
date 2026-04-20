# Paper II Safe Minimal Rerun Report

## Summary
- expected runs: 42
- successful runs: 42
- failed runs: 0
- schema inconsistency count: 0
- numerical inconsistency count: 11
- ready for fragile-sector join step: yes

## Output roots
- sweep: `C:\Users\fearl\OneDrive\Documents\ar\globalcy\outputs\cefalu_hard_regime_sweep_v1_regime_sweep_pointwise_safe_minimal`
- ablation: `C:\Users\fearl\OneDrive\Documents\ar\globalcy\outputs\cefalu_hard_regime_ablation_v1_cefalu_1_0_pointwise_safe_minimal`

## Frozen aggregate artifacts
- run index: `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_pointwise_diagnostics_run_index.csv`
- residual summary: `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_residual_summary.csv`
- manifest: `C:\Users\fearl\OneDrive\Documents\ar\globalcy\artifacts\paper2_pointwise_diagnostics_manifest.json`

## Validation
- every expected run completed successfully: yes
- every successful run includes `pointwise_diagnostics.parquet`: yes
- run index records paths for successful exports: yes
- max absolute mismatch between metrics negative fraction and pointwise recomputation: 5.960464e-09

## Schema inconsistencies
- none

## Numerical inconsistencies
```json
[
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_sweep_v1_regime_sweep_pointwise_safe_minimal\\runs\\cefalu_lambda_0_50\\seed_11\\global",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.11999999731779099,
    "pointwise_value": 0.12,
    "abs_diff": 2.682209010451686e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_sweep_v1_regime_sweep_pointwise_safe_minimal\\runs\\cefalu_lambda_0_90\\seed_11\\local",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.02499999850988388,
    "pointwise_value": 0.025,
    "abs_diff": 1.4901161207725444e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_sweep_v1_regime_sweep_pointwise_safe_minimal\\runs\\cefalu_lambda_0_90\\seed_11\\global",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.09999999403953552,
    "pointwise_value": 0.1,
    "abs_diff": 5.960464483090178e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_sweep_v1_regime_sweep_pointwise_safe_minimal\\runs\\cefalu_lambda_0_90\\seed_19\\global",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.02499999850988388,
    "pointwise_value": 0.025,
    "abs_diff": 1.4901161207725444e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_sweep_v1_regime_sweep_pointwise_safe_minimal\\runs\\cefalu_lambda_1_0\\seed_11\\global",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.08999999612569809,
    "pointwise_value": 0.09,
    "abs_diff": 3.8743019070697216e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_sweep_v1_regime_sweep_pointwise_safe_minimal\\runs\\cefalu_lambda_1_10\\seed_11\\global",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.07999999821186066,
    "pointwise_value": 0.08,
    "abs_diff": 1.7881393449270533e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_sweep_v1_regime_sweep_pointwise_safe_minimal\\runs\\cefalu_lambda_1_10\\seed_19\\global",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.02499999850988388,
    "pointwise_value": 0.025,
    "abs_diff": 1.4901161207725444e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_ablation_v1_cefalu_1_0_pointwise_safe_minimal\\runs\\cefalu_lambda_1_0\\seed_11\\baseline",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.08999999612569809,
    "pointwise_value": 0.09,
    "abs_diff": 3.8743019070697216e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_ablation_v1_cefalu_1_0_pointwise_safe_minimal\\runs\\cefalu_lambda_1_0\\seed_11\\baseline_plus_negativity",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.08999999612569809,
    "pointwise_value": 0.09,
    "abs_diff": 3.8743019070697216e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_ablation_v1_cefalu_1_0_pointwise_safe_minimal\\runs\\cefalu_lambda_1_0\\seed_11\\baseline_plus_projective",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.08999999612569809,
    "pointwise_value": 0.09,
    "abs_diff": 3.8743019070697216e-09,
    "tolerance": 1e-09
  },
  {
    "run_dir": "C:\\Users\\fearl\\OneDrive\\Documents\\ar\\globalcy\\outputs\\cefalu_hard_regime_ablation_v1_cefalu_1_0_pointwise_safe_minimal\\runs\\cefalu_lambda_1_0\\seed_11\\baseline_plus_both",
    "issue": "negative_fraction_mismatch",
    "metrics_value": 0.08999999612569809,
    "pointwise_value": 0.09,
    "abs_diff": 3.8743019070697216e-09,
    "tolerance": 1e-09
  }
]
```

## Failed runs
- none

## Readiness note
- The data are ready for the fragile-sector join step because all 42 expected runs completed, every successful run exported `pointwise_diagnostics.parquet`, and the join keys `case_id`, `seed`, and `point_id` are present in the validated schema.
- Immediate offline quantities now available include weighted fragile-vs-regular summaries of `min_eigenvalue`, `logdet_residual_proxy`, and `negative_flag` after joining to the GeoCYData fragility parquet.
