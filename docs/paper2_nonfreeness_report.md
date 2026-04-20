# Paper II Non-Freeness Report

This report builds the first **computational shadow of non-freeness** on top of the fragile-vs-regular support-shadow layer. The analysis treats each `(case_id, model_name, seed, fragile_flag)` unit as a response object and studies whether those response objects organize into a lower-dimensional collective structure.

The language here stays deliberately modest: this is evidence about **collective fragile-sector structure** and possible gluing-like organization in the frozen response channels, not a theorem-level sheaf, MHM, or Stokes computation.

## Baseline non-freeness summary
- combined response objects: effective rank `2.687` over full rank `12`, ratio `0.224`; PC1 explains `0.692` and PC1+PC2 explain `0.833`.
- fragile-only response objects: effective rank `2.990` over full rank `12`, ratio `0.249`; PC1 explains `0.630` and PC1+PC2 explain `0.797`.
- regular-only response objects: effective rank `2.047` over full rank `12`, ratio `0.171`; PC1 explains `0.793` and PC1+PC2 explain `0.917`.

## Questions answered
### 1. Do fragile-sector response objects lie in a lower-dimensional structure than the raw number of channels suggests?
- yes, modestly. The fragile-only response objects use `13` standardized channels but have effective rank `2.990` and full rank `12`. PC1 alone captures `0.630` of the standardized variance.

### 2. Is the fragile-sector structure more collective than the regular-sector structure?
- not strongly. The fragile-only effective-rank ratio `0.249` is not below the regular-only ratio `0.171`, so the non-freeness shadow is weak or mixed.

### 3. Which variables dominate the first collective modes?
- fragile-sector PC1 top loadings: `euler_density_proxy_mean` (-0.343), `logdet_g_mean` (-0.343), `euler_density_proxy_weighted_mean` (-0.335), `min_eigenvalue_q10` (-0.322), `logdet_residual_mean` (-0.320).
- fragile-sector PC2 top loadings: `logdet_residual_q90` (0.549), `logdet_target_mean` (-0.539), `logdet_residual_q95` (0.533), `logdet_residual_mean` (0.197), `determinant_g_mean` (-0.192).
- regular-sector PC1 top loadings: `euler_density_proxy_mean` (-0.310), `logdet_g_mean` (-0.310), `euler_density_proxy_weighted_mean` (-0.308), `logdet_residual_mean` (-0.304), `negative_frequency` (0.296).

### 4. Do instability, residual, and class-shadow channels align on the same dominant modes or separate?
- fragile-sector PC1 channel mix: instability `1`, residual `1`, class-shadow `3` among the top five absolute loadings.
- fragile-sector PC2 channel mix: instability `0`, residual `3`, class-shadow `2` among the top five absolute loadings.
- the dominant fragile-sector mode mixes instability, residual, and class-shadow channels on the same collective axis, which is modest evidence for gluing-like organization rather than completely separate channel families.

### 5. In the `lambda = 1.0` ablation subset, do objective variants materially change the collective-mode structure?
- ablation fragile-only effective-rank ratio: `0.154`, with PC1 explaining `0.795` and PC1+PC2 explaining `0.996`.
- ablation fragile-sector PC1 top loadings: `euler_density_proxy_mean` (0.310), `logdet_g_mean` (0.310), `logdet_residual_mean` (0.307), `logdet_residual_q95` (0.302), `euler_density_proxy_weighted_mean` (0.295).
- `baseline` / `fragile` mean scores: PC1 = 1.300, PC2 = -0.191.
- `baseline` / `regular` mean scores: PC1 = -1.077, PC2 = 0.233.
- `baseline_plus_both` / `fragile` mean scores: PC1 = 1.046, PC2 = -0.252.
- `baseline_plus_both` / `regular` mean scores: PC1 = -1.270, PC2 = 0.211.
- `baseline_plus_negativity` / `fragile` mean scores: PC1 = 1.046, PC2 = -0.252.
- `baseline_plus_negativity` / `regular` mean scores: PC1 = -1.270, PC2 = 0.211.
- `baseline_plus_projective` / `fragile` mean scores: PC1 = 1.300, PC2 = -0.191.
- `baseline_plus_projective` / `regular` mean scores: PC1 = -1.077, PC2 = 0.233.
- the ablation-mode shifts are present but small, so this remains modest evidence rather than a strong claim that the objective variants radically reorganize the collective fragile-sector structure.

## Interpretation
- The lower effective rank and the sizeable first-mode variance capture indicate that the response objects do not behave like fully free independent channels.
- The resulting structure is best described as a **computational shadow of non-freeness**: the fragile-sector responses show some collective organization, with instability, residual, and class-shadow channels partially locked together.
- That is compatible with modest evidence for gluing-like organization, but it is still a paper-facing statistical shadow, not an actual quotient/sheaf computation.
- If the current structure is used in the manuscript, it should be framed as collective-mode evidence and not as a theorem-level non-freeness result.
