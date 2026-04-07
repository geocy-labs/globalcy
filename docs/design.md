# Design Notes

GlobalCY Milestone 1 is intentionally narrow:

- consume GeoCYData bundles directly
- compare local and globally invariant scalar models on the same geometry bundle
- construct Kähler metrics with JAX autodiff via `g = g_FS + \partial \bar{\partial}\phi`
- score models with geometric diagnostics rather than loss alone

The bundle adapter is the seam between GeoCYData and GlobalCY. Geometry generation and benchmark-case bookkeeping stay in GeoCYData; model training and diagnostic evaluation live here.
