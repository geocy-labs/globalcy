# Paper II Fragile-Sector Compact Table Note

- source artifact used: `artifacts/paper2_fragile_vs_regular_summary.csv`
- grouping logic: use the already aggregated fragile-vs-regular sweep summary rows, filtered to `fragile_label = fragile` and `model_name in {local, global}`
- no ablation rows are included
- no regular-sector rows are included
- row ordering rule: sweep order `0.50, 0.75, 0.90, 1.00, 1.10`, with `Local` first and `Global` second inside each case
- displayed case labels: numeric lambda only
- numeric formatting rule: all three metric columns are formatted to exactly 6 decimal places
- validation checks performed:
  - exactly 10 rows written
  - exactly 2 rows per case
  - row order matches requested sweep order
  - all rows correspond to fragile sector only
  - values match source aggregates with no recomputation drift beyond formatting

This output is directly paste-ready for the manuscript caption: `Compact fragile-sector support-shadow summary across the Cefalu hard-regime sweep.`
