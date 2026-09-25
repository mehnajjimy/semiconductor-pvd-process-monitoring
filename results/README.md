# Results Guide

The six scripts in `src/` generate the CSV files in this folder. Each subfolder
holds one part: the main AlCu analysis, the separately fitted WTi replication,
cross-process comparisons, and data checks.

## Main tables

| Table | Contents |
|---|---|
| [`alcu/alcu_pca_monitoring_summary.csv`](alcu/alcu_pca_monitoring_summary.csv) | AlCu PCA size, empirical thresholds, and held-out alert rate |
| [`alcu/alcu_virtual_metrology_metrics.csv`](alcu/alcu_virtual_metrology_metrics.csv) | AlCu aggregate model errors for the ordinary and grouped splits |
| [`alcu/alcu_virtual_metrology_per_target.csv`](alcu/alcu_virtual_metrology_per_target.csv) | AlCu MAE, RMSE, and R² for each of the 17 released targets |
| [`wti/wti_pca_monitoring_summary.csv`](wti/wti_pca_monitoring_summary.csv) | Separately fitted WTi monitoring summary |
| [`wti/wti_virtual_metrology_metrics.csv`](wti/wti_virtual_metrology_metrics.csv) | WTi aggregate model errors for both split designs |
| [`comparison/cross_process_workflow_comparison.csv`](comparison/cross_process_workflow_comparison.csv) | Compact AlCu and WTi workflow comparison |
| [`audit/split_summary.csv`](audit/split_summary.csv) | Primary and grouped-sensitivity split counts and overlap checks |

## Supporting tables

- `audit/` has file checksums, alignment checks, feature screening, and
  released-scale quality summaries.
- `alcu/` has monitoring counts, excursion diagnostics, model-selection checks,
  output comparisons, and zero-record sensitivity checks for AlCu.
- `wti/` has the matching monitoring, prediction, and sensitivity results for
  the separately fitted WTi workflow.
- `comparison/` has the small summary behind the cross-process figure.

Row-level split assignments, monitoring scores, and predictions stay in
`data/processed/`, because later scripts read them. Figures are in `figures/`.
Run the scripts in number order to rebuild every result.
