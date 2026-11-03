# Processed data

Cleaned dataset versions and their change logs belong here.

## Current deliverable

- Final cleaned dataset: `off_52k_v5.parquet`
- Loadable profile: `config/profiles/off_52k_v5.json`
- Manifest: `off_52k_v5.manifest.json`
- Final change log: `change.off_52k_v5_precision_finalization.changes.parquet`
- Post-clean detection report: `logs/checks_v5/off_52k_v5_quality_summary.md`

`off_52k_v5` has 50,000 rows and 20 columns. Its post-clean detection reports:
duplicates 0, hard-range violations 0, IQR tails 0, and field conflicts 0.
Text missingness, the 197 non-standard `code` values, unavailable labels and
class imbalance remain and are reported rather than guessed or hidden.

## Version chain

| Version | Plan | Purpose |
|---|---|---|
| `v2` | `off_52k_v2_conservative` | Conservative baseline: deduplicate and fill numeric gaps; flag anomalies and conflicts |
| `v3` | `off_52k_v3_stratified_repair` | Quarantine and refill impossible hard-range values, winsorize IQR tails, recompute/cap conflicting fields |
| `v4` | `off_52k_v4_convergence` | Second IQR pass and relation-aware salt/sodium winsorization |
| `v5` | `off_52k_v5_precision_finalization` | Remove the final floating-point boundary conflict |

The full modification chain can be audited through the `v3`, `v4` and `v5`
change logs. The `v2` version remains available for comparison but is no longer
the downstream clustering input.

The frozen file under `data/base/` must never be used as the cleaning output.
Every processed manifest records its parent id, parent SHA256, complete change
log and matching profile. Files are not overwritten in place.
