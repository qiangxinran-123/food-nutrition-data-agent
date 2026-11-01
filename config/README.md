# Configuration

Dataset-specific mappings live in `config/profiles/`. The generic detection and
preprocessing code must not contain dataset column names.

The human-readable definition of every quality rule is in
`../docs/data_quality_definitions.md`.

`profiles/off_52k_v1.json` defines the frozen input; `profiles/off_52k_v2.json`
defines the current cleaned dataset and must be used for downstream clustering.
`plans/off_52k_v2_conservative.json` is the formal plan that produced `v2`.

`profiles/off_52k_v1.json` currently defines:

The current dataset has 52,000 rows: 50,000 original rows plus 2,000 injected
duplicate rows. Duplicate detection for this profile compares `code` only.

- the manifest and dataset format;
- row-id and primary-key fields;
- physical/logical types, units, and required-column rules;
- per-column missing-value tokens;
- allowed labels that later validation tools will use.
- configurable severity thresholds for missing values, labels, and imbalance;
- duplicate rules, outlier ranges, and cross-field constraints;
- optional boolean values, datetime parsing rules, and label aliases.

Declare missing tokens by column rather than globally. For example, `unknown` is
meaningful for `nutriscore_grade` but is not treated as missing in every field.

Run all seven quality checks from the project root:

```powershell
python scripts/run_quality_checks.py
```

Reports are written to `logs/checks/`:

- one JSON report per check;
- complete row/cell selections under `logs/checks/selections/`;
- `off_52k_v1_quality_summary.json` for machine use;
- `off_52k_v1_quality_summary.md` for human review.

Process findings by following each finding's `selection_artifact_id`; the JSON
samples are for inspection only and are intentionally truncated.

Every persisted cleaning action writes the next dataset version under
`data/processed/` and a loadable profile next to the source profile, for example
`config/profiles/off_52k_v2.json`. The generated profile points to the processed
manifest, so the normal loader and all seven detectors can run on the cleaned
version without changing detector code.

The missing-only runner remains available as `scripts/run_missing_check.py`.

To produce the formal single-version result, run the plan with
`--persist final --no-dry-run`. The default `per_step` mode is useful for
debugging but creates one intermediate version per action.
