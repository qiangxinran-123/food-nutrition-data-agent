# Processed data

Cleaned dataset versions and their change logs belong here.

The current `data/base/off_52k_v1` dataset is the active synthetic snapshot: it
contains 52,000 rows after adding 2,000 duplicate rows to the former 50,000-row
source. If a later transformation must overwrite it, record the parent hash in
the new manifest before replacing the file.

Cleaning tools write new versions here, for example `off_52k_v2.parquet` and
`off_52k_v2.manifest.json`. Every processed manifest records the parent dataset
id, parent SHA256, the complete change-log file, and the cleaning description.
The matching loadable profile is written to `config/profiles/off_52k_v2.json`;
the processed manifest records its path and SHA256 in `profile_file`.
The frozen file under `data/base/` must never be used as the cleaning output.

Multi-step plans support two persistence modes. `per_step` creates a version
chain such as `v2 -> v3 -> ...`; `final` runs every action in memory and writes
only one final dataset with one aggregate change log. The formal `off_52k_v2`
baseline was produced with `persist="final"`.

Later detection or clustering must load `config/profiles/off_52k_v2.json`, not
the `off_52k_v1` profile. A file is never overwritten in place.
