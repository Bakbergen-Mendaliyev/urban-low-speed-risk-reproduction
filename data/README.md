# Source files and integrity

No raw data are bundled. Download Chicago records from City of Chicago dataset IDs `sxs8-h27x` (historical) and `4g9f-3jbs` (later). Export the required periods as CSV or Parquet, preserving source timestamp precision and unavailable speeds. Use the Yandex Urban Traffic Benchmark repository's download links for `city-traffic-M-raw-speed.parquet`.

```text
data/raw/chicago/                Historical Chicago CSV/Parquet shards
data/raw/chicago_external/       Chicago Q1 2026 CSV/Parquet shards
data/raw/city_m/city-traffic-M-raw-speed.parquet
```

Chicago acquisition files must have `segment_id`, `time`, `speed` with `time` a naive America/Chicago local source timestamp. For portal CSV export, rename the corresponding source columns to these names without converting units or timestamps. Unavailable negative speeds become missing. DST-ambiguous raw wall times are excluded because their UTC offset cannot be recovered from a wall-time-only export. For exact historical reconstruction use audited canonical Parquet shards with `segment_id`, `source_timestamp_raw`, `network_snapshot_time`, `forecast_origin_utc`, and `speed_mph`, which preserve the resolved timestamp offsets. Original and canonical snapshot identities are listed in `checksums/source_inventory.csv` and `checksums/chicago_sources.csv`. A newly downloaded raw export is not guaranteed to be byte-identical to a canonical shard.

The frozen Chicago segment identities are in `configs/chicago_segment_ids.json`. The converter rounds nominal timestamps up to the next ten-minute grid point and retains the latest nominal row. It does not forward-fill missing speed. The feature reader masks source times against the origin or partition boundary. The historical panel contains the original September 2022 to September 2023 time grid; temporal folds start later as specified in configuration.

City-M input is the raw observed-speed matrix, 35,449 timestamps by 53,530 anonymous nodes, with `timestamp` and `node_0` through `node_53529`. No imputation or assumed road names are introduced. Values retain the benchmark's native speed units. Reference-relative ratios make the target unitless; within-dataset speed errors do not justify direct cross-city accuracy comparison.

Each `*_reference.npz` contains the frozen eligible node indices, IDs and training-only q85. Each `*_origins.npz` fixes the executed train/validation origin sets. Preprocessing does not silently override reference or cohort mismatches. The small feature fixture is processed data for implementation verification only, not a new evaluation sample. Its source and hash are documented in `checksums/fixture.json`.

`checksums/release.csv` lists every committed release file except itself (relative path, byte size, SHA-256). Git metadata, caches, virtual environments, raw/processed data and newly computed outputs are explicitly excluded. The manifest validator rejects duplicates, traversal, missing files, mismatched bytes/hashes and unlisted reviewable files.
