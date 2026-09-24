# Future low-speed state prediction and risk prioritization

This repository accompanies **Future Low-Speed State Prediction and Risk Prioritization in Urban Road Networks: Temporal Transportability and Cross-Network Replication**. The target is an observed future speed below a training-only segment reference, not a physically verified traffic jam. Emergence and persistence have separate evaluations. Operational readiness is not established.

## Reproduction scope

The default command checks the integrity and consistency of frozen result tables and exactly reconstructs the 328 Chicago input features for one saved real validation origin. It does not train a model, generate predictions, or estimate performance on a substitute sample. The fixture contains all 1,035 supported segments at the first chronological validation origin of the first Chicago fold and 145 preceding/current ten-minute slots.

The opt-in pipeline implements raw-file preprocessing, fixed-fold reference and feature preparation, six selected Direct CatBoost heads per fold, historical evaluation, and the fixed Chicago external replay. Model adapters are newly packaged code and have not been validated by a complete retraining in this release. The historical ablations, neural comparisons, spatial controls, and uncertainty estimates are supplied as frozen result tables; this entry point does not rerun their original model searches or regenerate their bootstrap distributions. Their displayed values remain independently inspectable. This is a frozen-evidence release with a selected-model reconstruction path, not a claim that every historical experiment was rerun from this checkout.

## Environment and quick verification

Python 3.12, a 64-bit CPU environment, and UTF-8 are required. From this repository root:

The original research environment used pandas 3.0.4, subsequently yanked by its maintainers for datetime-related crashes. This release pins 3.0.6 and checks exact equality to the saved real feature fixture. This packaging dependency correction does not alter any reported scientific value.

```sh
python -m venv .venv
```

Activate `.venv` using `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on POSIX systems, then run:

```sh
python -m pip install -r requirements.txt
python scripts/reproduce.py --help
python scripts/reproduce.py --config configs/reproduction.yaml
python -m pytest -q
python -m ruff check src scripts tests --select F
python scripts/reproduce.py --config configs/reproduction.yaml --stage figures
```

The last command renders the primary-horizon AP display and exports the road-priority display and manuscript tables from saved values. It does not recompute AP. The other figure-source tables are under `artifacts/verified_tables/figure_sources`; their scientific values, including intervals and deterministic cases, are retained without new metric estimation. Full original document typesetting is outside this code release.

## Data acquisition

Chicago historical source: [Chicago Data Portal, sxs8-h27x](https://data.cityofchicago.org/d/sxs8-h27x). Later source: [Chicago Data Portal, 4g9f-3jbs](https://data.cityofchicago.org/d/4g9f-3jbs). City-M is the anonymous M network from the [Yandex Urban Traffic Benchmark](https://github.com/yandex-research/urban-traffic-benchmark). Follow that project's raw-data download instructions and source terms; do not substitute its imputed speed product.

Read `data/README.md` for schemas and checksums before downloading. Raw data, models and predictions are not redistributed. Acquisition endpoints may return revisions different from the audited snapshots; an exact-reference mismatch is an error, not permission to silently change the cohort. The anonymized City-M network has no verified real street mapping.

## Configuration and stages

`configs/reproduction.yaml` contains the exact eight temporal folds as UTC nanoseconds, target thresholds, horizons, random seeds, CatBoost parameters, fixed external cutoffs and relative paths. References use the training-only linear 0.85 quantile, at least 100 observations and 20 dates. Chicago uses 10-minute intervals; City-M uses 5-minute intervals. Target timestamps are exact. The 90-minute end guard and saved deterministic training-origin samples are preserved.

The following commands are **expensive, opt-in commands for a later full reproduction**, not actions performed during packaging. Provide the raw sources first. Each feature/training/prediction output is new and refuses silent overwriting.

```sh
python scripts/reproduce.py --stage preprocess --dataset Chicago --allow-expensive
python scripts/reproduce.py --stage preprocess --dataset City-M --allow-expensive
python scripts/reproduce.py --stage features --dataset Chicago --fold 1 --allow-expensive
python scripts/reproduce.py --stage train --dataset Chicago --fold 1 --allow-expensive
python scripts/reproduce.py --stage predict --dataset Chicago --fold 1 --allow-expensive
```

Repeat `features`, `train`, and `predict` for folds 2, 3 and 4; repeat for `--dataset City-M` and folds 1 through 4. Then:

```sh
python scripts/reproduce.py --stage evaluate --dataset Chicago --allow-expensive
python scripts/reproduce.py --stage evaluate --dataset City-M --allow-expensive
python scripts/reproduce.py --stage preprocess --dataset Chicago-external --allow-expensive
python scripts/reproduce.py --stage external --dataset Chicago-external --allow-expensive
```

Historical operating points use the executed other-fold policy. These are retrospective blocked comparisons, not forward-only online selection. The external stage uses only Chicago fold 4, the saved training profile and six fixed thresholds. No 2026 fitting, calibration or threshold selection is available. Its period is local Chicago time `[2026-01-01, 2026-04-01)`, labelled retrospective temporal transportability because this period has prior project-level history. Existing failed outputs must be examined before a new output directory is configured; do not silently restart.

## Expected frozen headline values

For Chicago primary `r < 0.70`, external AP is 0.3847 at +30 min and 0.3767 at +60 min. Top-5% Direct CatBoost mean precision is 0.4724/0.4641 and mean recall is 0.1568/0.1537. These rounded values are checked against the supplied full-precision CSVs; they are not substituted into calculations. AP uses noninterpolated precision-weighted recall increments. Budget ranking uses within-origin eligible support and ascending segment ID for ties. Undefined recall/enrichment denominators remain explicit.

The priority display takes the first five existing rows for each primary horizon in the frozen per-segment table. Selection count, eligible-origin count and observed low-speed burden are different quantities. No new composite ranking or geometry is invented.

## Compute and limitations

Quick checks require roughly a few hundred MB of RAM and no GPU. Allow at least 16 GB RAM for the selected-model development workflow; City-M caches and source data require tens of GB of storage, potentially substantially more when retaining all folds. Tree fitting uses four CPU threads and 300 iterations per head. A full run can take hours or days depending on storage and CPU; no new runtime benchmark is claimed here.

Source timestamps are nominal retrospective timestamps, not measured publication availability. Missing outcomes are excluded on the original horizon-specific support. There is no deployment claim, causal congestion attribution, complete verified Chicago geometry or cross-city pooled metric. Frozen tables are the authoritative reported results until the opt-in reconstruction is independently run and compared. See `data/checksums/provenance.json` for original internal source identities, retained only for provenance.

## Layout

```text
configs/                 Fixed contracts and Chicago segment identities
src/data/                Source conversion and timestamp-aware panels
src/features/            Past-only profiles and fixed feature designs
src/models/              Selected CatBoost fit/score and reference architectures
src/evaluation/          Metrics, operating rules, ranking, replay and verification
src/visualization/       Saved-value publication displays
scripts/reproduce.py     Single entry point
tests/                   Unit, integrity, portability and safety checks
data/checksums/          Source hashes, fixed cohorts/origins, release manifest
artifacts/verified_tables/ Frozen tables, display sources, processed feature fixture
```

Raw inputs go to `data/raw/`, processed panels to `data/processed/`, and newly reproduced outputs to `artifacts/reproduced/`. All are ignored by Git. The small optional neural reference module requires `requirements-models.txt`; it is not used by the selected CatBoost entry point.
