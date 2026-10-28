# Frozen January-March 2026 evaluation protocol

These files are byte-identical copies of the records written before any 2026 outcome was read. They document how the Chicago transportability criteria were fixed.

| File here | Original record name | Content |
|---|---|---|
| decision_rules.json | FROZEN_EXTERNAL_DECISION_RULES | Pass conditions, development references and AP/F1 floors for +30 and +60 min |
| operating_points.csv | FROZEN_OPERATING_POINTS | Six fold-4 warning cutoffs (BALANCED_F1, raw scores) |
| calibration.csv | FROZEN_CALIBRATION | Uncalibrated scoring policy for all six heads |
| model_contract.json | FROZEN_MODEL_CONTRACT | Fold-4 model, reference and feature contract |
| selected_artifact_hashes.csv | SELECTED_ARTIFACT_HASHES | SHA-256 of the selected fold-4 artifacts |
| freeze_seal.json | PRE_EXTERNAL_FREEZE | Seal: SHA-256 of the five files above, external_outcome_reads 0 |
| freeze_seal.sha256 | PRE_EXTERNAL_FREEZE.sha256 | SHA-256 of the seal |
| evaluation_started.json | EXTERNAL_STARTED | Start of the single evaluation run, with the seal hash |
| evaluation_complete.json | EXTERNAL_COMPLETE | One inference pass, zero fit calls |

The seal lists the files under their original project paths. Their SHA-256 values match the copies in this folder. The freeze ran at 2026-09-21 19:32:55 UTC and the evaluation started at 19:33:20 UTC. The period has prior project-level history under other targets, so it is reported as retrospective temporal transportability.

Verify with:

```sh
python -c "import hashlib,json,pathlib;p=pathlib.Path('protocol/2026_evaluation');print(hashlib.sha256((p/'freeze_seal.json').read_bytes()).hexdigest())"
```

The printed value must equal the content of `freeze_seal.sha256`.
