"""Verify published results and a real processed feature fixture, with no scoring."""

import json
from types import SimpleNamespace
import numpy as np
import pandas as pd

from src.config import MINUTE, digest, resolve
from src.features.chicago import design


def reviewable(root):
    ignored = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and not any(part in ignored for part in p.relative_to(root).parts)
        and not p.relative_to(root)
        .as_posix()
        .startswith(("data/raw/", "data/processed/", "artifacts/reproduced/", "logs/"))
        and p.relative_to(root).as_posix() != "data/checksums/release.csv"
    }


def verify_manifest(root):
    records = pd.read_csv(root / "data/checksums/release.csv")
    assert not records.path.duplicated().any()
    assert set(records.path) == reviewable(root), "Missing or unlisted reviewable file"
    for row in records.itertuples():
        path = (root / row.path).resolve()
        assert path.is_relative_to(root.resolve())
        assert path.is_file(), row.path
        assert path.stat().st_size == row.bytes, row.path
        assert digest(path) == row.sha256, row.path
    return len(records)


def verify_feature_fixture(root):
    fixture = np.load(root / "artifacts/verified_tables/feature_fixture.npz")
    times = fixture["timestamps"]
    local = pd.to_datetime(times, utc=True).tz_convert("America/Chicago")

    def read(start, stop, cols, *, available_before):
        value = fixture["speed"][start:stop, cols].copy()
        nominal = fixture["nominal"][start:stop, cols]
        value[(nominal < int(fixture["historical_start"])) | (nominal >= available_before)] = np.nan
        return value

    panel = SimpleNamespace(
        time=times,
        phase=(local.hour * 6 + local.minute // 10).to_numpy(),
        weekday=local.dayofweek.to_numpy(),
        month=local.month.to_numpy(),
        step=10,
        read=read,
    )
    x, _ = design(
        panel,
        {"train_start": int(fixture["historical_start"])},
        144,
        fixture["profile"],
        np.arange(len(fixture["q85"])),
        fixture["q85"],
    )
    np.testing.assert_array_equal(x, fixture["expected"])
    assert np.all(np.diff(times) == 10 * MINUTE)
    return x.shape


def verify(config):
    root = config["_root"]
    files = verify_manifest(root)
    shape = verify_feature_fixture(root)
    tables = resolve(config, "tables")
    external = pd.read_csv(tables / "chicago_external.csv")
    primary = external[(external.state == 0.7) & external.horizon.isin([30, 60])].sort_values(
        "horizon"
    )
    ap_column = "PR_AUC" if "PR_AUC" in primary else "AP"
    assert np.allclose(primary[ap_column], [0.3847, 0.3767], atol=0.00005, rtol=0)
    ranking = pd.read_csv(tables / "chicago_ranking.csv")
    r = ranking[
        (ranking.scope == "all")
        & (ranking.method == "CAT")
        & (ranking.budget == 0.05)
        & ranking.horizon.isin([30, 60])
    ].sort_values("horizon")
    assert np.allclose(r.precision_mean, [0.4724, 0.4641], atol=0.00005, rtol=0)
    assert np.allclose(r.recall_mean, [0.1568, 0.1537], atol=0.00005, rtol=0)
    comparison = pd.read_csv(tables / "historical_discrimination.csv")
    z = comparison[
        (comparison.state == 0.7)
        & (comparison.policy == "BALANCED_F1")
        & comparison.model.isin(["CAT_C0", "M2_RAW", "Q10", "M0", "PROPENSITY"])
    ]
    for _, g in z.groupby(["fold", "horizon"]):
        assert len(g) == 5 and g.n.nunique() == g.positive.nunique() == 1
    result = dict(
        status="PASS",
        manifest_files=files,
        feature_fixture_shape=list(shape),
        training_calls=0,
        prediction_calls=0,
        AP_recomputed=False,
        scope="Frozen headline consistency and exact processed-feature replay; not a full experiment rerun.",
    )
    print(json.dumps(result, indent=2))
    return result
