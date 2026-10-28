from pathlib import Path
import ast
import importlib
import re
import subprocess
import sys
import csv
import hashlib

import numpy as np
import pytest

from src.config import read_config
from src.evaluation.frozen import verify, verify_feature_fixture, verify_manifest
from src.evaluation.discrimination import curve, select
from src.evaluation.operating import basic, transitions
from src.evaluation.ranking import rank_rows

ROOT = Path(__file__).resolve().parents[1]


def test_real_feature_replay():
    assert verify_feature_fixture(ROOT) == (1035, 328)


def test_frozen_headlines():
    assert verify(read_config(ROOT / "configs/reproduction.yaml"))["prediction_calls"] == 0


def test_manifest():
    assert verify_manifest(ROOT) > 80


@pytest.mark.parametrize("fault", ["missing", "duplicate", "bytes", "hash", "unlisted"])
def test_manifest_rejects_corruption(tmp_path, fault):
    target = tmp_path / "data/checksums"
    target.mkdir(parents=True)
    file = tmp_path / "sample.txt"
    file.write_bytes(b"abc")
    row = dict(path="sample.txt", bytes=3, sha256=hashlib.sha256(b"abc").hexdigest())
    if fault == "missing":
        file.unlink()
    elif fault == "bytes":
        row["bytes"] = 4
    elif fault == "hash":
        file.write_bytes(b"xyz")
    elif fault == "unlisted":
        (tmp_path / "extra.txt").write_bytes(b"not listed")
    with (target / "release.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerow(row)
        if fault == "duplicate":
            writer.writerow(row)
    with pytest.raises(AssertionError):
        verify_manifest(tmp_path)


def test_configuration():
    c = read_config(ROOT / "configs/reproduction.yaml")
    assert len(c["folds"]) == 8
    assert c["external"]["calibration"] == "UNCALIBRATED"
    assert len(c["external"]["cutoffs"]) == 6


@pytest.mark.parametrize(
    "stage", ["preprocess", "features", "train", "predict", "evaluate", "external"]
)
def test_explicit_opt_in(stage):
    p = subprocess.run(
        [sys.executable, "scripts/reproduce.py", "--stage", stage],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2 and "--allow-expensive" in p.stderr


def test_external_cannot_train():
    p = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce.py",
            "--dataset",
            "Chicago-external",
            "--stage",
            "train",
            "--allow-expensive",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2 and "no external fitting" in p.stderr


def test_help():
    p = subprocess.run(
        [sys.executable, "scripts/reproduce.py", "--help"], cwd=ROOT, capture_output=True, text=True
    )
    assert p.returncode == 0 and "verify" in p.stdout


def test_imports():
    for path in (ROOT / "src").rglob("*.py"):
        if path.name in ["__init__.py", "multitask.py"]:
            continue
        importlib.import_module(".".join(path.relative_to(ROOT).with_suffix("").parts))


def test_optional_architecture_syntax():
    ast.parse((ROOT / "src/models/multitask.py").read_text())


def test_tied_AP_unit_fixture():
    y = np.array([True, False, True, False])
    s = np.array([0.8, 0.8, 0.4, 0.1])
    c = curve(y, s, include_roc=False)
    assert c["AP"] == pytest.approx(0.5 * 0.5 + 0.5 * 2 / 3)
    assert select(c, "BALANCED_F1")["threshold"] == 0.4


def test_fixed_operating_and_transitions_unit_fixture():
    y = np.array([True, True, False, False])
    s = np.array([0.9, 0.1, 0.8, 0.2])
    b = basic(y, s, 0.5)
    assert [b[k] for k in ["TP", "FP", "FN", "TN"]] == [1, 1, 1, 1]
    t = transitions(y, s, 0.5, np.array([1.0, 0.5, 0.5, 1.0]), 0.7)
    assert [r["recall"] for r in t] == [1.0, 0.0, 0.0]


def test_ranking_tie_order_unit_fixture():
    rows, order = rank_rows(
        np.array([True, False, True]), np.ones(3), np.array([3, 1, 2]), [0.01, 0.5]
    )
    assert order.tolist() == [1, 2, 0]
    assert rows[0]["K"] == 1 and rows[1]["captured"] == 1


def test_no_paths_secrets_or_internal_API():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(
            part in [".git", "__pycache__", ".pytest_cache", ".ruff_cache"] for part in path.parts
        ):
            continue
        if path.suffix not in [".py", ".md", ".json", ".yaml", ".csv", ".txt", ".toml"]:
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[CD]:[\\/]", text), str(path)
        assert not re.search(
            r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})", text
        ), str(path)
        assert "-----BEGIN " + "PRIVATE KEY-----" not in text, str(path)
        if path.name != "provenance.json" and "2026_evaluation" not in path.parts:
            assert not re.search(r"V12[._]", text), str(path)


def test_no_duplicate_scripts():
    assert [p.name for p in (ROOT / "scripts").glob("*.py")] == ["reproduce.py"]
