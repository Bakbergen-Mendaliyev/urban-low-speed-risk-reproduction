"""Explicit configuration and output isolation for public reproduction."""

from pathlib import Path
import hashlib
import json

import yaml

ROOT = Path(__file__).resolve().parents[1]
MINUTE = 60_000_000_000


def read_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["horizons"] == [10, 30, 60]
    assert config["thresholds"] == [0.7, 0.6]
    assert config["seed"] == 20260909
    assert config["reference"]["quantile"] == 0.85
    for fold in config["folds"]:
        assert (
            fold["train_start"]
            < fold["train_end"]
            <= fold["validation_start"]
            < fold["validation_end"]
        )
    config["_root"] = path.parent.parent
    return config


def resolve(config, key):
    path = Path(config["paths"][key])
    return path if path.is_absolute() else config["_root"] / path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )


def slug(city):
    return "chicago" if city == "Chicago" else "city_m"
