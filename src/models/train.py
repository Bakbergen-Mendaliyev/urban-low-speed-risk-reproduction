"""Explicit opt-in selected-model training; never imported with side effects."""

import gc
import json
import time
import numpy as np
from catboost import CatBoostClassifier

from src.config import digest, resolve, slug, write_json


def train_fold(config, city, fold):
    directory = resolve(config, "work") / slug(city) / f"fold{fold}"
    if not (directory / "prepared.json").exists():
        raise FileNotFoundError("Prepare and verify the fixed feature cache first")
    receipt = json.loads((directory / "prepared.json").read_text())
    for name, expected in receipt["files"].items():
        assert digest(directory / name) == expected, f"Changed prepared input: {name}"
    x = np.load(directory / "train_x.npy", mmap_mode="r")
    y = np.load(directory / "train_y.npy", mmap_mode="r")
    step = 10 if city == "Chicago" else 5
    for state in config["thresholds"]:
        for horizon in config["horizons"]:
            name = f"r{state:g}_h{horizon}"
            job = directory / "models" / name
            job.mkdir(parents=True, exist_ok=False)
            future = y[:, :, horizon // step - 1].ravel()
            valid = np.isfinite(future)
            labels = future[valid] < state
            receipt = dict(
                city=city,
                fold=fold,
                state=state,
                horizon=horizon,
                parameters=config["tree"],
                feature_count=x.shape[-1],
                rows=len(labels),
                positives=int(labels.sum()),
            )
            write_json(job / "started.json", receipt)
            print(f"Training {city} fold {fold} {name}; {len(labels)} rows", flush=True)
            started = time.perf_counter()
            features = np.ascontiguousarray(x.reshape(-1, x.shape[-1])[valid])
            model = CatBoostClassifier(**config["tree"])
            model.fit(features, labels, verbose=50)
            model.save_model(str(job / "model.cbm"))
            receipt.update(
                seconds=time.perf_counter() - started, model_sha256=digest(job / "model.cbm")
            )
            write_json(job / "complete.json", receipt)
            del features, model
            gc.collect()
