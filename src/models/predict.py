"""Opt-in historical scoring, writing only to the reproduction output tree."""

import numpy as np
from catboost import CatBoostClassifier

from src.config import digest, resolve, slug, write_json


def predict_fold(config, city, fold):
    directory = resolve(config, "work") / slug(city) / f"fold{fold}"
    output = directory / "scores"
    output.mkdir(exist_ok=False)
    x = np.load(directory / "validation_x.npy", mmap_mode="r")
    for state in config["thresholds"]:
        for horizon in config["horizons"]:
            name = f"r{state:g}_h{horizon}"
            model = CatBoostClassifier()
            model.load_model(str(directory / "models" / name / "model.cbm"))
            path = output / (name + ".npy")
            scores = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=x.shape[:2])
            for a in range(0, len(x), 16):
                block = x[a : a + 16]
                scores[a : a + 16] = model.predict_proba(
                    np.ascontiguousarray(block.reshape(-1, x.shape[-1])), thread_count=4
                )[:, 1].reshape(block.shape[:2])
            scores.flush()
            write_json(path.with_suffix(".json"), dict(sha256=digest(path), shape=scores.shape))
            print(f"Scored {city} fold {fold} {name}", flush=True)
