"""Opt-in fixed fold-four replay. No fitting, adaptation, or cutoff selection."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import MINUTE, digest, resolve, write_json
from src.data.panel import Panel
from src.features.profiles import PrefixProfile
from src.features.chicago import design
from src.evaluation.discrimination import curve
from src.evaluation.operating import basic, transitions


def replay(config):
    c = config["external"]
    start = pd.Timestamp(c["start"], tz=c["timezone"]).value
    end = pd.Timestamp(c["end"], tz=c["timezone"]).value
    work = resolve(config, "work")
    fold = work / "chicago/fold4"
    output = work / "chicago_external"
    output.mkdir(parents=True, exist_ok=False)
    heads = [(s, h) for s in config["thresholds"] for h in config["horizons"]]
    models = []
    hashes = {}
    for s, h in heads:
        path = fold / f"models/r{s:g}_h{h}/model.cbm"
        hashes[path.relative_to(work).as_posix()] = digest(path)
        model = CatBoostClassifier()
        model.load_model(str(path))
        models.append(model)
    write_json(output / "pre_replay.json", dict(contract=c, models=hashes, fit_calls=0))
    p = Panel(resolve(config, "processed") / "chicago_external", "Chicago", start, end)
    p.a = 0  # Retain leading missing context, still masked by nominal-time floor.
    reference = np.load(fold / "reference.npz")
    mapping = {int(v): i for i, v in enumerate(p.ids)}
    nodes = np.array([mapping[int(v)] for v in reference["ids"]])
    q = reference["q85"]
    profile = PrefixProfile(len(nodes))
    with np.load(fold / "frozen_profile.npz") as z:
        for name in ["sums", "count", "days"]:
            for k in range(5):
                getattr(profile, name)[k] = z[f"{name}{k}"]
    f = next(f for f in config["folds"] if f["dataset"] == "Chicago" and f["fold"] == 4)
    origins = np.flatnonzero((p.time >= start) & (p.time + 90 * MINUTE < end))
    shape = (len(origins), len(nodes))
    scores = np.lib.format.open_memmap(
        output / "scores.npy", mode="w+", dtype="float32", shape=shape + (6,)
    )
    actual = np.lib.format.open_memmap(
        output / "future_ratio.npy", mode="w+", dtype="float64", shape=shape + (3,)
    )
    current = np.lib.format.open_memmap(
        output / "current_ratio.npy", mode="w+", dtype="float64", shape=shape
    )
    for first in range(0, len(origins), 32):
        chosen = origins[first : first + 32]
        v, s, g, count = profile.query(p.phase[chosen], p.weekday[chosen], p.month[chosen])
        blocks = []
        for j, o in enumerate(chosen):
            x, now = design(p, f, int(o), np.column_stack((v[j], s[j], g[j], count)), nodes, q)
            blocks.append(x)
            actual[first + j] = p.target(int(o), nodes, end)[:, [0, 2, 5]] / q[:, None]
            current[first + j] = now
        x = np.concatenate(blocks)
        for k, model in enumerate(models):
            scores[first : first + len(chosen), :, k] = (
                model.predict_proba(x, thread_count=4)[:, 1]
                .astype("float32")
                .reshape(len(chosen), len(nodes))
            )
        print(f"External replay {first + len(chosen)}/{len(origins)} origins", flush=True)
    for x in [scores, actual, current]:
        x.flush()
    rows, transition_rows = [], []
    for k, (state, horizon) in enumerate(heads):
        y = actual[:, :, [10, 30, 60].index(horizon)]
        valid = np.isfinite(y)
        labels = y[valid] < state
        risk = scores[:, :, k][valid]
        cut = c["cutoffs"][k]
        rows.append(
            dict(
                state=state,
                horizon=horizon,
                AP=curve(labels, risk, include_roc=False)["AP"],
                **basic(labels, risk, cut),
            )
        )
        transition_rows.extend(
            dict(state=state, horizon=horizon, **row)
            for row in transitions(labels, risk, cut, current[valid], state)
        )
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output / "transitions.csv", index=False)
    np.save(output / "origin_ns.npy", p.time[origins])
    np.save(output / "segment_ids.npy", reference["ids"])
    write_json(
        output / "complete.json",
        dict(
            fit_calls=0,
            interpretation=c["interpretation"],
            files={p.name: digest(p) for p in output.glob("*.npy")},
        ),
    )
