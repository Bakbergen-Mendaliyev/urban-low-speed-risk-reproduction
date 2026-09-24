"""Rebuild fixed reference and feature caches without selecting a new model."""

import warnings
import numpy as np

from src.config import MINUTE, digest, resolve, slug, write_json
from src.data.panel import open_history
from src.features.profiles import PrefixProfile
from src.features.chicago import design as chicago_design
from src.features.city import design as city_design, profile_query


def prepare_fold(config, city, fold):
    f = next(f for f in config["folds"] if f["dataset"] == city and f["fold"] == fold)
    p = open_history(config, city)
    path = resolve(config, "work") / slug(city) / f"fold{fold}"
    path.mkdir(parents=True, exist_ok=False)
    a, b = np.searchsorted(p.time, [f["train_start"], f["train_end"]])
    refpath = config["_root"] / f"data/checksums/{slug(city)}_fold{fold}_reference.npz"
    frozen = np.load(refpath)
    q = np.zeros(len(p.ids))
    counts = np.zeros(len(q), int)
    dates = counts.copy()
    for start in range(0, len(q), 128):
        end = min(start + 128, len(q))
        x = p.read(a, b, slice(start, end), available_before=f["train_end"])
        valid = np.isfinite(x)
        counts[start:end] = valid.sum(0)
        for day in np.unique(p.days[a:b]):
            dates[start:end] += valid[p.days[a:b] == day].any(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q[start:end] = np.nanquantile(x, 0.85, axis=0, method="linear")
    nodes = np.flatnonzero((counts >= 100) & (dates >= 20) & np.isfinite(q) & (q > 0))
    assert np.array_equal(nodes, frozen["nodes"]), "Training cohort changed"
    q = q[nodes]
    assert np.array_equal(q, frozen["q85"]), (
        "Reference differs from frozen inputs; stop, do not tune"
    )
    np.savez(path / "reference.npz", nodes=nodes, q85=q, ids=p.ids[nodes])
    original_origins = config["_root"] / f"data/checksums/{slug(city)}_fold{fold}_origins.npz"
    origins = dict(np.load(original_origins))
    np.savez(path / "origins.npz", **origins)
    for part in ["train", "validation"]:
        start = f["train_start"] if part == "train" else f["validation_start"]
        end = f["train_end"] if part == "train" else f["validation_end"]
        assert np.all(p.time[origins[part]] >= start)
        assert np.all(p.time[origins[part]] + 90 * MINUTE < end)
    n = len(nodes)
    width = 328 if city == "Chicago" else 75
    caches = {}
    for part in ["train", "validation"]:
        shape = (len(origins[part]), n)
        caches[part] = {
            "x": np.lib.format.open_memmap(
                path / f"{part}_x.npy", mode="w+", dtype="float32", shape=shape + (width,)
            ),
            "y": np.lib.format.open_memmap(
                path / f"{part}_y.npy", mode="w+", dtype="float64", shape=shape + (60 // p.step,)
            ),
            "current": np.lib.format.open_memmap(
                path / f"{part}_current.npy", mode="w+", dtype="float64", shape=shape
            ),
            "profile": np.lib.format.open_memmap(
                path / f"{part}_profile.npy",
                mode="w+",
                dtype="float32",
                shape=shape + ((11 if city == "Chicago" else 7),),
            ),
        }
    at = {int(o): (part, j) for part in origins for j, o in enumerate(origins[part])}
    profile = PrefixProfile(n) if city == "Chicago" else None
    sums, count, days = (
        np.zeros((24, n, 4)),
        np.zeros((24, n), np.int32),
        np.zeros((24, n), np.int16),
    )
    stop = np.searchsorted(p.time, f["validation_end"])
    for day in np.unique(p.days[a:stop]):
        ix = np.flatnonzero(p.days[a:stop] == day) + a
        for origin in ix:
            if int(origin) not in at:
                continue
            part, j = at[int(origin)]
            if city == "Chicago":
                values, support, globalmean, observations = profile.query(
                    p.phase[origin], p.weekday[origin], p.month[origin]
                )
                features = np.column_stack((values[0], support[0], globalmean[0], observations))
                xx, now = chicago_design(p, f, int(origin), features, nodes, q)
            else:
                features = profile_query(sums, count, days, p.hour[origin])
                xx = city_design(p, f, int(origin), nodes, q, features)
                now = (
                    p.read(
                        int(origin),
                        int(origin) + 1,
                        nodes,
                        available_before=int(p.time[origin]) + 1,
                    )[0]
                    / q
                )
            target = (
                p.target(
                    int(origin), nodes, f["train_end"] if part == "train" else f["validation_end"]
                )
                / q[:, None]
            )
            caches[part]["x"][j] = xx
            caches[part]["y"][j] = target
            caches[part]["current"][j] = now
            caches[part]["profile"][j] = features
        if ix[0] < b:
            last = min(int(ix[-1]) + 1, b)
            raw = p.read(
                int(ix[0]),
                last,
                nodes,
                available_before=min(f["train_end"], int(p.time[last - 1]) + p.step * MINUTE),
            )
            if city == "Chicago":
                profile.update_day(
                    raw / q,
                    p.phase[ix[: len(raw)]],
                    p.weekday[ix[: len(raw)]],
                    p.month[ix[: len(raw)]],
                )
            else:
                hours = p.hour[ix[: len(raw)]]
                for hour in np.unique(hours):
                    block = raw[hours == hour]
                    valid = np.isfinite(block)
                    sums[hour, :, 0] += np.where(valid, block, 0).sum(0, dtype=float)
                    for j, state in enumerate([0.9, 0.7, 0.6]):
                        sums[hour, :, j + 1] += (valid & (block / q < state)).sum(0)
                    count[hour] += valid.sum(0)
                    days[hour] += valid.any(0)
        print(f"{city} fold {fold}: completed date {day}", flush=True)
    for cache in caches.values():
        for value in cache.values():
            value.flush()
    if profile is not None:
        np.savez(
            path / "frozen_profile.npz",
            **{
                f"{name}{k}": getattr(profile, name)[k]
                for name in ["sums", "count", "days"]
                for k in range(5)
            },
        )
    write_json(
        path / "prepared.json",
        dict(status="COMPLETE", fold=f, files={p.name: digest(p) for p in path.glob("*.np*")}),
    )
