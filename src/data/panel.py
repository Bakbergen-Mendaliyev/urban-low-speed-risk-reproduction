"""Exact timestamp targets and documented retrospective source-time masking."""

import numpy as np
import pandas as pd

from src.config import MINUTE, resolve, slug


class Panel:
    def __init__(self, directory, city, start, end):
        self.city = city
        self.start, self.end = start, end
        self.step = 10 if city == "Chicago" else 5
        self.speed = np.load(directory / "speed.npy", mmap_mode="r")
        self.time = np.load(directory / "timestamps_ns.npy")
        self.ids = np.load(directory / "segment_ids.npy")
        nominal = directory / "nominal_ns.npy"
        self.nominal = np.load(nominal, mmap_mode="r") if nominal.exists() else None
        assert self.speed.shape == (len(self.time), len(self.ids))
        assert np.all(np.diff(self.time) == self.step * MINUTE)
        self.a, self.b = np.searchsorted(self.time, [start, end])
        assert self.a < self.b
        local = pd.to_datetime(self.time, utc=True)
        local = (
            local.tz_convert("America/Chicago") if city == "Chicago" else local.tz_localize(None)
        )
        self.local = local
        naive = local.tz_localize(None) if city == "Chicago" else local
        self.days = naive.normalize().as_unit("ns").asi8
        self.hour = local.hour.to_numpy()
        self.weekday = local.dayofweek.to_numpy()
        self.month = local.month.to_numpy()
        self.phase = self.hour * (60 // self.step) + local.minute.to_numpy() // self.step

    def read(self, start, stop, cols=slice(None), *, available_before=None, floor=None):
        if not self.a <= start < stop <= self.b:
            raise ValueError("Read outside the declared partition context")
        values = np.array(self.speed[start:stop, cols], dtype="float32", copy=True)
        if self.nominal is not None:
            nominal = self.nominal[start:stop, cols]
            low = self.start if floor is None else floor
            high = self.end if available_before is None else available_before
            values[(nominal < low) | (nominal >= high)] = np.nan
        if np.any(values[np.isfinite(values)] < 0):
            raise ValueError("Negative observed speed")
        return values

    def target(self, origin, nodes, end):
        count = 60 // self.step
        indices = origin + np.arange(1, count + 1)
        if self.time[indices[-1]] >= end:
            raise ValueError("Target crosses partition end")
        assert np.array_equal(
            self.time[indices], self.time[origin] + np.arange(1, count + 1) * self.step * MINUTE
        )
        return self.read(origin + 1, origin + count + 1, nodes, available_before=end).T


def open_history(config, city):
    folds = [f for f in config["folds"] if f["dataset"] == city]
    return Panel(
        resolve(config, "processed") / slug(city),
        city,
        folds[0]["train_start"],
        folds[-1]["validation_end"],
    )
