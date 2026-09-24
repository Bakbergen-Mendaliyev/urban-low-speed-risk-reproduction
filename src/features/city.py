"""Frozen 75-column City-M representation and past-date propensity."""

import warnings
import numpy as np


def profile_query(sums, counts, days, hour):
    globalcount = counts.sum(0)
    g = np.divide(
        sums.sum(0),
        globalcount[:, None],
        out=np.zeros_like(sums[0]),
        where=globalcount[:, None] > 0,
    )
    hourly = np.divide(
        sums[hour], counts[hour, :, None], out=g.copy(), where=counts[hour, :, None] > 0
    )
    weight = np.where(days[hour] >= 8, days[hour] / (days[hour] + 20), 0)[:, None]
    prop = (1 - weight) * g[:, 1:] + weight * hourly[:, 1:]
    return np.column_stack((hourly[:, 0], g[:, 0], prop, counts[hour], globalcount)).astype(
        "float32"
    )


def design(p, f, o, nodes, q, profile):
    n = len(nodes)
    a = int(np.searchsorted(p.time, f["train_start"]))
    first = max(a, o - 311)
    raw = p.read(first, o + 1, nodes, available_before=int(p.time[o]) + 1).astype(float) / q
    mask = np.isfinite(raw)
    current = raw[-1]
    trajectory = raw[-24:].T
    observed = mask[-24:].T
    previous = raw[-289] if len(raw) >= 289 else np.full(n, np.nan)
    last = np.max(np.where(mask, np.arange(len(raw))[:, None], -1), axis=0)
    age = np.minimum(len(raw) - 1 - last, 288) / 288
    expected = profile[:, 0] / q
    phase = p.phase[o] / 288
    A = np.column_stack(
        (
            np.arange(n) / max(n - 1, 1),
            q / 50,
            profile[:, 1] / q,
            np.log1p(profile[:, 6]),
            np.full(n, np.sin(2 * np.pi * phase)),
            np.full(n, np.cos(2 * np.pi * phase)),
            expected,
        )
    )
    B = np.column_stack(
        (
            np.where(observed, trajectory, expected[:, None]),
            observed,
            age,
            np.nan_to_num(previous),
            np.isfinite(previous),
        )
    )
    residual = current - expected
    trends = [current - raw[-1 - m // 5] for m in [10, 20, 30, 60]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        recent = raw[-12:]
        stats = [
            np.nanmin(recent, axis=0),
            np.nanmax(recent, axis=0),
            np.nanmedian(recent, axis=0),
            np.nanstd(recent, axis=0),
            np.nanmax(recent, axis=0) - np.nanmin(recent, axis=0),
        ]
    C = np.column_stack((residual * q / 50, residual, *trends, *stats, np.isfinite(current)))
    D = np.column_stack((profile[:, 2:5], np.log1p(profile[:, 5:7])))
    return np.nan_to_num(np.column_stack((A, B, C, D)), nan=0, posinf=0, neginf=0).astype("float32")
