"""Frozen 328-column Chicago D-prefix representation."""

import warnings
import numpy as np


def design(p, f, o, profile, nodes, q):
    n = len(nodes)
    a = int(np.searchsorted(p.time, f["train_start"]))
    length = 144
    history = p.read(max(a, o - length), o + 1, nodes, available_before=int(p.time[o]) + 1) / q
    raw = np.full((145, n), np.nan, np.float64)
    raw[-len(history) :] = history
    trajectory = raw[-144:].T
    mask = np.isfinite(trajectory)
    current = raw[-1]
    previous = raw[0]
    last = np.max(np.where(mask, np.arange(length), -1), axis=1)
    age = np.minimum(length - 1 - last, length) / length
    phase = p.phase[o] / 144
    week = p.weekday[o] / 7
    month = p.month[o] / 12
    expected = profile[:, 0]
    A = np.column_stack(
        (
            np.arange(n) / max(n - 1, 1),
            q / 50,
            profile[:, 9],
            np.log1p(profile[:, 10]),
            np.sin(2 * np.pi * phase) * np.ones(n),
            np.cos(2 * np.pi * phase) * np.ones(n),
            np.sin(2 * np.pi * week) * np.ones(n),
            np.cos(2 * np.pi * week) * np.ones(n),
            np.full(n, month),
            np.full(n, p.weekday[o] >= 5),
            expected,
            expected * q / 50,
            np.log1p(profile[:, 4:9]),
        )
    )
    B = np.column_stack(
        (np.where(mask, trajectory, 0), mask, age, np.nan_to_num(previous), np.isfinite(previous))
    )
    residual = current - expected
    trends = [current - raw[-1 - m // p.step] for m in (10, 20, 30, 60)]
    recent = raw[-6:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summaries = [
            np.nanmin(recent, axis=0),
            np.nanmax(recent, axis=0),
            np.nanmedian(recent, axis=0),
            np.nanstd(recent, axis=0),
            np.nanmax(recent, axis=0) - np.nanmin(recent, axis=0),
        ]
    C = np.column_stack((residual * q / 50, residual, *trends, *summaries, np.isfinite(residual)))
    D = np.column_stack((profile[:, 1:4], np.log1p(profile[:, 4:9])))
    return np.nan_to_num(np.column_stack((A, B, C, D)), nan=0, posinf=0, neginf=0).astype(
        "float32"
    ), current
