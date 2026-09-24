"""Prespecified City-M 101-point decision grid."""

import numpy as np

GRID = np.linspace(0, 1, 101)


def grid_counts(y, score):
    bins = np.searchsorted(GRID, score.astype(float), side="right")
    pos = np.bincount(bins, weights=y, minlength=102).astype(np.int64)
    total = np.bincount(bins, minlength=102)
    tp = np.cumsum(pos[::-1])[::-1][1:]
    selected = np.cumsum(total[::-1])[::-1][1:]
    fp = selected - tp
    fn = int(y.sum()) - tp
    tn = len(y) - tp - fp - fn
    return np.column_stack((tp, fp, fn, tn))
