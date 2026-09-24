"""Fixed-threshold operating and transition metrics."""

import numpy as np


def basic(y, s, cut):
    w = s.astype("float64") >= cut
    tp = int(np.sum(y & w))
    fp = int(np.sum(~y & w))
    fn = int(np.sum(y & ~w))
    tn = int(np.sum(~y & ~w))
    n = len(y)
    return dict(
        N=n,
        prevalence=float(np.mean(y)) if n else np.nan,
        TP=tp,
        FP=fp,
        FN=fn,
        TN=tn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        warnings_per_1000=1000 * (tp + fp) / n if n else np.nan,
        false_warnings_per_1000=1000 * fp / n if n else np.nan,
        missed_positives_per_1000=1000 * fn / n if n else np.nan,
    )


def transitions(y, s, cut, current, state):
    known = np.isfinite(current)
    low = current < state
    warn = s.astype("float64") >= cut
    out = []
    for name, mask, correct in [
        ("normal_to_low", known & ~low & y, warn),
        ("low_to_low", known & low & y, warn),
        ("low_to_normal", known & low & ~y, ~warn),
    ]:
        n = int(mask.sum())
        hit = int((mask & correct).sum())
        out.append(
            dict(transition=name, denominator=n, correct=hit, recall=hit / n if n else np.nan)
        )
    return out
