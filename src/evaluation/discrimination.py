"""Tie-aware noninterpolated AP and fixed decision rules."""

import numpy as np
from sklearn.metrics import roc_auc_score


def divide(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=np.asarray(b) != 0)


def metrics(y, w):
    tp = int(np.sum(y & w))
    fp = int(np.sum(~y & w))
    fn = int(np.sum(y & ~w))
    tn = int(np.sum(~y & ~w))
    n = len(y)
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        TN=tn,
        n=n,
        positive=int(y.sum()),
        prevalence=float(divide(y.sum(), n)),
        precision=float(divide(tp, tp + fp)),
        recall=float(divide(tp, tp + fn)),
        F1=float(divide(2 * tp, 2 * tp + fp + fn)),
        false_warnings_per_1000=float(divide(1000 * fp, n)),
        warnings_per_1000=float(divide(1000 * (tp + fp), n)),
    )


def curve(y, s, include_roc=True):
    assert len(y) > 0 and np.isfinite(s).all()
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    yy = y[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(yy, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    threshold = np.r_[np.inf, ss[ends]]
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    p = divide(tp, tp + fp)
    p[0] = 1.0
    r = divide(tp, y.sum())
    f1 = divide(2 * tp, tp + fp + y.sum())
    ap = float(np.sum(np.diff(r) * p[1:])) if y.any() else np.nan
    return dict(
        threshold=threshold,
        TP=tp,
        FP=fp,
        precision=p,
        recall=r,
        F1=f1,
        AP=ap,
        ROC_AUC=float(roc_auc_score(y, s)) if include_roc and 0 < y.sum() < len(y) else np.nan,
    )


def select(c, op):
    ok = np.arange(1, len(c["threshold"]))
    if op == "BALANCED_F1":
        keys = ["F1", "precision", "threshold"]
    elif op.startswith("P"):
        ok = ok[c["precision"][ok] >= int(op[1:]) / 100]
        keys = ["recall", "precision", "threshold"]
    elif op.startswith("R"):
        ok = ok[c["recall"][ok] >= int(op[1:]) / 100]
        keys = ["precision", "recall", "threshold"]
    else:
        raise ValueError(op)
    if not len(ok):
        return None
    for key in keys:
        ok = ok[c[key][ok] == np.max(c[key][ok])]
    i = ok[0]
    return {key: float(c[key][i]) for key in ["threshold", "precision", "recall", "F1"]}


def calibration_metrics(y, p):
    assert np.all((p >= 0) & (p <= 1))
    bins = np.minimum((p * 15).astype(int), 14)
    n = np.bincount(bins, minlength=15)
    sy = np.bincount(bins, weights=y, minlength=15)
    sp = np.bincount(bins, weights=p, minlength=15)
    reliability = [
        dict(
            bin=i,
            n=int(n[i]),
            mean_score=float(sp[i] / n[i]) if n[i] else np.nan,
            observed=float(sy[i] / n[i]) if n[i] else np.nan,
        )
        for i in range(15)
    ]
    return dict(
        Brier=float(np.mean((p - y) ** 2)), ECE=float(np.sum(np.abs(sp - sy)) / len(y))
    ), reliability
