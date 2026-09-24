"""Within-origin ranking with ascending identity tie breaks."""

import numpy as np
import pandas as pd


def budget_count(n, b):
    return min(n, max(1, int(np.ceil(b * n)))) if n else 0


def rank_rows(y, score, ids, budgets):
    n = len(y)
    assert n and np.isfinite(score).all()
    order = np.lexsort((ids, -score.astype(float)))
    truth = y[order].astype(int)
    positives = int(truth.sum())
    capture = np.cumsum(truth)
    discount = 1 / np.log2(np.arange(n) + 2)
    dcg = np.cumsum(truth * discount)
    ideal = np.cumsum(discount)
    rows = []
    for b in budgets:
        k = budget_count(n, b)
        tp = int(capture[k - 1])
        precision = tp / k
        prev = positives / n
        rows.append(
            dict(
                budget=b,
                N=n,
                K=k,
                positives=positives,
                captured=tp,
                false_prioritized=k - tp,
                precision=precision,
                recall=tp / positives if positives else np.nan,
                prevalence=prev,
                enrichment=precision / prev if positives else np.nan,
                hit=int(tp > 0),
                NDCG=dcg[k - 1] / ideal[min(k, positives) - 1] if positives else np.nan,
                random_precision=prev,
                random_recall=k / n if positives else np.nan,
            )
        )
    return rows, order


def summaries(frame, keys):
    metrics = [
        "N",
        "K",
        "positives",
        "captured",
        "false_prioritized",
        "precision",
        "recall",
        "prevalence",
        "enrichment",
        "hit",
        "NDCG",
        "random_precision",
        "random_recall",
    ]
    rows = []
    for key, g in frame.groupby(keys, dropna=False, sort=True):
        out = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        out["origins"] = len(g)
        for m in metrics:
            v = g[m].dropna()
            out[m + "_defined_origins"] = len(v)
            for label, fun in [
                ("mean", lambda x: x.mean()),
                ("median", lambda x: x.median()),
                ("P25", lambda x: x.quantile(0.25)),
                ("P75", lambda x: x.quantile(0.75)),
            ]:
                out[m + "_" + label] = fun(v)
        rows.append(out)
    return pd.DataFrame(rows)
