"""Historical-only cutoff policies; later-fold influence remains retrospective."""

import numpy as np
import pandas as pd

from src.config import resolve, slug
from src.evaluation.discrimination import curve, select
from src.evaluation.grid import GRID, grid_counts
from src.evaluation.operating import basic, transitions


def evaluate(config, city):
    root = resolve(config, "work") / slug(city)
    step = 10 if city == "Chicago" else 5
    rows, transition_rows, cutoffs = [], [], []
    for state in config["thresholds"]:
        for horizon in config["horizons"]:
            data = {}
            for fold in range(1, 5):
                d = root / f"fold{fold}"
                actual = np.load(d / "validation_y.npy", mmap_mode="r")[:, :, horizon // step - 1]
                valid = np.isfinite(actual)
                score = np.load(d / f"scores/r{state:g}_h{horizon}.npy", mmap_mode="r")[valid]
                data[fold] = (
                    actual[valid] < state,
                    score,
                    np.load(d / "validation_current.npy", mmap_mode="r")[valid],
                )
            for fold in data:
                others = [data[k] for k in data if k != fold]
                if city == "Chicago":
                    selected = select(
                        curve(
                            np.concatenate([v[0] for v in others]),
                            np.concatenate([v[1] for v in others]),
                            include_roc=False,
                        ),
                        "BALANCED_F1",
                    )
                    cutoff = selected["threshold"]
                else:
                    counts = np.stack([grid_counts(v[0], v[1]) for v in others])
                    denominator = 2 * counts[:, :, 0] + counts[:, :, 1] + counts[:, :, 2]
                    f1 = np.divide(
                        2 * counts[:, :, 0],
                        denominator,
                        out=np.zeros_like(denominator, dtype=float),
                        where=denominator > 0,
                    ).mean(0)
                    cutoff = float(GRID[np.flatnonzero(f1 == f1.max())[-1]])
                y, s, current = data[fold]
                row = dict(
                    fold=fold,
                    state=state,
                    horizon=horizon,
                    cutoff=cutoff,
                    AP=curve(y, s, include_roc=False)["AP"],
                    **basic(y, s, cutoff),
                )
                rows.append(row)
                cutoffs.append(dict(fold=fold, state=state, horizon=horizon, cutoff=cutoff))
                for t in transitions(y, s, cutoff, current, state):
                    transition_rows.append(dict(fold=fold, state=state, horizon=horizon, **t))
    pd.DataFrame(rows).to_csv(root / "historical_metrics.csv", index=False)
    pd.DataFrame(cutoffs).to_csv(root / "historical_cutoffs.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(root / "historical_transitions.csv", index=False)
