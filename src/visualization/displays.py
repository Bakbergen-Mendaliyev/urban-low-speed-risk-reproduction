"""Publication displays read verified values; never estimate model performance."""

import shutil
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.config import resolve


def render(config):
    tables = resolve(config, "tables")
    out = resolve(config, "work") / "displays"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(tables / "historical_discrimination.csv")
    names = {
        "CAT_C0": "Direct CatBoost",
        "M2_RAW": "Multitask",
        "Q10": "Lower quantile",
        "M0": "Speed-derived",
        "PROPENSITY": "Historical propensity",
    }
    frame = frame[
        (frame.state == 0.7)
        & (frame.policy == "BALANCED_F1")
        & frame.model.isin(names)
        & frame.horizon.isin([30, 60])
    ]
    colors = ["#216889", "#ae493b", "#36826e", "#775e89", "#876520"]
    low = min(frame.AP.min(), frame.prevalence.min())
    high = max(frame.AP.max(), frame.prevalence.max())
    padding = (high - low) * 0.07
    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.2), sharex=True, sharey=True, layout="constrained"
    )
    for ax, h in zip(axes, [30, 60]):
        g = frame[frame.horizon == h]
        for j, name in enumerate(names):
            values = g[g.model == name].sort_values("fold").AP.to_numpy()
            ax.plot([values.min(), values.max()], [j, j], color=colors[j], lw=1.5)
            for k, value in enumerate(values):
                ax.scatter(
                    value,
                    j + (k - 1.5) * 0.13,
                    marker=["o", "s", "^", "v"][k],
                    facecolors="none",
                    edgecolors=colors[j],
                    s=30,
                )
            ax.scatter(values.mean(), j, marker="D", color=colors[j], s=36)
        ax.axvline(g[g.model == "CAT_C0"].prevalence.mean(), color="#666666", ls="--")
        ax.set(
            title=f"+{h} min (primary)",
            xlabel="Average precision",
            yticks=range(5),
            yticklabels=list(names.values()),
            ylim=(4.5, -0.5),
            xlim=(low - padding, high + padding),
        )
    handles = [
        Line2D(
            [],
            [],
            marker=m,
            color="#555555",
            markerfacecolor="none",
            linestyle="none",
            label=f"Fold {i + 1}",
        )
        for i, m in enumerate(["o", "s", "^", "v"])
    ]
    handles.extend(
        [
            Line2D([], [], marker="D", color="#555555", linestyle="none", label="Equal-fold mean"),
            Line2D([], [], color="#666666", ls="--", label="Prevalence"),
        ]
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.savefig(out / "primary_discrimination.png", dpi=200)
    plt.close(fig)
    roads = pd.read_csv(tables / "chicago_priority_roads.csv")
    selected = pd.concat([roads[roads.horizon == h].head(5) for h in [30, 60]])
    burden = pd.read_csv(tables / "chicago_burden.csv")
    selected = selected.merge(
        burden[["horizon", "segment_id", "fraction_r_below_070"]],
        on=["horizon", "segment_id"],
        sort=False,
        validate="one_to_one",
    )
    selected.to_csv(out / "priority_segments.csv", index=False)
    destination = out / "manuscript_tables"
    destination.mkdir(exist_ok=True)
    for table in (tables / "manuscript").glob("*.csv"):
        shutil.copy2(table, destination / table.name)
    print(f"Frozen displays written to {out}")
