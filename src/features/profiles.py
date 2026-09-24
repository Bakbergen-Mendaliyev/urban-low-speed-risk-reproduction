"""Past-completed-date hierarchical Chicago profiles."""

import numpy as np

THRESHOLDS = [0.9, 0.7, 0.6]


class PrefixProfile:
    def __init__(self, n, phases=144):
        self.phases = phases
        self.sizes = [1, phases, 2 * phases, 7 * phases, 12 * phases]
        self.sums = [np.zeros((s, n, 4), np.float64) for s in self.sizes]
        self.count = [np.zeros((s, n), np.int32) for s in self.sizes]
        self.days = [np.zeros((s, n), np.int16) for s in self.sizes]

    def codes(self, phase, weekday, month):
        return [
            np.zeros_like(phase),
            phase,
            (weekday >= 5) * self.phases + phase,
            weekday * self.phases + phase,
            (month - 1) * self.phases + phase,
        ]

    def update_day(self, x, phase, weekday, month):
        valid = np.isfinite(x)
        values = np.stack(
            [np.where(valid, x, 0)] + [(valid & (x < t)).astype(float) for t in THRESHOLDS], axis=-1
        )
        for level, codes in enumerate(self.codes(phase, weekday, month)):
            for code in np.unique(codes):
                mask = codes == code
                self.sums[level][code] += values[mask].sum(0)
                self.count[level][code] += valid[mask].sum(0)
                self.days[level][code] += valid[mask].any(0)

    def query(self, phase, weekday, month):
        codes = self.codes(np.atleast_1d(phase), np.atleast_1d(weekday), np.atleast_1d(month))
        values = []
        support = []
        for level, index in enumerate(codes):
            count = self.count[level][index]
            avg = np.divide(
                self.sums[level][index],
                count[..., None],
                out=np.zeros_like(self.sums[level][index]),
                where=count[..., None] > 0,
            )
            values.append(avg)
            support.append(self.days[level][index])
        result = values[0].copy()
        for level in (1, 2, 3):
            weight = np.where(support[level] >= 8, support[level] / (support[level] + 20), 0)[
                ..., None
            ]
            result = (1 - weight) * result + weight * values[level]
        weight = np.where((support[4] >= 8) & (support[1] > 0), support[4] / (support[4] + 20), 0)[
            ..., None
        ]
        result += weight * (values[4] - values[1])
        result[..., 0] = np.maximum(result[..., 0], 0)
        result[..., 1:] = np.clip(result[..., 1:], 0, 1)
        return (
            result.astype("float32"),
            np.stack(support, axis=-1),
            values[0][..., 0].astype("float32"),
            self.count[0][0].copy(),
        )
