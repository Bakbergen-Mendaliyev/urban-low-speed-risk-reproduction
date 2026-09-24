"""Matched historical auxiliary-head architecture."""

import torch
from torch import nn


class MultiTask(nn.Module):
    def __init__(self, features, nodes):
        super().__init__()
        self.identity = nn.Embedding(nodes, 16)
        self.input = nn.Linear(features + 16, 128)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128)
                )
                for _ in range(2)
            ]
        )
        self.speed = nn.Linear(128, 6)
        self.state = nn.Linear(128, 9)

    def forward(self, x, node):
        z = self.input(torch.cat((x, self.identity(node)), dim=-1))
        for block in self.blocks:
            z = z + block(z)
        return self.speed(z), self.state(z).reshape(-1, 3, 3)


def auxiliary_loss(logits, y, mode, weights):
    target = y[:, [0, 2, 5]]
    mask = torch.isfinite(target)[..., None].expand(-1, -1, 3)
    truth = (
        target[..., None] < torch.tensor([0.9, 0.7, 0.6], device=y.device, dtype=y.dtype)
    ).float()
    if mode == "M2":
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, truth, reduction="none", pos_weight=weights
        )
    else:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    if mode == "M3":
        probability = torch.sigmoid(logits)
        pt = probability * truth + (1 - probability) * (1 - truth)
        alpha = 0.25 * truth + 0.75 * (1 - truth)
        loss = loss * alpha * (1 - pt).pow(2)
    total = logits.sum() * 0
    for k in range(3):
        if mask[:, :, k].any():
            total = total + 0.1 * loss[:, :, k][mask[:, :, k]].mean()
    return total
