# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.fabric_reduce_scatter (its docstring, fabric_reduce_scatter.py), for the fork's
model-case tests.

Groups as fabric_all_gather's: cluster_axis 0 -> one mesh column, 1 -> one mesh row, None -> the whole mesh. The input
of every chip is [1, 1, G * Sb, W] (block j = rows j * Sb ..); chip d gets block g (g = d's row-major rank in its group,
whatever order the group travels in) of the sum over the group's G inputs, [1, 1, Sb, W]."""

from __future__ import annotations

import torch


def group(mesh: tuple[int, int], d: int, cluster_axis: int | None) -> list[int]:
    """Row-major device indices of device d's group, in rank order."""
    rows, cols = mesh
    r, c = divmod(d, cols)
    if cluster_axis is None:
        return list(range(rows * cols))
    if cluster_axis == 0:
        return [i * cols + c for i in range(rows)]
    return [r * cols + j for j in range(cols)]


def reduce_scatter(xs, mesh, d, *, cluster_axis):
    """xs: every device's input (row-major list). -> device d's expected output, float64."""
    members = group(mesh, d, cluster_axis)
    G, g = len(members), members.index(d)
    sb = xs[0].shape[2] // G
    return sum(xs[m][:, :, g * sb : (g + 1) * sb].double() for m in members)
