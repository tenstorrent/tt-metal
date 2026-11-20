# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.fabric_reduce_scatter (its docstring, fabric_reduce_scatter.py), for the fork's
model-case tests.

Groups as fabric_all_gather's: cluster_axis 0 -> one mesh column, 1 -> one mesh row, None -> the whole mesh. The input
of every chip is [1, 1, G * Sb, W] (block j = rows j * Sb ..); chip d gets block g (g = d's row-major rank in its group,
whatever order the group travels in) of the sum over the group's G inputs, [1, 1, Sb, W]. dim 3: blocks of columns
instead (input [1, 1, S, G * Wb] -> chip d's [1, 1, S, Wb])."""

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


def blocks(xs, mesh, d, *, cluster_axis, dim=2):
    """Device d's block of every member's input, in the group's rank order (the addends of its output)."""
    members = group(mesh, d, cluster_axis)
    G, g = len(members), members.index(d)
    sb = xs[0].shape[dim] // G
    return [xs[m].narrow(dim, g * sb, sb) for m in members]


def reduce_scatter(xs, mesh, d, *, cluster_axis, dim=2):
    """xs: every device's input (row-major list). -> device d's expected output, float64."""
    return sum(b.double() for b in blocks(xs, mesh, d, cluster_axis=cluster_axis, dim=dim))
