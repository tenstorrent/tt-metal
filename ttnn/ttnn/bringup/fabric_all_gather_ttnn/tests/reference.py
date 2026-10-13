# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.fabric_all_gather (the contract of ttnn.experimental.high_bw_all_gather, README.md
and the op's docstring), for the fork's model-case tests.

Groups: cluster_axis 0 -> the chips of one mesh column, 1 -> one mesh row, None -> every chip; a group's output is
its members' inputs concatenated along `dim` in row-major chip order (rank g = the chip's row-major index in the group).
gathered_dim_size (a partial gather): rank g writes the first gathered_dim_size / G entries of its input along `dim`
into its fixed slot [g * L, g * L + gathered_dim_size / G) (L = the input's length along `dim`); every other output
byte keeps what the preallocated output held."""

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


def all_gather(xs, mesh, d, *, dim, cluster_axis, fill, gathered_dim_size=None):
    """xs: every device's input (row-major list); fill: the output's prior contents. -> device d's expected output."""
    members = group(mesh, d, cluster_axis)
    if gathered_dim_size is None:
        return torch.cat([xs[m] for m in members], dim=dim)
    out = fill.clone()
    L = xs[0].shape[dim]
    active = gathered_dim_size // len(members)
    assert gathered_dim_size % len(members) == 0 and active <= L
    for g, m in enumerate(members):
        out.narrow(dim, g * L, active).copy_(xs[m].narrow(dim, 0, active))
    return out
