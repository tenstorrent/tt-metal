# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.combine (fork of deepseek_prefill combine), for the fork's own tests.

Covered here: a dispatch group of one device (the fork's change: dispatch axis of size 1, no fabric). For every local
expert e of the device, rows regions[e] .. regions[e] + counts[e] - 1 of the flat expert-output buffer carry metadata
[linearized_coord, t, k]; the op writes that row to output[0, 0, t, k, :] ([1, 1, S, K, H], ROW_MAJOR). With
init_zeros every (t, k) no row targets reads 0. The routing helpers (the same rules as the dispatch fork's tests)
build a valid metadata / counts / regions triple from random top-k ids, as dispatch + offset_cumsum would.
Sources: models/demos/deepseek_v3_d_p/reference/tt/moe/combine.py (TorchCombineModule), the op's kernels.
"""

from __future__ import annotations

import torch

TILE = 32


def random_topk(seq: int, num_experts: int, k: int, g: torch.Generator) -> torch.Tensor:
    """[seq, k] int64: k distinct experts per token, uniformly at random (a token picks an expert at most once)."""
    return torch.rand(seq, num_experts, generator=g).argsort(dim=1)[:, :k]


def dispatch_table(num_experts: int, num_groups: int) -> torch.Tensor:
    """[num_groups, num_experts + 1] int32 for a 1-chip dispatch group (ExpertMapping.create_dispatch_table with
    dispatch_group_size=1): experts g*epc..(g+1)*epc-1 -> chip 0, the rest (and the sentinel column) -> -1."""
    epc = num_experts // num_groups
    t = torch.full((num_groups, num_experts + 1), -1, dtype=torch.int32)
    for g in range(num_groups):
        t[g, g * epc : (g + 1) * epc] = 0
    return t


def offsets_counts_regions(indices: torch.Tensor, table_row: torch.Tensor, experts_per_chip: int):
    """What masked_bincount + offset_cumsum produce for one device of a 1-device dispatch group.
    indices [S, K]; table_row [E+1]. Returns (offsets [E], counts [E], regions [E]) int64: counts = histogram of the
    present experts, regions = per-chip exclusive prefix sum of the tile-aligned counts, offsets = regions (a
    1-device group has no earlier source device)."""
    E = table_row.numel() - 1
    present = table_row[:E] != -1
    counts = torch.bincount(indices.reshape(-1), minlength=E)[:E] * present
    aligned = ((counts + TILE - 1) // TILE * TILE).reshape(-1, experts_per_chip)
    regions = (aligned.cumsum(-1) - aligned).reshape(E)
    return regions.clone(), counts, regions


def slots(indices: torch.Tensor, table_row: torch.Tensor, offsets: torch.Tensor):
    """Every (t, k) routed to a present expert and its buffer row. Returns (t, k, e, row) int64 tensors; within one
    expert, rows are filled in token order starting at offsets[e]."""
    E = table_row.numel() - 1
    present = torch.zeros(E + 1, dtype=torch.bool)
    present[:E] = table_row[:E] != -1
    t, k = present[indices].nonzero(as_tuple=True)  # row-major: sorted by t, then k
    e = indices[t, k]
    order = torch.sort(e, stable=True).indices  # group by expert, token order kept within an expert
    t, k, e = t[order], k[order], e[order]
    first = torch.searchsorted(e, e)  # index of the expert's first entry
    rank = torch.arange(e.numel()) - first
    return t, k, e, offsets[e] + rank


def metadata(indices, table_row, offsets, rows_total: int, group: int, num_groups: int, chip: int = 0):
    """What dispatch leaves in the metadata buffer [rows_total, 3] (int32; -1 where no (t, k) lands) and the written
    rows' (t, k, row)."""
    t, k, e, row = slots(indices, table_row, offsets)
    m = torch.full((rows_total, 3), -1, dtype=torch.int32)
    m[row, 0] = chip * num_groups + group
    m[row, 1] = t.to(torch.int32)
    m[row, 2] = k.to(torch.int32)
    return m, t, k, row


def combine(buffer, t, k, row, seq: int, top_k: int, init_zeros: bool = True):
    """Expected output [S, K, H] for one device: buffer[row] at (t, k); zeros elsewhere (init_zeros)."""
    assert init_zeros, "without init_zeros the untouched (t, k) are undefined"
    y = torch.zeros(seq, top_k, buffer.shape[-1], dtype=buffer.dtype)
    y[t, k] = buffer[row]
    return y
