# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.dispatch (fork of deepseek_prefill dispatch), for the fork's own tests.

Covered here: a dispatch group of one device (the fork's change: dispatch axis of size 1, no fabric), with the mesh's
other axis holding the dispatch groups. For group g (mesh column g) the dispatch table row maps the experts that live
on the group's chip to 0 and every other expert to -1 (the trailing sentinel column is -1 too). For every token t and
top-k slot k whose expert e is present (table[e] != -1), the op writes x[t] to the flat buffer row
offsets[e] + (number of earlier tokens routed to e), and metadata [linearized_coord, t, k] to the same row, with
linearized_coord = chip * num_dispatch_groups + group (chip = 0 here). Rows no (t, k) lands on are don't-care.
Sources: reference/tt/moe/dispatch.py (TorchDispatchModule), init_helpers.ExpertMapping, the op's reader kernel.
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


def dispatch(
    x: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    table_row: torch.Tensor,
    group: int,
    num_groups: int,
    chip: int = 0,
):
    """Expected (rows, buffer_rows [n, H], metadata_rows [n, 3]) for one device: only the rows the op writes."""
    t, k, e, row = slots(indices, table_row, offsets)
    lin = chip * num_groups + group
    meta = torch.stack([torch.full_like(t, lin), t, k], dim=1)
    return row, x[t], meta


# --- Dispatch groups of more than one device (the source op's own path, fabric on; mimo_v2_6_d_p_2x2 O.1) ----------
# The mesh columns are the dispatch groups and the rows the chips of a group (cluster_axis 0). Column c's experts are
# c*dgs*epc .. (c+1)*dgs*epc - 1, chip r of the group holds c*dgs*epc + r*epc .. + epc - 1 (ExpertMapping col-major).
# Source device (r, c) sends token t / slot k (expert e of group c) to chip table[c][e] of its column, at row
# offsets[r][e] + (number of r's earlier tokens routed to e); offsets from offset_cumsum over the group (regions of
# the group totals + the histograms of the group's earlier source devices). Metadata [r * num_groups + c, t, k].


def dispatch_table_groups(num_experts: int, dispatch_group_size: int, num_groups: int) -> torch.Tensor:
    """[num_groups, num_experts + 1] int32, ExpertMapping.create_dispatch_table: group c's experts -> their chip in
    the group (0 .. dispatch_group_size - 1), every other expert and the sentinel column -> -1."""
    epg = num_experts // num_groups
    epc = epg // dispatch_group_size
    t = torch.full((num_groups, num_experts + 1), -1, dtype=torch.int32)
    for c in range(num_groups):
        t[c, c * epg : (c + 1) * epg] = torch.arange(epg, dtype=torch.int32) // epc
    return t


def group_routing(indices: torch.Tensor, table_row: torch.Tensor, experts_per_chip: int):
    """masked_bincount + offset_cumsum over one dispatch group. indices [D, S, K] (the group's D source devices, in
    order); table_row [E + 1]. Returns (offsets [D, E], totals [E], regions [E]) int64."""
    D = indices.shape[0]
    E = table_row.numel() - 1
    present = (table_row[:E] != -1).to(torch.int64)
    hists = torch.stack([torch.bincount(indices[d].reshape(-1), minlength=E)[:E] * present for d in range(D)])
    totals = hists.sum(0)
    aligned = ((totals + TILE - 1) // TILE * TILE).reshape(-1, experts_per_chip)
    regions = (aligned.cumsum(-1) - aligned).reshape(E)
    before = hists.cumsum(0) - hists
    return before + regions, totals, regions


def group_slots(indices: torch.Tensor, table_row: torch.Tensor, offsets: torch.Tensor, dest_chip: int):
    """Every (source r, t, k) of the group routed to an expert on chip dest_chip, and its row in that chip's buffer.
    indices [D, S, K], offsets [D, E]. Returns a list over r of (t, k, e, row) int64 tensors."""
    E = table_row.numel() - 1
    mask_row = torch.full_like(table_row, -1)
    mask_row[:E] = torch.where(table_row[:E] == dest_chip, 0, -1)
    return [slots(indices[r], mask_row, offsets[r]) for r in range(indices.shape[0])]
