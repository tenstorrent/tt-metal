# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.offset_cumsum (fork of deepseek_prefill offset_cumsum), for the fork's tests.

Input: each device of a dispatch group (the devices along cluster_axis) holds a histogram [E] of how many of its
tokens go to each expert. Outputs, per device at position p of its group:
  - totals [1, E]  = sum of the group's histograms;
  - regions [1, E] = per chip (E / experts_per_chip chips), exclusive prefix sum of the totals rounded up to a tile (32);
  - offsets [1, E] = regions + sum of the histograms of the devices before p in the group.
The fork's change: a group of one device (cluster axis of size 1) skips the all_gather; the formulas are the same.
Source: models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_offset_cumsum.py (torch_offset_cumsum).
"""

import torch

TILE = 32


def offset_cumsum(hists: torch.Tensor, experts_per_chip: int):
    """hists [D, E] (the D devices of one dispatch group, in order) -> (offsets [D, E], totals [E], regions [E])."""
    D, E = hists.shape
    h = hists.to(torch.int64)
    totals = h.sum(0)
    aligned = ((totals + TILE - 1) // TILE * TILE).reshape(-1, experts_per_chip)
    regions = (aligned.cumsum(-1) - aligned).reshape(E)
    before = h.cumsum(0) - h
    return before + regions, totals, regions
