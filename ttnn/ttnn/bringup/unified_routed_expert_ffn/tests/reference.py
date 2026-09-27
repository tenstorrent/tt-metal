# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.unified_routed_expert_moe (fork of deepseek_prefill unified_routed_expert_ffn),
for the fork's own tests.

For every local expert slot le: g = global_expert_idx_table[le], n = expert_token_counts[g], r = expert_region_offsets[g];
rows r .. r + n - 1 of the dispatched buffer x go through the expert's gated FFN
    y = act(x @ gate_proj[le]) * (x @ up_proj[le]) @ down_proj[le]      (gate/up [emb, hidden], down [hidden, emb])
and land in the same rows of the output. Other rows (tile padding between regions, the tail) are don't-care.
act: Silu (source) or GeluTanh (fork change); high_precision (fork change) keeps x, the intermediates and the output in
bf16 at the requested fidelity / fp32 dest, which the reference approximates in float32 on the bf16 x and the
bfp8-rounded weights the device holds. The routing helpers (the dispatch fork's rules) build valid counts / regions.
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


ACT = {
    "Silu": torch.nn.functional.silu,
    "GeluTanh": lambda v: torch.nn.functional.gelu(v, approximate="tanh"),
}


def expert_ffn(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, activation: str) -> torch.Tensor:
    """x [n, emb]; wg, wu [emb, hidden]; wd [hidden, emb] -> [n, emb], float32."""
    x, wg, wu, wd = (t.float() for t in (x, wg, wu, wd))
    return (ACT[activation](x @ wg) * (x @ wu)) @ wd
