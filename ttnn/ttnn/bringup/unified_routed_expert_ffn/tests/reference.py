# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.unified_routed_expert_moe (fork of deepseek_prefill unified_routed_expert_ffn),
for the fork's own tests.

For every local expert slot le: g = global_expert_idx_table[le], n = expert_token_counts[g], r = expert_region_offsets[g];
rows r .. r + n - 1 of the dispatched buffer x go through the expert's gated FFN
    y = act(x @ gate_proj[le]) * (x @ up_proj[le]) @ down_proj[le]      (gate/up [emb, hidden], down [hidden, emb])
and land in the same rows of the output. Other rows (tile padding between regions, the tail) are don't-care.
act: Silu (source), GeluTanh (fork change) or ClampedSiluGlu (source; clamps gate and up, see GLU); high_precision
(fork change) keeps x, the intermediates and the output in bf16 at the requested fidelity / fp32 dest, which the
reference approximates in float32 on the bf16 x and the bfp8-rounded weights the device holds. The routing helpers (the dispatch fork's rules) build valid counts / regions.
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


# Gated activations that also clamp the up projection: act(gate, up). ClampedSiluGlu (DeepSeek V4, Hy4) =
# silu(min(gate, L)) * clamp(up, -L, L) with L = 10 (ClampedSiluGluConfigDsV4, baked in the kernel;
# device/unified_routed_expert_ffn_types.hpp).
CLAMP_LIMIT = 10.0
GLU = {
    "ClampedSiluGlu": lambda gt, up: torch.nn.functional.silu(gt.clamp(max=CLAMP_LIMIT))
    * up.clamp(-CLAMP_LIMIT, CLAMP_LIMIT),
}


def expert_ffn(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, activation: str) -> torch.Tensor:
    """x [n, emb]; wg, wu [emb, hidden]; wd [hidden, emb] -> [n, emb], float32."""
    x, wg, wu, wd = (t.float() for t in (x, wg, wu, wd))
    if activation in GLU:
        return GLU[activation](x @ wg, x @ wu) @ wd
    return (ACT[activation](x @ wg) * (x @ wu)) @ wd


# --------------------------------------------------------------------------- DeepSeek-V4.1 h options (P.2)
# unified_routed_expert_moe(..., routing_weights=R, fp8_act_quant=True, weights_transposed=True): per local expert le
# and dispatched row i of its region,
#     h_i = act(x_i @ W1^T, x_i @ W3^T) * R[i, le]                  W1, W3 [hidden, emb], W2 [emb, hidden] as stored
#     y_i = fp8_act_qdq(h_i) @ W2^T
# fp8_act_qdq = DeepSeek's act_quant(h, 32, 'ue8m0') followed by its dequantizing GEMM (inference kernel.py; the same
# arithmetic as models/demos/deepseek_v41_flash_d_p/reference/kernels.py:act_quant_dequant).
FP8_MAX = 448.0
FP8_AMAX_MIN = 1e-4
INV_FP8_MAX = torch.tensor(1.0 / 448.0, dtype=torch.float32)  # fp32(1/448), the kernels' constant


def pow2_ceil(t: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2 t) of positive normal fp32 t, from the bits (ue8m0)."""
    b = t.float().contiguous().view(torch.int32)
    return ((b + 0x7FFFFF) & 0x7F800000).view(torch.float32)


def fp8_act_scale(h: torch.Tensor, block: int = 32) -> torch.Tensor:
    """The per-block ue8m0 scale s [..., n/block, 1] of fp32 h [..., n]."""
    g = h.float().reshape(*h.shape[:-1], h.shape[-1] // block, block)
    return pow2_ceil(g.abs().amax(dim=-1, keepdim=True).clamp_min(FP8_AMAX_MIN) * INV_FP8_MAX)


def fp8_act_qdq(h: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp32 e4m3(clamp(h / s, +-448)) * s per block of `block` values along the last dim (round to nearest even)."""
    g = h.float().reshape(*h.shape[:-1], h.shape[-1] // block, block)
    s = fp8_act_scale(h, block)
    return ((g / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s).reshape(h.shape)


def routed_h(x, w1, w3, r, activation: str = "ClampedSiluGlu") -> torch.Tensor:
    """x [n, emb] bf16; w1, w3 [hidden, emb] (stored); r [n] fp32 or None -> h [n, hidden] fp32 (before the quant).
    The GEMMs in float64, rounded once to fp32 (the device accumulates in fp32)."""
    g = (x.double() @ w1.double().T).float()
    u = (x.double() @ w3.double().T).float()
    h = GLU[activation](g, u)
    return h if r is None else h * r.float()[:, None]
