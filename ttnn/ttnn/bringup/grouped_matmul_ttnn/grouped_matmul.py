# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.grouped_matmul: a batched matmul with one weight per batch (group), y[g] = x[g] @ w[g], laid out one
grid row per group (deepseek_v41_flash_d_p P.81, ATTN_ROOFLINE: the attention's grouped wo_a, 8 groups x [160, 4096] @
[4096, 1024] a chip).

The operation is the native C++ ttnn.grouped_matmul (ttnn/cpp/ttnn/operations/grouped_matmul), which plans the
program, checks its inputs (refusals raise AssertionError, as the Python op's asserts did) and caches the program.
This module keeps ``supported`` (whether a call has a layout, asked by models before they call) and ``column_split``
(bank.py's column split), and the public entry point.

    grid      G rows (one a group) x GX columns (GX <= the grid width, <= Nt); core (cx, g): all MT row tiles x the
              PN(cx) column tiles from n0(cx) of group g (column_split: as even as possible)
    in0       x[g] streamed in K blocks of BW tiles, multicast along the row from its column-0 core
    in1       each core reads its own [BW x PN] weight block
    compute   fp32 partials over the K blocks, one pack to the output dtype at the end
    out       y[g] tiles written by each core
"""

from __future__ import annotations

import ttnn

L1_BUDGET = 1024 * 1024
_TILE_BYTES = {}


def _tile_bytes(dtype) -> int:
    if dtype not in _TILE_BYTES:
        _TILE_BYTES[dtype] = ttnn.tile_size(dtype)
    return _TILE_BYTES[dtype]


def column_split(nt: int, gx: int) -> list[int]:
    """Output column tiles per grid column, as even as possible (the wider ones first)."""
    base, rem = divmod(nt, gx)
    return [base + (1 if i < rem else 0) for i in range(gx)]


def _grid(device, max_grid=None) -> tuple[int, int]:
    g = device.compute_with_storage_grid_size()
    if max_grid is None:
        return g.x, g.y
    mx, my = (max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid)
    return max(1, min(g.x, mx)), max(1, min(g.y, my))


def plan(x_shape, w_shape, x_dtype, w_dtype, out_dtype, device, max_grid=None, block_w=None):
    """(G, MT, KT, NT, GX, pns, BW) of a call, or None when the layout does not apply (the C++ op's planner rule:
    more groups than grid rows, a batch shape other than [1, G], dims not tile multiples, nothing fits L1)."""
    if len(x_shape) != 4 or len(w_shape) != 4 or x_shape[0] != 1 or w_shape[0] != 1:
        return None
    G, M, K = x_shape[1], x_shape[2], x_shape[3]
    if w_shape[1] != G or w_shape[2] != K or M % 32 or K % 32 or w_shape[3] % 32:
        return None
    MT, KT, NT = M // 32, K // 32, w_shape[3] // 32
    gx, gy = _grid(device, max_grid)
    if G > gy:
        return None
    GX = min(gx, NT)
    pns = column_split(NT, GX)
    pn = max(pns)
    a, b, o = _tile_bytes(x_dtype), _tile_bytes(w_dtype), _tile_bytes(out_dtype)
    part = MT * pn * _tile_bytes(ttnn.float32)
    for bw in (block_w,) if block_w else (32, 16, 8, 4, 2, 1):
        if KT % bw:
            continue
        if 2 * MT * bw * a + 2 * bw * pn * b + part + MT * pn * o <= L1_BUDGET:
            return G, MT, KT, NT, GX, pns, bw
    return None


_SUPPORTED = {}


def supported(x_shape, w_shape, x_dtype, w_dtype, out_dtype, device, max_grid=None, block_w=None) -> bool:
    """Whether the call has a layout (``plan``); cached by its arguments (a model asks on every call)."""
    mg = None if max_grid is None else ((max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid))
    key = (tuple(x_shape), tuple(w_shape), str(x_dtype), str(w_dtype), str(out_dtype), device.id(), mg, block_w)
    hit = _SUPPORTED.get(key)
    if hit is None:
        hit = _SUPPORTED[key] = (
            plan(x_shape, w_shape, x_dtype, w_dtype, out_dtype, device, max_grid, block_w) is not None
        )
    return hit


# Public entry point: the registered C++ operation (its binding returns the caller's output_tensor object as is).
grouped_matmul = ttnn.grouped_matmul
