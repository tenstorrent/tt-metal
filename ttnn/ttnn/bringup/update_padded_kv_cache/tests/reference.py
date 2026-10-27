# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.update_padded_kv_cache in torch: which cache rows a call writes, and from which input row.

Semantics (the op's docstring, device operation and reader / writer kernels; the source op's tests
_rotated_chip_positions and the TP-sharded readback), in page rows (a token row for ROW_MAJOR, the only layout here):
- A call writes the global rows [start, start + SP W) of one cache slab (batch row slot num_layers + layer), W = the
  input's rows per chip. With valid_global (either path) only the rows below ceil32(valid) are written; the 32-row
  block holding the last real row is written whole.
- Input: mesh row s (the cluster_axis coordinate) holds the global rows of that range it owns (row g belongs to mesh row
  (g / W) mod SP), in increasing order: the chunk's start rotates the start's mesh row (its rows past the start first,
  then those of the next slab). With tp_axis the input is replicated over the TP axis; without, over the other axis.
- Cache: block-cyclic over L = SP TP chips with stripe c = W / TP rows a chip (TP = 1 without tp_axis): row g lives on
  chip (g / c) mod L, local row (g / (c L)) c + g mod c. Chip (r, col) is linear chip r TP + col with tp_axis, r
  without (the TP-replicated chips of a mesh row then hold the same rows).
Every other cache element keeps its value.
"""

from __future__ import annotations

import torch

ROW_BLOCK = 32  # the valid_global clamp's granularity, in page rows


def written_rows(start: int, valid: int | None, sp: int, w: int) -> torch.Tensor:
    """The global rows the call writes."""
    end = start + sp * w
    if valid is not None:
        end = min(end, -(-valid // ROW_BLOCK) * ROW_BLOCK)
    return torch.arange(start, max(start, end))


def input_rows(start: int, sp: int, w: int) -> list[torch.Tensor]:
    """For each mesh row s: the global row each of its w input rows holds, in input order."""
    g = torch.arange(start, start + sp * w)
    owner = (g // w) % sp
    rows = [g[owner == s] for s in range(sp)]
    assert all(r.numel() == w for r in rows), [r.numel() for r in rows]
    return rows


def chip_of(g: torch.Tensor, stripe: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(linear chip, local row) of global rows g in the block-cyclic cache of n chips, stripe rows a chip."""
    return (g // stripe) % n, (g // (stripe * n)) * stripe + g % stripe


def update(
    before: list[torch.Tensor],
    x: torch.Tensor,
    mesh: tuple[int, int],
    *,
    start: int,
    valid: int | None,
    cluster_axis: int,
    tp_axis: int | None,
    slot: int,
    layer: int,
    num_layers: int,
) -> list[torch.Tensor]:
    """The cache after the call. ``before``: every chip's cache [B, heads, rows, width], chip d = r cols + col of a
    (rows, cols) mesh; ``x`` [SP, heads, W, width]: mesh row s's input (the same on each chip of the row)."""
    assert cluster_axis == 0 and tp_axis in (None, 1), (cluster_axis, tp_axis)
    sp, cols = mesh
    tp = cols if tp_axis is not None else 1
    w = x.shape[-2]
    assert w % tp == 0, (w, tp)
    stripe, n = w // tp, sp * tp
    batch = slot * num_layers + layer
    want = [t.clone() for t in before]
    rows_in = input_rows(start, sp, w)
    wr = written_rows(start, valid, sp, w)
    for s in range(sp):
        pos = torch.full((sp * w,), -1, dtype=torch.long)  # global row - start -> input row of mesh row s
        pos[rows_in[s] - start] = torch.arange(w)
        g = wr[pos[wr - start] >= 0]  # the written rows mesh row s supplies
        i = pos[g - start]
        chip, local = chip_of(g, stripe, n)
        assert not local.numel() or int(local.max()) < before[0].shape[-2], "a written row is past the cache end"
        for col in range(cols):
            q = s * tp + (col if tp_axis is not None else 0)
            m = chip == q
            want[s * cols + col][batch][:, local[m]] = x[s][:, i[m]]
    return want
