# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side data + torch reference for the compact-partial combine bake-off.

The resident input CB holds exactly the pages ONE call of the variant consumes, and the kernel
re-exposes those same physical pages for every call (see combine_bench.py's docstring), so the
reference below walks the same logical stream the kernel does -- the check stays exact even where
the modelled logical stream is far larger than L1.

Value scale is realistic: a member's partial is sum(x^2) over its W/GROUP_SIZE-wide slice of a
W = 1024 row of N(0,1) activations, so a partial averages 1024/GROUP_SIZE and the group total
averages W -- which puts the finalize's rsqrt argument at ~1.0, where the real op runs it.
"""

import struct


TILE = 32
W_LOGICAL = 1024
EPS = 1e-5


def bits(f):
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


INV_W_BITS = bits(1.0 / W_LOGICAL)
EPS_BITS = bits(EPS)


def _rsqrt_ref(total):
    import torch

    return torch.rsqrt(total.double() * (1.0 / W_LOGICAL) + EPS)


def _col_partial_tiles(vals, garbage):
    """`vals` [n, 32] -> a [n*32, 32] fp32 tile column with vals in COLUMN 0.

    Columns 1..15 carry finite GARBAGE (the op's reduce leaves partial sums there) and columns
    16..31 are 0 (the writer's boot-zeroing of faces 1 and 3).  A variant that reads anything
    outside column 0 will fail the correctness gate.
    """
    import torch

    n = vals.shape[0]
    t = torch.zeros(n * TILE, TILE, dtype=torch.float32)
    for k in range(n):
        t[k * TILE : (k + 1) * TILE, 0] = vals[k]
        t[k * TILE : (k + 1) * TILE, 1:16] = garbage[k]  # [32, 15]
    return t


def _compact_tiles(vals):
    """`vals` [n, rows, 32] -> a [n*32, 32] fp32 tile column, tile k COLUMN r = vals[k, r, :].

    Columns >= rows are exactly 0 (what the pack's matmul leaves there).
    """
    import torch

    n, rows, _ = vals.shape
    t = torch.zeros(n * TILE, TILE, dtype=torch.float32)
    for k in range(n):
        for r in range(rows):
            t[k * TILE : (k + 1) * TILE, r] = vals[k, r]
    return t


def make_case(variant, group_size, rows, seed=0):
    """-> (part_torch [P*32,32], bank_torch [(2*rows+1)*32,32], expect) where `expect` is a list of
    (out_tile_index, column, reference_vector[32]) triples to check in the fp32 output."""
    import torch

    g = torch.Generator().manual_seed(seed + 1000 * group_size + rows)
    mean = W_LOGICAL / group_size

    def partials(*shape):
        import torch

        return (mean + (mean / 8.0) * torch.randn(*shape, generator=g)).clamp_min(mean / 4)

    def garbage(n):
        import torch

        return mean * torch.rand(n, TILE, 15, generator=g)

    bank = torch.zeros((2 * rows + 1) * TILE, TILE, dtype=torch.float32)
    for r in range(rows):
        bank[r * TILE + 0, r] = 1.0  # E_r: pack   column 0 -> column r
        bank[(rows + r) * TILE + r, 0] = 1.0  # F_r: un-pack column r -> column 0
    # bank page 2*rows stays all-zero (the optional explicit DEST seed).

    if variant == "base_phase0":
        # rows physical column-partials, re-read by all GROUP_SIZE fold calls.
        p = partials(rows, TILE)
        part = _col_partial_tiles(p, garbage(rows))
        stat = _rsqrt_ref(group_size * p)
        expect = [(r, 0, stat[r]) for r in range(rows)]
    elif variant == "base_l1acc":
        # GROUP_SIZE physical column-partials, re-read by all `rows` fold calls.
        p = partials(group_size, TILE)
        part = _col_partial_tiles(p, garbage(group_size))
        stat = _rsqrt_ref(p.sum(dim=0))
        expect = [(r, 0, stat) for r in range(rows)]
    elif variant in ("cand_root", "cand_recv"):
        v = partials(group_size, rows, TILE)  # [g][r][i]
        part = _compact_tiles(v)
        stat = _rsqrt_ref(v.sum(dim=0))  # [rows, 32]
        if variant == "cand_recv":
            expect = [(0, r, stat[r]) for r in range(rows)]  # ONE compact output tile
        else:
            expect = [(r, 0, stat[r]) for r in range(rows)]  # un-packed on the root
    elif variant == "member_pack":
        p = partials(rows, TILE)
        part = _col_partial_tiles(p, garbage(rows))
        expect = [(0, r, p[r].double()) for r in range(rows)]
        # Columns >= rows of the packed tile must be EXACTLY 0 (nothing else may leak in).
        expect += [(0, c, torch.zeros(TILE, dtype=torch.float64)) for c in range(rows, TILE)]
    elif variant == "member_copy":
        p = partials(rows, TILE)
        part = _col_partial_tiles(p, garbage(rows))
        expect = [(r, 0, p[r].double()) for r in range(rows)]
    elif variant == "recv_unpack":
        v = partials(1, rows, TILE)
        part = _compact_tiles(v)
        expect = [(r, 0, v[0, r].double()) for r in range(rows)]
    else:
        raise ValueError(variant)
    return part, bank, expect


def pcc(a, b):
    import torch

    a = a.double().flatten()
    b = b.double().flatten()
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel_rms(got, ref):
    import torch

    got = got.double().flatten()
    ref = ref.double().flatten()
    return float(torch.sqrt(((got - ref) ** 2).mean()) / torch.sqrt((ref**2).mean()))


def eff_pcc(got, ref):
    """PCC the way the OP is gated, not the way a variance-starved stat vector would be.

    The bench's output is 1/rms, a 32-vector whose values all sit near 1.0 (the group total is ~W
    by construction), so a raw PCC on it is dominated by its own tiny variance and reads ~0.996
    even for the op's CURRENT combine.  The op's soft gate is on `out = x * (1/rms) * gamma`,
    whose PCC is set by the stat's RELATIVE error against activation-scale variance -- so multiply
    both by the same fixed N(0,1) draw before correlating.  Same number the golden test would see.
    """
    import torch

    x = torch.randn(got.numel(), generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    return pcc(x * got.double().flatten(), x * ref.double().flatten())
