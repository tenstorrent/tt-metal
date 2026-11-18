# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn): per core, the TILE slab rows owned by a higher-index input shard (the split fill's writer
share), mirroring the kernels' tile_slab_plan. python3 hi_rows.py"""
TILE = 32


def div_up(a, b):
    return -(-a // b)


def plan_hi_rows(H, W, s, p, d, o_first, o_last, shard_rows, my_shard):
    wo_out = (W + 2 * p - 2 * d - 1) // s + 1
    ho0, wo0 = divmod(o_first, wo_out)
    ho1, wo1 = divmod(o_last, wo_out)
    band_len = (ho1 - ho0) * s * W + (wo1 - wo0) * s + 2 * d + 1
    f = (ho0 * s - p) * W + wo0 * s - p
    prev = None
    hi_rows = rows = 0
    for kh in range(3):
        v_lo = f + kh * d * W
        lo, hi = max(v_lo, 0), min(v_lo + band_len, H * W)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // TILE, div_up(hi, TILE)
        first_new = prev[1] if prev is not None and t_lo < prev[1] else t_lo
        for r in range(first_new, t_hi):
            rows += 1
            hi_rows += r // shard_rows > my_shard
        prev = (t_lo, max(t_hi, prev[1]) if prev and t_lo < prev[1] else t_hi)
    return rows, hi_rows


# (name, H, W, C_in, stride, pad, dil, shard_px, grid_rows, slice_tiles, tile_bytes)
CASES = [
    ("aspp1", 32, 64, 2048, 1, 6, 6, 256, 8, 6, 1088),
    ("aspp2", 32, 64, 2048, 1, 12, 12, 256, 8, 6, 1088),
    ("aspp3", 32, 64, 1024, 1, 18, 18, 256, 8, 3, 1088),
    ("res5_0", 32, 64, 512, 1, 2, 2, 224, 10, 2, 1088),
    ("res5_1", 32, 64, 512, 1, 4, 4, 224, 10, 2, 1088),
    ("res5_2", 32, 64, 512, 1, 8, 8, 224, 10, 2, 1088),
    ("res4", 32, 64, 256, 1, 1, 1, 256, 8, 1, 1088),
    ("res4_0", 64, 128, 256, 2, 1, 1, 1024, 8, 1, 1088),
]
for name, H, W, C, s, p, d, shard_px, rows_g, ct, tb in CASES:
    wo_out = (W + 2 * p - 2 * d - 1) // s + 1
    ho_out = (H + 2 * p - 2 * d - 1) // s + 1
    n_out = ho_out * wo_out
    core_out = div_up(div_up(n_out, TILE), rows_g) * TILE
    shard_rows = shard_px // TILE
    res = []
    for m in range(rows_g):
        o_first = m * core_out
        o_last = min((m + 1) * core_out, n_out) - 1
        if o_first > o_last:
            continue
        res.append(plan_hi_rows(H, W, s, p, d, o_first, o_last, shard_rows, m))
    print(f"{name:7} rows/hi per core {res}  max hi bytes {max(h for _, h in res) * ct * tb}")
