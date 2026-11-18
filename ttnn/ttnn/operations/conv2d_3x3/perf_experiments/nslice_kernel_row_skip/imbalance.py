# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn): per grid row of the BLOCK N-slice walk, kernel-row work in units of (tile-row x kernel row)
(x K slice x N-slice per step, the same factor on every row of a case):
  base  = 3 x bm (every kernel row, every row of the block, padding rows included)
  skip  = sum over ring row groups of popcount(group mask) x group rows (NSLICE_KH_SKIP, this experiment)
  real  = sum over valid tile-rows of their in-image kernel rows (the floor a per-tile-row skip would reach)
and the balanced row: total real / rows, what a non-uniform M split (whole tile-rows) could approach."""
TILE = 32


def masks(H, W, d, p, s, bm, sb_h, rows):
    Ho = (H + 2 * p - 2 * d - 1) // s + 1
    Wo = (W + 2 * p - 2 * d - 1) // s + 1
    m_tiles = -(-Ho * Wo // TILE)
    out = []
    for row in range(rows):
        t0 = row * bm
        groups, reals = [], []
        for g0 in range(0, bm, sb_h):
            m, rr = 0, []
            for r in range(g0, min(g0 + sb_h, bm)):
                t = t0 + r
                if t >= m_tiles:
                    rr.append(0)
                    continue
                o_a, o_b = t * TILE, min(t * TILE + TILE, Ho * Wo)
                mt = 0
                for oy in range(o_a // Wo, (o_b - 1) // Wo + 1):
                    for kh in range(3):
                        if 0 <= oy * s + kh * d - p < H:
                            mt |= 1 << kh
                m |= mt
                rr.append(bin(mt).count("1"))
            groups.append((m or 2, min(g0 + sb_h, bm) - g0))
            reals.append(sum(rr))
        out.append((groups, sum(reals)))
    return out


CASES = {
    "aspp3 (d 18)": (32, 64, 18, 18, 1),
    "aspp2 (d 12)": (32, 64, 12, 12, 1),
    "aspp1 (d 6)": (32, 64, 6, 6, 1),
    "res5_2 (d 8)": (32, 64, 8, 8, 1),
    "res5_1 (d 4)": (32, 64, 4, 4, 1),
    "res5_0 (d 2)": (32, 64, 2, 2, 1),
    "res4_x (d 1)": (32, 64, 1, 1, 1),
    "res4_0 (s 2)": (64, 128, 1, 1, 2),
}
for name, (H, W, d, p, s) in CASES.items():
    bm, sb_h, rows = 7, 4, 10
    rs = masks(H, W, d, p, s, bm, sb_h, rows)
    base = [3 * bm] * rows
    skip = [sum(bin(m).count("1") * n for m, n in g) for g, _ in rs]
    real = [r for _, r in rs]
    total_real = sum(real)
    print(
        f"{name:<14} base {base[0]:>2}/row | skip per row {skip} max {max(skip)} | real per row {real} "
        f"max {max(real)} | balanced {total_real / rows:.1f} (max-row cut vs skip: {1 - total_real / rows / max(skip):.0%})"
    )
