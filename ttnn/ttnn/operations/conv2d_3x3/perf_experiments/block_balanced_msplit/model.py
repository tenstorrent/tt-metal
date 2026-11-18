# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn) cost model of one grid row of the BLOCK N-slice ring walk, and the work-balanced M split it
drives. Mirrors the op copy's balanced_m_ranges (op_copy/conv2d_3x3_program_descriptor.py); this file prints, per case,
the base uniform split's per-row costs next to the balanced split's.

    python3 model.py [case ...]
"""
import sys

TILE = 32
CASES = {
    # name: (H, W, C_in, C_out, d, stride, k_cols, c_in_p, rows)
    "aspp3": (32, 64, 1024, 256, 18, 1, 11, 96, 10),
    "aspp2": (32, 64, 2048, 256, 12, 1, 11, 192, 10),
    "aspp1": (32, 64, 2048, 256, 6, 1, 11, 192, 10),
    "res5_2": (32, 64, 512, 512, 8, 1, 8, 64, 10),
    "res5_1": (32, 64, 512, 512, 4, 1, 8, 64, 10),
    "res5_0": (32, 64, 512, 512, 2, 1, 8, 64, 10),
    "res4_x": (32, 64, 256, 256, 1, 1, 8, 32, 10),
}


def div_up(a, b):
    return -(-a // b)


class G:
    def __init__(self, H, W, c_in, c_out, d, s, k_cols, c_in_p):
        self.H, self.W, self.d, self.s, self.p = H, W, d, s, d
        self.Ho = (H + 2 * d - 2 * d - 1) // s + 1
        self.Wo = (W + 2 * d - 2 * d - 1) // s + 1
        self.m_tiles = div_up(self.Ho * self.Wo, TILE)
        self.ct = c_in_p // TILE
        self.kkh = div_up(3 * c_in_p, TILE)
        self.n_tiles = div_up(c_out, TILE)
        self.ns = div_up(self.n_tiles, k_cols)
        self.steps = div_up(self.n_tiles, self.ns)
        self.num_px = self.Ho * self.Wo
        self.num_in = H * W

    def band_len(self, o_first, o_last):
        ho0, wo0 = divmod(o_first, self.Wo)
        ho1, wo1 = divmod(o_last, self.Wo)
        return (ho1 - ho0) * self.s * self.W + (wo1 - wo0) * self.s + 2 * self.d + 1

    def first_tap(self, o_first):
        ho0, wo0 = divmod(o_first, self.Wo)
        return (ho0 * self.s - self.p) * self.W + wo0 * self.s - self.p

    def slab_rows(self, t0, n):
        o_first = t0 * TILE
        o_last = min((t0 + n) * TILE, self.num_px) - 1
        bl = self.band_len(o_first, o_last)
        f = self.first_tap(o_first)
        next_free, prev = 0, None
        for kh in range(3):
            v_lo = f + kh * self.d * self.W
            lo, hi = max(v_lo, 0), min(v_lo + bl, self.num_in)
            if lo >= hi:
                continue
            t_lo, t_hi = lo // TILE, div_up(hi, TILE)
            slab_row = prev[2] + (t_lo - prev[0]) if prev is not None and t_lo < prev[1] else next_free
            next_free = slab_row + (t_hi - t_lo)
            prev = (t_lo, t_hi, slab_row)
        return next_free

    def tile_mask(self, t):
        o_a, o_b = t * TILE, min(t * TILE + TILE, self.num_px)
        m = 0
        for oy in range(o_a // self.Wo, (o_b - 1) // self.Wo + 1):
            for kh in range(3):
                if 0 <= oy * self.s + kh * self.d - self.p < self.H:
                    m |= 1 << kh
        return m

    def groups(self, t0, n, sb_h):
        """(rows, kernel rows walked) per ring row group over the n valid rows (the walk's contiguous range)."""
        out = []
        for g0 in range(0, n, sb_h):
            r = min(sb_h, n - g0)
            m = 0
            for t in range(t0 + g0, t0 + g0 + r):
                m |= self.tile_mask(t)
            m = m or 0b010
            m = 0b111 if m == 0b101 else m
            out.append((r, bin(m).count("1")))
        return out


# Cost coefficients, cycles (Blackhole p150b, LoFi, fitted on the base op's zones: see README / runs/base0).
U_TILE = 115  # untilize of one slab tile (fill-paced)
T_UNIT = 150  # tap-direct gather + tilize of one unit (tile-row x K-block), plus per in0 tile:
T_TILE = 30
# walk: cycles per k-tile of one ring row group of r rows (an r x 1 subblock with runtime dims), fitted on aspp.convs.2's
# per-row walks (8 steps): 4 + 4 rows 30.0 us, 4 + 3 rows 28.0 us, 4 + 1 rows 38.2 us (3 kernel rows)
W_GROUP_KSTEP = {1: 46, 2: 48, 3: 61, 4: 70}  # CT group heights (ct_group_rows)
W_GROUP = 180  # walk: per ring row group per step (token, pack, init)


def row_cost(g, t0, n, sb_h=4, ub=1, center=True):
    if n == 0:
        return 0.0, {}
    s = div_up(g.slab_rows(t0, n), ub) * ub
    in0_per_unit = (2 if center else 3) * g.ct
    til = n * 3 * (T_UNIT + T_TILE * in0_per_unit)
    unt = U_TILE * g.ct * s
    grp = g.groups(t0, n, sb_h)
    walk = g.steps * sum(g.kkh * W_GROUP_KSTEP[r] * k for r, k in grp) * g.ns + g.steps * len(grp) * W_GROUP
    return (unt + til + walk) / 1350.0, dict(
        S=s, T=n, W=sum(r * k for r, k in grp), unt=unt / 1350, til=til / 1350, walk=walk / 1350
    )


def balanced(g, rows, cap=None, sb_h=4):
    """Contiguous split of the m_tiles tile-rows over at most `rows` rows, every row >= 1 tile-row and <= cap, minimizing
    the most expensive row (ties: fewer tile-rows on the largest row, then the lexicographically flattest split)."""
    M = g.m_tiles
    cap = cap or M
    INF = float("inf")
    cost = {}
    for a in range(M):
        for n in range(1, min(cap, M - a) + 1):
            cost[(a, n)] = row_cost(g, a, n, sb_h)[0]
    # best[r][a]: min over splits of [a, M) into exactly r rows of the max row cost
    best = [[INF] * (M + 1) for _ in range(rows + 1)]
    choice = [[None] * (M + 1) for _ in range(rows + 1)]
    best[0][M] = 0.0
    for r in range(1, rows + 1):
        for a in range(M - 1, -1, -1):
            for n in range(1, min(cap, M - a) + 1):
                v = max(cost[(a, n)], best[r - 1][a + n])
                if v < best[r][a] - 1e-9:
                    best[r][a] = v
                    choice[r][a] = n
    r = min(range(1, rows + 1), key=lambda k: (best[k][0], -k))
    out, a = [], 0
    while a < M:
        n = choice[r][a]
        out.append((a, n))
        a += n
        r -= 1
    return out


def main():
    names = sys.argv[1:] or list(CASES)
    for name in names:
        H, W, c_in, c_out, d, s, k_cols, c_in_p, rows = CASES[name]
        g = G(H, W, c_in, c_out, d, s, k_cols, c_in_p)
        per = div_up(g.m_tiles, rows)
        base = [(a, min(per, g.m_tiles - a)) for a in range(0, g.m_tiles, per)]
        bm = per
        print(f"== {name}: m_tiles {g.m_tiles} ct {g.ct} kkh {g.kkh} ns {g.ns} steps {g.steps}")
        for label, split in (("base(CT bm)", base), ("base(rt)", base)):
            costs = []
            for a, n in split:
                c, info = row_cost(g, a, n)
                costs.append(c)
            print(
                f"  {label:<12} max {max(costs):6.2f}  "
                + " ".join(f"{a}+{n}:{c:.1f}" for (a, n), c in zip(split, costs))
            )
        for cap in (None, 7, 8, 9):
            split = balanced(g, rows, cap)
            costs = [row_cost(g, a, n)[0] for a, n in split]
            print(
                f"  bal cap {str(cap):<4} max {max(costs):6.2f}  bm {max(n for _, n in split)}  "
                + " ".join(f"{a}+{n}:{c:.1f}" for (a, n), c in zip(split, costs))
            )


if __name__ == "__main__":
    main()
