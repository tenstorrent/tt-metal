# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only check of the ring plan (ttnn stubbed): simulates the reader's ring cursors (conv2d_3x3_rm_ring.hpp) on
every core of a shape at op_copy's solved blocking, and checks that no band wait needs a pixel past `allowed` (its slot
still live: the wait would never land) and that every block's pixels lie within one wrap of its base.

    python3 -I sim_ring.py            # the focus cases plus a sweep of odd shapes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from l1_plan import BUDGET, FOCUS, load_pd  # noqa: E402

INF = (1 << 31) - 1


def geometry(pd, H, W, c_in, c_out, pad=(0, 0, 1, 1), dilation=1):
    top, bottom, left, right = pad
    return pd.Conv3x3Geometry(
        input_height=H,
        input_width=W,
        c_in_p=c_in,
        c_out=c_out,
        padding=pad,
        stride=1,
        dilation=dilation,
        output_height=H + top + bottom - 2 * dilation,
        output_width=W + left + right - 2 * dilation,
        c_in=c_in,
        tile_input=False,
        in0_tile_bytes=2048,
        weight_tile_bytes=2048,
        out_tile_bytes=2048,
        has_bias=True,
        l1_small_bytes=65536,
        bank_run_input=True,
    )


def check(pd, g, b, grid_cores=110):
    ring = b.rm_ring_px
    bm, bk = b.block_m_tiles, b.block_k_tiles
    kkh, nkb = g.tensor_kkh_tiles, b.num_k_blocks
    bad = 0
    for start, actual in pd.core_m_ranges(g, b.core_m_tiles):
        wins = []
        for bs in range(0, actual, bm):
            tiles = min(bm, actual - bs)
            o0 = (start + bs) * 32
            o1 = min((start + bs + tiles) * 32, g.num_out_px)
            bl = pd._block_band_len(g, o0, o1 - 1)
            f = pd._block_first_tap(g, o0)
            bands = []
            for kh in range(3):
                v = f + kh * g.dilation * g.input_width
                a, z = max(v, 0), min(v + bl, g.num_in_px)
                bands.append((a, z) if a < z else None)
            wins.append(bands)
        # core segments
        o0 = start * 32
        o1 = min((start + actual) * 32, g.num_out_px)
        bl = pd._block_band_len(g, o0, o1 - 1)
        f = pd._block_first_tap(g, o0)
        segs = []
        for kh in range(3):
            v = f + kh * g.dilation * g.input_width
            a, z = max(v, 0), min(v + bl, g.num_in_px)
            if a >= z:
                continue
            if segs and a <= segs[-1][1]:
                segs[-1] = (segs[-1][0], max(segs[-1][1], z))
            else:
                segs.append((a, z))
        lo_of = [min(bd[0] for bd in w if bd) for w in wins]
        first_px = min(lo_of)
        origin = min(first_px, segs[0][0])
        allowed = first_px + ring
        copied = segs[0][0]
        prev_last = INF

        def fbl(w, kh0):
            for x in range(kh0, 3):
                if w[x]:
                    return w[x][0]
            return INF

        for m, w in enumerate(wins):
            keep_next = min(lo_of[m + 1 :], default=INF)
            lo = lo_of[m]
            hi = max(bd[1] for bd in w if bd)
            q = (lo - origin) // ring
            base = -origin - q * ring
            if not (0 <= lo + base and hi + base <= 2 * ring and hi - lo <= ring):
                bad += 1
            for k in range(nkb):
                kh = k * bk // kkh
                live = min(prev_last, lo, keep_next) if k == 0 else min(keep_next, fbl(w, (k - 1) * bk // kkh))
                if live < INF:
                    allowed = max(allowed, live + ring)
                if k * bk == kh * kkh and w[kh]:
                    # The band wait issues chunks cut at the wait's end and at `allowed`: a wait past `allowed` would
                    # never land (the kernel ASSERTs it).
                    x = w[kh][1]
                    if x > allowed and copied < x:
                        bad += 1
                    copied = max(copied, x)
            prev_last = fbl(w, (nkb - 1) * bk // kkh)
    return bad


def main():
    pd = load_pd("op_copy")
    shapes = [(tag, H, W, ci, co, (0, 0, 1, 1), 1) for _, (tag, H, W, ci, co) in FOCUS.items()]
    shapes += [
        ("w64", 16384, 64, 32, 64, (1, 1, 1, 1), 1),
        ("w32", 32768, 32, 64, 64, (1, 1, 1, 1), 1),
        ("w40", 20000, 40, 32, 32, (1, 1, 1, 1), 1),
        ("w8", 40000, 8, 32, 32, (1, 1, 1, 1), 1),
        ("p0", 2048, 1024, 64, 64, (0, 0, 0, 0), 1),
        ("asym", 2000, 1000, 32, 64, (2, 0, 1, 0), 1),
        ("asym2", 2000, 1000, 64, 64, (0, 2, 0, 1), 1),
        ("d2", 2000, 1000, 32, 64, (2, 2, 2, 2), 1 + 1),
        ("d3", 1500, 700, 64, 64, (1, 1, 1, 1), 3),
        ("big", 8000, 100, 128, 32, (1, 1, 1, 1), 1),
        ("odd", 1333, 1333, 96, 96, (1, 1, 1, 1), 1),
        ("c256", 1026, 1024, 256, 256, (1, 1, 1, 1), 1),
    ]
    for tag, H, W, ci, co, pad, d in shapes:
        g = geometry(pd, H, W, ci, co, pad, d)
        try:
            b = pd.select_blocking(g, (11, 10), BUDGET)
        except ValueError as e:
            print(tag, "no blocking", e)
            continue
        if not b.rm_ring_px:
            print(f"{tag:5s} no ring (bm {b.block_m_tiles}, blocks {b.max_m_blocks_per_core})")
            continue
        print(f"{tag:5s} bm {b.block_m_tiles} bk {b.block_k_tiles} ring {b.rm_ring_px} bad {check(pd, g, b)}")


if __name__ == "__main__":
    main()
