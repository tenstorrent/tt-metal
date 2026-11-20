# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (no device, ttnn stubbed): blocking, L1 footprint and per-core fill reads of the FLUX.2 bh2x4 focus cases,
for the per-M-block slab (base) and the ROW_MAJOR ring slab (rm_ring_slab).

    python3 -I l1_plan.py

main(): the base blocking and the ring's L1 at it; solve_report(): a standalone (bm, bk) solve with the ring counted;
final_report(): the descriptor's own solve with RM_RING off / on, L1 and fill reads per core.
Budget = 1,334,368 B (l1_cb_budget_bytes(0, 65536) on the p150b, from the round brief)."""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
BUDGET = 1334368
UNRESERVED = BUDGET + 128 * 1024 + 65536

# (idx, tag, H, W, C_in, C_out)
FOCUS = [
    (41, "P1", 1026, 2048, 32, 128),
    (42, "P2", 514, 1024, 64, 256),
    (35, "P3", 1026, 2048, 128, 3),
    (36, "P4", 514, 1024, 32, 128),
    (44, "P5", 1026, 2048, 64, 128),
    (46, "P6", 1026, 2048, 64, 256),
    (43, "P7", 258, 512, 128, 512),
    (47, "i47", 514, 1024, 128, 512),
]


def load_pd():
    fake = types.ModuleType("ttnn")
    fake.get_max_worker_l1_unreserved_size = lambda: UNRESERVED
    fake.MathFidelity = types.SimpleNamespace(LoFi="LoFi", HiFi2="HiFi2", HiFi3="HiFi3", HiFi4="HiFi4")
    fake.ProgramDescriptor = object
    sys.modules.setdefault("ttnn", fake)
    spec = importlib.util.spec_from_file_location("rmr_pd", HERE / "op_copy" / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["rmr_pd"] = m
    spec.loader.exec_module(m)
    return m


def geometry(pd, H, W, c_in, c_out):
    return pd.Conv3x3Geometry(
        input_height=H,
        input_width=W,
        c_in_p=-(-c_in // 32) * 32,
        c_out=c_out,
        padding=(0, 0, 1, 1),
        stride=1,
        dilation=1,
        output_height=H - 2,
        output_width=W,
        c_in=c_in,
        tile_input=False,
        in0_tile_bytes=2048,
        weight_tile_bytes=2048,
        out_tile_bytes=2048,
        has_bias=True,
        l1_small_bytes=65536,
    )


def reads_per_core(pd, g, b):
    """Max over cores of (base fill reads, ring fill reads): base = every block's in-image band pixels (bands merged where
    they overlap), ring = the core's in-image union (each pixel once)."""
    worst = (0, 0)
    for start, actual in pd.core_m_ranges(g, b.core_m_tiles):
        base = 0
        lo_all, hi_all = None, None
        for bs in range(0, actual, b.block_m_tiles):
            tiles = min(b.block_m_tiles, actual - bs)
            o0 = (start + bs) * 32
            o1 = min((start + bs + tiles) * 32, g.num_out_px) - 1
            band = pd._block_band_len(g, o0, o1)
            f = pd._block_first_tap(g, o0)
            ivs = []
            for kh in range(3):
                v = f + kh * g.input_width
                lo, hi = max(v, 0), min(v + band, g.num_in_px)
                if lo < hi:
                    ivs.append((lo, hi))
            # merge overlapping bands (the slab reads each overlapped pixel once)
            ivs.sort()
            merged = []
            for lo, hi in ivs:
                if merged and lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
                else:
                    merged.append((lo, hi))
            base += sum(hi - lo for lo, hi in merged)
            lo_all = ivs[0][0] if lo_all is None else min(lo_all, ivs[0][0])
            hi_all = max(hi for _, hi in ivs) if hi_all is None else max(hi_all, max(hi for _, hi in ivs))
        ring = hi_all - lo_all
        if base > worst[0]:
            worst = (base, ring)
    return worst


def main():
    pd = load_pd()
    for idx, tag, H, W, c_in, c_out in FOCUS:
        g = geometry(pd, H, W, c_in, c_out)
        b = pd.select_blocking(g, (11, 10), BUDGET)
        px = g.c_in_p * 2
        slab_px = 2 * b.band_stride_px + b.band_len_px
        window = 2 * g.input_width + b.band_len_px  # d = 1: band 0 first pixel .. band 2 last pixel
        la = b.block_m_tiles * 32
        base_r, ring_r = reads_per_core(pd, g, b)
        left = BUDGET - b.footprint_bytes
        print(
            f"idx{idx} {tag}: bm {b.block_m_tiles} bk {b.block_k_tiles} nb {b.max_m_blocks_per_core} band {b.band_len_px} "
            f"slab {slab_px} px = {slab_px * px / 1024:.0f} KB; foot {b.footprint_bytes / 1024:.0f} KB, left {left / 1024:.0f} KB; "
            f"ring {window} px = {window * px / 1024:.0f} KB (+{(window - slab_px) * px / 1024:.0f}), "
            f"ring+la {window + la} px = {(window + la) * px / 1024:.0f} KB (+{(window + la - slab_px) * px / 1024:.0f}); "
            f"reads/core base {base_r} ring {ring_r} ({base_r / ring_r:.2f}x)"
        )


def ring_px(pd, g, core_m_tiles, bm, lookahead):
    """Largest M-block window over every core: band 0's first in-image pixel .. band 2's last (+ lookahead px)."""
    worst = 0
    for start, actual in pd.core_m_ranges(g, core_m_tiles):
        for bs in range(0, actual, bm):
            tiles = min(bm, actual - bs)
            o0 = (start + bs) * 32
            o1 = min((start + bs + tiles) * 32, g.num_out_px) - 1
            band = pd._block_band_len(g, o0, o1)
            f = pd._block_first_tap(g, o0)
            lo = max(f, 0)
            hi = min(f + 2 * g.dilation * g.input_width + band, g.num_in_px)
            worst = max(worst, hi - lo)
    return worst + lookahead * bm * 32


def ring_solve(pd, g, lookahead):
    """select_blocking's (bm, bk) order with the ring counted in the solve instead of the per-block slab."""
    core_m_tiles, _ = pd.m_split(g, 110)
    bn = g.tensor_n_tiles
    kkh = g.tensor_kkh_tiles
    divs = [d for d in range(kkh, 0, -1) if kkh % d == 0 and d % g.c_in_tiles == 0]
    bms = sorted({-(-core_m_tiles // n) for n in range(1, core_m_tiles + 1)}, reverse=True)
    for bm in bms:
        for bk in divs:
            fit = pd._footprint(g, core_m_tiles, bm, bk, bn)
            band_len, _ = pd._block_extents(g, core_m_tiles, bm)
            slab = (2 * min(g.input_width, band_len) + band_len) * g.c_in_p * 2
            ring = ring_px(pd, g, core_m_tiles, bm, lookahead) * g.c_in_p * 2
            total = fit["footprint_bytes"] - slab + ring
            if total <= BUDGET:
                return bm, bk, -(-core_m_tiles // bm), ring, total
    return None


def solve_report():
    pd = load_pd()
    print("\nring counted in the solve (bm, bk, M-blocks/core, ring KB, footprint KB):")
    for idx, tag, H, W, c_in, c_out in FOCUS:
        g = geometry(pd, H, W, c_in, c_out)
        b = pd.select_blocking(g, (11, 10), BUDGET)
        parts = []
        for la in (0, 1):
            r = ring_solve(pd, g, la)
            parts.append(
                "none fits"
                if r is None
                else f"bm {r[0]} bk {r[1]} nb {r[2]} ring {r[3] / 1024:.0f} KB foot {r[4] / 1024:.0f} KB"
            )
        print(
            f"idx{idx} {tag}: base bm {b.block_m_tiles} bk {b.block_k_tiles} nb {b.max_m_blocks_per_core} | "
            f"ring: {parts[0]} | ring+la: {parts[1]}"
        )


def ring_reads_per_core(pd, g, b):
    """Max over cores of the ring's fill reads: the union of the core's three kh bands (each pixel once)."""
    worst = 0
    for start, actual in pd.core_m_ranges(g, b.core_m_tiles):
        o0 = start * 32
        o1 = min((start + actual) * 32, g.num_out_px) - 1
        band = pd._block_band_len(g, o0, o1)
        f = pd._block_first_tap(g, o0)
        ivs = sorted(
            (max(f + kh * g.dilation * g.input_width, 0), min(f + kh * g.dilation * g.input_width + band, g.num_in_px))
            for kh in range(3)
        )
        merged = []
        for lo, hi in ivs:
            if lo >= hi:
                continue
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        worst = max(worst, sum(hi - lo for lo, hi in merged))
    return worst


def final_report():
    """The descriptor's own solve (RM_RING off / on): blocking, L1 and fill reads per core."""
    pd = load_pd()
    print("\nfinal (descriptor solve, RM_RING off -> on):")
    print(
        "case  bm bk nb -> bm bk nb | slab KB -> ring KB | im2col rows | foot KB | reads/core base -> ring (x) | reads/M-block"
    )
    for idx, tag, H, W, c_in, c_out in FOCUS:
        g = geometry(pd, H, W, c_in, c_out)
        pd.RM_RING = False
        pd.select_blocking.cache_clear()
        b0 = pd.select_blocking(g, (11, 10), BUDGET)
        pd.RM_RING = True
        pd.select_blocking.cache_clear()
        b1 = pd.select_blocking(g, (11, 10), BUDGET)
        px = g.c_in_p * 2
        r0, _ = reads_per_core(pd, g, b0)
        r1 = ring_reads_per_core(pd, g, b1)
        print(
            f"idx{idx} {tag}: {b0.block_m_tiles} {b0.block_k_tiles} {b0.max_m_blocks_per_core} -> "
            f"{b1.block_m_tiles} {b1.block_k_tiles} {b1.max_m_blocks_per_core} | "
            f"{b0.slab_px * px / 1024:.0f} -> {b1.slab_px * px / 1024:.0f} | {b0.im2col_rm_depth_rows} -> {b1.im2col_rm_depth_rows} | "
            f"{b0.footprint_bytes / 1024:.0f} -> {b1.footprint_bytes / 1024:.0f} | {r0} -> {r1} ({r0 / r1:.2f}x) | "
            f"{r0 / b0.max_m_blocks_per_core:.0f} -> {r1 / b1.max_m_blocks_per_core:.0f}"
        )


if __name__ == "__main__":
    main()
    solve_report()
    final_report()
