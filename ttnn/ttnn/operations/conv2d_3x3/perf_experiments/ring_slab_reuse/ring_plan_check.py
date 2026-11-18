# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (no device, no real ttnn import): the blocking each variant selects for the domain cases, and the ring
slab's untilize rows per core against the per-M-block slab's.

    python3 ring_plan_check.py

ttnn is stubbed (importing the real one starts the UMD cluster outside the device lock); the L1 budget uses the
Blackhole unreserved size observed in round 3 (1367.1 KB interleaved budget + 128 KB headroom), so it is approximate.
"""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
UNRESERVED = int(1367.1 * 1024) + 128 * 1024
TILE = 32
BF16_TILE, BF8_TILE = 2048, 1088


def load_pd():
    fake = types.ModuleType("ttnn")
    fake.get_max_worker_l1_unreserved_size = lambda: UNRESERVED
    sys.modules.setdefault("ttnn", fake)
    spec = importlib.util.spec_from_file_location("ring_pd", HERE / "op_copy" / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ring_pd"] = m
    spec.loader.exec_module(m)
    return m


def cases():
    # (id, (H, W, C_in), C_out, padding, stride, in_tile_bytes, sharded (shard_rows, cores) or None)
    out = []
    height = [
        ("stem_conv3", (256, 512, 64), 128, 1, 1, (1216, 110)),
        ("res2_conv2", (128, 256, 64), 64, 1, 1, (320, 103)),
        ("res3_0_conv2", (128, 256, 128), 128, 1, 2, (384, 86)),
        ("res3_conv2", (64, 128, 128), 128, 1, 1, (96, 86)),
        ("sem_res2_fuse_conv0", (128, 256, 288), 256, 1, 1, (320, 103)),
        ("stem_conv2", (256, 512, 64), 64, 1, 1, (1216, 110)),
        ("sem_res3_fuse_conv0", (64, 128, 320), 256, 1, 1, (96, 86)),
        ("inst_res3_fuse_conv0", (64, 128, 320), 128, 1, 1, (96, 86)),
        ("sem_res3_fuse_conv1", (64, 128, 256), 256, 1, 1, (96, 86)),
        ("inst_res2_fuse_conv0", (128, 256, 160), 128, 1, 1, (320, 103)),
        ("sem_res2_fuse_conv1", (128, 256, 256), 256, 1, 1, (320, 103)),
        ("inst_res2_fuse_conv1", (128, 256, 128), 128, 1, 1, (320, 103)),
        ("center_head1", (128, 256, 128), 32, 1, 1, (320, 103)),
    ]
    for name, hwc, c_out, p, s, sh in height:
        out.append((name, hwc, c_out, p, s, BF8_TILE, sh))
    perf_shapes = [
        ((32, 64, 128), 128, 1),
        ((16, 16, 64), 128, 1),
        ((8, 16, 512), 256, 1),
        ((8, 16, 2048), 256, 1),
        ((64, 128, 128), 128, 1),
        ((128, 256, 64), 64, 1),
        ((128, 256, 256), 256, 1),
        ((128, 256, 512), 256, 1),
        ((256, 512, 128), 128, 1),
        ((64, 128, 1024), 512, 1),
    ]
    for hwc, c_out, p in perf_shapes:
        for tag, tb in (("bf16", BF16_TILE), ("bf8b", BF8_TILE)):
            out.append((f"dram_H{hwc[0]}_W{hwc[1]}_Ci{hwc[2]}_Co{c_out}_tile_{tag}", hwc, c_out, p, 1, tb, None))
    return out


def main():
    pd = load_pd()
    grid = (11, 10)
    for name, (H, W, c_in), c_out, p, s, tb, sh in cases():
        c_in_p = -(-c_in // TILE) * TILE
        h_out = (H + 2 * p - 2 - 1) // s + 1
        w_out = (W + 2 * p - 2 - 1) // s + 1
        g = pd.Conv3x3Geometry(
            input_height=H,
            input_width=W,
            c_in_p=c_in_p,
            c_out=c_out,
            padding=p,
            stride=s,
            dilation=1,
            output_height=h_out,
            output_width=w_out,
            c_in=c_in,
            tile_input=True,
            in0_tile_bytes=tb,
            weight_tile_bytes=BF8_TILE,
            out_tile_bytes=tb,
            has_bias=True,
            sharded_output=sh is not None,
        )
        if sh is not None:
            shard_rows, cores = sh
            work = (cores, 1)
            core_m, _ = pd.m_split(g, cores)
            resident = (shard_rows // TILE) * (c_in_p // TILE) * tb + core_m * g.out_n_tiles * tb
        else:
            work = grid
            resident = 0
        budget = UNRESERVED - 128 * 1024 - 2 * resident
        res = {}
        for ring in (False, True):
            pd.RING_SLAB = ring
            pd.RING_SLAB_SINGLE = ring and "--single" in sys.argv
            pd.RING_IN_SOLVE = ring and "--solve" in sys.argv
            pd.select_blocking.cache_clear()
            b = pd.select_blocking(g, work, budget)
            res[ring] = b
        b0, b1 = res[False], res[True]
        nb = b0.max_m_blocks_per_core
        line = (
            f"{name:<34} nb {nb:>2} bm {b0.block_m_tiles:>2} bk {b0.block_k_tiles:>2} ub {b0.untilize_batch_rows:>2} "
            f"slab {b0.slab_tile_rows:>3} c{b0.center_mode} free {(budget - b0.footprint_bytes) / 1024:7.1f}K"
        )
        if b1.ring:
            plan = pd.ring_plan(g, b1.core_m_tiles, b1.block_m_tiles)
            base_rows = max(len(core) for core in plan) * b0.slab_tile_rows
            ring_rows = max(sum(n for *_, n in core) for core in plan)
            line += f" [ring bm {b1.block_m_tiles} bk {b1.block_k_tiles} ub {b1.untilize_batch_rows} im2col {b1.im2col_rm_depth_rows} c{b1.center_mode}; base im2col {b0.im2col_rm_depth_rows}]"
            same = (
                b0.block_m_tiles,
                b0.block_k_tiles,
                b0.untilize_batch_rows,
                b0.im2col_rm_depth_rows,
                b0.center_mode,
            ) == (
                b1.block_m_tiles,
                b1.block_k_tiles,
                b1.untilize_batch_rows,
                b1.im2col_rm_depth_rows,
                b1.center_mode,
            )
            line += (
                f" | RING S {b1.slab_tile_rows:>3} (+{(b1.footprint_bytes - b0.footprint_bytes) / 1024:6.1f}K) "
                f"untilize rows/core {base_rows} -> {ring_rows}  same_blocking {same}"
            )
        else:
            line += " | ring off"
        print(line)


if __name__ == "__main__":
    main()
