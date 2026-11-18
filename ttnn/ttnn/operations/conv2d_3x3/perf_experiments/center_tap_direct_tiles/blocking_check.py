# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device): the solved blocking of every HEIGHT TILE guard layer and the BLOCK focus layers,
whether the center-tap direct path applies, and the L1 it costs against select_blocking's budget.

Stubs `ttnn` so the descriptor module's solve imports without starting UMD. Blackhole p150b:
get_max_worker_l1_unreserved_size() = 1530976 B.

    python3 -I blocking_check.py
"""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
MAX_UNRESERVED = 1530976
BF8_TILE = 1088


def load_descriptor():
    stub = types.ModuleType("ttnn")
    stub.get_max_worker_l1_unreserved_size = lambda: MAX_UNRESERVED
    sys.modules.setdefault("ttnn", stub)
    spec = importlib.util.spec_from_file_location("ctd_pd", HERE / "op_copy" / "conv2d_3x3_program_descriptor.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ctd_pd"] = mod
    spec.loader.exec_module(mod)
    return mod


def div_up(a, b):
    return -(-a // b)


HEIGHT = [
    ("stem_conv3", (256, 512, 64), 128, 1, 1, [1216, 64], 110),
    ("res2_conv2", (128, 256, 64), 64, 1, 1, [320, 64], 103),
    ("res3_0_conv2", (128, 256, 128), 128, 1, 2, [384, 128], 86),
    ("res3_conv2", (64, 128, 128), 128, 1, 1, [96, 128], 86),
    ("sem_res2_fuse_conv0", (128, 256, 288), 256, 1, 1, [320, 288], 103),
    ("stem_conv2", (256, 512, 64), 64, 1, 1, [1216, 64], 110),
    ("sem_res3_fuse_conv0", (64, 128, 320), 256, 1, 1, [96, 320], 86),
    ("inst_res3_fuse_conv0", (64, 128, 320), 128, 1, 1, [96, 320], 86),
    ("sem_res3_fuse_conv1", (64, 128, 256), 256, 1, 1, [96, 256], 86),
    ("inst_res2_fuse_conv0", (128, 256, 160), 128, 1, 1, [320, 160], 103),
    ("sem_res2_fuse_conv1", (128, 256, 256), 256, 1, 1, [320, 256], 103),
    ("inst_res2_fuse_conv1", (128, 256, 128), 128, 1, 1, [320, 128], 103),
    ("center_head1", (128, 256, 128), 32, 1, 1, [320, 128], 103),
]
BLOCK = [
    ("res5_1_conv2", (32, 64, 512), 512, 4, 4, [224, 64], (8, 10)),
    ("res5_2_conv2", (32, 64, 512), 512, 8, 8, [224, 64], (8, 10)),
    ("aspp_convs_1", (32, 64, 2048), 256, 6, 6, [256, 192], (11, 8)),
    ("aspp_convs_2", (32, 64, 2048), 256, 12, 12, [256, 192], (11, 8)),
    ("res4_conv2", (32, 64, 256), 256, 1, 1, [256, 32], (8, 8)),
]


def geometry(pd, H, W, c_in, c_in_p, c_out, pad, dil, stride, k_cols):
    return pd.Conv3x3Geometry(
        input_height=H,
        input_width=W,
        c_in_p=c_in_p,
        c_out=c_out,
        padding=pad,
        stride=stride,
        dilation=dil,
        output_height=(H + 2 * pad - (2 * dil + 1)) // stride + 1,
        output_width=(W + 2 * pad - (2 * dil + 1)) // stride + 1,
        c_in=c_in,
        tile_input=True,
        in0_tile_bytes=BF8_TILE,
        weight_tile_bytes=BF8_TILE,
        out_tile_bytes=BF8_TILE,
        has_bias=True,
        sharded_output=True,
        k_cols=k_cols,
    )


def cases(pd):
    out = []
    for name, (H, W, c_in), c_out, pad, stride, shard, cores in HEIGHT:
        g = geometry(pd, H, W, c_in, div_up(c_in, 32) * 32, c_out, pad, 1, stride, 1)
        cm, _ = pd.m_split(g, cores)
        res = (shard[0] // 32) * (shard[1] // 32) * BF8_TILE + cm * g.out_n_tiles * BF8_TILE
        out.append(("H:" + name, g, (cores, 1), pd.l1_cb_budget_bytes(res)))
    for name, (H, W, c_in), c_out, pad, dil, shard, (gx, gy) in BLOCK:
        g = geometry(pd, H, W, c_in, shard[1], c_out, pad, dil, 1, div_up(c_in, shard[1]))
        cm, _ = pd.m_split(g, gy)
        res = (shard[0] // 32) * (shard[1] // 32) * BF8_TILE + cm * g.out_n_tiles * BF8_TILE
        out.append(("B:" + name, g, (gy, 1), pd.l1_cb_budget_bytes(res)))
    return out


def main():
    pd = load_descriptor()
    print(
        f"{'case':24s} {'bm':>3s} {'bk':>3s} {'Ct':>3s} {'nkb':>3s} {'nsl':>3s} {'ub':>3s} {'ring':>4s} "
        f"{'base KB':>8s} {'budget':>8s} {'slack':>7s}  mode slots {'ctd KB':>7s} {'+KB':>6s}"
    )
    for name, g, grid, budget in cases(pd):
        pd.CENTER_TAP_DIRECT = False
        pd.select_blocking.cache_clear()
        b = pd.select_blocking(g, grid, budget)
        pd.CENTER_TAP_DIRECT = True
        pd.select_blocking.cache_clear()
        c = pd.select_blocking(g, grid, budget)
        assert (b.block_m_tiles, b.block_k_tiles, b.untilize_batch_rows, b.im2col_rm_depth_rows, b.nslice) == (
            c.block_m_tiles,
            c.block_k_tiles,
            c.untilize_batch_rows,
            c.im2col_rm_depth_rows,
            c.nslice,
        ), name
        print(
            f"{name:24s} {b.block_m_tiles:3d} {b.block_k_tiles:3d} {g.c_in_tiles:3d} {b.num_k_blocks:3d} "
            f"{int(b.nslice):3d} {b.untilize_batch_rows:3d} {b.im2col_rm_depth_rows:4d} "
            f"{b.footprint_bytes / 1024:8.1f} {budget / 1024:8.1f} {(budget - b.footprint_bytes) / 1024:7.1f}  "
            f"{c.center_mode:4d} {c.center_slot_blocks:5d} {c.footprint_bytes / 1024:7.1f} "
            f"{(c.footprint_bytes - b.footprint_bytes) / 1024:6.1f}"
        )


if __name__ == "__main__":
    main()
