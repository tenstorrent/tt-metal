# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only L1 table for weights_resident (no device, no real ttnn: a stub module stands in for it).
Per focus case: the streaming solve's blocking and footprint, the resident weights' extra bytes, whether they fit in the
L1 left (WR_MODE leftover), and the blocking WR_MODE solve picks. Budget = l1_cb_budget_bytes(0, 65536) on BH p150b."""
import importlib
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


class _Any:
    def __getattr__(self, k):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


stub = types.ModuleType("ttnn")
stub.__getattr__ = lambda k: _Any()
sys.modules["ttnn"] = stub
pkg = types.ModuleType("wr_op_copy")
pkg.__path__ = [str(HERE / "op_copy")]
sys.modules["wr_op_copy"] = pkg
pd = importlib.import_module("wr_op_copy.conv2d_3x3_program_descriptor")

BUDGET = 1_334_368
CASES = {
    "idx41": (1026, 2048, 32, 128),
    "idx42": (514, 1024, 64, 256),
    "idx35": (1026, 2048, 128, 3),
    "idx36": (514, 1024, 32, 128),
    "idx44": (1026, 2048, 64, 128),
    "idx46": (1026, 2048, 64, 256),
    "idx43": (258, 512, 128, 512),
    "idx47": (514, 1024, 128, 512),
    "c66x128": (66, 128, 32, 128),
    "c130x256": (130, 256, 32, 128),
    "c258x512_64": (258, 512, 64, 256),
    "c258x512_128_256": (258, 512, 128, 256),
    "c514x1024_128_3": (514, 1024, 128, 3),
}


def geom(H, W, ci, co, tile=False, pad=(0, 0, 1, 1), stride=1):
    c_in_p = -(-ci // 32) * 32 if tile else -(-ci // 8) * 8
    top, bottom, left, right = (pad,) * 4 if isinstance(pad, int) else pad
    return pd.Conv3x3Geometry(
        input_height=H,
        input_width=W,
        c_in_p=c_in_p,
        c_out=co,
        padding=pad,
        stride=stride,
        dilation=1,
        output_height=(H + top + bottom - 3) // stride + 1,
        output_width=(W + left + right - 3) // stride + 1,
        c_in=ci,
        tile_input=tile,
        in0_tile_bytes=2048,
        weight_tile_bytes=2048,
        out_tile_bytes=2048,
        has_bias=True,
        sharded_output=False,
        k_cols=1,
        l1_small_bytes=65536,
    )


def solve(g, mode):
    pd.WR_MODE = mode
    pd.select_blocking.cache_clear()
    return pd.select_blocking(g, (11, 10), BUDGET)


print(
    "case       | streaming: bm bk nkb mb im2col footprint left | wts KB extra KB lo | trim: res im2col | solve: bm mb res im2col"
)
import os

if os.environ.get("WR_TABLE") == "tile":
    # The bh2x4 convs on TILE input and the FLUX.2 single-device (1x1) convs (TILE, padding 1; stride-2 downsamplers
    # pad (0, 1, 0, 1)).
    CASES = {k + "_T": v + (True,) for k, v in CASES.items()}
    CASES.update(
        {
            "1x1_128x128_32_512": (128, 128, 32, 512, True, 1),
            "1x1_128x128_512_32": (128, 128, 512, 32, True, 1),
            "1x1_1024_128_3": (1024, 1024, 128, 3, True, 1),
            "1x1_128x128_512_512": (128, 128, 512, 512, True, 1),
            "1x1_512_128_256": (512, 512, 128, 256, True, 1),
            "1x1_256_256_512": (256, 256, 256, 512, True, 1),
            "1x1_256_512_512": (256, 256, 512, 512, True, 1),
            "1x1_512_256_256": (512, 512, 256, 256, True, 1),
            "1x1_1024_128_128": (1024, 1024, 128, 128, True, 1),
            "1x1_1024_128_128_s2": (1024, 1024, 128, 128, True, (0, 1, 0, 1), 2),
            "1x1_512_256_256_s2": (512, 512, 256, 256, True, (0, 1, 0, 1), 2),
            "1x1_256_512_512_s2": (256, 256, 512, 512, True, (0, 1, 0, 1), 2),
            "1x1_512_512_256": (512, 512, 512, 256, True, 1),
            "1x1_1024_256_128": (1024, 1024, 256, 128, True, 1),
            "1x1_512_512_512": (512, 512, 512, 512, True, 1),
            "1x1_1024_256_256": (1024, 1024, 256, 256, True, 1),
        }
    )
for name, spec in CASES.items():
    g = geom(*spec)
    b = solve(g, "off")
    wts = g.tensor_k_tiles * b.block_n_tiles * g.weight_tile_bytes
    extra = (b.num_k_blocks - b.weight_depth) * b.block_k_tiles * b.block_n_tiles * g.weight_tile_bytes
    left = BUDGET - b.footprint_bytes
    lo = solve(g, "leftover")
    tr = solve(g, "trim")
    so = solve(g, "solve")
    print(
        f"{name:16s} | {b.block_m_tiles:3d} {b.block_k_tiles} {b.num_k_blocks} {b.max_m_blocks_per_core:3d}"
        f" {b.im2col_rm_depth_rows:3d} {b.footprint_bytes:8d} {left:7d} | {wts / 1024:5.0f} {extra / 1024:4.0f}"
        f" {lo.weights_resident:d} | {tr.weights_resident:d} {tr.im2col_rm_depth_rows:3d} bm{tr.block_m_tiles}"
        f" | {so.block_m_tiles:3d} {so.max_m_blocks_per_core:3d} {so.weights_resident:d} {so.im2col_rm_depth_rows:3d} bk{so.block_k_tiles}"
    )
