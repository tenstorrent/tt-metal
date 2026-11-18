# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device): the solved blocking of the BLOCK N-slice shapes, the in0 / input-ring alias, and the
L1 left, for the center-tap variants of this experiment.

Stubs `ttnn` so the descriptor module imports without starting UMD. Blackhole p150b:
get_max_worker_l1_unreserved_size() = 1530976 B.

    python3 -I blocking_check.py [op_copy]
"""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
MAX_UNRESERVED = 1530976
BF8_TILE = 1088
TILE = 32


def load_descriptor(sub):
    stub = types.ModuleType("ttnn")
    stub.get_max_worker_l1_unreserved_size = lambda: MAX_UNRESERVED
    sys.modules.setdefault("ttnn", stub)
    spec = importlib.util.spec_from_file_location("nct_pd", HERE / sub / "conv2d_3x3_program_descriptor.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nct_pd"] = mod
    spec.loader.exec_module(mod)
    return mod


def div_up(a, b):
    return -(-a // b)


# (name, (H, W, C_in), C_out, padding(=dilation), stride, shard_shape, (gx, gy), bias)
BLOCK = [
    ("res4_x", (32, 64, 256), 256, 1, 1, [256, 32], (8, 8), True),
    ("res5_0", (32, 64, 512), 512, 2, 1, [224, 64], (8, 10), True),
    ("res5_1", (32, 64, 512), 512, 4, 1, [224, 64], (8, 10), True),
    ("res5_2", (32, 64, 512), 512, 8, 1, [224, 64], (8, 10), True),
    ("aspp1", (32, 64, 2048), 256, 6, 1, [256, 192], (11, 8), True),
    ("aspp2", (32, 64, 2048), 256, 12, 1, [256, 192], (11, 8), True),
    ("aspp3", (32, 64, 1024), 256, 18, 1, [256, 96], (11, 8), False),
    ("res4_0", (64, 128, 256), 256, 1, 2, [1024, 32], (8, 8), True),
]


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "op_copy"
    pd = load_descriptor(sub)
    # knob=value overrides (e.g. NSLICE_CENTER=5 NCT_COMPACT=1)
    for kv in sys.argv[2:]:
        k, v = kv.split("=")
        setattr(pd, k, None if v == "None" else int(v))
    for name, (H, W, c_in), c_out, p, s, shard, (gx, gy), bias in BLOCK:
        d = p
        c_in_p = shard[1]
        k_cols = div_up(c_in, c_in_p)
        h_out = (H + 2 * p - 2 * d - 1) // s + 1
        w_out = (W + 2 * p - 2 * d - 1) // s + 1
        g = pd.Conv3x3Geometry(
            input_height=H,
            input_width=W,
            c_in_p=c_in_p,
            c_out=c_out,
            padding=p,
            stride=s,
            dilation=d,
            output_height=h_out,
            output_width=w_out,
            c_in=c_in,
            tile_input=True,
            in0_tile_bytes=BF8_TILE,
            weight_tile_bytes=BF8_TILE,
            out_tile_bytes=BF8_TILE,
            has_bias=bias,
            sharded_output=True,
            k_cols=k_cols,
        )
        core_m_tiles, _ = pd.m_split(g, gy)
        in_pages = (shard[0] // TILE) * (shard[1] // TILE)
        resident = in_pages * BF8_TILE + core_m_tiles * g.out_n_tiles * BF8_TILE
        budget = pd.l1_cb_budget_bytes(resident)
        b = pd.select_blocking(g, (gy, 1), budget)
        bm = b.block_m_tiles
        in0_pages = bm * (g.tensor_k_tiles if b.nslice else b.block_k_tiles)
        slab_pages = b.slab_tile_rows * g.c_in_tiles
        center_pages = bm * 3 * g.c_in_tiles
        print(
            f"{name:7s} k_cols {k_cols:2d} ct {g.c_in_tiles} bm {bm} bk {b.block_k_tiles} nkb {b.num_k_blocks} "
            f"nslice {int(b.nslice)} slab_rows {b.slab_tile_rows} ub {b.untilize_batch_rows} ring {b.im2col_rm_depth_rows} "
            f"in0 {in0_pages} slab_pages {slab_pages} center {center_pages} "
            f"alias {slab_pages <= in0_pages} alias2/3 {slab_pages <= in0_pages - center_pages} "
            f"fp {b.footprint_bytes} budget {budget} free {budget - b.footprint_bytes} "
            f"td {int(pd._tap_direct(g, bm, b.block_k_tiles, b.nslice))} nct {pd.nslice_center_mode(g, b)} "
            f"unit {pd._unit_tiles(g, bm, b.block_k_tiles, b.nslice)}"
        )


if __name__ == "__main__":
    main()
