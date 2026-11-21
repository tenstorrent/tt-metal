# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (ttnn stubbed, no device): per FLUX.2 bh2x4 focus case, the base blocking (op_base) against the kh-reuse
blocking (op_copy, KH_REUSE), and the tap-direct units / tilized tiles per core.

    python3 -I ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kh_tile_reuse/khr_plan.py [op_copy | op_final]

Budget = l1_cb_budget_bytes(0, 65536) on the p150b (1,334,368 B)."""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
BUDGET = 1334368
UNRESERVED = BUDGET + 128 * 1024 + 65536
GRID = (11, 10)

# (idx, tag, H, W, C_in, C_out)
FOCUS = [
    (41, "P1", 1026, 2048, 32, 128),
    (42, "P2", 514, 1024, 64, 256),
    (35, "P3", 1026, 2048, 128, 3),
    (36, "P4", 514, 1024, 32, 128),
    (44, "P5", 1026, 2048, 64, 128),
    (46, "P6", 1026, 2048, 64, 256),
    (43, "P7", 258, 512, 128, 512),
    (37, "P8", 258, 512, 64, 256),
    (33, "P9", 514, 1024, 128, 3),
    (38, "P10", 130, 256, 128, 512),
    (39, "P11", 514, 1024, 64, 128),
    (45, "P12", 514, 1024, 128, 256),
    (47, "P13", 514, 1024, 128, 512),
    (40, "P14", 258, 512, 128, 256),
    (34, "P15", 66, 128, 128, 512),
    (32, "P16", 130, 256, 32, 128),
    (31, "P17", 66, 128, 32, 128),
]


def load(name):
    fake = types.ModuleType("ttnn")
    fake.get_max_worker_l1_unreserved_size = lambda: UNRESERVED
    fake.MathFidelity = types.SimpleNamespace(LoFi="LoFi", HiFi2="HiFi2", HiFi3="HiFi3", HiFi4="HiFi4")
    fake.ProgramDescriptor = object
    sys.modules.setdefault("ttnn", fake)
    spec = importlib.util.spec_from_file_location(f"khr_{name}", HERE / name / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"khr_{name}"] = m
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
        bank_run_input=True,
    )


def describe(g, b):
    ng = getattr(b, "n_groups", 1)
    khr = getattr(b, "khr_ring_rows", 0)
    tw = g.c_in_tiles
    if khr:
        units = b.max_m_blocks_per_core * b.block_m_tiles + 2 * (g.input_width // 32)
        tiles = units * 3 * tw
    else:
        units = b.max_m_blocks_per_core * b.block_m_tiles * b.num_k_blocks
        tiles = units * b.block_k_tiles
    return (
        f"ng {ng} core_m {b.core_m_tiles} bm {b.block_m_tiles} bk {b.block_k_tiles} bn {b.block_n_tiles} "
        f"nb {b.max_m_blocks_per_core} res {int(b.weights_resident)} ring_px {b.rm_ring_px} "
        f"depth {b.im2col_rm_depth_rows} khr_R {khr} units {units} tiles {tiles} "
        f"foot {b.footprint_bytes / 1024:.0f} KB"
    )


def main():
    base, copy = load("op_base"), load(sys.argv[1] if len(sys.argv) > 1 else "op_copy")
    for idx, tag, H, W, c_in, c_out in FOCUS:
        gb = geometry(base, H, W, c_in, c_out)
        rows = [f"idx{idx} {tag} {H}x{W} {c_in}->{c_out}"]
        rows.append(f"  base: {describe(gb, base.select_blocking(gb, GRID, BUDGET))}")
        if hasattr(copy, "KH_REUSE"):
            gc = geometry(copy, H, W, c_in, c_out)
            copy.select_blocking.cache_clear()
            rows.append(f"  khr:  {describe(gc, copy.select_blocking(gc, GRID, BUDGET))}")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
