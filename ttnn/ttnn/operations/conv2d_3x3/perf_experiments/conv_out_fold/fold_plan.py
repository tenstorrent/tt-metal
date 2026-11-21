# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (ttnn stubbed, no device): the blocking the C_out fold solve picks per focus / sweep case against the base
(op_base) blocking.

    python3 -I ttnn/ttnn/operations/conv2d_3x3/perf_experiments/conv_out_fold/fold_plan.py

Budget = l1_cb_budget_bytes(0, 65536) on the p150b (1,334,368 B)."""

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
BUDGET = 1334368
UNRESERVED = BUDGET + 128 * 1024 + 65536
GRID = (11, 10)

# (tag, H, W, C_in, C_out, padding)
CASES = [
    ("idx35 P3", 1026, 2048, 128, 3, (0, 0, 1, 1)),
    ("idx33 P9", 514, 1024, 128, 3, (0, 0, 1, 1)),
    ("r1 258x512 128->3", 258, 512, 128, 3, (0, 0, 1, 1)),
    ("r2 130x256 128->3", 130, 256, 128, 3, (0, 0, 1, 1)),
    ("r3 514x1024 64->3", 514, 1024, 64, 3, (0, 0, 1, 1)),
    ("r4 512x1024 128->3 p1", 512, 1024, 128, 3, (1, 1, 1, 1)),
    ("r5 514x1024 128->4", 514, 1024, 128, 4, (0, 0, 1, 1)),
    ("r6 1026x2048 256->3", 1026, 2048, 256, 3, (0, 0, 1, 1)),
]


def load(name):
    fake = types.ModuleType("ttnn")
    fake.get_max_worker_l1_unreserved_size = lambda: UNRESERVED
    fake.MathFidelity = types.SimpleNamespace(LoFi="LoFi", HiFi2="HiFi2", HiFi3="HiFi3", HiFi4="HiFi4")
    fake.ProgramDescriptor = object
    sys.modules.setdefault("ttnn", fake)
    spec = importlib.util.spec_from_file_location(f"cof_{name}", HERE / name / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"cof_{name}"] = m
    spec.loader.exec_module(m)
    return m


def geometry(pd, H, W, c_in, c_out, pad):
    return pd.Conv3x3Geometry(
        input_height=H,
        input_width=W,
        c_in_p=c_in,
        c_out=c_out,
        padding=pad if len(set(pad)) > 1 else pad[0],
        stride=1,
        dilation=1,
        output_height=H + pad[0] + pad[1] - 2,
        output_width=W + pad[2] + pad[3] - 2,
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
    return (
        f"core_m {b.core_m_tiles} bm {b.block_m_tiles} bk {b.block_k_tiles} nkb {b.num_k_blocks} "
        f"nb {b.max_m_blocks_per_core} sb {b.out_subblock_h}x{b.out_subblock_w} res {int(b.weights_resident)} "
        f"ring {b.rm_ring_px} foot {b.footprint_bytes / 1024:.0f} KB"
    )


def main():
    base, copy = load("op_base"), load("op_copy")
    for tag, H, W, c_in, c_out, pad in CASES:
        gb = geometry(base, H, W, c_in, c_out, pad)
        bb = base.select_blocking(gb, GRID, BUDGET)
        gc = geometry(copy, H, W, c_in, c_out, pad)
        print(f"{tag}\n  base: {describe(gb, bb)}")
        if not copy.cout_fold_capable(gc):
            print("  fold: not capable")
            continue
        gf = dataclasses.replace(gc, fold=True)
        bf = copy.select_blocking(gf, GRID, BUDGET)
        print(f"  fold: {describe(gf, bf)}")


if __name__ == "__main__":
    main()
