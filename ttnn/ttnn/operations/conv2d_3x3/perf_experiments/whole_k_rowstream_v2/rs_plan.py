# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (ttnn stubbed, no device): per FLUX.2 bh2x4 focus case, the base blocking (op_base) and, for the row-streamed
walk (op_copy, ROWSTREAM, descriptor defaults), its roofline, output stream rate and plan (None: not applied).

    python3 -I ttnn/ttnn/operations/conv2d_3x3/perf_experiments/whole_k_rowstream/rs_plan.py [op_copy|op_base]

Budget = l1_cb_budget_bytes(0, 65536) on the p150b (1,334,368 B)."""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
BUDGET = 1334368
UNRESERVED = BUDGET + 128 * 1024 + 65536
GRID = (11, 10)

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
    spec = importlib.util.spec_from_file_location(f"rsp2_{name}", HERE / name / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"rsp2_{name}"] = m
    spec.loader.exec_module(m)
    return m


def geometry(pd, H, W, c_in, c_out):
    g = pd.Conv3x3Geometry(
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
    if pd.cout_fold_capable(g):
        import dataclasses

        g = dataclasses.replace(g, fold=True)
    return g


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "op_copy"
    pd = load(which)
    for idx, tag, H, W, ci, co in FOCUS:
        g = geometry(pd, H, W, ci, co)
        b = pd.select_blocking(g, GRID, BUDGET)
        line = (
            f"{tag:4s} idx{idx} {H}x{W} {ci}->{co}: bm {b.block_m_tiles} bk {b.block_k_tiles} bn {b.block_n_tiles} "
            f"nkb {b.num_k_blocks} mb {b.max_m_blocks_per_core} sb {b.out_subblock_h}x{b.out_subblock_w} "
            f"ring {b.rm_ring_px} res {int(b.weights_resident)} ng {b.n_groups} fp {b.footprint_bytes} "
            f"depth {b.im2col_rm_depth_rows}"
        )
        line += f" khr {b.khr_ring_rows}"
        plan = getattr(pd, "rowstream_plan", None)
        if plan is not None:
            p = plan(g, b, True, BUDGET)
            c, d = pd.rowstream_roofline(g, b)
            line += f"\n      roofline: matmul {c * 1e6:.0f} us, DRAM {d * 1e6:.0f} us, DRAM / matmul {d / c:.2f}"
            line += f", output stream {pd.rowstream_store_rate(g, b) / 1e9:.0f} GB/s"
            line += f"\n      rowstream: {p}"
        print(line)


if __name__ == "__main__":
    main()
