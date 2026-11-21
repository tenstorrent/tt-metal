# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (ttnn stubbed, no device): per FLUX.2 bh2x4 focus case, the base (op_base) and candidate (op_copy) blocking,
the L1 slack after the solve and the output CB depth in row groups.

    python3 -I ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kblock_store_decouple/plan.py [COPY=op_final] [knob=value ...]

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
    spec = importlib.util.spec_from_file_location(f"ksd_{name}", HERE / name / "conv2d_3x3_program_descriptor.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"ksd_{name}"] = m
    spec.loader.exec_module(m)
    return m


def geometry(pd, H, W, c_in, c_out, tile=False):
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
        tile_input=tile,
        in0_tile_bytes=2048,
        weight_tile_bytes=2048,
        out_tile_bytes=2048,
        has_bias=True,
        l1_small_bytes=65536,
        bank_run_input=not tile,
    )


def describe(g, b):
    rg = b.block_m_tiles // b.out_subblock_h
    return (
        f"cores {b.num_m_groups * getattr(b, 'n_groups', 1)} core_m {b.core_m_tiles} bm {b.block_m_tiles} "
        f"bk {b.block_k_tiles} bn {b.block_n_tiles} nb {b.max_m_blocks_per_core} sb {b.out_subblock_h}x"
        f"{b.out_subblock_w} res {int(b.weights_resident)} ring_px {b.rm_ring_px} im2col_d {b.im2col_rm_depth_rows} "
        f"out_rg {b.out_depth_row_groups}/{rg} foot {b.footprint_bytes / 1024:.0f} KB "
        f"slack {(BUDGET - b.footprint_bytes) / 1024:.0f} KB"
    )


def main(knobs):
    base, copy = load("op_base"), load(knobs.pop("COPY", "op_copy"))
    for k, v in knobs.items():
        setattr(copy, k, eval(v))
    for idx, tag, H, W, c_in, c_out in FOCUS:
        rows = [f"idx{idx} {tag} {H}x{W} {c_in}->{c_out}"]
        for name, pd in (("base", base), ("copy", copy)):
            g = geometry(pd, H, W, c_in, c_out)
            try:
                rows.append(f"  {name}: {describe(g, pd.select_blocking(g, GRID, BUDGET))}")
            except ValueError as e:
                rows.append(f"  {name}: {e}")
        print("\n".join(rows))


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
