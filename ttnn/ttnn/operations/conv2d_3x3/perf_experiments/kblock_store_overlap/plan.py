# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (ttnn stubbed, no device): the blocking of each FLUX.2 bh2x4 focus case under op_copy's knobs.

    python3 -I plan.py [KNOB=value ...]     e.g. plan.py OUT_BLOCK_DEPTH=1.0   plan.py OUT_DEPTH_FREE=1 OUT_DEEP_CIN32=0"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "whole_k_rowstream_v2"))
import rs_plan  # noqa: E402

rs_plan.HERE = HERE
pd = rs_plan.load("op_copy")
for kv in sys.argv[1:]:
    k, v = kv.split("=")
    setattr(pd, k, type(getattr(pd, k))(float(v)) if not isinstance(getattr(pd, k), bool) else bool(int(v)))
for idx, tag, H, W, ci, co in rs_plan.FOCUS:
    g = rs_plan.geometry(pd, H, W, ci, co)
    b = pd.select_blocking(g, rs_plan.GRID, rs_plan.BUDGET)
    nominal = b.max_m_blocks_per_core * b.block_m_tiles
    print(
        f"{tag:4s} idx{idx} {H}x{W} {ci}->{co}: bm {b.block_m_tiles} mb {b.max_m_blocks_per_core} core_m {b.core_m_tiles}"
        f" (nominal {nominal}) outdepth {b.out_depth_row_groups}/{-(-b.block_m_tiles // b.out_subblock_h)}"
        f" im2col {b.im2col_rm_depth_rows} ring {b.rm_ring_px} khr {b.khr_ring_rows} fp {b.footprint_bytes}"
    )
