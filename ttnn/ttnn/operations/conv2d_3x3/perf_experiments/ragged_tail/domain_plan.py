# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (ttnn stubbed): per ROW_MAJOR regime (padding (0, 0, 1, 1)), the blocking, kh reuse, and the padding rows
the ragged tail removes on the busiest core and over all cores.

    python3 -I domain_plan.py H,W,C_in,C_out [...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "whole_k_rowstream_v2"))
import rs_plan  # noqa: E402

rs_plan.HERE = Path(__file__).parent
pd = rs_plan.load("op_copy")
for spec in sys.argv[1:]:
    H, W, ci, co = (int(v) for v in spec.split(","))
    g = rs_plan.geometry(pd, H, W, ci, co)
    b = pd.select_blocking(g, rs_plan.GRID, rs_plan.BUDGET)
    bm = b.block_m_tiles
    ranges = pd.core_m_ranges(g, b.core_m_tiles)[: b.num_m_groups]
    pad = [(-(-a // bm)) * bm - a for _, a in ranges]
    busiest = max(a for _, a in ranges)
    wt = W // 32
    print(
        f"{H}x{W} {ci}->{co}: bm {bm} sb {b.out_subblock_h}x{b.out_subblock_w} mb {b.max_m_blocks_per_core} "
        f"core_m {b.core_m_tiles} busiest {busiest} (pad {(-(-busiest // bm)) * bm - busiest}) ng {b.n_groups} "
        f"khr {b.khr_ring_rows} wt {wt} ring {b.rm_ring_px} fold {int(g.fold)} cores {len(ranges)} "
        f"min_actual {min(a for _, a in ranges)} pad_rows_total {sum(pad)}"
    )
