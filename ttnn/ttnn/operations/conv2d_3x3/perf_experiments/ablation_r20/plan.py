# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (ttnn stubbed, no device): the blocking of each FLUX.2 bh2x4 focus case for an op dir.

    python3 -I ablation_r20/plan.py [op dir, default: the real op]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "whole_k_rowstream_v2"))
import rs_plan  # noqa: E402

op = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
rs_plan.HERE = op.parent
pd = rs_plan.load(op.name)
for idx, tag, H, W, ci, co in rs_plan.FOCUS:
    g = rs_plan.geometry(pd, H, W, ci, co)
    b = pd.select_blocking(g, rs_plan.GRID, rs_plan.BUDGET)
    print(
        f"{tag:4s} idx{idx} {H}x{W} {ci}->{co}: bm {b.block_m_tiles} bk {b.block_k_tiles} bn {b.block_n_tiles} "
        f"nkb {b.num_k_blocks} mb {b.max_m_blocks_per_core} core_m {b.core_m_tiles} sb {b.out_subblock_h}x{b.out_subblock_w} "
        f"outdepth {b.out_depth_row_groups} ring {b.rm_ring_px} res {int(b.weights_resident)} ng {b.n_groups} "
        f"khr {b.khr_ring_rows} fp {b.footprint_bytes}"
    )
