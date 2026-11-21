# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (ttnn stubbed, no device): which geometries take the kh-reuse walk, with their blocking.

    python3 -I khrk_plan.py [op dir name, default op_base] [sweep]
Without `sweep`: the FLUX.2 bh2x4 focus cases. With it: a grid of ROW_MAJOR interleaved geometries, printing the
kh-reuse ones (and flagging any with streamed, non-resident weights)."""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "whole_k_rowstream_v2"))
import rs_plan  # noqa: E402

rs_plan.HERE = Path(__file__).parent
name = sys.argv[1] if len(sys.argv) > 1 else "op_base"
pd = rs_plan.load(name)


def show(tag, g):
    b = pd.select_blocking(g, rs_plan.GRID, rs_plan.BUDGET)
    if b is None:
        return None
    return (
        f"{tag}: bm {b.block_m_tiles} bk {b.block_k_tiles} bn {b.block_n_tiles} nkb {b.num_k_blocks} "
        f"mb {b.max_m_blocks_per_core} core_m {b.core_m_tiles} sb {b.out_subblock_h}x{b.out_subblock_w} "
        f"res {int(b.weights_resident)} ng {b.n_groups} khr {b.khr_ring_rows} fp {b.footprint_bytes}"
    ), b


if "sweep" not in sys.argv:
    for idx, tag, H, W, ci, co in rs_plan.FOCUS:
        r = show(f"{tag:4s} idx{idx} {H}x{W} {ci}->{co}", rs_plan.geometry(pd, H, W, ci, co))
        print(r[0] if r else f"{tag} none")
else:
    n = nres = 0
    for H in (66, 258, 1026):
        for W in (64, 256, 1024, 2048):
            for ci in (64, 128, 256, 512):
                for co in (64, 256, 1024):
                    for pad in ((0, 0, 1, 1),):
                        g = rs_plan.geometry(pd, H, W, ci, co)
                        g = dataclasses.replace(g, padding=pad, output_height=H + pad[0] + pad[1] - 2)
                        try:
                            r = show(f"{H}x{W} {ci}->{co} pad{pad}", g)
                        except Exception as e:  # noqa: BLE001
                            continue
                        if r and r[1].khr_ring_rows > 0:
                            n += 1
                            if not r[1].weights_resident:
                                nres += 1
                                print("NONRES", r[0])
                            elif "-v" in sys.argv:
                                print(r[0])
    print(f"kh-reuse geometries: {n}, non-resident weights: {nres}")
