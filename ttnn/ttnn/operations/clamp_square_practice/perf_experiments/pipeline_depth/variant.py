# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pipeline_depth experiment: op descriptor with the DM transfer block decoupled from the DEST block.

Builds on the op's own create_program_descriptor (work split, RT args, compute kernel untouched),
then swaps reader/writer to the parametrized kernels here and resizes the two CBs.

  rd_tiles / wr_tiles  - reader/writer tiles per barrier/flush & per CB push/pop (multiple of the
                         compute block cb_block, which stays the op's DEST block = min(8, ...)).
  in_depth / out_depth - CB capacity in DM blocks (rd_tiles / wr_tiles units).
"""
from pathlib import Path

import ttnn
from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

KDIR = Path(__file__).parent / "kernels"


def build(x, out, min_val, max_val, rd_tiles=8, wr_tiles=8, in_depth=2, out_depth=2):
    desc = pd.create_program_descriptor(x, out, min_val, max_val)
    rk, wk, ck = desc.kernels
    cb_in, cb_out = desc.cbs
    cb_block = ck.compile_time_args[2]  # the op's DEST/compute block
    page = x.buffer_page_size()
    grid = x.device().compute_with_storage_grid_size()
    tiles_g1 = ttnn.split_work_to_cores(grid, x.buffer_num_pages(), row_wise=True)[4]  # largest per-core assignment
    cap = -(-tiles_g1 // cb_block) * cb_block  # no point in a DM block above the padded assignment

    def clamp(t):
        t = max(cb_block, (t // cb_block) * cb_block)
        return min(t, cap)

    rd, wr = clamp(rd_tiles), clamp(wr_tiles)
    rk.kernel_source = str(KDIR / "reader.cpp")
    wk.kernel_source = str(KDIR / "writer.cpp")
    ck.kernel_source = str(KDIR / "compute.cpp")
    ra, wa = list(rk.compile_time_args), list(wk.compile_time_args)
    rk.compile_time_args = [ra[0], rd, ra[2], cb_block] + ra[3:]
    wk.compile_time_args = [wa[0], wr, wa[2], cb_block] + wa[3:]
    # Capacity capped at the (padded) per-core assignment: a CB that already holds everything a
    # core will ever push never blocks its producer, so more L1 buys nothing (small shapes).
    in_tiles = min(in_depth * rd, -(-cap // rd) * rd)
    out_tiles = min(out_depth * wr, -(-cap // wr) * wr)
    cb_in.total_size = in_tiles * page
    cb_out.total_size = out_tiles * page
    desc.kernels = [rk, wk, ck]
    desc.cbs = [cb_in, cb_out]
    return desc, dict(cb_block=cb_block, rd=rd, wr=wr, in_tiles=in_tiles, out_tiles=out_tiles)
