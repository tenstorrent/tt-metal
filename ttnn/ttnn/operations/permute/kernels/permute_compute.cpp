// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// permute compute (tile_transpose only): transpose_block over this NEO's tiles, block_tiles per DEST window.
//
// Raw compute API, not a kernel-library helper (op_design.md "Helpers considered and rejected"): the eltwise
// chain has no transpose element, and its CopyTile path is barred from transposing on Quasar
// (tt_metal/hw/inc/api/compute/tile_move_copy.h: "Transpose not supported on Quasar").
//
// NEO t of C takes this cluster's tiles t, t+C, ... (its slice of the STRIDED rings), block_tiles at a time;
// the last window of a NEO may be shorter. Ring capacity per NEO is a whole number of windows, so a window
// never straddles the ring end. bf16 unpack -> datacopy (16-bit DEST) -> pack is bit-exact.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);
    static_assert(block_tiles >= 1 && block_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT, "DEST window overflow");

    const uint32_t t = get_my_thread_id();
    const uint32_t my_tiles = num_pages / C + (t < num_pages % C ? 1u : 0u);

    DataflowBuffer tiles_in(dfb::tiles_in);
    DataflowBuffer tiles_out(dfb::tiles_out);

    compute_kernel_hw_startup(dfb::tiles_in, dfb::tiles_out);
    transpose_init(dfb::tiles_in);
    pack_init(dfb::tiles_out);

    for (uint32_t done = 0; done < my_tiles; done += block_tiles) {
        const uint32_t n = my_tiles - done < block_tiles ? my_tiles - done : block_tiles;

        tiles_in.wait_front(n);
        tile_regs_acquire();
        transpose_block(dfb::tiles_in, 0, 0, n);
        tile_regs_commit();
        tiles_in.pop_front(n);

        tiles_out.reserve_back(n);
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dfb::tiles_out, i);
        }
        tile_regs_release();
        tiles_out.push_back(n);
    }
}
