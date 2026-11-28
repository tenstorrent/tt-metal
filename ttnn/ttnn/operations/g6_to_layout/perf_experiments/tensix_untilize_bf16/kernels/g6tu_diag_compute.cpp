// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf-1 experiment tensix_untilize_bf16: diagnostic compute (precision census only, never timed).
// One tile per tile-row (block_cols_tiles == 1). Modes (CT arg):
//   1  copy_init (Src zero flag = preserve) + copy_tile -> DEST, pack_untilize_dest -> rows
//   2  copy_init (Src zero flag = preserve) + copy_tile -> DEST, pack_tile -> TILE bytes (host re-orders)
//   3  as 2 with the flag forced to flush after copy_init (the pre-existing default)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(Geometry::next_ct_arg + 1);
    constexpr uint32_t mode = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
    static_assert(Geometry::block_cols_tiles == 1 && Geometry::block_rows_tiles == 1);

    CoreBlocks<false> blocks;
    Block block;
    uint32_t tiles = 0;
    while (blocks.next(block)) {
        ++tiles;
    }

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    if constexpr (mode == 3) {
        MATH((ckernel::math::_configure_src_zero_flag_(false)));
    }
    if constexpr (mode == 1) {
        pack_untilize_dest_init<1, 1>(cb_out);
    }
    for (uint32_t t = 0; t < tiles; ++t) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        if constexpr (mode == 1) {
            pack_untilize_dest<1, 1>(cb_out, 1, 0);
        } else {
            pack_tile(0, cb_out);
        }
        tile_regs_release();
        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
    if constexpr (mode == 1) {
        pack_untilize_uninit(cb_out);
    }
}
