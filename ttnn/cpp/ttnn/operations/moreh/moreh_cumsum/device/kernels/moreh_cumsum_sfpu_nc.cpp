// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#define APPROX false
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    const auto num_tiles_to_cumsum = get_arg_val<uint32_t>(0);
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_intermed = tt::CBIndex::c_24;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    unary_op_init_common(cb_in0, cb_out0);

    cb_wait_front(cb_in1, 1);

    for (uint32_t i = 0; i < num_output_tiles_per_core; i++) {
        // Initialize cb_intermed with cb_in1 (0)
        tile_regs_acquire();  // MATH ACQ
        copy_tile_to_dst_init_short(cb_in1);
        copy_tile(cb_in1, first_tile, dst0);
        tile_regs_commit();  // MATH REL

        tile_regs_wait();  // isto to PACK
        cb_reserve_back(cb_intermed, 1);
        pack_tile(dst0, cb_intermed);
        cb_push_back(cb_intermed, 1);
        tile_regs_release();

        for (uint32_t j = 0; j < num_tiles_to_cumsum; ++j) {
            // Copy intermediate sum to dst0
            cb_wait_front(cb_intermed, 1);  // Wait for one tile
            tile_regs_acquire();  // has to be after wait front, otherwise MATH will acquire lock and PACK wond be able
                                  // to signal cb_intermed
            copy_tile_to_dst_init_short(cb_intermed);
            copy_tile(cb_intermed, first_tile, dst0);
            cb_pop_front(cb_intermed, 1);  // Pop one tile

            // Copy input tile to dst1
            cb_wait_front(cb_in0, 1);
            copy_tile_to_dst_init_short(cb_in0);
            copy_tile(cb_in0, first_tile, dst1);
            cb_pop_front(cb_in0, 1);

            // Add tiles in dst0 and dst1. Store result to dst0
            add_int_tile_init();
            add_int32_tile(dst0, dst1);
            tile_regs_commit();

            // Copy sum to intermediate CB
            tile_regs_wait();
            cb_reserve_back(cb_intermed, 1);
            pack_tile(dst0, cb_intermed);
            cb_push_back(cb_intermed, 1);

            // Copy sum to output
            cb_reserve_back(cb_out0, 1);
            pack_tile(dst0, cb_out0);
            cb_push_back(cb_out0, 1);
            tile_regs_release();
        }
        cb_wait_front(cb_intermed, 1);
        cb_pop_front(cb_intermed, 1);  // this solves blocking in cases where outer loop has multiple iterations
    }
    cb_pop_front(cb_in1, 1);
}
}  // namespace NAMESPACE
