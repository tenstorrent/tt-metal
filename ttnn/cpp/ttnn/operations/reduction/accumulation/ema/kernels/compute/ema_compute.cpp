// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/transpose_wh_dest.h"
#include "compute_kernel_api/ema.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    //-------------------------------------------------------------------------
    // Compile time args
    // -----------------
    constexpr auto total_batches_per_core = get_compile_time_arg_val(0);
    constexpr auto tiles_per_channel = get_compile_time_arg_val(1);

    //-------------------------------------------------------------------------
    // CB indices
    // -----------------
    constexpr auto src_cb = tt::CBIndex::c_0;
    constexpr auto dst_cb = tt::CBIndex::c_1;
    constexpr auto prev_cb = tt::CBIndex::c_2;

    constexpr auto input_dst_index = 0;
    constexpr auto prev_dst_index = input_dst_index + 1;

    //-------------------------------------------------------------------------
    // Main loop - compute ema for each batch
    binary_op_init_common(src_cb, src_cb, dst_cb);
    add_tiles_init(src_cb, src_cb);
    ema_init();
    // transpose_wh_init(src_cb, dst_cb);
    // copy_tile_init(src_cb);

    for (uint32_t batch_id = 0; batch_id < total_batches_per_core; ++batch_id) {
        for (uint32_t tile_id = 0; tile_id < tiles_per_channel; ++tile_id) {
            cb_wait_front(src_cb, 1);
            tile_regs_acquire();
            // transpose_wh_init_short(src_cb);
            // transpose_wh_tile(src_cb, 0, 0);
            // transpose_wh_dest_init_short();
            // transpose_wh_dest(0);
            // copy_tile(src_cb, 0, 0);
            // tt::compute::common::print_tile_rows(src_cb, 32, 0);
            add_tiles(src_cb, src_cb, 0, 0, input_dst_index);
            ema_tile<input_dst_index>((tile_id == 0));
            // dprint_tensix_dest_reg(0);
            tile_regs_commit();
            cb_pop_front(src_cb, 1);

            cb_reserve_back(dst_cb, 1);
            tile_regs_wait();
            pack_tile(input_dst_index, dst_cb);
            // pack_tile(prev_dst_index, prev_cb);
            tile_regs_release();
            cb_push_back(dst_cb, 1);
        }
    }
}
}  // namespace NAMESPACE
