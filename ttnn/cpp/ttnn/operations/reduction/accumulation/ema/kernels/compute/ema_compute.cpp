// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    //-------------------------------------------------------------------------
    // Compile time args
    // -----------------
    constexpr auto batches_per_core = get_compile_time_arg_val(0);
    constexpr auto pages_per_batch = get_compile_time_arg_val(1);

    //-------------------------------------------------------------------------
    // CB indices
    // -----------------
    constexpr auto src_cb = tt::CBIndex::c_0;
    constexpr auto dst_cb = tt::CBIndex::c_1;

    //-------------------------------------------------------------------------
    // Main loop - compute ema for each batch
    binary_op_init_common(src_cb, src_cb, dst_cb);
    add_tiles_init(src_cb, src_cb);
    // copy_tile_init(src_cb);
    for (uint32_t batch_id = 0; batch_id < batches_per_core; ++batch_id) {
        for (uint32_t page_id = 0; page_id < pages_per_batch; ++page_id) {
            cb_wait_front(src_cb, 1);
            tile_regs_acquire();
            // copy_tile(src_cb, 0, 0);
            // tt::compute::common::print_tile_rows(src_cb, 32, 0);
            add_tiles(src_cb, src_cb, 0, 0, 0);
            // dprint_tensix_dest_reg(0);
            tile_regs_commit();
            cb_pop_front(src_cb, 1);

            cb_reserve_back(dst_cb, 1);
            tile_regs_wait();
            pack_tile(0, dst_cb);
            tile_regs_release();
            cb_push_back(dst_cb, 1);
        }
    }
}
}  // namespace NAMESPACE
