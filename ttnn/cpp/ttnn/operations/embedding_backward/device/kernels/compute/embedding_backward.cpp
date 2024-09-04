// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/reshuffle.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t tiles_per_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t max_tiles_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t input_height = get_compile_time_arg_val(1);

    constexpr uint32_t cb_grad = tt::CB::c_in0;
    constexpr uint32_t cb_index = tt::CB::c_in1;
    constexpr uint32_t cb_out_intermed = tt::CB::c_in2;
    constexpr uint32_t cb_mask = tt::CB::c_intermed0;
    constexpr uint32_t cb_chunk_count_scratch = tt::CB::c_intermed1;
    constexpr uint32_t cb_out = tt::CB::c_out0;

    unary_op_init_common(cb_grad);

    for (uint32_t i = 0; i < input_height; ++i) {
        cb_wait_front(cb_grad, max_tiles_per_core);

        // Get chunk_count from reader
        volatile uint32_t *chunk_addr_ptr;
        cb_get_tile(cb_chunk_count_scratch, 0, &chunk_addr_ptr);
        uint32_t chunk_count = chunk_addr_ptr[4];  // Need to shift because read ptr is off by 1 << 4 in BBE
        cb_release_tile(cb_chunk_count_scratch);

        for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {  // chunk_count
            cb_wait_front(cb_mask, 1);
            // get cb_index pointer from unpack to math thread
            volatile uint *idx_addr_ptr;
            uint32_t tile_to_get = 0;
            cb_get_tile(cb_mask, tile_to_get, &idx_addr_ptr);
            uint32_t idx_addr = reinterpret_cast<uint32_t>(idx_addr_ptr);

            cb_wait_front(cb_out_intermed, max_tiles_per_core);

            cb_reserve_back(cb_out, max_tiles_per_core);

            for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                tile_regs_acquire();
                tile_regs_wait();

                copy_tile(cb_grad, hidden_dim, 0);

                copy_tile(cb_out_intermed, hidden_dim, 1);

                reshuffle_rows_tile_init();
                reshuffle_rows_tile(0, idx_addr);

                pack_tile(1, cb_out, hidden_dim);  // reshuffle puts output into Tile 1 in DEST

                tile_regs_commit();
                tile_regs_release();
            }

            cb_push_back(cb_out, max_tiles_per_core);
            cb_pop_front(cb_out_intermed, max_tiles_per_core);

            cb_release_tile(cb_mask);
            cb_pop_front(cb_mask, 1);
        }

        cb_pop_front(cb_grad, max_tiles_per_core);
    }
}
}  // namespace NAMESPACE
