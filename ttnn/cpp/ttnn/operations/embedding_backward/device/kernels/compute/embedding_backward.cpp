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

    constexpr uint32_t cb_grad = tt::CBIndex::c_0;
    constexpr uint32_t cb_index = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_intermed = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask = tt::CBIndex::c_24;
    constexpr uint32_t cb_chunk_count_scratch = tt::CBIndex::c_25;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_grad, cb_out);

    for (uint32_t i = 0; i < input_height; ++i) {
        cb_wait_front(cb_grad, max_tiles_per_core);

        // Get chunk_count from CB using mailbox-based synchronization (issue #27979)
        uint32_t chunk_count = read_tile_value(cb_chunk_count_scratch, 0, 0);

        for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
            cb_wait_front(cb_mask, 1);

            // Get idx_addr from CB using mailbox-based synchronization (issue #27979)
            uint32_t idx_addr = get_tile_address(cb_mask, 0);
#if defined(ARCH_BLACKHOLE)
            // Workaround for tt-metal issue #11816:
            // Flush the cache, forces going to L1 to access data
            asm volatile("fence");
#endif

            cb_wait_front(cb_out_intermed, max_tiles_per_core);

            cb_reserve_back(cb_out, max_tiles_per_core);

            for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                tile_regs_acquire();
                tile_regs_wait();

                copy_tile(cb_grad, hidden_dim, 0);

                copy_tile(cb_out_intermed, hidden_dim, 1);

                reshuffle_rows_tile_init();
                // reshuffle_rows_tile expects that tiles have a header of 16 bytes.
                // This isn't true, so we have to substract 16 bytes from the address.
                // Check implementation of reshuffle_rows_tile in LLK for more details.
                // tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_reshuffle_rows.h
                reshuffle_rows_tile(0, idx_addr - 16);

                pack_tile(1, cb_out, hidden_dim);  // reshuffle puts output into Tile 1 in DEST

                tile_regs_commit();
                tile_regs_release();
            }

            cb_push_back(cb_out, max_tiles_per_core);
            cb_pop_front(cb_out_intermed, max_tiles_per_core);

            cb_pop_front(cb_mask, 1);
        }

        cb_pop_front(cb_grad, max_tiles_per_core);
    }
}
}  // namespace NAMESPACE
