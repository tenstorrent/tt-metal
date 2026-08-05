// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reshuffle.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t tiles_per_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t max_tiles_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t input_height = get_compile_time_arg_val(1);

    constexpr uint32_t cb_grad_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_index_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_intermed_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_idx = tt::CBIndex::c_24;
    constexpr uint32_t cb_chunk_count_scratch_idx = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_idx = tt::CBIndex::c_16;

    CircularBuffer cb_grad(cb_grad_idx);
    CircularBuffer cb_out_intermed(cb_out_intermed_idx);
    CircularBuffer cb_mask(cb_mask_idx);
    CircularBuffer cb_out(cb_out_idx);

    compute_kernel_hw_startup(cb_grad_idx, cb_out_idx);
    copy_init(cb_grad_idx);

    for (uint32_t i = 0; i < input_height; ++i) {
        cb_grad.wait_front(max_tiles_per_core);

        // Get chunk_count from CB using mailbox-based synchronization (issue #27979)
        uint32_t chunk_count = read_tile_value(cb_chunk_count_scratch_idx, 0, 0);

        for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
            cb_mask.wait_front(1);

            // Get idx_addr from CB using mailbox-based synchronization (issue #27979)
            uint32_t idx_addr = get_tile_address(cb_mask_idx, 0);
#if defined(ARCH_BLACKHOLE)
            // Workaround for tt-metal issue #11816:
            // Flush the cache, forces going to L1 to access data
            asm volatile("fence");
#endif

            cb_out_intermed.wait_front(max_tiles_per_core);

            cb_out.reserve_back(max_tiles_per_core);

            for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                tile_regs_acquire();
                tile_regs_wait();

                copy_tile(cb_grad_idx, hidden_dim, 0);

                copy_tile(cb_out_intermed_idx, hidden_dim, 1);

                reshuffle_rows_tile_init();
                // reshuffle_rows_tile expects that tiles have a header of 16 bytes.
                // This isn't true, so we have to subtract 16 bytes from the address.
                // Check implementation of reshuffle_rows_tile in LLK for more details.
                // tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reshuffle_rows.h
                reshuffle_rows_tile(0, idx_addr - 16);

                pack_tile(1, cb_out_idx, hidden_dim);  // reshuffle puts output into Tile 1 in DEST

                tile_regs_commit();
                tile_regs_release();
            }

            cb_out.push_back(max_tiles_per_core);
            cb_out_intermed.pop_front(max_tiles_per_core);

            cb_mask.pop_front(1);
        }

        cb_grad.pop_front(max_tiles_per_core);
    }
}
