// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t num_tiles = num_blocks * block_num_tiles;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    // Wait for all tiles upfront from both input CBs
    cb_wait_front(cb_in0, num_tiles);
    cb_wait_front(cb_in1, num_tiles);
    cb_reserve_back(cb_out0, num_tiles);

    // Process tiles in batches of max_dst_tiles for efficiency
    constexpr uint32_t max_dst_tiles = 4;
    constexpr uint32_t num_batches = (num_tiles + max_dst_tiles - 1) / max_dst_tiles;

    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        uint32_t start_tile = batch * max_dst_tiles;
        uint32_t batch_size = (start_tile + max_dst_tiles <= num_tiles) ? max_dst_tiles : (num_tiles - start_tile);

        tile_regs_acquire();
        for (uint32_t i = 0; i < batch_size; ++i) {
            add_tiles(cb_in0, cb_in1, start_tile + i, start_tile + i, i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < batch_size; ++i) {
            pack_tile<true>(i, cb_out0, start_tile + i);
        }
        tile_regs_release();
    }

    // Pop all tiles at once after processing
    cb_pop_front(cb_in0, num_tiles);
    cb_pop_front(cb_in1, num_tiles);
    cb_push_back(cb_out0, num_tiles);
}
}  // namespace NAMESPACE
