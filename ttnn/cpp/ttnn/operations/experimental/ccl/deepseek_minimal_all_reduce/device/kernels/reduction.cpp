// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    // cb_in0: remote data (intermediate tensor from other device)
    // cb_in1: local data (input tensor)
    // cb_out0: output tensor
    // cb_residual: residual tensor (optional, for fused residual add)
    // cb_temp: scratch buffer for (local + residual) intermediate result
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_residual = get_compile_time_arg_val(3);
    constexpr uint32_t cb_temp = get_compile_time_arg_val(4);
    constexpr uint32_t has_residual = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(6);

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    constexpr uint32_t max_dst_tiles = 4;
    constexpr uint32_t num_batches = (num_tiles + max_dst_tiles - 1) / max_dst_tiles;

    if constexpr (has_residual) {
        // Fused residual add: (local + residual) + remote → output
        // Step 1: Add local + residual while waiting for remote data

        cb_wait_front(cb_in1, num_tiles);       // local data
        cb_wait_front(cb_residual, num_tiles);  // residual data
        cb_reserve_back(cb_temp, num_tiles);    // temp storage for (local + residual)

        // First add: local + residual → temp
        for (uint32_t batch = 0; batch < num_batches; ++batch) {
            uint32_t start_tile = batch * max_dst_tiles;
            uint32_t batch_size = (start_tile + max_dst_tiles <= num_tiles) ? max_dst_tiles : (num_tiles - start_tile);

            tile_regs_acquire();
            for (uint32_t i = 0; i < batch_size; ++i) {
                add_tiles(cb_in1, cb_residual, start_tile + i, start_tile + i, i);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < batch_size; ++i) {
                pack_tile(i, cb_temp, start_tile + i);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_in1, num_tiles);
        cb_pop_front(cb_residual, num_tiles);
        cb_push_back(cb_temp, num_tiles);

        // Step 2: Wait for remote data, then add (local+residual) + remote → output
        cb_wait_front(cb_in0, num_tiles);   // remote data
        cb_wait_front(cb_temp, num_tiles);  // (local + residual) result
        cb_reserve_back(cb_out0, num_tiles);

        for (uint32_t batch = 0; batch < num_batches; ++batch) {
            uint32_t start_tile = batch * max_dst_tiles;
            uint32_t batch_size = (start_tile + max_dst_tiles <= num_tiles) ? max_dst_tiles : (num_tiles - start_tile);

            tile_regs_acquire();
            for (uint32_t i = 0; i < batch_size; ++i) {
                add_tiles(cb_temp, cb_in0, start_tile + i, start_tile + i, i);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < batch_size; ++i) {
                pack_tile(i, cb_out0, start_tile + i);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_in0, num_tiles);
        cb_pop_front(cb_temp, num_tiles);
        cb_push_back(cb_out0, num_tiles);
    } else {
        // Simple all-reduce: local + remote → output
        cb_wait_front(cb_in0, num_tiles);
        cb_wait_front(cb_in1, num_tiles);
        cb_reserve_back(cb_out0, num_tiles);

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
                pack_tile(i, cb_out0, start_tile + i);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_in0, num_tiles);
        cb_pop_front(cb_in1, num_tiles);
        cb_push_back(cb_out0, num_tiles);
    }
}
}  // namespace NAMESPACE
