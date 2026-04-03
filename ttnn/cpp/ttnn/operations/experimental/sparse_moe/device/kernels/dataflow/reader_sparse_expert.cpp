// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Sparse MoE reader: streams expert weight tiles from DRAM, skipping inactive experts.
//
// Push order:
//   1. Input tiles (k_tiles, held for all experts)
//   2. Per expert, per output sub-block, per K row: out_sub weight tiles
//
// For inactive experts: push uninitialized tiles (no DRAM read).
// The existing mask multiply in the pipeline zeros out garbage output.
//
// Active flag: runtime arg per expert (0 = inactive, 1 = active).
// Host pre-computes these flags from the routing weights.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t weights_addr = get_arg_val<uint32_t>(1);
    uint32_t expert_start = get_arg_val<uint32_t>(2);
    uint32_t num_local_experts = get_arg_val<uint32_t>(3);
    // Active flags follow as runtime args 4..4+num_local_experts-1
    // (1 = active, 0 = skip DRAM reads)

    // Compile-time args
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);           // 64
    constexpr uint32_t tiles_per_expert = get_compile_time_arg_val(1);  // 32
    constexpr uint32_t out_sub = get_compile_time_arg_val(2);           // 8
    constexpr uint32_t total_col_tiles = get_compile_time_arg_val(3);   // 2048 (65536/32)
    constexpr auto input_acc_args = TensorAccessorArgs<4>();
    constexpr auto weights_acc_args = TensorAccessorArgs<input_acc_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_weights = tt::CBIndex::c_1;

    uint32_t input_tile_bytes = get_tile_size(cb_input);
    uint32_t weight_tile_bytes = get_tile_size(cb_weights);

    const auto s_input = TensorAccessor(input_acc_args, input_addr, input_tile_bytes);
    const auto s_weights = TensorAccessor(weights_acc_args, weights_addr, weight_tile_bytes);

    constexpr uint32_t num_sub_blocks = tiles_per_expert / out_sub;

    // === Step 1: Read input tiles (held in CB for all experts) ===
    cb_reserve_back(cb_input, k_tiles);
    uint32_t l1_addr = get_write_ptr(cb_input);
    for (uint32_t k = 0; k < k_tiles; k++) {
        noc_async_read_tile(k, s_input, l1_addr);
        l1_addr += input_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_input, k_tiles);

    // === Step 2: Per-expert weight streaming ===
    for (uint32_t e = 0; e < num_local_experts; e++) {
        uint32_t global_expert = expert_start + e;
        uint32_t is_active = get_arg_val<uint32_t>(4 + e);

        // Expert e's weight tile at (K row r, local col c) has global tile_id:
        //   r * total_col_tiles + global_expert * tiles_per_expert + c

        for (uint32_t sb = 0; sb < num_sub_blocks; sb++) {
            uint32_t col_start = sb * out_sub;

            for (uint32_t k = 0; k < k_tiles; k++) {
                cb_reserve_back(cb_weights, out_sub);

                if (is_active) {
                    // Read out_sub weight tiles from DRAM
                    uint32_t w_l1 = get_write_ptr(cb_weights);
                    for (uint32_t j = 0; j < out_sub; j++) {
                        uint32_t tile_id = k * total_col_tiles + global_expert * tiles_per_expert + col_start + j;
                        noc_async_read_tile(tile_id, s_weights, w_l1);
                        w_l1 += weight_tile_bytes;
                    }
                    noc_async_read_barrier();
                }
                // If inactive: push uninitialized tiles (garbage zeroed by mask multiply later)

                cb_push_back(cb_weights, out_sub);
            }
        }
    }
}
