// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Sparse MoE compute kernel: matmul input × expert_weights per expert.
//
// For each expert:
//   For each output sub-block of OUT_SUB tiles:
//     acquire_dst() — holds OUT_SUB partial accumulators
//     For each K row:
//       Wait for OUT_SUB weight tiles from reader
//       matmul_tiles: DST[j] += input[k] * weight[j] for j in 0..OUT_SUB-1
//       Pop weight tiles
//     Pack OUT_SUB output tiles → cb_out
//     release_dst()
//
// Input tiles (cb_input, k_tiles) stay in CB for all experts.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);           // 64 (hidden_dim/32)
    constexpr uint32_t tiles_per_expert = get_compile_time_arg_val(2);  // 32 (expert_width/32)
    constexpr uint32_t out_sub = get_compile_time_arg_val(3);           // 8 (output sub-block size, fits in DST)

    constexpr auto cb_input = tt::CBIndex::c_0;    // input tiles (k_tiles, held)
    constexpr auto cb_weights = tt::CBIndex::c_1;  // weight tiles (out_sub per push)
    constexpr auto cb_out = tt::CBIndex::c_16;     // output tiles

    // Wait for input tiles (held by reader for all experts)
    cb_wait_front(cb_input, k_tiles);

    mm_init(cb_input, cb_weights, cb_out);

    constexpr uint32_t num_sub_blocks = tiles_per_expert / out_sub;

    for (uint32_t e = 0; e < num_local_experts; e++) {
        for (uint32_t sb = 0; sb < num_sub_blocks; sb++) {
            acquire_dst();

            for (uint32_t k = 0; k < k_tiles; k++) {
                // Wait for out_sub weight tiles (one K row, sub-block columns)
                cb_wait_front(cb_weights, out_sub);

                // Accumulate: DST[j] += input[k] * weight[j]
                for (uint32_t j = 0; j < out_sub; j++) {
                    matmul_tiles(cb_input, cb_weights, k, j, j);
                }

                cb_pop_front(cb_weights, out_sub);
            }

            // Pack output tiles
            for (uint32_t j = 0; j < out_sub; j++) {
                cb_reserve_back(cb_out, 1);
                pack_tile(j, cb_out);
                cb_push_back(cb_out, 1);
            }

            release_dst();
        }
    }

    // Pop input tiles
    cb_pop_front(cb_input, k_tiles);
}
