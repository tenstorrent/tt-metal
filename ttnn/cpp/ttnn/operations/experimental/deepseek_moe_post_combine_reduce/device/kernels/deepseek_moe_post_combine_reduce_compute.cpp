// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"

/**
 * Fused post-combine reduce compute kernel.
 *
 * This kernel performs the following operations in a single pass:
 * 1. Read ROW_MAJOR combine output (no padding!)
 * 2. Read weights and broadcast across embedding dimension
 * 3. Multiply expert outputs by weights
 * 4. Reduce (sum) across expert dimension
 * 5. Output tiled result ready for reduce_scatter
 *
 * Input CBs:
 * - CB_combine_input: ROW_MAJOR combine output [seq_len, num_experts, emb_dim]
 * - CB_weights: Gate weights [seq_len, num_experts]
 *
 * Output CB:
 * - CB_output: TILE_LAYOUT reduced result [seq_len, emb_dim]
 */

// Compile-time constants
constexpr uint32_t CB_combine_input = tt::CBIndex::c_0;
constexpr uint32_t CB_weights = tt::CBIndex::c_1;
constexpr uint32_t CB_temp_tiled = tt::CBIndex::c_2;      // Temporary tiled data
constexpr uint32_t CB_temp_weighted = tt::CBIndex::c_3;   // After multiply
constexpr uint32_t CB_output = tt::CBIndex::c_4;

// Runtime arguments
constexpr uint32_t num_tokens = get_compile_time_arg_val(0);
constexpr uint32_t num_experts = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(2);  // emb_dim / 32

void kernel_main() {
    uint32_t tokens_per_core = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);

    // Initialize compute operations
    tilize_init_short(CB_combine_input, CB_temp_tiled);
    mul_tiles_init(CB_temp_tiled, CB_weights, CB_temp_weighted);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(CB_temp_weighted, CB_output);

    // Process tokens assigned to this core
    for (uint32_t token_idx = 0; token_idx < tokens_per_core; ++token_idx) {

        // Initialize accumulator for this token
        cb_reserve_back(CB_output, emb_dim_tiles);

        // Initialize output tiles to zero
        for (uint32_t emb_tile = 0; emb_tile < emb_dim_tiles; ++emb_tile) {
            tile_regs_acquire();
            // Zero out destination register
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(emb_tile, CB_output);
            tile_regs_release();
        }

        // Process each expert for this token
        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {

            // 1. Read and tilize expert output for this token
            cb_wait_front(CB_combine_input, emb_dim_tiles);
            cb_reserve_back(CB_temp_tiled, emb_dim_tiles);

            // Tilize expert output: ROW_MAJOR → TILE_LAYOUT
            for (uint32_t emb_tile = 0; emb_tile < emb_dim_tiles; ++emb_tile) {
                tilize_block(CB_combine_input, emb_tile, CB_temp_tiled, emb_tile);
            }
            cb_push_back(CB_temp_tiled, emb_dim_tiles);
            cb_pop_front(CB_combine_input, emb_dim_tiles);

            // 2. Read weight for this (token, expert) pair
            cb_wait_front(CB_weights, 1);  // Single weight value

            // 3. Broadcast multiply: expert_output * weight
            cb_wait_front(CB_temp_tiled, emb_dim_tiles);
            cb_reserve_back(CB_temp_weighted, emb_dim_tiles);

            for (uint32_t emb_tile = 0; emb_tile < emb_dim_tiles; ++emb_tile) {
                tile_regs_acquire();

                // Broadcast multiply: one weight tile broadcasts to emb_dim_tiles
                mul_tiles(CB_temp_tiled, CB_weights, emb_tile, 0, 0);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_temp_weighted, emb_tile);
                tile_regs_release();
            }
            cb_push_back(CB_temp_weighted, emb_dim_tiles);
            cb_pop_front(CB_temp_tiled, emb_dim_tiles);
            cb_pop_front(CB_weights, 1);

            // 4. Accumulate into output (reduce across experts)
            cb_wait_front(CB_temp_weighted, emb_dim_tiles);

            for (uint32_t emb_tile = 0; emb_tile < emb_dim_tiles; ++emb_tile) {
                tile_regs_acquire();

                // Load current accumulator value
                unpack_regs_acquire();
                unpack_tile(CB_output, emb_tile);
                unpack_regs_commit();

                // Add weighted expert contribution
                add_tiles(CB_output, CB_temp_weighted, emb_tile, emb_tile, 0);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_output, emb_tile);  // Update accumulator
                tile_regs_release();
            }
            cb_pop_front(CB_temp_weighted, emb_dim_tiles);
        }

        // Output final result for this token
        cb_push_back(CB_output, emb_dim_tiles);
    }
}