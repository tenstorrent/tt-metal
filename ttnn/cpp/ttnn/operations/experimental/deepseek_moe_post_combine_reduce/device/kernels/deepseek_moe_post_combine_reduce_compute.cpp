// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

/**
 * Optimized post-combine reduce compute kernel (Approach 1).
 *
 * Key optimizations:
 * 1. NO TILIZE: Treats ROW_MAJOR input as "fake tiles" directly
 * 2. DST BATCHING: Process all 7 embedding tiles in parallel using DST[0..6]
 * 3. SCALAR BROADCAST: Each weight multiplies all 7 embedding tiles for one expert
 * 4. ACCUMULATION: Reduce across 8 experts using binary_dest_reuse_tiles
 *
 * Input:
 * - cb_combine_input (c_0): ROW_MAJOR expert outputs [seq_len, num_experts, emb_dim] as "fake tiles"
 * - cb_weights (c_1): Single weight scalar per expert (loaded one at a time)
 *
 * Output:
 * - cb_output (c_16): TILE_LAYOUT reduced result [seq_len, emb_dim]
 *
 * Performance:
 * - Processes 7 tiles/batch (1 batch per expert, perfect DST utilization!)
 * - Single tile_regs_acquire/commit/release per expert (9 total: 1 init + 8 experts)
 * - No intermediate CBs for tilization
 * - Minimal sync overhead
 */
constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;  // Expert outputs (fake tiles)
constexpr uint32_t cb_weights = tt::CBIndex::c_1;        // Weight scalar per expert
constexpr uint32_t cb_accumulator = tt::CBIndex::c_24;   // Intermediate accumulation buffer
constexpr uint32_t cb_output = tt::CBIndex::c_16;        // Final output

constexpr uint32_t num_tokens = get_compile_time_arg_val(0);
constexpr uint32_t num_experts = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(2);

void kernel_main() {
    uint32_t tokens_per_core = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);

    binary_op_init_common(cb_combine_input, cb_weights, cb_output);

    for (uint32_t token_idx = 0; token_idx < tokens_per_core; ++token_idx) {
        uint32_t total_expert_tiles = num_experts * emb_dim_tiles;
        cb_wait_front(cb_combine_input, total_expert_tiles);
        cb_wait_front(cb_weights, num_experts);

        SliceRange sr = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

        cb_reserve_back(cb_accumulator, emb_dim_tiles);

        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            if (expert_idx > 0) {
                pack_reconfig_l1_acc(1);
            } else {
                pack_reconfig_l1_acc(0);
            }

            uint32_t tile_offset = expert_idx * emb_dim_tiles;

            tile_regs_acquire();

            mul_tiles_bcast_scalar_init_short(cb_combine_input, cb_weights);

            for (uint32_t j = 0; j < emb_dim_tiles; j++) {
                mul_tiles_bcast<BroadcastType::SCALAR>(cb_combine_input, cb_weights, tile_offset + j, expert_idx, j);
            }

            tile_regs_commit();
            tile_regs_wait();

            for (uint32_t j = 0; j < emb_dim_tiles; j++) {
                pack_tile<true>(j, cb_accumulator, j);  // out_of_order_output=true to write to explicit tile index
            }

            tile_regs_release();
        }

        cb_push_back(cb_accumulator, emb_dim_tiles);

        pack_reconfig_l1_acc(0);

        cb_wait_front(cb_accumulator, emb_dim_tiles);
        cb_reserve_back(cb_output, emb_dim_tiles);

        tile_regs_acquire();

        copy_tile_init(cb_accumulator);
        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            copy_tile(cb_accumulator, j, j);
        }

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            pack_tile(j, cb_output, j);
        }
        DPRINT_PACK(DPRINT << "final CB outputs -- " << "\n" << ENDL());
        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            DPRINT_PACK(DPRINT << "tile " << j << " values = " << TileSlice(cb_output, j, sr, true, false) << ENDL());
        }

        tile_regs_release();

        cb_pop_front(cb_accumulator, emb_dim_tiles);
        cb_push_back(cb_output, emb_dim_tiles);

        cb_pop_front(cb_combine_input, total_expert_tiles);
        cb_pop_front(cb_weights, num_experts);
    }
}
