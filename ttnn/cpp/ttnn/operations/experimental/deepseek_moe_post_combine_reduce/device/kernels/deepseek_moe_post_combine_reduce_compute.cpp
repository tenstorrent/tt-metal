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

// Circular buffer indices
constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;  // Expert outputs (fake tiles)
constexpr uint32_t cb_weights = tt::CBIndex::c_1;        // Weight scalar per expert
constexpr uint32_t cb_output = tt::CBIndex::c_16;        // Final output

// Compile-time arguments
constexpr uint32_t num_tokens = get_compile_time_arg_val(0);
constexpr uint32_t num_experts = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(2);  // 7 for 7168/1024

// DST register batch size (hardware max)
constexpr uint32_t DST_BATCH_SIZE = 8;

void kernel_main() {
    uint32_t tokens_per_core = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);

    // Initialize binary operations
    binary_op_init_common(cb_combine_input, cb_weights, cb_output);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Process each token assigned to this core
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for (uint32_t token_idx = 0; token_idx < tokens_per_core; ++token_idx) {
        DPRINT << "COMPUTE: Processing token " << token_idx << ENDL();

        // Reserve output space (7 tiles - the final reduced result for this token)
        cb_reserve_back(cb_output, emb_dim_tiles);

        // ──────────────────────────────────────────────────────────────────────
        // Step 1: Initialize accumulator to ZERO (all 7 tiles in one batch)
        // ──────────────────────────────────────────────────────────────────────
        tile_regs_acquire();

        zero_tile_init_short();
        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            zero_tile(j);  // Zero DST[0..6]
        }

        tile_regs_commit();
        tile_regs_wait();

        // Pack zeros to output (so we can accumulate into them)
        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            pack_tile(j, cb_output, j);
        }

        tile_regs_release();

        DPRINT << "  COMPUTE: Initialized accumulator to zero" << ENDL();

        // ──────────────────────────────────────────────────────────────────────
        // Step 2: For each expert, multiply by weight and accumulate
        // ──────────────────────────────────────────────────────────────────────
        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            DPRINT << "  COMPUTE: Processing expert " << expert_idx << ENDL();

            // Wait for this expert's weight (loaded by reader)
            cb_wait_front(cb_weights, 1);

            // Wait for this expert's output tiles (7 tiles)
            cb_wait_front(cb_combine_input, emb_dim_tiles);

            // Debug: Print input data before processing
            SliceRange sr_input = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 5, .ws = 1};
            DPRINT << "    Input tile 0: " << TileSlice(cb_combine_input, 0, sr_input, true, false) << ENDL();
            DPRINT << "    Weight: " << TileSlice(cb_weights, 0, sr_input, true, false) << ENDL();

            // ──────────────────────────────────────────────────────────────────
            // Process all 7 tiles in a single batch (fits perfectly in DST[0..6])
            // ──────────────────────────────────────────────────────────────────
            tile_regs_acquire();

            // ─────────────────────────────────────────────────────────────────
            // Multiply: expert_output[0..6] × weight → DST[0..6]
            // ─────────────────────────────────────────────────────────────────
            mul_tiles_bcast_scalar_init_short(cb_combine_input, cb_weights);

            for (uint32_t j = 0; j < emb_dim_tiles; j++) {
                // input[j] × weight → DST[j]
                mul_tiles_bcast<BroadcastType::SCALAR>(
                    cb_combine_input,
                    cb_weights,
                    j,  // Source tile index (0-6)
                    0,  // Weight tile (always 0)
                    j   // DST register (0-6)
                );
            }

            // ─────────────────────────────────────────────────────────────────
            // Accumulate: DST[j] = cb_output[j] + DST[j]
            // (Add current accumulator value to the weighted expert output)
            // ─────────────────────────────────────────────────────────────────
            binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_output);

            for (uint32_t j = 0; j < emb_dim_tiles; j++) {
                // cb_output[j] + DST[j] → DST[j]
                binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_output,
                    j,  // Accumulator tile index (0-6)
                    j   // DST register (has weighted output)
                );
            }

            tile_regs_commit();  // ⚡ Execute all 7 ops in parallel!
            tile_regs_wait();

            // ─────────────────────────────────────────────────────────────────
            // Pack updated accumulator back to cb_output
            // ─────────────────────────────────────────────────────────────────
            for (uint32_t j = 0; j < emb_dim_tiles; j++) {
                pack_tile(j, cb_output, j);
            }

            tile_regs_release();

            // Debug: Print accumulator after this expert
            DPRINT << "    Accumulator tile 0 after expert " << expert_idx << ": "
                   << TileSlice(cb_output, 0, sr_input, true, false) << ENDL();

            // Done with this expert's data
            cb_pop_front(cb_combine_input, emb_dim_tiles);
            cb_pop_front(cb_weights, 1);
        }

        // ──────────────────────────────────────────────────────────────────────
        // Step 3: Output final accumulated result
        // ──────────────────────────────────────────────────────────────────────
        SliceRange sr_final = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 10, .ws = 1};
        DPRINT << "  COMPUTE: Final output tile 0: " << TileSlice(cb_output, 0, sr_final, true, false) << ENDL();

        cb_push_back(cb_output, emb_dim_tiles);
    }
}
