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
        // DPRINT << "COMPUTE: Processing token " << token_idx << ENDL();

        // Reserve output space (7 tiles - the final reduced result for this token)
        cb_reserve_back(cb_output, emb_dim_tiles);

        // ──────────────────────────────────────────────────────────────────────
        // Step 2: Wait for ALL data (bulk loaded by reader)
        // ──────────────────────────────────────────────────────────────────────
        uint32_t total_expert_tiles = num_experts * emb_dim_tiles;
        cb_wait_front(cb_combine_input, total_expert_tiles);  // All expert tiles
        cb_wait_front(cb_weights, num_experts);               // All weights

        // 2. COMPUTE START: Show first value of first 2 input tiles
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};
        DPRINT_UNPACK(
            DPRINT << "COMPUTE tile values for expert 0 -- " << "\n"
                   << TileSlice(cb_combine_input, 0, sr, true, false) << ", "
                   << TileSlice(cb_combine_input, 1, sr, true, false) << ", "
                   << TileSlice(cb_combine_input, 2, sr, true, false) << ", "
                   << TileSlice(cb_combine_input, 3, sr, true, false) << ", "
                   << TileSlice(cb_combine_input, 4, sr, true, false) << ", "
                   << TileSlice(cb_combine_input, 5, sr, true, false) << ", "
                   << TileSlice(cb_combine_input, 6, sr, true, false) << ENDL());

        // ──────────────────────────────────────────────────────────────────────
        // Step 3: Process first expert (no accumulation, just multiply and pack)
        // ──────────────────────────────────────────────────────────────────────
        // DPRINT_UNPACK(DPRINT << "  COMPUTE: Processing expert 0 (initial)" << ENDL());

        // SliceRange sr_input = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 5, .ws = 1};
        // DPRINT_MATH(DPRINT << "    Input tile offset 0: "
        //         << TileSlice(cb_combine_input, 0, sr_input, true, false) << ENDL());
        DPRINT_UNPACK(DPRINT << ENDL());
        DPRINT_UNPACK(
            DPRINT << "COMPUTE weight values for expert 0 ================== "
                   << TileSlice(cb_weights, 0, sr, true, false) << ENDL());

        tile_regs_acquire();

        // Multiply expert 0 output by weight 0
        mul_tiles_bcast_scalar_init_short(cb_combine_input, cb_weights);

        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            mul_tiles_bcast<BroadcastType::SCALAR>(
                cb_combine_input,
                cb_weights,
                j,  // Expert 0's tiles start at offset 0
                0,  // Weight 0
                j   // DST register (0-6)
            );
        }

        tile_regs_commit();
        tile_regs_wait();

        // Pack to output (this becomes our initial accumulator value)
        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            pack_tile(j, cb_output, j);
        }

        tile_regs_release();

        DPRINT_UNPACK(
            DPRINT << "COMPUTE output for expert 0 -- " << token_idx << " => " << "\n"
                   << TileSlice(cb_output, 0, sr, true, false) << "               , "
                   << TileSlice(cb_output, 1, sr, true, false) << "               , "
                   << TileSlice(cb_output, 2, sr, true, false) << "               , "
                   << TileSlice(cb_output, 3, sr, true, false) << "               , "
                   << TileSlice(cb_output, 4, sr, true, false) << "               , "
                   << TileSlice(cb_output, 5, sr, true, false) << "               , "
                   << TileSlice(cb_output, 6, sr, true, false) << ENDL());

        // ──────────────────────────────────────────────────────────────────────
        // Step 4: For remaining experts (1-7), multiply by weight and accumulate
        // ──────────────────────────────────────────────────────────────────────
        for (uint32_t expert_idx = 1; expert_idx < num_experts; ++expert_idx) {
            // DPRINT_UNPACK(DPRINT << "  COMPUTE: Processing expert " << expert_idx << ENDL());

            uint32_t tile_offset = expert_idx * emb_dim_tiles;  // Offset into CB for this expert's tiles

            // Debug: Print input data before processing
            // SliceRange sr_input = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 5, .ws = 1};
            // DPRINT_UNPACK(DPRINT << "    Input tile offset " << tile_offset << ": "
            //        << TileSlice(cb_combine_input, tile_offset, sr, true, false) << ENDL());
            DPRINT_UNPACK(
                DPRINT << "COMPUTE tile values for expert " << expert_idx << " -- " << "\n"
                       << TileSlice(cb_combine_input, tile_offset + 0, sr, true, false) << "               , "
                       << TileSlice(cb_combine_input, tile_offset + 1, sr, true, false) << "               , "
                       << TileSlice(cb_combine_input, tile_offset + 2, sr, true, false) << "               , "
                       << TileSlice(cb_combine_input, tile_offset + 3, sr, true, false) << "               , "
                       << TileSlice(cb_combine_input, tile_offset + 4, sr, true, false) << "               , "
                       << TileSlice(cb_combine_input, tile_offset + 5, sr, true, false) << "               , "
                       << TileSlice(cb_combine_input, tile_offset + 6, sr, true, false) << ENDL());
            DPRINT_UNPACK(
                DPRINT << "COMPUTE weight values for expert " << expert_idx
                       << " ================== " << TileSlice(cb_weights, expert_idx, sr, true, false) << ENDL());

            // ──────────────────────────────────────────────────────────────────
            // Process all 7 tiles in a single batch (fits perfectly in DST[0..6])
            // ──────────────────────────────────────────────────────────────────
            tile_regs_acquire();

            // ─────────────────────────────────────────────────────────────────
            // Multiply: expert_output[tile_offset..tile_offset+6] × weight[expert_idx] → DST[0..6]
            // ─────────────────────────────────────────────────────────────────
            mul_tiles_bcast_scalar_init_short(cb_combine_input, cb_weights);

            for (uint32_t j = 0; j < emb_dim_tiles; j++) {
                // Read from CB at correct offset for this expert
                mul_tiles_bcast<BroadcastType::SCALAR>(
                    cb_combine_input,
                    cb_weights,
                    tile_offset + j,  // Source tile index (expert's j-th tile)
                    expert_idx,       // Weight tile index (expert's weight)
                    j                 // DST register (0-6)
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
            DPRINT_PACK(
                DPRINT << "COMPUTE output for expert " << expert_idx << " -- " << "\n"
                       << TileSlice(cb_output, tile_offset + 0, sr, true, false) << "               , "
                       << TileSlice(cb_output, tile_offset + 1, sr, true, false) << "               , "
                       << TileSlice(cb_output, tile_offset + 2, sr, true, false) << "               , "
                       << TileSlice(cb_output, tile_offset + 3, sr, true, false) << "               , "
                       << TileSlice(cb_output, tile_offset + 4, sr, true, false) << "               , "
                       << TileSlice(cb_output, tile_offset + 5, sr, true, false) << "               , "
                       << TileSlice(cb_output, tile_offset + 6, sr, true, false) << ENDL());
        }

        // ──────────────────────────────────────────────────────────────────────
        // Step 5: Pop all data at once (after processing all experts)
        // ──────────────────────────────────────────────────────────────────────
        cb_pop_front(cb_combine_input, total_expert_tiles);
        cb_pop_front(cb_weights, num_experts);

        // ──────────────────────────────────────────────────────────────────────
        // Step 6: Output final accumulated result
        // ──────────────────────────────────────────────────────────────────────
        // 3. COMPUTE END: Show first value of first 2 output tiles
        DPRINT_PACK(
            DPRINT << "COMPUTE final output -- " << "\n"
                   << TileSlice(cb_output, 0, sr, true, false) << "               , "
                   << TileSlice(cb_output, 1, sr, true, false) << "               , "
                   << TileSlice(cb_output, 2, sr, true, false) << "               , "
                   << TileSlice(cb_output, 3, sr, true, false) << "               , "
                   << TileSlice(cb_output, 4, sr, true, false) << "               , "
                   << TileSlice(cb_output, 5, sr, true, false) << "               , "
                   << TileSlice(cb_output, 6, sr, true, false) << "               , "
                   << TileSlice(cb_output, 7, sr, true, false) << "               , "
                   << TileSlice(cb_output, 8, sr, true, false) << "               , "
                   << TileSlice(cb_output, 9, sr, true, false) << "               , "
                   << TileSlice(cb_output, 10, sr, true, false) << ENDL());

        cb_push_back(cb_output, emb_dim_tiles);
    }
}
