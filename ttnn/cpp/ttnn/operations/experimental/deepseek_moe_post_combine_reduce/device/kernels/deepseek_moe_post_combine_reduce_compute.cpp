// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
// #include "api/compute/pack_untilize.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;
constexpr uint32_t cb_weights = tt::CBIndex::c_1;
constexpr uint32_t cb_accumulator = tt::CBIndex::c_24;
constexpr uint32_t cb_output = tt::CBIndex::c_16;
constexpr uint32_t cb_rowmajor = tt::CBIndex::c_17;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(1);

void kernel_main() {
    SliceRange sr = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};
    uint32_t token_start_idx = get_arg_val<uint32_t>(0);

    binary_op_init_common(cb_combine_input, cb_weights, cb_output);

    // Each core processes exactly 32 tokens for hardware tilization
    constexpr uint32_t TOKENS_PER_CORE = 32;
    static_assert(TOKENS_PER_CORE == 32, "Hardware tilize requires exactly 32 tokens per core");

    // Reserve row-major buffer for all 32 tokens (tile-sized pages: 32 tokens × 7 tiles each = 224)
    cb_reserve_back(cb_rowmajor, TOKENS_PER_CORE * emb_dim_tiles);

    // Process each token
    for (uint32_t i = 0; i < TOKENS_PER_CORE; ++i) {
        uint32_t total_expert_tiles = num_experts * emb_dim_tiles;
        cb_wait_front(cb_combine_input, total_expert_tiles);
        cb_wait_front(cb_weights, num_experts);

        cb_reserve_back(cb_accumulator, emb_dim_tiles);

        // === STAGE 1: Print input data for token 0 ===
        // if (i == 0) {
        //     DPRINT_UNPACK({ DPRINT << "=== STAGE 1: INPUT DATA (token 0) ===" << ENDL(); });
        //     for (uint32_t e = 0; e < num_experts; ++e) {
        //         for (uint32_t j = 0; j < emb_dim_tiles; ++j) {
        //             uint32_t tile_idx = e * emb_dim_tiles + j;
        //             DPRINT_UNPACK({ DPRINT << "  expert " << e << " tile " << j << ": "
        //                 << TileSlice(cb_combine_input, tile_idx, sr, true, false) << ENDL(); });
        //         }
        //     }
        //     DPRINT_UNPACK({ DPRINT << "  weights: "; });
        //     for (uint32_t e = 0; e < num_experts; ++e) {
        //         DPRINT_UNPACK({ DPRINT << TileSlice(cb_weights, e, sr, true, false) << " "; });
        //     }
        //     DPRINT_UNPACK({ DPRINT << ENDL(); });
        // }

        // Weighted sum across experts (unchanged logic)
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

        // Copy accumulator result to row-major buffer
        cb_wait_front(cb_accumulator, emb_dim_tiles);
        // === STAGE 2: Print accumulator result for token 0 ===
        if (i == 0) {
            DPRINT_UNPACK({ DPRINT << "=== STAGE 2: ACCUMULATOR (token 0) ===" << ENDL(); });
            for (uint32_t j = 0; j < emb_dim_tiles; ++j) {
                DPRINT_UNPACK(
                    { DPRINT << "  tile " << j << ": " << TileSlice(cb_accumulator, j, sr, true, false) << ENDL(); });
            }
        }

        tile_regs_acquire();

        copy_tile_init(cb_accumulator);
        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            copy_tile(cb_accumulator, j, j);
        }

        tile_regs_commit();
        tile_regs_wait();

        // Pack to row-major CB17 at position for this token in batch
        for (uint32_t j = 0; j < emb_dim_tiles; j++) {
            // if (i == 0 || i == 1  || i == 2) {
            //     DPRINT_UNPACK({ DPRINT << "packing data from cb_accumulator to cb_rowmajor from tile " << j << " to
            //     tile " << (i * emb_dim_tiles + j) << ENDL(); });
            // }
            pack_tile(j, cb_rowmajor, i * emb_dim_tiles + j);
        }

        tile_regs_release();

        if (i == 0) {
            DPRINT_UNPACK({ DPRINT << "=== STAGE 3: ROW-MAJOR BUFFER (token 0) ===" << ENDL(); });
            for (uint32_t j = 0; j < emb_dim_tiles; ++j) {
                DPRINT_UNPACK(
                    { DPRINT << "  tile " << j << ": " << TileSlice(cb_rowmajor, j, sr, true, false) << ENDL(); });
            }
        }

        cb_pop_front(cb_accumulator, emb_dim_tiles);
        cb_pop_front(cb_combine_input, total_expert_tiles);
        cb_pop_front(cb_weights, num_experts);
    }

    cb_push_back(cb_rowmajor, TOKENS_PER_CORE * emb_dim_tiles);

    // DPRINT_UNPACK({ DPRINT << "data row major" << ENDL(); });
    // for (uint32_t i = 0; i < emb_dim_tiles*32; ++i) {
    //     DPRINT_UNPACK({ DPRINT << "tile " << i << " data: " << TileSlice(cb_rowmajor, i, sr, true, false) << ENDL();
    //     });
    // }
    using namespace compute_kernel_lib::tilize_config;
    compute_kernel_lib::tilize<emb_dim_tiles * 32, cb_rowmajor, cb_output>(1);

    // === STAGE 4: Print tilized output (first few tiles) ===
    // DPRINT_UNPACK({
    //     DPRINT << "=== STAGE 4: TILIZED OUTPUT (cb_output) ===" << ENDL();
    //     // Print first 7 tiles (= first tile-row = token 0..31's data for tile-col 0..6)
    //     for (uint32_t t = 0; t < emb_dim_tiles; ++t) {
    //         DPRINT << "  tile " << t << ": " << TileSlice(cb_output, t, sr, true, false) << ENDL();
    //     }
    // });
}
