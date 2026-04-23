// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"

constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;
constexpr uint32_t cb_weights = tt::CBIndex::c_1;
constexpr uint32_t cb_dispatch_table = tt::CBIndex::c_2;
constexpr uint32_t cb_indices = tt::CBIndex::c_3;
constexpr uint32_t cb_output = tt::CBIndex::c_16;
constexpr uint32_t cb_rowmajor = tt::CBIndex::c_17;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
// The following two CT args are only meaningful when use_dispatch_table_skip
// is true; they carry zeros from the program factory otherwise.
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(2);
constexpr uint32_t indices_pages_per_core = get_compile_time_arg_val(3);
constexpr bool use_dispatch_table_skip = get_compile_time_arg_val(4) != 0;

void kernel_main() {
    constexpr uint32_t TOKENS_PER_CORE = 32;
    uint32_t token_start_idx = get_arg_val<uint32_t>(0);
    constexpr uint32_t total_token_tiles = TOKENS_PER_CORE * emb_dim_cb_tiles;

    if constexpr (use_dispatch_table_skip) {
        // Wait for writer to finish loading scratch data
        cb_wait_front(cb_dispatch_table, dispatch_table_num_pages);
        cb_wait_front(cb_indices, indices_pages_per_core);
    }

    binary_op_init_common(cb_combine_input, cb_weights, cb_output);

    cb_reserve_back(cb_rowmajor, total_token_tiles);

    // Process one expert at a time: both input (c_0) and weight (c_1) are streamed
    // one expert at a time by reader and writer respectively.
    for (uint32_t i = 0; i < TOKENS_PER_CORE; ++i) {
        mul_tiles_bcast_scalar_init_short(cb_combine_input, cb_weights);

        // In the DeepSeek path, "first_local" tracks whether we've started the
        // accumulator with a truly local expert; in the GPT-OSS path,
        // "first_active" tracks whether we've picked a non-zero-weight expert.
        // In both cases a single pass through num_experts is made, skipping
        // inactive experts except when a per-token fallback is needed to
        // initialise the accumulator.
        bool first_active = true;
        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            cb_wait_front(cb_combine_input, emb_dim_cb_tiles);
            cb_wait_front(cb_weights, 1);

            bool skip_expert = false;
            bool must_zero_init = false;
            if constexpr (use_dispatch_table_skip) {
                // Look up global expert ID from indices CB
                uint32_t expert_id = read_tile_value(cb_indices, i, expert_idx);
                // Check dispatch table: -1 (0xFFFFFFFF) means non-local
                uint32_t chip_id = read_tile_value(cb_dispatch_table, 0, expert_id);
                bool is_local = (chip_id != 0xFFFFFFFF);

                // On the last expert, if none were local, we must process it to
                // initialize the accumulator. Writer guarantees the weight is zero
                // for this case, so multiply produces zeros.
                bool is_last = (expert_idx == num_experts - 1);
                must_zero_init = is_last && first_active;
                skip_expert = !is_local && !must_zero_init;
            } else {
                // Read weight value — if zero, skip (but always process at least one
                // expert so the accumulator gets initialized with valid data)
                uint32_t weight_val = read_tile_value(cb_weights, 0, 0);
                skip_expert = (weight_val == 0) && !first_active;
            }

            if (skip_expert) {
                cb_pop_front(cb_combine_input, emb_dim_cb_tiles);
                cb_pop_front(cb_weights, 1);
                continue;
            }

            if (!first_active) {
                pack_reconfig_l1_acc(1);  // accumulate
            } else {
                pack_reconfig_l1_acc(0);  // overwrite
                first_active = false;
            }

            tile_regs_acquire();

            for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                mul_tiles_bcast<BroadcastType::SCALAR>(cb_combine_input, cb_weights, j, 0, j);
            }

            tile_regs_commit();
            tile_regs_wait();

            for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                pack_tile<true>(j, cb_rowmajor, i * emb_dim_cb_tiles + j);
            }

            tile_regs_release();

            cb_pop_front(cb_combine_input, emb_dim_cb_tiles);
            cb_pop_front(cb_weights, 1);
        }
        pack_reconfig_l1_acc(0);
    }
    cb_push_back(cb_rowmajor, total_token_tiles);

    // Tilize all 32 tokens' row-major scratch into the output CB as one block.
    ckernel::tilize_init(cb_rowmajor, total_token_tiles, cb_output);
    cb_wait_front(cb_rowmajor, total_token_tiles);
    cb_reserve_back(cb_output, total_token_tiles);
    ckernel::tilize_block(cb_rowmajor, total_token_tiles, cb_output);
    cb_pop_front(cb_rowmajor, total_token_tiles);
    cb_push_back(cb_output, total_token_tiles);
    ckernel::tilize_uninit(cb_rowmajor, cb_output);
}
