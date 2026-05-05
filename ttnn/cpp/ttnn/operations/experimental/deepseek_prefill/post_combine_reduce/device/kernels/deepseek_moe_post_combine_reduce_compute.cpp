// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;
constexpr uint32_t cb_weights = tt::CBIndex::c_1;
constexpr uint32_t cb_dispatch_table = tt::CBIndex::c_2;
constexpr uint32_t cb_indices = tt::CBIndex::c_3;
constexpr uint32_t cb_output = tt::CBIndex::c_16;
constexpr uint32_t cb_rowmajor = tt::CBIndex::c_17;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
// dispatch_table_num_pages is only meaningful when use_dispatch_table_skip
// is true; it carries zero from the program factory otherwise.
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(2);
constexpr bool use_dispatch_table_skip = get_compile_time_arg_val(3) != 0;

void kernel_main() {
    constexpr uint32_t TOKENS_PER_CHUNK = 32;
    uint32_t token_start_idx = get_arg_val<uint32_t>(0);
    uint32_t num_chunks = get_arg_val<uint32_t>(1);
    constexpr uint32_t total_token_tiles = TOKENS_PER_CHUNK * emb_dim_cb_tiles;

    if constexpr (use_dispatch_table_skip) {
        // Wait for writer to finish pre-loading dispatch table (loaded once for all chunks)
        cb_wait_front(cb_dispatch_table, dispatch_table_num_pages);
    }

    binary_op_init_common(cb_combine_input, cb_weights, cb_output);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        if constexpr (use_dispatch_table_skip) {
            // Wait for writer to load this chunk's indices (one chunk at a time)
            cb_wait_front(cb_indices, TOKENS_PER_CHUNK);
        }

        cb_reserve_back(cb_rowmajor, total_token_tiles);

        // Process one expert at a time: both input (c_0) and weight (c_1) are streamed
        // one expert at a time by reader and writer respectively.
        for (uint32_t i = 0; i < TOKENS_PER_CHUNK; ++i) {
            mul_tiles_bcast_scalar_init_short(cb_combine_input, cb_weights);

            // first_active tracks whether we've picked the accumulator-initializing
            // expert yet: the DeepSeek path looks for a locally-mapped expert via
            // the dispatch table; the GPT-OSS path looks for a non-zero routing
            // weight. A single pass skips inactive experts; if none qualified,
            // the last expert is forced through to initialise the accumulator.
            bool first_active = true;
            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                cb_wait_front(cb_combine_input, emb_dim_cb_tiles);
                cb_wait_front(cb_weights, 1);

                bool skip_expert = false;
                bool must_zero_init = false;
                if constexpr (use_dispatch_table_skip) {
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

        if constexpr (use_dispatch_table_skip) {
            // Release this chunk's indices so writer can load the next chunk's
            cb_pop_front(cb_indices, TOKENS_PER_CHUNK);
        }

        cb_push_back(cb_rowmajor, total_token_tiles);

        compute_kernel_lib::tilize<total_token_tiles, cb_rowmajor, cb_output>(1);
    }
}
