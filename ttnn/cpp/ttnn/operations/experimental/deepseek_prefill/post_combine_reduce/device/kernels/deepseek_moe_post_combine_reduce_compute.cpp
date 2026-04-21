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
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(2);
constexpr uint32_t indices_pages_per_core = get_compile_time_arg_val(3);

void kernel_main() {
    constexpr uint32_t TOKENS_PER_CORE = 32;
    uint32_t token_start_idx = get_arg_val<uint32_t>(0);
    constexpr uint32_t total_expert_tiles = num_experts * emb_dim_tiles;
    constexpr uint32_t total_token_tiles = TOKENS_PER_CORE * emb_dim_tiles;

    // Wait for writer to finish loading scratch data
    cb_wait_front(cb_dispatch_table, dispatch_table_num_pages);
    cb_wait_front(cb_indices, indices_pages_per_core);

    binary_op_init_common(cb_combine_input, cb_weights, cb_output);

    cb_reserve_back(cb_rowmajor, total_token_tiles);

    for (uint32_t i = 0; i < TOKENS_PER_CORE; ++i) {
        mul_tiles_bcast_scalar_init_short(cb_combine_input, cb_weights);

        bool first_local = true;
        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            cb_wait_front(cb_combine_input, emb_dim_tiles);
            cb_wait_front(cb_weights, 1);

            // Look up global expert ID from indices CB
            uint32_t expert_id = read_tile_value(cb_indices, i, expert_idx);
            // Check dispatch table: -1 (0xFFFFFFFF) means non-local
            uint32_t chip_id = read_tile_value(cb_dispatch_table, 0, expert_id);
            bool is_local = (chip_id != 0xFFFFFFFF);

            // On the last expert, if none were local, we must process it to
            // initialize the accumulator. Writer guarantees the weight is zero
            // for this case, so multiply produces zeros.
            bool is_last = (expert_idx == num_experts - 1);
            bool must_zero_init = is_last && first_local;

            if (!is_local && !must_zero_init) {
                cb_pop_front(cb_combine_input, emb_dim_tiles);
                cb_pop_front(cb_weights, 1);
                continue;
            }

            if (!first_local) {
                pack_reconfig_l1_acc(1);  // accumulate
            } else {
                pack_reconfig_l1_acc(0);  // overwrite
                first_local = false;
            }

            tile_regs_acquire();

            for (uint32_t j = 0; j < emb_dim_tiles; j++) {
                mul_tiles_bcast<BroadcastType::SCALAR>(cb_combine_input, cb_weights, j, 0, j);
            }

            tile_regs_commit();
            tile_regs_wait();

            for (uint32_t j = 0; j < emb_dim_tiles; j++) {
                pack_tile<true>(j, cb_rowmajor, i * emb_dim_tiles + j);
            }

            tile_regs_release();

            cb_pop_front(cb_combine_input, emb_dim_tiles);
            cb_pop_front(cb_weights, 1);
        }
        pack_reconfig_l1_acc(0);
    }
    cb_push_back(cb_rowmajor, total_token_tiles);

    using namespace compute_kernel_lib::tilize_config;
    compute_kernel_lib::tilize<total_token_tiles, cb_rowmajor, cb_output>(1);
}
