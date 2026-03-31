// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;
constexpr uint32_t cb_weights = tt::CBIndex::c_1;
constexpr uint32_t cb_accumulator = tt::CBIndex::c_24;
constexpr uint32_t cb_output = tt::CBIndex::c_16;

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

        tile_regs_release();

        cb_pop_front(cb_accumulator, emb_dim_tiles);
        cb_push_back(cb_output, emb_dim_tiles);

        cb_pop_front(cb_combine_input, total_expert_tiles);
        cb_pop_front(cb_weights, num_experts);
    }
}
