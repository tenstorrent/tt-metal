// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;
constexpr uint32_t cb_weights_id = tt::CBIndex::c_1;
constexpr uint32_t cb_token_counts_id = tt::CBIndex::c_4;
constexpr uint32_t cb_output_id = tt::CBIndex::c_16;
constexpr uint32_t cb_rowmajor_id = tt::CBIndex::c_17;

constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(0);

void kernel_main() {
    constexpr uint32_t TOKENS_PER_CHUNK = 32;
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);
    constexpr uint32_t total_token_tiles = TOKENS_PER_CHUNK * emb_dim_cb_tiles;

    CircularBuffer cb_combine_input(cb_combine_input_id);
    CircularBuffer cb_weights(cb_weights_id);
    CircularBuffer cb_token_counts(cb_token_counts_id);
    CircularBuffer cb_rowmajor(cb_rowmajor_id);

    compute_kernel_hw_startup(cb_combine_input_id, cb_weights_id, cb_output_id);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // The reader publishes each token's active-slot count, then streams exactly that many
        // (combine row, weight scalar) pairs per token.
        cb_token_counts.wait_front(1);
        cb_rowmajor.reserve_back(total_token_tiles);

        for (uint32_t i = 0; i < TOKENS_PER_CHUNK; ++i) {
            const uint32_t num_active = read_tile_value(cb_token_counts_id, 0, i);
            pack_reconfig_l1_acc(0);

            if (num_active == 0) {
                // No active slot: the token's row is exactly zero.
                tile_regs_acquire();
                fill_tile_init();
                fill_tile(0, 0.0f);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                    pack_tile<true>(0, cb_rowmajor_id, i * emb_dim_cb_tiles + j);
                }
                tile_regs_release();
                continue;
            }

            mul_bcast_scalar_init(cb_combine_input_id, cb_weights_id);
            for (uint32_t slot = 0; slot < num_active; ++slot) {
                cb_combine_input.wait_front(emb_dim_cb_tiles);
                cb_weights.wait_front(1);

                if (slot == 1) {
                    pack_reconfig_l1_acc(1);  // the first slot overwrites, the rest accumulate
                }

                tile_regs_acquire();
                for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                    mul_tiles_bcast<BroadcastType::SCALAR>(cb_combine_input_id, cb_weights_id, j, 0, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < emb_dim_cb_tiles; j++) {
                    pack_tile<true>(j, cb_rowmajor_id, i * emb_dim_cb_tiles + j);
                }
                tile_regs_release();

                cb_combine_input.pop_front(emb_dim_cb_tiles);
                cb_weights.pop_front(1);
            }
            pack_reconfig_l1_acc(0);
        }

        cb_token_counts.pop_front(1);
        cb_rowmajor.push_back(total_token_tiles);

        compute_kernel_lib::tilize<total_token_tiles, cb_rowmajor_id, cb_output_id>(1);
    }
}
