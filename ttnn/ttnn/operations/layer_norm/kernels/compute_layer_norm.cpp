// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel (Stage 1: Identity Passthrough)
//
// Pass 1: Copy input tiles from c_0 to c_16 (identity).
// Pass 2: Consume and discard tiles from c_0.
// Pass 3: Consume and discard tiles from c_0.
//
// Compile-time args:
//   [0] num_rows_per_core : uint32  Tile-rows assigned to this core
//   [1] Wt               : uint32  Width in tiles
//   [2] has_gamma        : uint32  1 if gamma tensor is present
//   [3] has_beta         : uint32  1 if beta tensor is present

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    constexpr uint32_t cb_input = 0;    // c_0
    constexpr uint32_t cb_output = 16;  // c_16

    unary_op_init_common(cb_input, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Pass 1: Copy c_0 -> c_16 (identity passthrough)
        for (uint32_t col = 0; col < Wt; ++col) {
            acquire_dst();

            cb_wait_front(cb_input, 1);
            cb_reserve_back(cb_output, 1);
            copy_tile(cb_input, 0, 0);

            pack_tile(0, cb_output);

            cb_pop_front(cb_input, 1);
            cb_push_back(cb_output, 1);

            release_dst();
        }

        // Pass 2: Consume and discard tiles from c_0
        for (uint32_t col = 0; col < Wt; ++col) {
            cb_wait_front(cb_input, 1);
            cb_pop_front(cb_input, 1);
        }

        // Pass 3: Consume and discard tiles from c_0
        for (uint32_t col = 0; col < Wt; ++col) {
            cb_wait_front(cb_input, 1);
            cb_pop_front(cb_input, 1);
        }
    }
}
