// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Compute Kernel
// Stage 1: data_passthrough - copies input tiles to output unchanged.
//
// Compile-time args:
//   [0] Wt              : uint32_t - Width in tiles
//   [1] num_rows        : uint32_t - Number of tile-rows to process
//   [2] gamma_has_value : uint32_t - 1 if gamma is provided
//   [3] beta_has_value  : uint32_t - 1 if beta is provided

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t num_rows = get_compile_time_arg_val(1);
    constexpr uint32_t gamma_has_value = get_compile_time_arg_val(2);
    constexpr uint32_t beta_has_value = get_compile_time_arg_val(3);

    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_out = 16;

    // Initialize unpacker (input) and packer (output)
    unary_op_init_common(cb_input, cb_out);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Wait for reader to push Wt input tiles
        cb_wait_front(cb_input, Wt);

        // Copy each tile from input to output
        for (uint32_t w = 0; w < Wt; ++w) {
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            copy_tile(cb_input, w, 0);  // copy tile at index w to dst register 0
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);  // pack from dst register 0 to cb_out
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }

        // Done with this row's input tiles
        cb_pop_front(cb_input, Wt);
    }
}
