// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

// Compute kernel: copies tiles from input CB to output CB.
// This kernel runs on receiver cores and demonstrates the compute pipeline stage.
//
// In a real application, this is where you would perform operations on the tiles
// (e.g., element-wise math, matrix operations, activations, etc.) before passing
// them to the writer kernel. For this example, we simply copy the tiles unchanged
// to illustrate the dataflow pattern.
void kernel_main() {
    uint32_t n_tiles = get_compile_time_arg_val(0);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;    // Input from mcast_receiver kernel
    constexpr tt::CBIndex cb_out0 = tt::CBIndex::c_16;  // Output to write_tiles kernel

    // Destination register index for tile operations
    constexpr uint32_t dst_reg_idx = 0;

    // Initialize for tile copy operation, which is a two step init:
    unary_op_init_common(cb_in0, cb_out0);
    copy_tile_init(cb_in0);

    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        // Wait for a tile to be available in the input CB (from mcast_receiver kernel)
        cb_wait_front(cb_in0, 1);

        // Acquire destination register for tile operations
        tile_regs_acquire();

        // Copy tile from input CB to destination register.
        // In a real application, this would be replaced with actual computation:
        // e.g., add_tiles(), mul_tiles(), exp_tile(), etc.
        copy_tile(cb_in0, 0, dst_reg_idx);

        // Free the input CB slot (tile has been copied to dest register)
        cb_pop_front(cb_in0, 1);

        // Release destination register (computation complete)
        tile_regs_commit();

        // Wait for destination register to be ready for packing
        tile_regs_wait();

        // Reserve space in the output CB for the result
        cb_reserve_back(cb_out0, 1);

        // Pack the result from destination register to output CB
        pack_tile(dst_reg_idx, cb_out0);

        // Mark output tile as ready for the writer kernel
        cb_push_back(cb_out0, 1);

        // Release destination register for next iteration
        tile_regs_release();
    }
}
