// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Note: The argument index to get_compile_time_arg_val() must be a compile time constant.
    const uint32_t num_k_blocks = get_compile_time_arg_val(0);

    // Read parameters from the kernel's runtime arguments.
    int arg_idx = 0;
    uint32_t num_tiles = get_arg_val<uint32_t>(arg_idx++);

    // We are going to read from these two circular buffers.
    // Note that indices have to be in sync with the reader kernel.
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    // And write to this circular buffer.
    // Note that indices have to be in sync with the writer kernel.
    constexpr tt::CBIndex cb_out0 = tt::CBIndex::c_16;

    // FPU has a destination register, which is an array that can fit multiple tiles (details vary on data type).
    // For our case, FPU will add two tiles and produce a result that is a single tile.
    // We will instruct FPU to store the result in the destination register array at index 0.
    constexpr uint32_t dst_reg_idx = 0;

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    mm_init(cb_in0, cb_in1, cb_out0);

    // The simplest possible version of outer product blocked matmul.
    // The reader is expected to read the A's and B's tile rows and tile
    // columns for each output tile
    // Loop over all the MtxNt tiles in the output matrix and compute each tile of the result.
    for (uint32_t out_tile = 0; out_tile < num_tiles; ++out_tile) {
        // Make sure destination register array is ready for FPU to write its result to.
        // Note that this will also initialize all the tiles in the destination register array to 0,
        // so it's important that this is done before the Kt loop, since that loop needs to
        // accumulate sum of partial products into the destination tile.
        tile_regs_acquire();

        // Loop over all the tiles along the K dimension to accumulate partial products.
        for (uint32_t kt = 0; kt < Kt; kt++) {
            // Wait until there is a tile in each of the input circular buffers.
            // In more advanced applications we could wait for multiple tiles in each buffer and use them to
            // perform a more complex operation or to improve performance.
            // These are blocking calls.
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            // Perform the matrix multiplication for the current tile.
            // NOTE: This function accumulates the result into the destination tile.
            matmul_tiles(cb_in0, cb_in1, 0, 0, dst_reg_idx);

            // Mark the tiles in the input circular buffers as consumed.
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        // Release the destination register array because the computation is done.
        tile_regs_commit();

        // Make sure destination register array is ready for packer RISC core to read from.
        tile_regs_wait();
        // Make sure there is space in the output circular buffer to write result to.
        cb_reserve_back(cb_out0, 1);
        // Copy the result of addition from destination register to the output circular buffer.
        pack_tile(dst_reg_idx, cb_out0);
        // Mark the tile in the output circular buffer as ready.
        cb_push_back(cb_out0, 1);

        // Release the destination register array because packing is done.
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
