// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"

void kernel_main() {
    // Note: The argument index to get_compile_time_arg_val() must be a compile time constant.
    const uint32_t num_k_blocks = get_compile_time_arg_val(0);
    const uint32_t M_block_tiles = get_compile_time_arg_val(1);
    const uint32_t N_block_tiles = get_compile_time_arg_val(2);
    const uint32_t K_block_tiles = get_compile_time_arg_val(3);
    const uint32_t A_slab_tiles = get_compile_time_arg_val(4);
    const uint32_t B_slab_tiles = get_compile_time_arg_val(5);
    const uint32_t C_block_tiles = get_compile_time_arg_val(6);

    // We are going to read from these two circular buffers.
    // Note that indices have to be in sync with the reader kernel.
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    // And write to this circular buffer.
    // Note that indices have to be in sync with the writer kernel.
    constexpr tt::CBIndex cb_out0 = tt::CBIndex::c_16;

    constexpr tt::CBIndex cb_intermediate = tt::CBIndex::c_24;

    // FPU has a destination register, which is an array that can fit multiple tiles (details vary by data type).
    // For our case, FPU will multiply two tiles and produce a result that is a single tile.
    // We will instruct FPU to store the result in the destination register array at index 0.
    constexpr uint32_t dst_reg_idx = 0;

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    mm_init(cb_in0, cb_in1, cb_out0);

    // Code here largely follows the pseudocode in the lab writeup.
    // Loop over all the K-blocks.
    for (uint32_t b = 0; b < num_k_blocks; ++b) {
        cb_wait_front(cb_in0, A_slab_tiles);  // Ensure that A_slab(b) is in CB0
        cb_wait_front(cb_in1, B_slab_tiles);  // Ensure that B_slab(b) is in CB1

        // For every output tile (i,j) in this C_block:
        for (uint32_t i = 0; i < M_block_tiles; i++) {
            for (uint32_t j = 0; j < N_block_tiles; j++) {
                tile_regs_acquire();
                // Get the current accumulator tile for C(i,j)
                if (b != 0) {
                    // Middle or last K-block: partial result for C(i, j) already exists.
                    // Load the partial result built so far into the destination register.
                    // First run the initialization step for the copy operation.
                    copy_tile_to_dst_init_short(cb_intermediate);

                    // Pop just one tile.
                    // This is possible because the i, j loop goes in the same order for every block,
                    // so values are popped in the same order that they were pushed.
                    cb_wait_front(cb_intermediate, 1);
                    copy_tile(cb_intermediate, 0, dst_reg_idx);
                    cb_pop_front(cb_intermediate, 1);

                    // Reinitialize the FPU for matmul.
                    mm_init_short(cb_in0, cb_in1);
                }

                // Compute partial result for block b, and accumulate it into the destination register.
                for (uint32_t k_local = 0; k_local < K_block_tiles; k_local++) {
                    matmul_tiles(cb_in0, cb_in1, i * K_block_tiles + k_local, k_local * N_block_tiles + j, dst_reg_idx);
                }

                // Release the destination register array because the computation is done.
                tile_regs_commit();

                // Make sure destination register array is ready for packer RISC core to read from.
                tile_regs_wait();
                // Store updated result for C(i,j)
                if (b == num_k_blocks - 1) {
                    // Last K-block: acc_tile has the final result for C(i,j)
                    cb_reserve_back(cb_out0, 1);
                    // Copy the result of matrix multiplication from destination register to the output circular buffer.
                    pack_tile(dst_reg_idx, cb_out0);
                    // Mark the tile in the output circular buffer as ready.
                    cb_push_back(cb_out0, 1);
                } else {
                    // Not last K-block: acc_tile is a partial result to be reused later
                    cb_reserve_back(cb_intermediate, 1);
                    // Copy the partial result from destination register to the intermediate circular buffer.
                    pack_tile(dst_reg_idx, cb_intermediate);
                    // Mark the tile in the output circular buffer as ready.
                    cb_push_back(cb_intermediate, 1);
                }

                // Release the destination register array because packing is done.
                tile_regs_release();
            }
        }

        cb_pop_front(cb_in0, A_slab_tiles);  // Done with A_slab(b), free space in CB0
        cb_pop_front(cb_in1, B_slab_tiles);  // Done with B_slab(b), free space in CB1
    }
}
