// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/debug/dprint.h"
using namespace ckernel;

/**
 * @brief Compute kernel for matmul + add: output = a @ b + c
 *
 * Performs tiled matrix multiplication followed by element-wise addition.
 * For each output tile, it:
 * - Accumulates the dot product across K tiles
 * - Adds the bias tile from c
 * - Packs the result to the output circular buffer
 */
void kernel_main() {
    uint32_t num_output_tiles = get_arg_val<uint32_t>(0);  // number of output tiles to produce
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    uint32_t coreX = get_arg_val<uint32_t>(0);  // core X coordinate
    uint32_t coreY = get_arg_val<uint32_t>(1);  // core Y coordinate

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_in2 = tt::CBIndex::c_2;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    mm_init(cb_in0, cb_in1, cb_out);

    DPRINT << "Compute kernel started. Core: (" << coreX << ", " << coreY << ")" << ENDL();

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        // Make sure registers can be used for the output tile. This also sets the registers to zero.
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; kt++) {
            // Wait for the input tiles to be available in the input circular buffers.
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            // Perform the matrix multiplication for the current tile.
            // NOTE: This function also accumulates the result into the destination tile.
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

            // Mark the input tiles as used by popping them from the front of the circular buffers.
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        // Add bias from c (reusing matmul result in DST)
        cb_wait_front(cb_in2, 1);
        binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in2);
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in2, 0, 0);
        cb_pop_front(cb_in2, 1);

        // Commit and wait for the registers are populated with the results from the FPU
        tile_regs_commit();
        tile_regs_wait();

        // Ensure the output circular buffer has space for the result tile.
        cb_reserve_back(cb_out, 1);
        // Pack the result tile into the output circular buffer.
        pack_tile(0, cb_out);
        // Mark the output tile as ready so the writer can read it.
        cb_push_back(cb_out, 1);

        // We don't need the registers anymore, so we can release them and prepare for the next output tile.
        tile_regs_release();
    }
}
