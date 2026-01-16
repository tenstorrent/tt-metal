// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Note: The argument index to get_compile_time_arg_val() must be a compile time constant.
    uint32_t n_tiles = get_compile_time_arg_val(0);

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

    // Initialize the Tensix Engine to perform an elementwise binary operation using circular buffers c_in0, c_in1 and c_out0.
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    // Initialize FPU for elementwise add operation specifically. This function is called any time we want to switch
    // operation to a different elementwise operation. Since we are only adding tiles, this function is called
    // only once before the loop.
    add_tiles_init(cb_in0, cb_in1);

    // Loop over all the tiles and perform the computation.
    // it's important to keep in mind that compute kernel runs on three different RISC-V processors.
    // One for unpacking, one for computing, and one for packing.
    // The compiler automatically compiles the same compute kernel code for all three processors,
    // relieving programmer from having to write different code for each core.
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait until there is a tile in each of the input circular buffers.
        // In more advanced applications we could wait for multiple tiles in each buffer and use them to
        // perform a more complex operation or to improve performance.
        // These are blocking calls.
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // Make sure destination register array is ready for FPU to write its result to.
        // Note that this will also initialize all the tiles in the destination register array to 0.
        // The initialization is not needed for this example, but is quite useful for matrix multiply.
        tile_regs_acquire();

        // Add the tiles from the input circular buffers: 0, 0 are tile indices into cb_in0 and cb_in1
        // to read the tiles from. Since we only waited for a single tile in each buffer above, there is
        // is only one tile to read from each buffer, so both indices are 0.
        // Write the result of FPU computation at specified index in the destination register array.
        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg_idx);

        // Mark the tiles in the input circular buffers as consumed.
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

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
