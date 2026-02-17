// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Read parameters from the kernel's runtime arguments.
    uint32_t out0_base_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    // The circular buffer that contains the result, which this kernel will
    // read from and then write to device memory.
    constexpr tt::CBIndex cb_out0 = tt::CBIndex::c_16;
    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (which is most often the case).
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_out0);

    // Create address generator for the output buffer. Address generators can determine
    // physical address based on the provided data layout and base address.
    // Start by extracting the tensor layout parameters from the compile-time arguments.
    // Recall that compile-time arguments are stored as a vector of uint32_t values. TensorAccessorArgs is a clean way
    // to extract the appropriate number of these uint32_t values and store them in an object.
    constexpr auto out0_layout_args = TensorAccessorArgs<0>();
    // Then, construct the address generator for the output buffer.
    // Observe that here we are just constructing the address generator object, but not using it yet.
    // It will be used in the loop below to determine the address to write the tiles to.
    const auto out0_addr_gen = TensorAccessor(out0_layout_args, out0_base_addr, tile_size_bytes);

    // Loop over all the tiles and write them to the output buffer.
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Make sure there is a tile ready in the circular buffer. This is a blocking call.
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);
        // Write the tile to device memory. This is a non-blocking call.
        noc_async_write_tile(i, out0_addr_gen, cb_out0_addr);

        // Wait until the write is done. This is a blocking call.
        noc_async_write_barrier();
        // Mark the tile in the circular buffer as consumed, freeing up space for the next tile.
        cb_pop_front(cb_out0, 1);
    }
}
