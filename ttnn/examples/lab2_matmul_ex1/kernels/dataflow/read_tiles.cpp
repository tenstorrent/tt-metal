// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Read parameters from the kernel's runtime arguments.
    int arg_idx = 0;
    uint32_t src0_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t src1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Kt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Nt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_offset = get_arg_val<uint32_t>(arg_idx++);

    // The circular buffers to read the tiles into
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (which is most often the case).
    constexpr uint32_t tile_size_bytes_0 = get_tile_size(cb_in0);
    constexpr uint32_t tile_size_bytes_1 = get_tile_size(cb_in1);

    // Create address generators for the input buffers. Address generators can determine
    // physical address based on the provided data layout and base address.
    // Start by extracting the tensor layout parameters from the compile-time arguments.
    // Recall that compile-time arguments are stored as a vector of uint32_t values.
    // TensorAccessorArgs is a clean way to extract the appropriate number of
    // these uint32_t values and store them in an object.
    constexpr auto src0_layout_args = TensorAccessorArgs<0>();
    // Then, construct the address generator for the input buffer.
    // Observe that here we are just constructing the address generator object, but not using it yet.
    // It will be used in the loop below to determine the address to read the tiles from.
    const auto src0_addr_gen = TensorAccessor(src0_layout_args, src0_base_addr, tile_size_bytes_0);
    // Repeat for the second input buffer. next_compile_time_args_offset() is a clean way to
    // determine the index of the next compile-time argument (after TensorAccessorArgs read
    // the number of arguments it required above) without having to hard-code the index.
    constexpr auto src1_layout_args = TensorAccessorArgs<src0_layout_args.next_compile_time_args_offset()>();
    // Finally, construct the address generator for the second input buffer.
    const auto src1_addr_gen = TensorAccessor(src1_layout_args, src1_base_addr, tile_size_bytes_1);

    // Loop over output tiles 
    for (uint32_t output_tile = 0; output_tile < num_output_tiles; output_tile++) {
        // Convert linear output tile ID to 2D coordinates.
        uint32_t row = output_tile / Nt;
        uint32_t col = output_tile % Nt;
        
        // Read all K tiles for this output position
        for (uint32_t kt = 0; kt < Kt; kt++) {
            // First make sure there is space for one tile in each of the circular buffers to be written to.
            // Deciding how large the buffers should be is a tradeoff.
            // These are blocking calls.
            cb_reserve_back(cb_in0, 1);
            cb_reserve_back(cb_in1, 1);

            uint32_t cb_in0_addr = get_write_ptr(cb_in0);
            uint32_t cb_in1_addr = get_write_ptr(cb_in1);
            // Read the tiles from DRAM into the circular buffers.

            // Read A's tile at (mt, kt). A has MtxKt tiles, so we stride by Kt
            uint32_t a_tile_index = row * Kt + kt;
            // Read B's tile at (kt, nt). B has KtxNt tiles, so we stride by Nt
            uint32_t b_tile_index = kt * Nt + col;
            // Recall that src0_addr_gen and src1_addr_gen are address generators for the input buffers.
            // They are used to determine the address to read the tiles from. i is the index of the tile to read.
            // cb_in0_addr and cb_in1_addr are the circular buffer addresses to write the tiles to.
            // These are non-blocking calls, so they both proceed in parallel.
            noc_async_read_tile(a_tile_index, src0_addr_gen, cb_in0_addr);
            noc_async_read_tile(b_tile_index, src1_addr_gen, cb_in1_addr);

            // Wait until both reads are done before signaling the circular buffers that the tiles are ready.
            noc_async_read_barrier();
            // Mark the tiles in circular buffers as ready.
            // After this, any kernel (e.g. compute kernel) calling `cb_wait_front` will see this tile.
            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }  // Kt loop
    }  // output tile loop
}
