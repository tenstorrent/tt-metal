// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
/**
 * Generate index tiles for TopK multicore local processing phase.
 *
 * This function creates index tiles that track the original positions of elements
 * within each width tile. The generated indices enable proper tracking of element
 * origins after sorting operations.
 *
 * Index Generation Pattern:
 * - Each tile contains indices for 32x32 = 1024 elements arranged in tile format
 * - For tile at position 'wt' along the width: elements have indices [wt*32, wt*32+31]
 * - Layout: First 32 elements get indices {wt*32, wt*32+1, ..., wt*32+31}
 *           Second 32 elements get indices {wt*32+32, wt*32+33, ..., wt*32+63}, etc.
 *
 * @param cb_id Circular buffer index to write the generated index tile
 * @param wt    Width tile position [0, Wt_local) identifying which tile position along width
 */
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // TODO: investigate moving to compile time (binary size is at risk)
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    uint16_t wt_offset = wt << 5;

    uint32_t count = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t k = 0; k < 16; ++k) {
                for (uint32_t l = 0; l < 16; l += 2) {
                    uint16_t value = l + 16 * j + wt_offset;
                    ptr[count] = (value + 1) << 16 | value;
                    count++;
                }
            }
        }
    }
    cb_push_back(cb_id, 1);
}

/**
 * TopK Multicore Reader Kernel - Local Core Input Data and Index Generation
 *
 * This kernel runs on each local processing core and is responsible for:
 * 1. Reading the assigned width chunk of input tensor data from DRAM
 * 2. Generating corresponding index tiles to track element positions
 * 3. Streaming data to the local compute kernel for bitonic sorting
 *
 * Memory Organization:
 * - Double-buffered input to support continuous data flow to compute kernel
 * - Each local core processes Wt_local consecutive width tiles
 * - Index generation happens on-demand to minimize DRAM access
 */
void kernel_main() {
    // Runtime arguments - core-specific work assignment
    uint32_t src_addr = get_arg_val<uint32_t>(0);  // DRAM address of input tensor
    uint32_t start_ht = get_arg_val<uint32_t>(1);  // Starting height tile index
    uint32_t start_wt = get_arg_val<uint32_t>(2);  // Starting width tile index for this core

    // Compile-time configuration
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);  // Input values circular buffer
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);  // Generated indices circular buffer
    // Note: Skipping indices tensor arg get_compile_time_arg_val(3) as we generate indices locally

    constexpr uint32_t Ht = get_compile_time_arg_val(2);        // Total height tiles in tensor
    constexpr uint32_t Wt_local = get_compile_time_arg_val(3);  // Width tiles assigned to this core
    constexpr uint32_t Wt = get_compile_time_arg_val(4);        // Total width tiles in tensor

    // DRAM tensor accessor configuration
    constexpr auto s_args = TensorAccessorArgs<5>();
    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

    // MAIN DATA STREAMING LOOP
    // Process each height row sequentially, streaming this core's assigned width chunk
    // to the compute kernel. The double-buffered circular buffers allow compute operations
    // to proceed while the next tile is being fetched from DRAM.
    //
    // Memory Access Pattern:
    // - Linear access within each height row for optimal DRAM bandwidth
    // - Each core accesses a contiguous range [start_wt, start_wt + Wt_local)
    // - Index generation eliminates need for separate DRAM reads
    for (uint32_t i = start_ht; i < Ht; ++i) {                       // For each height row
        for (uint32_t j = start_wt; j < start_wt + Wt_local; ++j) {  // For each width tile in chunk
            // Stream input value tile from DRAM to local circular buffer
            cb_reserve_back(cb_id_in0, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile(i * Wt + j, s, l1_write_addr);  // Read tile at (i,j) position
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);

            // Generate corresponding index tile for position tracking during sort
            generate_index_tile(cb_id_in1, j);  // Generate indices for width position j
        }
    }
}
