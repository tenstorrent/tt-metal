// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Outbound kernel: writes received tiles back to DRAM for verification.
// This kernel runs on receiver cores (e.g., logical cores 1,0 through 3,0).
// Each receiver writes its complete copy of the tensor to a separate section of the output buffer.
void kernel_main() {
    ////////// RUNTIME ARGS //////////
    uint32_t dst_base_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t receiver_idx = get_arg_val<uint32_t>(2);

    ////////// BUFFER SETUP //////////
    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;  // CBIndex::c_16
    const uint32_t tile_size_bytes = get_tile_size(cb_id_out0);

    // Create address generator for the output buffer using TensorAccessorArgs.
    // TensorAccessorArgs extracts data distribution details from compile-time arguments.
    constexpr auto dst_layout_args = TensorAccessorArgs<0>();
    const auto dst_addr_gen = TensorAccessor(dst_layout_args, dst_base_addr, tile_size_bytes);

    // Calculate the starting tile offset for this receiver.
    // Each receiver writes n_tiles tiles, so receiver 0 writes tiles 0..n_tiles-1,
    // receiver 1 writes tiles n_tiles..2*n_tiles-1, etc.
    uint32_t tile_offset = receiver_idx * n_tiles;

    ////////// MAIN LOOP: WRITE EACH TILE TO DRAM //////////
    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        // Wait for a tile to be available in the output CB
        cb_wait_front(cb_id_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        // Write tile to DRAM at the appropriate offset
        uint32_t dram_tile_id = tile_offset + tile_idx;
        noc_async_write_tile(dram_tile_id, dst_addr_gen, l1_read_addr);
        noc_async_write_barrier();

        // Free the CB slot for next tile
        cb_pop_front(cb_id_out0, 1);
    }
}
