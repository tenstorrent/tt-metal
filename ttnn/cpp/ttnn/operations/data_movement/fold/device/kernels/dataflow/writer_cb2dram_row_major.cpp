// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t batch_size = get_compile_time_arg_val(0);      // Number of batches
    constexpr uint32_t input_height = get_compile_time_arg_val(1);    // Height of input tensor
    constexpr uint32_t input_width = get_compile_time_arg_val(2);     // Width of input tensor
    constexpr uint32_t stride_height = get_compile_time_arg_val(3);   // Vertical stride
    constexpr uint32_t stride_width = get_compile_time_arg_val(4);    // Horizontal stride
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(5);    // Size of each stick in bytes
    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(6);  // Number of tiles per row
    constexpr uint32_t element_size = get_compile_time_arg_val(7);    // Size of each element in bytes
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(8);       // Input compute buffer ID

    // Runtime arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);          // Destination address in DRAM
    uint32_t start_block_id = get_arg_val<uint32_t>(1);    // Starting block ID for processing
    uint32_t num_blocks = get_arg_val<uint32_t>(2);        // Number of blocks to process
    uint32_t nblocks_per_core = get_arg_val<uint32_t>(3);  // Number of blocks per core

    // Constants for tile dimensions
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;

    // Initialize DRAM address generator
    const InterleavedAddrGen<true> d = {.bank_base_address = dst_addr, .page_size = stick_nbytes};

    // Calculate output dimensions and patch size
    uint32_t OH = input_height / stride_height;                         // Output height
    uint32_t OW = input_width / stride_width;                           // Output width
    uint32_t patch_size = stride_height * stride_width;                 // Size of each patch
    const uint32_t W_PAD = TILE_HEIGHT;                                 // Width padding
    const uint32_t C_PAD = TILE_WIDTH * element_size * ntiles_per_row;  // Channel padding

    // Calculate tile dimensions
    uint32_t tile_cols = (input_width + W_PAD - 1) / W_PAD;  // Number of tile columns
    uint32_t tiles_per_batch = input_height * tile_cols;     // Tiles per batch

    uint32_t end_block_id = start_block_id + num_blocks;
    for (uint32_t i = start_block_id; i < end_block_id; ++i) {
        // Wait for input data to be available
        cb_wait_front(cb_id_in1, ntiles_per_row);
        uint64_t l1_read_addr = get_read_ptr(cb_id_in1);

        // Calculate batch and position indices
        int batch_idx = i / tiles_per_batch;    // Batch index
        int bh_index = i % tiles_per_batch;     // Index within batch
        int height_idx = bh_index / tile_cols;  // Height index
        int tile_col = bh_index % tile_cols;    // Tile column index

        // Process each element in the tile
        int w_start = tile_col * W_PAD;
        int w_end = (w_start + W_PAD > input_width) ? input_width : w_start + W_PAD;
        for (int w_local = 0; w_start + w_local < w_end; ++w_local) {
            int width_idx = w_start + w_local;
            uint64_t src = l1_read_addr + w_local * C_PAD;

            // Calculate output indices based on stride
            int out_height_idx = height_idx / stride_height;
            int out_width_idx = width_idx / stride_width;
            int kernel_height_idx = height_idx % stride_height;
            int kernel_width_idx = width_idx % stride_width;

            // Calculate destination address
            int dst_row = batch_idx * OH * OW + out_height_idx * OW + out_width_idx;
            int dst_col = (kernel_height_idx * stride_width + kernel_width_idx);
            uint64_t dst = dst_row * patch_size + dst_col;
            uint64_t dst_addr_ = get_noc_addr(dst, d);

            // Write data to DRAM
            noc_async_write(src, dst_addr_, stick_nbytes);
        }
        // Ensure all writes are completed
        noc_async_write_barrier();

        // Pop processed data buffer
        cb_pop_front(cb_id_in1, ntiles_per_row);
    }
}
