// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_mm_scaler.hpp"

void kernel_main() {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);  // softmax_output
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);  // upstream_grad
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(2);  // ones vector for matmul reduction
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(3);

    // Set up tensor accessors
    constexpr auto softmax_output_args = TensorAccessorArgs<4>();
    constexpr auto upstream_grad_args = TensorAccessorArgs<softmax_output_args.next_compile_time_args_offset()>();

    // Common runtime args (shared across all cores)
    const uint32_t softmax_output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t upstream_grad_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t total_num_rows = get_common_arg_val<uint32_t>(2);
    const uint32_t num_cores_x = get_common_arg_val<uint32_t>(3);
    const uint32_t num_cores_y = get_common_arg_val<uint32_t>(4);

    // Calculate work assignment for this core based on coordinates
    const uint32_t core_id_x = get_absolute_logical_x();
    const uint32_t core_id_y = get_absolute_logical_y();
    // Match factory's column-major indexing: core_idx = x * num_cores_y + y
    const uint32_t core_id = core_id_x * num_cores_y + core_id_y;

    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t rows_per_core = (total_num_rows + num_cores - 1) / num_cores;

    const uint32_t start_tile = core_id * rows_per_core;
    const uint32_t end_tile =
        ((start_tile + rows_per_core) < total_num_rows) ? (start_tile + rows_per_core) : total_num_rows;
    const uint32_t num_tiles = (start_tile < total_num_rows) ? (end_tile - start_tile) : 0;

    // Get tile sizes
    const uint32_t src0_tile_size = get_tile_size(src0_cb_id);
    const uint32_t src1_tile_size = get_tile_size(src1_cb_id);

    // Create tensor accessors
    const auto softmax_output_accessor = TensorAccessor(softmax_output_args, softmax_output_addr, src0_tile_size);
    const auto upstream_grad_accessor = TensorAccessor(upstream_grad_args, upstream_grad_addr, src1_tile_size);

    // Early exit if this core has no work
    if (num_tiles == 0) {
        return;
    }

    // Generate a BF16 column vector of ones for matmul-based reduction
    constexpr uint32_t identity_scalar_packed = 0x3f803f80;  // BF16(1.0) packed twice into u32
    generate_mm_scaler(ones_cb_id, identity_scalar_packed);

    // Process rows - batch read all tiles per row for better performance
    for (uint32_t row_idx = 0; row_idx < num_tiles; ++row_idx) {
        const uint32_t row_start_tile = (start_tile + row_idx) * num_tiles_per_row;

        // Batch read all tiles for this row from softmax_output and upstream_grad
        cb_reserve_back(src0_cb_id, num_tiles_per_row);
        cb_reserve_back(src1_cb_id, num_tiles_per_row);
        const uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_id);
        const uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_id);

        for (uint32_t w = 0; w < num_tiles_per_row; ++w) {
            const uint32_t curr_tile = row_start_tile + w;
            noc_async_read_page(curr_tile, softmax_output_accessor, l1_write_addr_src0 + w * src0_tile_size);
            noc_async_read_page(curr_tile, upstream_grad_accessor, l1_write_addr_src1 + w * src1_tile_size);
        }

        // Single barrier for all reads in this batch
        noc_async_read_barrier();

        // Push all tiles at once
        cb_push_back(src0_cb_id, num_tiles_per_row);
        cb_push_back(src1_cb_id, num_tiles_per_row);
    }
}
