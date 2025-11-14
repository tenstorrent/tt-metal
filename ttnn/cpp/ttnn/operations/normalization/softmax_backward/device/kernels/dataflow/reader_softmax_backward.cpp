// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>
#include <type_traits>
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_mm_scaler.hpp"

// Generate a column vector of ones for matmul-based reduction using BF16 scaler tile
// Uses optimized helper to place 1.0 at the first column of faces 0 and 2
// Packed BF16 1.0 is 0x3f80; double-packed into u32 as 0x3f803f80

void kernel_main() {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);  // softmax_output
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);  // upstream_grad
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(2);  // ones vector for matmul reduction
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(3);

    // Adjustable block size - push this many tiles at once to L1 cache
    // Can be tuned based on L1 cache size and tile size
    constexpr uint32_t tiles_per_block = 4;

    // Set up tensor accessors
    constexpr auto softmax_output_args = TensorAccessorArgs<4>();
    constexpr auto upstream_grad_args = TensorAccessorArgs<softmax_output_args.next_compile_time_args_offset()>();

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_args_idx++);

    // Common runtime args (shared across all cores)
    const uint32_t softmax_output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t upstream_grad_addr = get_common_arg_val<uint32_t>(1);

    // Get tile sizes
    const uint32_t src0_tile_size = get_tile_size(src0_cb_id);
    const uint32_t src1_tile_size = get_tile_size(src1_cb_id);

    // Create tensor accessors
    const auto softmax_output_accessor = TensorAccessor(softmax_output_args, softmax_output_addr, src0_tile_size);
    const auto upstream_grad_accessor = TensorAccessor(upstream_grad_args, upstream_grad_addr, src1_tile_size);

    // Generate a BF16 column vector of ones for matmul-based reduction
    constexpr uint32_t identity_scalar_packed = 0x3f803f80;  // BF16(1.0) packed twice into u32
    generate_mm_scaler(ones_cb_id, identity_scalar_packed);

    // Two-pass streaming algorithm to minimize L1 memory
    // Read the same data twice: once for sum computation, once for final output
    for (uint32_t row_idx = 0; row_idx < num_tiles; ++row_idx) {
        const uint32_t row_start_tile = (start_tile + row_idx) * num_tiles_per_row;

        // Read entire row twice (2 passes)
        for (uint32_t pass = 0; pass < 2; ++pass) {
            // Read row in blockes
            for (uint32_t block_start = 0; block_start < num_tiles_per_row; block_start += tiles_per_block) {
                const uint32_t current_block_size = (block_start + tiles_per_block <= num_tiles_per_row)
                                                        ? tiles_per_block
                                                        : (num_tiles_per_row - block_start);

                cb_reserve_back(src0_cb_id, current_block_size);
                const uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_id);

                cb_reserve_back(src1_cb_id, current_block_size);
                const uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_id);

                for (uint32_t i = 0; i < current_block_size; ++i) {
                    const uint32_t curr_tile = row_start_tile + block_start + i;
                    noc_async_read_page(curr_tile, softmax_output_accessor, l1_write_addr_src0 + i * src0_tile_size);
                    noc_async_read_page(curr_tile, upstream_grad_accessor, l1_write_addr_src1 + i * src1_tile_size);
                }

                noc_async_read_barrier();
                cb_push_back(src0_cb_id, current_block_size);
                cb_push_back(src1_cb_id, current_block_size);
            }
        }
    }
}
