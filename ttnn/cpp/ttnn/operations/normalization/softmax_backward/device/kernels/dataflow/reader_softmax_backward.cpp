// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>

void kernel_main() {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);  // softmax_output
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);  // upstream_grad
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(2);

    // Set up tensor accessors
    constexpr auto softmax_output_args = TensorAccessorArgs<3>();
    constexpr auto upstream_grad_args = TensorAccessorArgs<softmax_output_args.next_compile_time_args_offset()>();

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t softmax_output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t upstream_grad_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_args_idx++);

    // Get tile sizes
    const uint32_t src0_tile_size = get_tile_size(src0_cb_id);
    const uint32_t src1_tile_size = get_tile_size(src1_cb_id);

    // Create tensor accessors
    const auto softmax_output_accessor = TensorAccessor(softmax_output_args, softmax_output_addr, src0_tile_size);
    const auto upstream_grad_accessor = TensorAccessor(upstream_grad_args, upstream_grad_addr, src1_tile_size);

    // Process tiles one by one for simplicity
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint32_t current_tile = start_tile + tile_idx;

        // Read one tile from softmax_output
        cb_reserve_back(src0_cb_id, 1);
        uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_id);
        noc_async_read(softmax_output_accessor.get_noc_addr(current_tile), l1_write_addr_src0, src0_tile_size);
        noc_async_read_barrier();
        cb_push_back(src0_cb_id, 1);

        // Read one tile from upstream_grad
        cb_reserve_back(src1_cb_id, 1);
        uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_id);
        noc_async_read(upstream_grad_accessor.get_noc_addr(current_tile), l1_write_addr_src1, src1_tile_size);
        noc_async_read_barrier();
        cb_push_back(src1_cb_id, 1);
    }
}
