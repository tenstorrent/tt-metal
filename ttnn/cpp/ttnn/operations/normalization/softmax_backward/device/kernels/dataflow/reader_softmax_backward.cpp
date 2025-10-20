// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>
#include <type_traits>
#include <algorithm>

// Function to generate broadcast scaler tile for reduction operations
template <typename T = uint16_t>
inline void generate_bcast_scaler(uint32_t cb_scaler, uint32_t scaler) {
    cb_reserve_back(cb_scaler, 1);
    auto ptr = reinterpret_cast<T*>(get_write_ptr(cb_scaler));

    std::fill_n(ptr, 1024, T(0));

    // Fill first 16 elements of each of 4 faces with the scaler value
    const T scaler_value = std::is_same_v<T, uint16_t> ? T(scaler >> 16) : T(scaler);
    for (int k = 0; k < 4; k++) {
        std::fill_n(ptr + k * 256, 16, scaler_value);
    }

    cb_push_back(cb_scaler, 1);
}

void kernel_main() {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);  // softmax_output
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);  // upstream_grad
    constexpr uint32_t scaler_cb_id = get_compile_time_arg_val(2);  // scaler for reduction
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(3);

    // Set up tensor accessors
    constexpr auto softmax_output_args = TensorAccessorArgs<4>();
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

    // Generate scaler value of 1.0 for SUM reduction
    constexpr uint32_t scaler = 0x3f800000;  // == 1.0f
    generate_bcast_scaler(scaler_cb_id, scaler);

    // Process rows - batch read all tiles per row for better performance
    for (uint32_t row_idx = 0; row_idx < num_tiles; ++row_idx) {
        const uint32_t row_start_tile = (start_tile + row_idx) * num_tiles_per_row;

        // Batch read all tiles for this row from softmax_output
        cb_reserve_back(src0_cb_id, num_tiles_per_row);
        const uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_id);

        for (uint32_t w = 0; w < num_tiles_per_row; ++w) {
            const uint32_t curr_tile = row_start_tile + w;
            noc_async_read(
                softmax_output_accessor.get_noc_addr(curr_tile),
                l1_write_addr_src0 + w * src0_tile_size,
                src0_tile_size);
        }

        // Batch read all tiles for this row from upstream_grad
        cb_reserve_back(src1_cb_id, num_tiles_per_row);
        const uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_id);

        for (uint32_t w = 0; w < num_tiles_per_row; ++w) {
            const uint32_t curr_tile = row_start_tile + w;
            noc_async_read(
                upstream_grad_accessor.get_noc_addr(curr_tile),
                l1_write_addr_src1 + w * src1_tile_size,
                src1_tile_size);
        }

        // Single barrier for all reads in this batch
        noc_async_read_barrier();

        // Push all tiles at once
        cb_push_back(src0_cb_id, num_tiles_per_row);
        cb_push_back(src1_cb_id, num_tiles_per_row);
    }
}
