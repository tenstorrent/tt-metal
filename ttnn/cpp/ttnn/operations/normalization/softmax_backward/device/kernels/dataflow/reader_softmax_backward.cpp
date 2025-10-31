// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>
#include <type_traits>
#include <algorithm>

// Function to generate a column vector of ones for matmul-based reduction
// Creates a tile where the first column contains 1.0 and all other elements are 0.0
// When used in matmul: [32x32] @ [32x1] -> [32x1] where output is row-wise sum
template <typename T = uint16_t>
inline void generate_ones_vector(uint32_t cb_ones) {
    cb_reserve_back(cb_ones, 1);
    auto ptr = reinterpret_cast<T*>(get_write_ptr(cb_ones));

    // Initialize entire tile with zeros
    std::fill_n(ptr, 1024, T(0));

    // Set one value based on data type
    constexpr T one = std::is_same_v<T, uint8_t> ? T(0x38) : (std::is_same_v<T, uint16_t> ? T(0x3f80) : T(0x3f800000));

    // Tile layout: 4 faces of 16x16, stored row-major within each face
    // Face 0: top-left (rows 0-15, cols 0-15)
    // Face 1: top-right (rows 0-15, cols 16-31)
    // Face 2: bottom-left (rows 16-31, cols 0-15)
    // Face 3: bottom-right (rows 16-31, cols 16-31)

    // For a column vector of ones at column 0:
    // - Face 0 (offset 0): set first element of each row (every 16th element)
    // - Face 2 (offset 512): set first element of each row (every 16th element)

    for (int el = 0; el < 16 * 16; el += 16) {
        ptr[el] = one;        // First element of each row in face 0
        ptr[512 + el] = one;  // First element of each row in face 2
    }

    cb_push_back(cb_ones, /*ntiles*/ 1);
}

void kernel_main() {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);  // softmax_output
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);  // upstream_grad
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(2);  // ones vector for matmul reduction
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(3);

    // Adjustable batch size - push this many tiles at once to L1 cache
    // Can be tuned based on L1 cache size and tile size
    constexpr uint32_t tiles_per_batch = 4;

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

    // Generate a tile filled with 1.0 values for matmul-based reduction
    generate_ones_vector(ones_cb_id);

    // Two-pass streaming algorithm to minimize L1 memory
    // Read the same data twice: once for sum computation, once for final output
    for (uint32_t row_idx = 0; row_idx < num_tiles; ++row_idx) {
        const uint32_t row_start_tile = (start_tile + row_idx) * num_tiles_per_row;

        // Read entire row twice (2 passes)
        for (uint32_t pass = 0; pass < 2; ++pass) {
            // Read row in batches
            for (uint32_t batch_start = 0; batch_start < num_tiles_per_row; batch_start += tiles_per_batch) {
                const uint32_t current_batch_size = (batch_start + tiles_per_batch <= num_tiles_per_row)
                                                        ? tiles_per_batch
                                                        : (num_tiles_per_row - batch_start);

                cb_reserve_back(src0_cb_id, current_batch_size);
                const uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_id);

                cb_reserve_back(src1_cb_id, current_batch_size);
                const uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_id);

                for (uint32_t i = 0; i < current_batch_size; ++i) {
                    const uint32_t curr_tile = row_start_tile + batch_start + i;
                    noc_async_read_page(curr_tile, softmax_output_accessor, l1_write_addr_src0 + i * src0_tile_size);
                    noc_async_read_page(curr_tile, upstream_grad_accessor, l1_write_addr_src1 + i * src1_tile_size);
                }

                noc_async_read_barrier();
                cb_push_back(src0_cb_id, current_batch_size);
                cb_push_back(src1_cb_id, current_batch_size);
            }
        }
    }
}
