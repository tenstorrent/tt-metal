// SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_common.hpp"
#include "accessor/tensor_accessor.h"
#include "dataflow_api.h"

#include <stdint.h>

void kernel_main() {
    // Compile time args
    // -----------------

    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);

    constexpr uint32_t src_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(3);

    constexpr uint32_t tile_height = get_compile_time_arg_val(4);
    constexpr uint32_t tile_width = get_compile_time_arg_val(5);

    // Input padded size (last two dims) in tiles
    constexpr uint32_t input_height = get_compile_time_arg_val(6);
    constexpr uint32_t input_width = get_compile_time_arg_val(7);

    // Input logical size (last two dims) in data elements
    constexpr uint32_t logical_height = get_compile_time_arg_val(8);
    constexpr uint32_t logical_width = get_compile_time_arg_val(9);

    constexpr bool reduce_all = (bool)get_compile_time_arg_val(10);
    constexpr bool keepdim = (bool)get_compile_time_arg_val(11);

    constexpr uint32_t num_c_time_args = 12;

    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);

    // Tensor Accessors
    // ----------------
    constexpr auto s_src_args = TensorAccessorArgs<num_c_time_args>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();

    auto s_src = TensorAccessor(s_src_args, src_base_addr, src_page_size);
    auto s_dst = TensorAccessor(s_dst_args, dst_base_addr, dst_page_size);

    // CB for storing input data.
    const uint32_t src_cb_addr = get_write_ptr(src_cb_idx);
    constexpr DataFormat src_data_format = get_dataformat(src_cb_idx);

    // C++ type representation of the input data format
    using src_element_type = decltype(get_default_value<src_data_format>());

    auto src_ptr = get_tt_l1_ptr_based_on_data_format<src_data_format>(src_cb_addr);

    // CB in L1 memory for storing output
    const uint32_t dst_cb_addr = get_write_ptr(dst_cb_idx);
    volatile tt_l1_ptr uint32_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb_addr);

    const auto default_val = get_default_value<src_data_format>();

    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_height = 16;

    // This assumes the reduction is along the 'W' dimension
    src_element_type max_values[tile_height] = {default_val};
    uint32_t arg_max[tile_height] = {0};

    // To detect the boundary with padding
    const uint32_t width_rem = logical_width % tile_width;
    const uint32_t height_rem = logical_height % tile_height;

    const bool skip_right_face = width_rem != 0 && (width_rem < face_width);
    const bool skip_bottom_face = height_rem != 0 && (height_rem < face_height);

    // Iterate over the tiles of the input tensor, in (tile) row-major order.
    for (uint32_t i = 0; i < input_height; i++) {
        // Initialize max value, index buffers
        for (uint32_t row = 0; row < tile_height; row++) {
            max_values[row] = default_val;
            arg_max[row] = 0;
        }

        for (uint32_t j = 0; j < input_width; j++) {
            const int tile_id = i * input_width + j;

            // Fetch a tile into cb0
            const uint64_t src_noc_addr = get_noc_addr(tile_id, s_src);
            noc_async_read(src_noc_addr, src_cb_addr, src_page_size);
            noc_async_read_barrier();

            // To check if we are at a boundary with the padding
            int face_w_rem = 0;
            int face_h_rem = 0;
            if ((i == (input_height - 1)) || (j == (input_width - 1))) {
                if (width_rem != 0 || height_rem != 0) {
                    face_w_rem = logical_width % face_width;
                    face_h_rem = logical_height % face_height;
                }
            }

            // Iterate over the faces of the tile
            for (uint32_t k = 0; k < 4; k++) {
                // Full face dimensions - updated when face intersects the boundary with padding
                uint32_t rows_to_process = face_width;
                uint32_t cols_to_process = face_height;

                // Checks to avoid processing the padding
                const bool is_right_face = (k == 1 || k == 3);
                const bool is_bottom_face = (k == 2 || k == 3);
                if (i == (input_height - 1)) {
                    if (is_bottom_face) {
                        if (skip_bottom_face) {
                            continue;
                        }
                        if (height_rem != 0) {
                            rows_to_process = face_h_rem;
                        }
                    } else {
                        // One of the upper faces
                        if (height_rem != 0 && height_rem < face_height) {
                            rows_to_process = height_rem;
                        }
                    }
                }
                if (j == (input_width - 1)) {
                    if (is_right_face) {
                        if (skip_right_face) {
                            continue;
                        }
                        if (width_rem != 0) {
                            cols_to_process = face_w_rem;
                        }
                    } else {
                        if (width_rem != 0 && width_rem < face_width) {
                            cols_to_process = face_w_rem;
                        }
                    }
                }

                // Get an offset to the face
                uint32_t face_offset = k * face_width * face_height;
                volatile tt_l1_ptr decltype(get_default_value<src_data_format>())* face_ptr = src_ptr + face_offset;

                // Go over the rows of the face. Update the maximum values in each row.
                for (uint32_t m = 0; m < rows_to_process; m++) {
                    // Row index in the tile
                    const uint32_t row_index = (k < 2) ? m : m + face_height;

                    src_element_type curr_max = max_values[row_index];
                    uint32_t curr_arg_max = arg_max[row_index];

                    // Go over elements in the current row, current face
                    for (uint32_t n = 0; n < cols_to_process; n++) {
                        // Index within the face
                        uint32_t index = m * face_width + n;

                        src_element_type value = face_ptr[index];

                        bool new_max = false;
                        if constexpr (src_data_format == DataFormat::Float16_b) {
                            new_max = bfloat16_greater(value, curr_max);
                        } else if constexpr (src_data_format == DataFormat::Float32) {
                            new_max = float32_greater(value, curr_max);
                        }

                        if (new_max) {
                            const bool is_left_side_face = (k == 0 || k == 2);
                            const uint32_t new_arg_max = j * tile_width + (is_left_side_face ? 0 : face_width) + n;
                            curr_max = value;
                            curr_arg_max = new_arg_max;
                        }
                    }
                    max_values[row_index] = curr_max;
                    arg_max[row_index] = curr_arg_max;
                }
            }
        }

        // Write-out one tile of argmax data into the output tensor.
        // Only one column (keepdim == true) or one row (keepdim == false) holds the actual data; the reset is padding.
        uint32_t units_to_write = tile_height;
        const uint32_t boundary_tile_units = height_rem;

        // Check if the (output) tile is at the boundary with padding.
        // Recall that reduction only on the last dim is currently supported in this kernel.
        const bool is_boundary_tile = (i == input_height - 1) && (height_rem != 0);
        if (is_boundary_tile) {
            units_to_write = boundary_tile_units;
        }

        for (uint32_t idx = 0; idx < units_to_write; idx++) {
            // Find the output face and offset for the current element.
            uint32_t face_id = 0;
            // row within face
            uint32_t row_f = 0;
            // column within face
            uint32_t col_f = 0;

            if (keepdim) {
                // Write to the left-most column of the tile.
                face_id = (idx < face_height) ? 0 : 2;
                row_f = (face_id == 0) ? idx : idx - 16;
                col_f = 0;
            } else {
                // Write to the top row of the tile.
                face_id = (idx < face_width) ? 0 : 1;
                row_f = 0;
                col_f = (face_id == 0) ? idx : idx - 16;
            }

            // Face is a row-major chunk of memory.
            // Calculate the index of the current element within the face.

            // First, obtain the start offset of the face,
            // within the tile.
            const uint32_t tile_offset = face_id * (face_width * face_height);

            // Find element offset within the (row-major) 16x16 face.
            uint32_t face_offset = 0;
            if (keepdim) {
                // Each element is stored in the first column of
                // the subsequent rows.
                face_offset = row_f * face_width;
            } else {
                // Each element is stored in subsequent columns
                // of the first row.
                face_offset = col_f;
            }

            uint32_t index = tile_offset + face_offset;
            dst_ptr[index] = arg_max[idx];
        }

        // Send contents of cb1 to the output tensor
        uint64_t dst_noc_addr = get_noc_addr(i, s_dst);
        uint32_t write_size = tile_width * tile_height * sizeof(uint32_t);
        noc_async_write(dst_cb_addr, dst_noc_addr, write_size);
        noc_async_write_barrier();
    }
}
