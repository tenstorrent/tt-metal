// SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_tile_layout.hpp"
#include "argmax_common.hpp"
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"

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

    // Size of all dims combined, excluding the last two dims.
    constexpr uint32_t outer_dim_size = get_compile_time_arg_val(10);

    constexpr bool reduce_all = (bool)get_compile_time_arg_val(11);
    constexpr bool keepdim = (bool)get_compile_time_arg_val(12);

    constexpr uint32_t num_c_time_args = 13;

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

    using dst_accessor_type = decltype(s_dst);

    // CB for input data.
    const uint32_t src_cb_addr = get_write_ptr(src_cb_idx);
    constexpr DataFormat src_data_format = get_dataformat(src_cb_idx);

    // CB for output data.
    const uint32_t dst_cb_addr = get_write_ptr(dst_cb_idx);

    auto default_val = get_default_value<src_data_format>();
    // C++ type representation of the src/dst data formats
    using src_element_type = decltype(default_val);

    // This assumes the reduction is along the 'W' dimension
    src_element_type max_values[tile_height] = {default_val};
    uint32_t arg_max[tile_height] = {0};

    // Only ROW_MAJOR output layout is supported.
    //
    // When keep_dim==true
    // Each output row contains 1 element. We will accumulate tile_height output rows,
    // and then write them out in one pass.
    //
    // When keep_dim=false
    // Each output row contains logical_height elements. We will accumulate values
    // for a single output row, then write all of them out in a single noc message.

    // Number of data elements in one output page (ROW_MAJOR layout)
    constexpr uint32_t output_page_elements = keepdim ? 1 : logical_height;

    // Array for accumulating final argmax values. Used only when keepdim==true.
    uint32_t accumulated_arg_max[tile_height] = {0};

    constexpr uint32_t tile_height_rem = logical_height % tile_height;
    constexpr uint32_t tile_width_rem = logical_width % tile_width;
    constexpr uint32_t face_height_rem = logical_height % face_height;
    constexpr uint32_t face_width_rem = logical_width % face_width;

    const InputContext input_ctx(
        tile_height,
        tile_width,
        input_height,
        input_width,
        logical_height,
        logical_width,
        tile_height_rem,
        tile_width_rem,
        face_height_rem,
        face_width_rem,
        src_data_format,
        src_cb_addr);

    OutputContext output_ctx((uint32_t*)accumulated_arg_max, tile_height, dst_cb_addr, output_page_elements, keepdim);

    // Iterate over the initial dimensions combined together
    for (uint32_t outer_index = 0; outer_index < outer_dim_size; outer_index++) {
        // For a given outer index,
        // iterate over the tiles of the input tensor, in (tile) row-major order.
        // Each horizontal pass over the input generates tile_height output values,
        // or 'no more than tile_height' when padding is considered.
        for (uint32_t i = 0; i < input_height; i++) {
            // Initialize max values and index buffers
            for (uint32_t row = 0; row < tile_height; row++) {
                max_values[row] = default_val;
                arg_max[row] = 0;
            }

            // Number of output units to be generated in this iteration
            const uint32_t units_generated =
                (tile_height_rem == 0 || i < input_height - 1) ? tile_height : tile_height_rem;

            for (uint32_t j = 0; j < input_width; j++) {
                // Number of input tiles in the last two dimensions.
                constexpr uint32_t inner_size = input_height * input_width;
                const int src_tile_id = outer_index * inner_size + i * input_width + j;

                // Fetch the next tile
                const uint64_t src_noc_addr = get_noc_addr(src_tile_id, s_src);
                noc_async_read(src_noc_addr, src_cb_addr, src_page_size);
                noc_async_read_barrier();

                uint32_t tile_rows_processed = 0;
                process_input_tile<src_element_type, src_data_format>(
                    input_ctx, j, i, max_values, arg_max, tile_height, tile_rows_processed);
                ASSERT(tile_rows_processed == units_generated);
            }

            // The rate at which argmax values are generated in the loop above
            // might be different than the rate of writing them out to the output tensor.
            // Buffer the data into an intermediate storage.
            collect_row_major_output<keepdim>(arg_max, units_generated, output_ctx);

            if (output_ctx.collected_count >= output_page_elements) {
                write_to_output<dst_accessor_type, keepdim>(s_dst, output_ctx);
            }
        }
    }
}

void get_face_data_range(
    uint32_t& data_rows,
    uint32_t& data_cols,
    uint32_t tile_x,
    uint32_t tile_y,
    uint32_t face_id,
    const InputContext& ctx) {
    const bool is_bottom_tile = tile_y == (ctx.input_height - 1);
    const bool is_right_most_tile = tile_x == (ctx.input_width - 1);

    // Initialize the range as full face
    data_rows = face_height;
    data_cols = face_width;

    if (!ctx.has_padding) {
        return;
    }

    if (!is_bottom_tile && !is_right_most_tile) {
        // Only marginal tiles may contain the padding
        return;
    }

    const bool is_right_face = (face_id == 1 || face_id == 3);
    const bool is_bottom_face = (face_id == 2 || face_id == 3);

    const uint32_t height_rem = ctx.tile_h_rem;
    if (is_bottom_tile && height_rem != 0) {
        if (is_bottom_face) {
            const bool skip_bottom_face = height_rem < face_height;
            if (skip_bottom_face) {
                data_rows = 0;
                data_cols = 0;
                return;
            }
            data_rows = ctx.face_h_rem;
        } else {
            // One of the upper faces
            if (height_rem < face_height) {
                data_rows = height_rem;
            }
        }
    }

    const uint32_t width_rem = ctx.tile_w_rem;
    if (is_right_most_tile && width_rem != 0) {
        if (is_right_face) {
            const bool skip_right_face = width_rem < face_width;
            if (skip_right_face) {
                data_rows = 0;
                data_cols = 0;
                return;
            }
            data_cols = ctx.face_w_rem;
        } else {
            if (width_rem < face_width) {
                data_cols = width_rem;
            }
        }
    }
}
