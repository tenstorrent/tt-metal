// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_tile_layout.hpp"
#include "argmax_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <stdint.h>

void kernel_main() {
    // Compile time args
    // -----------------
    constexpr auto src_page_size = get_arg(args::src_page_size);

    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto tile_width = get_arg(args::tile_width);

    // Input padded size (last two dims) in tiles
    constexpr auto input_height = get_arg(args::input_height);
    constexpr auto input_width = get_arg(args::input_width);

    // Input logical size (last two dims) in data elements
    constexpr auto logical_height = get_arg(args::logical_height);
    constexpr auto logical_width = get_arg(args::logical_width);

    // Size of all dims combined, excluding the last two dims.
    constexpr auto outer_dim_size = get_arg(args::outer_dim_size);

    constexpr bool reduce_all = (bool)get_arg(args::reduce_all);
    constexpr bool keepdim = (bool)get_arg(args::keepdim);

    // Tensor Accessors
    // ----------------
    auto s_src = TensorAccessor(tensor::src);
    auto s_dst = TensorAccessor(tensor::dst);

    using dst_accessor_type = decltype(s_dst);

    Noc noc;
    DataflowBuffer src_dfb(dfb::src);
    DataflowBuffer dst_dfb(dfb::dst);

    // DFB for input data.
    const uint32_t src_dfb_addr = src_dfb.get_write_ptr();
    constexpr DataFormat src_data_format = get_dataformat(dfb::src);

    // DFB for output data.
    const uint32_t dst_dfb_addr = dst_dfb.get_write_ptr();

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
        src_dfb_addr);

    OutputContext output_ctx((uint32_t*)accumulated_arg_max, tile_height, dst_dfb_addr, output_page_elements);

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
                const uint32_t src_tile_id = outer_index * inner_size + i * input_width + j;

                // Fetch the next tile
                noc.async_read(s_src, src_dfb, src_page_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
                noc.async_read_barrier();

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
                write_to_output<dst_accessor_type, keepdim>(noc, s_dst, output_ctx);
            }
        }
    }
}
