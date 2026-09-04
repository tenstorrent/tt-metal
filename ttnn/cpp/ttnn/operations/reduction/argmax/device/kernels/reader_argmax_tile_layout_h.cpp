// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_tile_h_col.hpp"
#include "argmax_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include <stdint.h>

/**
 * Argmax over the H (height) dimension for TILE layout, without transposing the input.
 * For each (outer, global_w) position, scan all H tiles and rows; index is 0..logical_height-1.
 *
 * Loop order: for fixed (outer, w_tile), load each (h_tile, w_tile) tile once; one pass over the
 * tile in L1 updates all in-tile columns (avoids repeated NOC reads and repeated full-tile scans).
 */

void kernel_main() {
    constexpr auto src_page_size = get_arg(args::src_page_size);

    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto tile_width = get_arg(args::tile_width);

    constexpr auto input_height = get_arg(args::input_height);
    constexpr auto input_width = get_arg(args::input_width);

    constexpr auto logical_height = get_arg(args::logical_height);
    constexpr auto logical_width = get_arg(args::logical_width);

    constexpr auto outer_dim_size = get_arg(args::outer_dim_size);

    // This reader takes no reduce_all/keepdim arguments; the width reader adds those.

    auto s_src = TensorAccessor(tensor::src);
    auto s_dst = TensorAccessor(tensor::dst);
    using dst_accessor_type = decltype(s_dst);

    DataflowBuffer src_dfb(dfb::src);
    const uint32_t src_dfb_addr = src_dfb.get_write_ptr();
    constexpr DataFormat src_data_format = get_dataformat(dfb::src);
    DataflowBuffer dst_dfb(dfb::dst);
    const uint32_t dst_dfb_addr = dst_dfb.get_write_ptr();

    auto default_val = get_default_value<src_data_format>();
    using src_element_type = decltype(default_val);

    // Required by OutputContext; unused with collect_row_major_output<false> (values go to output DFB).
    uint32_t stack_unused[1] = {0};

    // Batching must match the output buffer page size. Do not use keepdim ? 1 : width (one uint32
    // per NOC) or page_ids misalign with the interleaved row-major distribution spec.
    constexpr uint32_t output_page_elements = logical_width;

    const uint32_t tile_height_rem = logical_height % tile_height;
    const uint32_t tile_width_rem = logical_width % tile_width;
    const uint32_t face_height_rem = logical_height % face_height;
    const uint32_t face_width_rem = logical_width % face_width;

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

    OutputContext output_ctx((uint32_t*)stack_unused, 1, dst_dfb_addr, output_page_elements);

    Noc noc;

    constexpr uint32_t inner_size = input_height * input_width;

    for (uint32_t outer_index = 0; outer_index < outer_dim_size; outer_index++) {
        for (uint32_t w_tile = 0; w_tile < input_width; w_tile++) {
            src_element_type max_vals[tile_width];
            uint32_t arg_maxs[tile_width];
            for (uint32_t lw = 0; lw < tile_width; lw++) {
                max_vals[lw] = default_val;
                arg_maxs[lw] = 0;
            }

            for (uint32_t h_tile = 0; h_tile < input_height; h_tile++) {
                const uint32_t src_tile_id = outer_index * inner_size + h_tile * input_width + w_tile;

                noc.async_read(s_src, src_dfb, src_page_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
                noc.async_read_barrier();

                process_loaded_tile_all_h_columns<src_element_type, src_data_format>(
                    input_ctx, w_tile, h_tile, max_vals, arg_maxs);
            }

            for (uint32_t local_w = 0; local_w < tile_width; local_w++) {
                const uint32_t global_w = w_tile * tile_width + local_w;
                if (global_w >= logical_width) {
                    continue;
                }

                collect_row_major_output<false>(&arg_maxs[local_w], 1, output_ctx);

                if (output_ctx.collected_count >= output_page_elements) {
                    write_to_output<dst_accessor_type, false>(noc, s_dst, output_ctx);
                }
            }
        }
    }
}
