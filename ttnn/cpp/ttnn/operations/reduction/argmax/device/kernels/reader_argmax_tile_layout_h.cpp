// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_tile_h_col.hpp"
#include "argmax_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

#include <stdint.h>

/**
 * Argmax over the H (height) dimension for TILE layout, without transposing the input.
 * For each (outer, global_w) position, scan all H tiles and rows; index is 0..logical_height-1.
 *
 * Loop order: for fixed (outer, w_tile), load each (h_tile, w_tile) tile once; one pass over the
 * tile in L1 updates all in-tile columns (avoids repeated NOC reads and repeated full-tile scans).
 */

void kernel_main() {
    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);

    constexpr uint32_t src_page_size = get_compile_time_arg_val(2);

    constexpr uint32_t tile_height = get_compile_time_arg_val(4);
    constexpr uint32_t tile_width = get_compile_time_arg_val(5);

    constexpr uint32_t input_height = get_compile_time_arg_val(6);
    constexpr uint32_t input_width = get_compile_time_arg_val(7);

    constexpr uint32_t logical_height = get_compile_time_arg_val(8);
    constexpr uint32_t logical_width = get_compile_time_arg_val(9);

    constexpr uint32_t outer_dim_size = get_compile_time_arg_val(10);

    // TensorAccessor follows the kernel-specific compile-time args (no reduce_all/keepdim; width reader adds those).
    constexpr uint32_t num_c_time_args = 11;

    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);

    constexpr auto s_src_args = TensorAccessorArgs<num_c_time_args>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();

    auto s_src = TensorAccessor(s_src_args, src_base_addr);
    auto s_dst = TensorAccessor(s_dst_args, dst_base_addr);
    using dst_accessor_type = decltype(s_dst);

    const uint32_t src_cb_addr = get_write_ptr(src_cb_idx);
    constexpr DataFormat src_data_format = get_dataformat(src_cb_idx);
    const uint32_t dst_cb_addr = get_write_ptr(dst_cb_idx);

    auto default_val = get_default_value<src_data_format>();
    using src_element_type = decltype(default_val);

    // Required by OutputContext; unused with collect_row_major_output<false> (values go to output CB).
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
        src_cb_addr);

    OutputContext output_ctx((uint32_t*)stack_unused, 1, dst_cb_addr, output_page_elements);

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
                const int src_tile_id = outer_index * inner_size + h_tile * input_width + w_tile;

                const uint64_t src_noc_addr = get_noc_addr(src_tile_id, s_src);
                noc_async_read(src_noc_addr, src_cb_addr, src_page_size);
                noc_async_read_barrier();

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
