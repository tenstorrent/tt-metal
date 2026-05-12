// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

#include <stdint.h>

void kernel_main() {
    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);

    // Compile time args
    // -----------------
    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t src_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(3);

    // This is the number of elements in the output, excluding the last two dimensions.
    // i.e. for an input tensor of shape (.., N, C, H, W), this is (.. * N * C)
    // It also depends on the `keepdim`
    constexpr uint32_t outer_dim_units = get_compile_time_arg_val(4);

    // This is the number of elements in the last dimension of the output
    // i.e. for an input tensor of shape (.., N, C, H, W), this is H.
    // This dictates the page size in the output cb
    constexpr uint32_t inner_dim_units = get_compile_time_arg_val(5);

    // This is the number of elements in the input tensor along the reduction dim (W)
    constexpr uint32_t red_dim_units = get_compile_time_arg_val(6);

    // Boolean to indicate if we reduce across _all_ dimensions or just on the reduction dim (last dim)
    constexpr bool reduce_all = (bool)get_compile_time_arg_val(7);

    constexpr auto s_src_args = TensorAccessorArgs<8>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();

    //-------------------------------------------------------------------------
    const auto s_src = TensorAccessor(s_src_args, src_base_addr);
    const auto s_dst = TensorAccessor(s_dst_args, dst_base_addr);

    experimental::Noc noc;
    experimental::CircularBuffer src_cb(src_cb_idx);
    experimental::CircularBuffer dst_cb(dst_cb_idx);

    // CB in L1 memory for storing input
    const uint32_t src_cb_addr = src_cb.get_write_ptr();
    constexpr DataFormat src_cb_addr_data_format = get_dataformat(src_cb_idx);

    // CB in L1 memory for storing output
    const uint32_t dst_cb_addr = dst_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* out_idxs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb_addr);

    uint32_t max_idx = 0;
    auto max_val = get_default_value<src_cb_addr_data_format>();

    //-------------------------------------------------------------------------
    // Main loop - run by all cores
    for (uint32_t k = 0; k < outer_dim_units; ++k) {
        for (uint32_t j = 0; j < inner_dim_units; ++j) {
            noc.async_read(s_src, src_cb, src_page_size, {.page_id = k * inner_dim_units + j}, {.offset_bytes = 0});
            noc.async_read_barrier();

            // Reset max_val for each new output
            if constexpr (not reduce_all) {
                max_idx = 0;
                max_val = get_default_value<src_cb_addr_data_format>();
            }

            for (uint32_t i = 0; i < red_dim_units; ++i) {
                compare_values<src_cb_addr_data_format>(
                    src_cb_addr, max_val, max_idx, i, j, k, red_dim_units, reduce_all, inner_dim_units);
            }
            if constexpr (not reduce_all) {
                out_idxs[j] = max_idx;
            }
        }

        if constexpr (not reduce_all) {
            noc.async_write(dst_cb, s_dst, dst_page_size, {.offset_bytes = 0}, {.page_id = k});
            noc.async_write_barrier();
        }
    }

    // TODO: Generalize write for argmax for other dims
    if constexpr (reduce_all) {
        out_idxs[0] = max_idx;
        noc.async_write(dst_cb, s_dst, dst_page_size, {.offset_bytes = 0}, {.page_id = 0});
        noc.async_write_barrier();
    }
}
