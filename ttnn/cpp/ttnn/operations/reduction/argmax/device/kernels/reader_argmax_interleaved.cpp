// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "argmax_common.hpp"

#include "debug/dprint.h" 

void kernel_main() {
    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);

    // Compile time args
    // -----------------
    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t w2r_cb_idx = get_compile_time_arg_val(2);

    constexpr uint32_t src_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(4);

    // This is the number of elements in the output, excluding the last two dimensions.
    // i.e. for an input tensor of shape (.., N, C, H, W), this is (.. * N * C)
    // It also depends on the `keepdim`
    constexpr uint32_t outer_dim_units = get_compile_time_arg_val(5);

    // This is the number of elements in the last dimension of the output
    // i.e. for an input tensor of shape (.., N, C, H, W), this is H.
    // This dictates the page size in the output cb
    constexpr uint32_t inner_dim_units = get_compile_time_arg_val(6);

    // This is the number of elements in the input tensor along the reduction dim (W)
    constexpr uint32_t red_dim_units = get_compile_time_arg_val(7);

    // Boolean to indicate if we reduce across _all_ dimensions or just on the reduction dim (last dim)
    constexpr bool reduce_all = (bool)get_compile_time_arg_val(8);

    constexpr bool is_reader = (bool)get_compile_time_arg_val(9);

    constexpr auto s_src_args = TensorAccessorArgs<10>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();

    //-------------------------------------------------------------------------
    const auto s_src = TensorAccessor(s_src_args, src_base_addr, src_page_size);
    const auto s_dst = TensorAccessor(s_dst_args, dst_base_addr, dst_page_size);

    // CB in L1 memory for storing input
    const uint32_t src_cb_addr = get_write_ptr(src_cb_idx) + (src_page_size * (is_reader ? 0 : 1));
    constexpr DataFormat src_cb_addr_data_format = get_dataformat(src_cb_idx);

    // CB in L1 memory for storing output
    const uint32_t dst_cb_addr = get_write_ptr(dst_cb_idx) + (dst_page_size * (is_reader ? 0 : 1));
    volatile tt_l1_ptr uint32_t* out_idxs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb_addr);

    uint32_t max_idx = 0;
    auto max_val = get_default_value<src_cb_addr_data_format>();

    constexpr uint32_t start_outer_dim_units = is_reader ? 0 : (outer_dim_units  +  1) / 2;
    constexpr uint32_t end_outer_dim_units = is_reader ? (outer_dim_units + 1) / 2 : outer_dim_units;

    //-------------------------------------------------------------------------
    // Main loop - run by all cores
    for (uint32_t k = start_outer_dim_units; k < end_outer_dim_units; ++k) {
        for (uint32_t j = 0; j < inner_dim_units; ++j) {
            const uint64_t src_noc_addr = get_noc_addr(k * inner_dim_units + j, s_src);
            noc_async_read(src_noc_addr, src_cb_addr, src_page_size);
            noc_async_read_barrier();

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
            uint64_t dst_noc_addr = get_noc_addr(k, s_dst);
            noc_async_write(dst_cb_addr, dst_noc_addr, dst_page_size);
            noc_async_write_barrier();
        }
    }

    if constexpr (reduce_all) {
        constexpr bool need_w2r_sync = ((outer_dim_units + 1) / 2) < outer_dim_units;
        if constexpr (is_reader && need_w2r_sync) {
            cb_wait_front(w2r_cb_idx, 1);
            const uint32_t w2r_cb_addr = get_read_ptr(w2r_cb_idx);
            volatile tt_l1_ptr uint32_t* idx_val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w2r_cb_addr);
            auto idx = idx_val[0]; 
            auto raw_val = idx_val[1];
            auto val = get_value<src_cb_addr_data_format>(reinterpret_cast<void*>(&raw_val));
            compare_values<src_cb_addr_data_format>(val, idx, max_val, max_idx);
            cb_pop_front(w2r_cb_idx, 1);
        }else if constexpr (!is_reader && need_w2r_sync) {  
            cb_reserve_back(w2r_cb_idx, 1);
            const uint32_t w2r_cb_addr = get_write_ptr(w2r_cb_idx);
            volatile tt_l1_ptr uint32_t* idx_val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w2r_cb_addr);

            idx_val[0] = max_idx;
            idx_val[1] = max_val; 
            cb_push_back(w2r_cb_idx, 1);
        }
        else {
            // nothing to do
        }

        out_idxs[0] = max_idx;
        const uint64_t dst_noc_addr = get_noc_addr(0, s_dst);
        noc_async_write(dst_cb_addr, dst_noc_addr, dst_page_size);
        noc_async_write_barrier();
    }
}
