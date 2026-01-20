// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t srcA_addr = get_arg_val<uint32_t>(0);
    uint32_t srcB_addr = get_arg_val<uint32_t>(1);
    uint32_t srcC_addr = get_arg_val<uint32_t>(2);
    uint32_t num_pages = get_arg_val<uint32_t>(3);
    uint32_t start_page_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_srcA_index = get_compile_time_arg_val(0);
    constexpr uint32_t cb_srcB_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_srcC_index = get_compile_time_arg_val(2);
    constexpr uint32_t aligned_input_unit_size = get_compile_time_arg_val(3);

    constexpr auto srcA_args = TensorAccessorArgs<4>();
    constexpr auto srcB_args = TensorAccessorArgs<5>();
    constexpr auto srcC_args = TensorAccessorArgs<6>();
    const auto srcA_accessor = TensorAccessor(srcA_args, srcA_addr, aligned_input_unit_size);
    const auto srcB_accessor = TensorAccessor(srcB_args, srcB_addr, aligned_input_unit_size);
    const auto srcC_accessor = TensorAccessor(srcC_args, srcC_addr, aligned_input_unit_size);

    const uint32_t end_id = start_page_id + num_pages;

    // reader copied the data from DRAM to CB buffer.
    for (uint32_t i = start_page_id; i < end_id; ++i) {
        uint32_t l1_write_addr = 0;  // or something invalid.
        uint64_t src_noc_addr = 0;   // or something invalid.

        // Read a page for A
        cb_reserve_back(cb_srcA_index, 1);
        l1_write_addr = get_write_ptr(cb_srcA_index);
        src_noc_addr = srcA_accessor.get_noc_addr(i);
        noc_async_read(src_noc_addr, l1_write_addr, aligned_input_unit_size);
        noc_async_read_barrier();
        cb_push_back(cb_srcA_index, 1);

        // Read a page for B
        cb_reserve_back(cb_srcB_index, 1);
        l1_write_addr = get_write_ptr(cb_srcB_index);
        src_noc_addr = srcB_accessor.get_noc_addr(i);
        noc_async_read(src_noc_addr, l1_write_addr, aligned_input_unit_size);
        noc_async_read_barrier();
        cb_push_back(cb_srcB_index, 1);

        // Read a page for C
        cb_reserve_back(cb_srcC_index, 1);
        l1_write_addr = get_write_ptr(cb_srcC_index);
        src_noc_addr = srcC_accessor.get_noc_addr(i);
        noc_async_read(src_noc_addr, l1_write_addr, aligned_input_unit_size);
        noc_async_read_barrier();
        cb_push_back(cb_srcC_index, 1);
    }
}
