// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_stick_nbytes_dram = get_compile_time_arg_val(2);
    constexpr uint32_t stride_h = get_compile_time_arg_val(3);
    constexpr uint32_t stride_w = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    constexpr uint32_t work_per_core = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const auto s_in = TensorAccessor(src_args, src_addr, stick_nbytes);

    uint32_t src_index = get_arg_val<uint32_t>(1);
    uint32_t curr_src_row_index = get_arg_val<uint32_t>(2);
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        uint32_t curr_src_offset = src_index;
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t i = 0; i < stride_h; i++) {
            for (uint32_t j = 0; j < stride_w; j++) {
                uint64_t src_noc_addr = get_noc_addr(curr_src_offset, s_in);
                noc_async_read(src_noc_addr, l1_write_addr, stick_nbytes);
                curr_src_offset++;
                l1_write_addr += aligned_stick_nbytes_dram;
            }
            curr_src_offset += input_width - stride_w;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);

        curr_src_row_index += stride_w;
        if (curr_src_row_index >= (input_width)) {
            src_index += input_width * (stride_h - 1);
            curr_src_row_index = 0;
        }
        src_index += stride_w;
    }
}
