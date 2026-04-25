// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_stick_nbytes_dram = get_compile_time_arg_val(2);
    constexpr uint32_t stride_h = get_compile_time_arg_val(3);
    constexpr uint32_t stride_w = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    constexpr uint32_t work_per_core = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<9>();

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const auto s_in = TensorAccessor(src_args, src_addr);

    experimental::Noc noc;
    experimental::CB cb_in0(cb_id_in0);

    uint32_t src_index = get_arg_val<uint32_t>(1);
    uint32_t curr_src_row_index = get_arg_val<uint32_t>(2);
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        uint32_t curr_src_offset = src_index;
        cb_in0.reserve_back(1);
        uint32_t l1_offset = 0;
        for (uint32_t i = 0; i < stride_h; i++) {
            for (uint32_t j = 0; j < stride_w; j++) {
                noc.async_read(s_in, cb_in0, stick_nbytes, {.page_id = curr_src_offset}, {.offset_bytes = l1_offset});
                curr_src_offset++;
                l1_offset += aligned_stick_nbytes_dram;
            }
            curr_src_offset += input_width - stride_w;
        }
        noc.async_read_barrier();
        cb_in0.push_back(1);

        curr_src_row_index += stride_w;
        if (curr_src_row_index >= (input_width)) {
            src_index += input_width * (stride_h - 1);
            curr_src_row_index = 0;
        }
        src_index += stride_w;
    }
}
