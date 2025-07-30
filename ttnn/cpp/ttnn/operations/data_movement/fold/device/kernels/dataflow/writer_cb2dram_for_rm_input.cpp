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
    constexpr auto dst_args = TensorAccessorArgs<7>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const auto s_out = TensorAccessor(dst_args, dst_addr, stick_nbytes);
    uint32_t dst_index = get_arg_val<uint32_t>(1);
    constexpr uint32_t patch_size = stride_h * stride_w;
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        cb_wait_front(cb_id_in0, 1);
        uint32_t l1_addr = get_read_ptr(cb_id_in0);
        for (uint32_t i = 0; i < patch_size; i++) {
            uint64_t dst_noc_addr = get_noc_addr(dst_index, s_out);
            noc_async_write(l1_addr, dst_noc_addr, stick_nbytes);
            dst_index++;
            l1_addr += aligned_stick_nbytes_dram;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_in0, 1);
    }
}
