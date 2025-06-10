// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t batch_size = get_compile_time_arg_val(0);
    constexpr uint32_t input_width = get_compile_time_arg_val(1);
    constexpr uint32_t input_height = get_compile_time_arg_val(2);
    constexpr uint32_t stride_height = get_compile_time_arg_val(3);
    constexpr uint32_t stride_width = get_compile_time_arg_val(4);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(5);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(6);
    constexpr bool dst_stick_size_is_power_of_two = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t dst_log2_stick_size = get_compile_time_arg_val(8);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_input_work = get_arg_val<uint32_t>(1);
    uint32_t end_input_work = get_arg_val<uint32_t>(2);

    const auto s_out =
        get_interleaved_addr_gen<true, dst_stick_size_is_power_of_two>(dst_addr, stick_nbytes, dst_log2_stick_size);

    constexpr uint32_t Oh = input_height / stride_height;
    constexpr uint32_t Ow = input_width / stride_width;
    constexpr uint32_t patch_size = stride_height * stride_width;
    constexpr uint32_t input_hw = input_height * input_width;

    for (uint32_t input_idx = start_input_work; input_idx < end_input_work; input_idx++) {
        const uint32_t b = input_idx / input_hw;
        const uint32_t hw = input_idx % input_hw;
        const uint32_t h = hw / input_width;
        const uint32_t w = hw % input_width;

        const uint32_t oh = h / stride_height;
        const uint32_t ow = w / stride_width;
        const uint32_t kh = h % stride_height;
        const uint32_t kw = w % stride_width;

        int dst_row = (b * Oh + oh) * Ow + ow;
        int dst_col = (kh * stride_width + kw);
        int dst_index = dst_row * patch_size + dst_col;
        cb_wait_front(cb_id_in0, 1);
        uint32_t l1_addr = get_read_ptr(cb_id_in0);
        uint64_t dst_noc_addr = get_noc_addr(dst_index, s_out);
        noc_async_write(l1_addr, dst_noc_addr, stick_nbytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_in0, 1);
    }
}
