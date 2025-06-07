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
    constexpr uint32_t pad_height = get_compile_time_arg_val(5);
    constexpr uint32_t pad_width = get_compile_time_arg_val(6);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(8);
    constexpr bool dst_stick_size_is_power_of_two = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t dst_log2_stick_size = get_compile_time_arg_val(10);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_input_work = get_arg_val<uint32_t>(1);
    uint32_t end_input_work = get_arg_val<uint32_t>(2);
    uint32_t start_padding_work = get_arg_val<uint32_t>(3);
    uint32_t end_padding_work = get_arg_val<uint32_t>(4);
    const auto s_out =
        get_interleaved_addr_gen<true, dst_stick_size_is_power_of_two>(dst_addr, stick_nbytes, dst_log2_stick_size);
    constexpr uint32_t Oh = (input_height + pad_height) / stride_height;
    constexpr uint32_t Ow = (input_width + pad_width) / stride_width;
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

    // Handle padding: Write zeros to padding locations
    // Create a zero buffer
    uint8_t zero_buffer[stick_nbytes];
    for (uint32_t i = 0; i < stick_nbytes; i++) {
        zero_buffer[i] = 0;
    }

    // Calculate total output size
    constexpr uint32_t total_output_hw = Oh * Ow * stride_height * stride_width;

    // Process padding locations assigned to this core
    for (uint32_t padding_idx = start_padding_work; padding_idx < end_padding_work; padding_idx++) {
        // Map padding_idx to b, oh, ow, kh, kw
        const uint32_t output_hw_size = Oh * Ow * stride_height * stride_width;
        const uint32_t b = padding_idx / output_hw_size;
        const uint32_t hw_idx = padding_idx % output_hw_size;

        const uint32_t oh_ow_size = Oh * Ow;
        const uint32_t patch_idx = hw_idx / (stride_height * stride_width);
        const uint32_t k_idx = hw_idx % (stride_height * stride_width);

        const uint32_t oh = patch_idx / Ow;
        const uint32_t ow = patch_idx % Ow;
        const uint32_t kh = k_idx / stride_width;
        const uint32_t kw = k_idx % stride_width;

        // Calculate corresponding input coordinates
        int h = oh * stride_height + kh;
        int w = ow * stride_width + kw;

        // Check if this is a padding location
        bool is_padding = (h < 0 || h >= static_cast<int>(input_height) || w < 0 || w >= static_cast<int>(input_width));

        if (is_padding) {
            int dst_row = (b * Oh + oh) * Ow + ow;
            int dst_col = (kh * stride_width + kw);
            int dst_index = dst_row * patch_size + dst_col;
            uint64_t dst_noc_addr = get_noc_addr(dst_index, s_out);

            // Write zeros to this padding location
            noc_async_write(reinterpret_cast<uint32_t>(zero_buffer), dst_noc_addr, stick_nbytes);
        }
    }
    noc_async_write_barrier();
}
