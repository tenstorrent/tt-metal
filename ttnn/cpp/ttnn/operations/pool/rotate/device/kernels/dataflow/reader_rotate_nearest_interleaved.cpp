// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include <ttnn/operations/pool/device/kernels/pool_kernels_common.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    int32_t cos_angle = static_cast<int32_t>(get_arg_val<uint32_t>(3));
    int32_t sin_angle = static_cast<int32_t>(get_arg_val<uint32_t>(4));
    int32_t center_x = static_cast<int32_t>(get_arg_val<uint32_t>(5));
    int32_t center_y = static_cast<int32_t>(get_arg_val<uint32_t>(6));
    uint32_t fill_value_bf16 = get_arg_val<uint32_t>(7);

    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t input_batch = get_compile_time_arg_val(2);
    constexpr uint32_t input_height = get_compile_time_arg_val(3);
    constexpr uint32_t input_width = get_compile_time_arg_val(4);
    constexpr uint32_t input_channels = get_compile_time_arg_val(5);
    constexpr uint32_t num_cb_pages = get_compile_time_arg_val(6);
    constexpr uint32_t fill_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t input_stick_nbytes_unaligned = get_compile_time_arg_val(8);
    constexpr bool fill_is_zero = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t batch_size = get_compile_time_arg_val(10);

    constexpr auto src_args = TensorAccessorArgs<11>();
    const auto input_tensor_accessor = TensorAccessor(src_args, input_addr, input_stick_nbytes_unaligned);

    uint32_t fill_stick_addr = get_write_ptr(fill_cb_index);
    if constexpr (fill_is_zero) {
        zero_out_page<fill_cb_index>(fill_stick_addr);
    } else {
        volatile tt_l1_ptr uint32_t* fill_ptr32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fill_stick_addr);
        const uint32_t fill_value_packed = (fill_value_bf16 << 16) | fill_value_bf16;
        const uint32_t num_pairs = input_channels / 2;
        for (uint32_t c = 0; c < num_pairs; c++) {
            fill_ptr32[c] = fill_value_packed;
        }
        if (input_channels & 1) {
            volatile tt_l1_ptr uint16_t* fill_ptr16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(fill_stick_addr);
            fill_ptr16[input_channels - 1] = static_cast<uint16_t>(fill_value_bf16);
        }
    }

    for (uint32_t local_stick_idx = 0; local_stick_idx < num_sticks;) {
        uint32_t sticks_this_batch =
            (num_sticks - local_stick_idx) < batch_size ? (num_sticks - local_stick_idx) : batch_size;
        cb_reserve_back(output_cb_index, sticks_this_batch);
        uint32_t l1_write_addr = get_write_ptr(output_cb_index);

        for (uint32_t i = 0; i < sticks_this_batch; i++, local_stick_idx++) {
            const uint32_t global_stick_idx = start_stick_id + local_stick_idx;

            const uint32_t batch_idx = global_stick_idx / (input_height * input_width);
            const uint32_t spatial_idx = global_stick_idx % (input_height * input_width);
            const uint32_t y_out = spatial_idx / input_width;
            const uint32_t x_out = spatial_idx % input_width;

            const int32_t x_out_fixed = fixed_point_arithmetic::int_to_fixed(x_out);
            const int32_t y_out_fixed = fixed_point_arithmetic::int_to_fixed(y_out);
            const int32_t x_centered = x_out_fixed - center_x;
            const int32_t y_centered = y_out_fixed - center_y;

            const int32_t x_in =
                fixed_point_arithmetic::fixed_mul_sub_add(x_centered, cos_angle, y_centered, sin_angle, center_x);
            const int32_t y_in =
                fixed_point_arithmetic::fixed_mul_add_add(x_centered, sin_angle, y_centered, cos_angle, center_y);

            const int32_t nearest_x = fixed_point_arithmetic::fixed_to_int_round(x_in);
            const int32_t nearest_y = fixed_point_arithmetic::fixed_to_int_round(y_in);

            const bool x_valid = nearest_x >= 0 && nearest_x < static_cast<int32_t>(input_width);
            const bool y_valid = nearest_y >= 0 && nearest_y < static_cast<int32_t>(input_height);

            if (x_valid && y_valid) {
                const uint32_t input_stick_index =
                    batch_idx * (input_height * input_width) + nearest_y * input_width + nearest_x;
                const uint64_t input_noc_addr = input_tensor_accessor.get_noc_addr(input_stick_index);
                noc_async_read(input_noc_addr, l1_write_addr, input_stick_nbytes_unaligned);
            } else {
                noc_async_read(get_noc_addr(fill_stick_addr), l1_write_addr, input_stick_nbytes_unaligned);
            }
            l1_write_addr += input_stick_nbytes;
        }

        noc_async_read_barrier();
        cb_push_back(output_cb_index, sticks_this_batch);
    }
}
