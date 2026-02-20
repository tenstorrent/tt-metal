// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#include "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/grid_sample_reader_common.hpp"
#include "ttnn/cpp/ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp"
#include "ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp"

using namespace fixed_point_arithmetic;

#define ALWI inline __attribute__((always_inline))

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    uint32_t cos_angle_bits = get_arg_val<uint32_t>(3);
    uint32_t sin_angle_bits = get_arg_val<uint32_t>(4);
    uint32_t center_x_bits = get_arg_val<uint32_t>(5);
    uint32_t center_y_bits = get_arg_val<uint32_t>(6);
    uint32_t fill_value_bits = get_arg_val<uint32_t>(7);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t input_batch = get_compile_time_arg_val(3);
    constexpr uint32_t input_height = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    constexpr uint32_t fill_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t input_channels = get_compile_time_arg_val(7);
    constexpr bool fill_is_zero = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t element_size = get_compile_time_arg_val(9);

    const int32_t cos_angle_q16 = static_cast<int32_t>(cos_angle_bits);
    const int32_t sin_angle_q16 = static_cast<int32_t>(sin_angle_bits);
    const int32_t center_x_q16 = static_cast<int32_t>(center_x_bits);
    const int32_t center_y_q16 = static_cast<int32_t>(center_y_bits);

    constexpr auto src_args = TensorAccessorArgs<10>();
    const auto input_tensor_accessor = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    constexpr uint32_t hw_size = input_height * input_width;

    const uint32_t end_stick_id = start_stick_id + num_sticks;

    uint32_t fill_stick_addr = get_write_ptr(fill_cb_index);
    if constexpr (fill_is_zero) {
        zero_out_page<fill_cb_index>(fill_stick_addr);
    } else {
        volatile tt_l1_ptr uint32_t* fill_ptr32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fill_stick_addr);
        if constexpr (element_size == 2) {
            const uint32_t fill_value_packed = (fill_value_bits << 16) | fill_value_bits;
            const uint32_t num_pairs = input_channels / 2;
            for (uint32_t c = 0; c < num_pairs; c++) {
                fill_ptr32[c] = fill_value_packed;
            }
            if (input_channels & 1) {
                volatile tt_l1_ptr uint16_t* fill_ptr16 =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(fill_stick_addr);
                fill_ptr16[input_channels - 1] = static_cast<uint16_t>(fill_value_bits);
            }
        } else {
            for (uint32_t c = 0; c < input_channels; c++) {
                fill_ptr32[c] = fill_value_bits;
            }
        }
    }
    noc_async_read_barrier();

    uint32_t curr_batch = start_stick_id / hw_size;
    uint32_t spatial_pos_in_batch = start_stick_id % hw_size;
    uint32_t batch_offset = curr_batch * hw_size;

    for (uint32_t stick_id = start_stick_id; stick_id < end_stick_id; ++stick_id) {
        const uint32_t y_out = spatial_pos_in_batch / input_width;
        const uint32_t x_out = spatial_pos_in_batch % input_width;

        const int32_t x_out_q16 = int_to_fixed(static_cast<int32_t>(x_out));
        const int32_t y_out_q16 = int_to_fixed(static_cast<int32_t>(y_out));

        const int32_t x_centered_q16 = fixed_sub(x_out_q16, center_x_q16);
        const int32_t y_centered_q16 = fixed_sub(y_out_q16, center_y_q16);

        const int32_t term1 = fixed_mul(x_centered_q16, cos_angle_q16);
        const int32_t term2 = fixed_mul(y_centered_q16, sin_angle_q16);
        const int32_t x_in_q16 = fixed_add(fixed_sub(term1, term2), center_x_q16);

        const int32_t term3 = fixed_mul(x_centered_q16, sin_angle_q16);
        const int32_t term4 = fixed_mul(y_centered_q16, cos_angle_q16);
        const int32_t y_in_q16 = fixed_add(fixed_add(term3, term4), center_y_q16);

        const int32_t h0 = fixed_to_int(y_in_q16);
        const int32_t h1 = h0 + 1;
        const int32_t w0 = fixed_to_int(x_in_q16);
        const int32_t w1 = w0 + 1;

        const int32_t h_frac_q16 = fixed_frac(y_in_q16);
        const int32_t w_frac_q16 = fixed_frac(x_in_q16);

        const int32_t h_frac_inv_q16 = fixed_one_minus(h_frac_q16);
        const int32_t w_frac_inv_q16 = fixed_one_minus(w_frac_q16);

        const int32_t weight_nw_q16 = fixed_mul(h_frac_inv_q16, w_frac_inv_q16);
        const int32_t weight_ne_q16 = fixed_mul(h_frac_inv_q16, w_frac_q16);
        const int32_t weight_sw_q16 = fixed_mul(h_frac_q16, w_frac_inv_q16);
        const int32_t weight_se_q16 = fixed_mul(h_frac_q16, w_frac_q16);

        const uint16_t weight_nw_bf = fixed_to_bf16(weight_nw_q16);
        const uint16_t weight_ne_bf = fixed_to_bf16(weight_ne_q16);
        const uint16_t weight_sw_bf = fixed_to_bf16(weight_sw_q16);
        const uint16_t weight_se_bf = fixed_to_bf16(weight_se_q16);

        cb_reserve_back(input_cb_index, 1);
        const uint32_t l1_write_input_addr = get_write_ptr(input_cb_index);

        read_four_corner_inputs_with_fill(
            input_tensor_accessor,
            batch_offset,
            input_width,
            input_stick_nbytes,
            h0,
            h1,
            w0,
            w1,
            input_height,
            l1_write_input_addr,
            fill_stick_addr);

        cb_reserve_back(scalar_cb_index, 1);
        const uint32_t l1_write_scalar_addr = get_write_ptr(scalar_cb_index);
        fill_four_val(l1_write_scalar_addr, weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf);
        cb_push_back(scalar_cb_index, 1);

        noc_async_read_barrier();
        cb_push_back(input_cb_index, 1);

        ++spatial_pos_in_batch;
        if (spatial_pos_in_batch == hw_size) {
            spatial_pos_in_batch = 0;
            ++curr_batch;
            batch_offset = curr_batch * hw_size;
        }
    }
}
