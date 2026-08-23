// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/pool_kernels_common.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>

template <
    uint32_t input_stick_nbytes,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t input_channels,
    uint32_t input_stick_nbytes_unaligned,
    uint32_t fill_is_zero,
    uint32_t burst_size>
TT_KERNEL void reader_rotate_nearest(
    uint32_t num_sticks,
    uint32_t start_stick_id,
    uint32_t cos_angle_bits,
    uint32_t sin_angle_bits,
    uint32_t center_x_bits,
    uint32_t center_y_bits,
    uint32_t fill_value_bf16) {
    const int32_t cos_angle = static_cast<int32_t>(cos_angle_bits);
    const int32_t sin_angle = static_cast<int32_t>(sin_angle_bits);
    const int32_t center_x = static_cast<int32_t>(center_x_bits);
    const int32_t center_y = static_cast<int32_t>(center_y_bits);
    const auto input_tensor_accessor = TensorAccessor(tensor::input);

    DataflowBuffer output_dfb(dfb::output);
    DataflowBuffer fill_dfb(dfb::fill);
    Noc noc;
    UnicastEndpoint self_ep;

    uint32_t fill_stick_addr = fill_dfb.get_write_ptr();
    if constexpr (fill_is_zero != 0) {
        zero_out_page(noc, fill_dfb);
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
        uint32_t sticks_this_burst =
            (num_sticks - local_stick_idx) < burst_size ? (num_sticks - local_stick_idx) : burst_size;
        output_dfb.reserve_back(sticks_this_burst);
        uint32_t write_offset = 0;

        for (uint32_t i = 0; i < sticks_this_burst; i++, local_stick_idx++) {
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
                noc.async_read(
                    input_tensor_accessor,
                    output_dfb,
                    input_stick_nbytes_unaligned,
                    {.page_id = input_stick_index},
                    {.offset_bytes = write_offset});
            } else {
                noc.async_read(
                    self_ep,
                    output_dfb,
                    input_stick_nbytes_unaligned,
                    experimental::local_addr(fill_stick_addr, noc.get_noc_id()),
                    {.offset_bytes = write_offset});
            }
            write_offset += input_stick_nbytes;
        }

        noc.async_read_barrier();
        output_dfb.push_back(sticks_this_burst);
    }
}
