// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

// #pragma GCC optimize("fast-math")
// #pragma GCC optimize("O3")

#define ALWI inline __attribute__((always_inline))

ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);

    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t grid_addr = get_arg_val<uint32_t>(1);
    uint32_t num_pages = get_arg_val<uint32_t>(2);
    uint32_t start_page_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(2);
    constexpr bool src_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool grid_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(5);
    constexpr bool input_size_is_power_of_two = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t input_log2_size = get_compile_time_arg_val(7);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(8);
    constexpr bool grid_size_is_power_of_two = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t grid_log2_size = get_compile_time_arg_val(10);
    constexpr uint32_t input_height = get_compile_time_arg_val(11);
    constexpr uint32_t input_width = get_compile_time_arg_val(12);

    DPRINT << "height" << input_height << " width" << input_width << "\n";

    const auto s0 =
        get_interleaved_addr_gen<grid_is_dram, grid_size_is_power_of_two>(grid_addr, grid_stick_nbytes, grid_log2_size);

    const auto s1 = get_interleaved_addr_gen<src_is_dram, input_size_is_power_of_two>(
        input_addr, input_stick_nbytes, input_log2_size);

    const uint32_t end_id = start_page_id + num_pages;

    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);

    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;

    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;

    auto read_input_stick = [&](uint32_t curr_batch, int32_t h_coord, int32_t w_coord) {
        // cb_reserve_back(input_cb_index, 1);
        uint32_t l1_write_addr = get_write_ptr(input_cb_index);

        if (h_coord < 0) {
            h_coord = 0;
        }
        if (h_coord >= static_cast<int32_t>(input_height)) {
            h_coord = input_height - 1;
        }
        if (w_coord < 0) {
            w_coord = 0;
        }
        if (w_coord >= static_cast<int32_t>(input_width)) {
            w_coord = input_width - 1;
        }

        uint32_t read_index = curr_batch * input_width * input_height + static_cast<uint32_t>(h_coord) * input_width +
                              static_cast<uint32_t>(w_coord);
        uint32_t input_noc_addr = get_noc_addr(read_index, s1);
        noc_async_read(input_noc_addr, l1_write_addr, input_stick_nbytes);

        noc_async_read_barrier();

        // cb_push_back(input_cb_index, 1);
    };

    // auto compute_weight_low  = [&] (float coord_f, uint32_t coord_int){

    // }
    // auto compute_weight_high = [&] (float)

    // reader copied the data from DRAM to CB buffer.
    for (uint32_t i = start_page_id; i < end_id; ++i) {
        // cb_reserve_back(grid_cb_index, 1);

        uint32_t l1_write_stick_addr = get_write_ptr(grid_cb_index);
        uint64_t grid_noc_addr = get_noc_addr(i, s0);

        {
            DeviceZoneScopedN("DRAM");
            noc_async_read(grid_noc_addr, l1_write_stick_addr, grid_stick_nbytes);

            noc_async_read_barrier();
        }

        // cb_reserve_back(scalar_cb_index, 1);

        // Read the first two bfloat16 values (grid coordinates) from the L1 buffer

        float h_coord_rel, w_coord_rel;

        {
            DeviceZoneScopedN("Logi0");
            volatile tt_l1_ptr uint16_t* grid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_stick_addr);
            uint16_t h_coord_raw = grid_ptr[0];  // First bfloat16 coordinate (x)
            uint16_t w_coord_raw = grid_ptr[1];  // Second bfloat16 coordinate (y)

            h_coord_rel = bfloat16_to_float(h_coord_raw);
            w_coord_rel = bfloat16_to_float(w_coord_raw);
        }

        DPRINT << h_coord_rel << " " << w_coord_rel << "\n";

        float h_coord_image, w_coord_image;
        int32_t h0, h1, w0, w1;

        {
            DeviceZoneScopedN("Logi1");
            h_coord_image = h_coord_rel * height_scale + height_offset;
            w_coord_image = w_coord_rel * width_scale + width_offset;

            // Use explicit type casts for better optimization
            h0 = static_cast<int32_t>(h_coord_image);
            h1 = h0 + 1;
            w0 = static_cast<int32_t>(w_coord_image);
            w1 = w0 + 1;
        }
        DPRINT << h_coord_image << " " << w_coord_image << "\n";
        DPRINT << "L1\n";

        // read the input sticks

        read_input_stick(0, h0, w0);
        read_input_stick(0, h0, w1);
        read_input_stick(0, h1, w0);
        read_input_stick(0, h1, w1);

        // compute scalars

        float weight_h0, weight_h1, weight_w0, weight_w1;

        // Optimized weight calculation with cached conversions
        {
            DeviceZoneScopedN("Logi2");
            // Cache integer-to-float conversions once
            float h0_f = static_cast<float>(h0);
            float w0_f = static_cast<float>(w0);

            // Precompute fractions and their inverses
            float h_frac = h_coord_image - h0_f;
            float w_frac = w_coord_image - w0_f;
            float h_frac_inv = 1.0f - h_frac;
            float w_frac_inv = 1.0f - w_frac;

            // Efficient boundary checks - now meaningful with signed integers
            bool h0_valid = (h0 >= 0) && (h0 < static_cast<int32_t>(input_height));
            bool h1_valid = (h1 >= 0) && (h1 < static_cast<int32_t>(input_height));
            bool w0_valid = (w0 >= 0) && (w0 < static_cast<int32_t>(input_width));
            bool w1_valid = (w1 >= 0) && (w1 < static_cast<int32_t>(input_width));

            // Assign weights using precomputed values
            weight_h0 = h0_valid ? h_frac_inv : 0.0f;
            weight_h1 = h1_valid ? h_frac : 0.0f;
            weight_w0 = w0_valid ? w_frac_inv : 0.0f;
            weight_w1 = w1_valid ? w_frac : 0.0f;
        }

        DPRINT << "L2\n";

        DPRINT << "weight_w0 " << weight_w0 << " weight_w1 " << weight_w1 << " weight_h0 " << weight_h0 << " weight_h1 "
               << weight_h1 << "\n";

        // cb_reserve_back(scalar_cb_index, 1);

        // uint32_t scalar_cb_addr = get_write_ptr(scalar_cb_index);

        float wei1, wei2, wei3, wei4;
        {
            DeviceZoneScopedN("Logi3");
            wei1 = weight_h0 * weight_w0;
            wei2 = weight_h0 * weight_w1;
            wei3 = weight_h1 * weight_w0;
            wei4 = weight_h1 * weight_w1;
            fill_four_val(
                get_write_ptr(scalar_cb_index),
                float_to_bfloat16(wei1),
                float_to_bfloat16(wei2),
                float_to_bfloat16(wei3),
                float_to_bfloat16(wei4));
        }

        {
            DeviceZoneScopedN("Nista");
        }

        DPRINT << "L3\n";
        // wei1 = wei1 + 1.0;
        // wei2 = wei2 + 1.0;
        // wei3 = wei3 + 1.0;
        // wei4 = wei4 + 1.0;
        DPRINT << wei1 << " " << wei2 << " " << wei3 << " " << wei4 << '\n';

        // cb_pop_front(scalar_cb_index, 1);

        // cb_push_back(scalar_cb_index, 1);
    }
    DPRINT << "READER FINISHED" << "\n";
}
