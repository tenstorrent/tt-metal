// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

#include "debug/dprint.h"

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

    const auto s0 =
        get_interleaved_addr_gen<grid_is_dram, grid_size_is_power_of_two>(grid_addr, grid_stick_nbytes, grid_log2_size);

    const auto s1 = get_interleaved_addr_gen<src_is_dram, input_size_is_power_of_two>(
        input_addr, input_stick_nbytes, input_log2_size);

    const uint32_t end_id = start_page_id + num_pages;

    auto read_input_stick = [&](uint32_t curr_batch, uint32_t h_coord, uint32_t w_coord) {
        cb_reserve_back(input_cb_index, 1);
        uint32_t l1_write_addr = get_write_ptr(input_cb_index);

        if (h_coord <= 0) {
            h_coord = 0;
        }
        if (h_coord >= input_height - 1) {
            h_coord = input_height - 1;
        }
        if (w_coord <= 0) {
            w_coord = 0;
        }
        if (h_coord >= input_width + 1) {
            w_coord = input_width + 1;
        }

        uint32_t read_index = curr_batch * input_width * input_height + h_coord * input_width + w_coord;
        uint32_t input_noc_addr = get_noc_addr(read_index, s1);
        noc_async_read(input_noc_addr, l1_write_addr, input_stick_nbytes);

        noc_async_read_barrier();

        cb_push_back(input_cb_index, 1);
    };

    // auto compute_weight_low  = [&] (float coord_f, uint32_t coord_int){

    // }
    // auto compute_weight_high = [&] (float)

    // reader copied the data from DRAM to CB buffer.
    for (uint32_t i = start_page_id; i < end_id; ++i) {
        // cb_reserve_back(grid_cb_index, 1);
        uint32_t l1_write_stick_addr = get_write_ptr(grid_cb_index);
        uint64_t grid_noc_addr = get_noc_addr(i, s0);

        noc_async_read(grid_noc_addr, l1_write_stick_addr, grid_stick_nbytes);

        noc_async_read_barrier();

        // Read the first two bfloat16 values (grid coordinates) from the L1 buffer
        volatile tt_l1_ptr uint16_t* grid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_stick_addr);
        uint16_t h_coord_raw = grid_ptr[0];  // First bfloat16 coordinate (x)
        uint16_t w_coord_raw = grid_ptr[1];  // Second bfloat16 coordinate (y)

        float h_coord_rel = bfloat16_to_float(h_coord_raw);
        float w_coord_rel = bfloat16_to_float(w_coord_raw);

        DPRINT << h_coord_rel << " " << w_coord_rel << "\n";

        float h_coord_image = (h_coord_rel + 1) * input_height / 2 - 0.5;
        float w_coord_image = (w_coord_rel + 1) * input_width / 2 - 0.5;
        uint32_t h0 = int(h_coord_image);
        uint32_t h1 = h0 + 1;
        uint32_t w0 = int(w_coord_image);
        uint32_t w1 = w0 + 1;

        // read the input sticks

        read_input_stick(0, h0, w0);
        read_input_stick(0, h0, w1);
        read_input_stick(0, h1, w0);
        read_input_stick(0, h1, w1);

        // compute scalars

        float weight_h0, weight_h1, weight_w0, weight_w1;

        // petak u 6 racunanje weightova, mnogo scuffed ali dobro, za sad zelim da radi

        if (h0 < 0 || h0 >= input_height) {
            weight_h0 = 0;
        } else {
            weight_h0 = 1 - (h_coord_image - h0);
        }

        if (h1 < 0 || h1 >= input_height) {
            weight_h1 = 0;
        } else {
            weight_h1 = h_coord_image - h0;
        }

        if (w0 < 0 || w0 >= input_width) {
            weight_w0 = 0;
        } else {
            weight_w0 = 1 - (w_coord_image - w0);
        }

        if (w1 < 0 || w1 >= input_width) {
            weight_w1 = 0;
        } else {
            weight_w1 = w_coord_image - w0;
        }

        cb_reserve_back(scalar_cb_index, 1);

        uint32_t scalar_cb_addr = get_write_ptr(scalar_cb_index);

        fill_four_val(
            get_write_ptr(scalar_cb_addr),
            float_to_bfloat16(weight_h0 * weight_w0),
            float_to_bfloat16(weight_h0 * weight_w1),
            float_to_bfloat16(weight_h1 * weight_w0),
            float_to_bfloat16(weight_h1 * weight_w1));

        cb_push_back(scalar_cb_index, 1);
    }
    DPRINT << "READER FINISHED" << "\n";
}
