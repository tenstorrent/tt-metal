// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

#define ALWI inline __attribute__((always_inline))

// Fill given four values into the memory starting at the given address.
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);

    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

ALWI float uint32_to_float(uint32_t f) {
    float ret;
    std::memcpy(&ret, &f, sizeof(float));
    return ret;
}

void kernel_main() {
    uint32_t stick_nbytes = get_arg_val<uint32_t>(0);
    uint32_t in_image_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t scale_h = get_arg_val<uint32_t>(2);
    uint32_t scale_w = get_arg_val<uint32_t>(3);
    uint32_t in_w = get_arg_val<uint32_t>(4);
    uint32_t out_w = get_arg_val<uint32_t>(5);
    uint32_t src1_addr = get_arg_val<uint32_t>(6);
    uint32_t read_offset = get_arg_val<uint32_t>(8);
    uint32_t is_last_row = get_arg_val<uint32_t>(9);
    uint32_t in_h = 1;
    constexpr bool src1_is_dram = false;

    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = tt::CB::c_in1;
    // constexpr uint32_t is_reader = get_compile_time_arg_val(2);
    constexpr uint32_t scale_h_inv_comp = get_compile_time_arg_val(3);
    constexpr uint32_t scale_w_inv_comp = get_compile_time_arg_val(4);
    constexpr uint32_t y_index_comp = get_compile_time_arg_val(5);
    constexpr uint32_t x_index_compute_comp = get_compile_time_arg_val(6);

    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in4;

    // assuming shard begins with a new row. TODO: generalize?
    float scale_h_inv = uint32_to_float(scale_h_inv_comp);
    float scale_w_inv = uint32_to_float(scale_w_inv_comp);
    float x, y, x_index, y_index, dx, dy;
    y_index = uint32_to_float(y_index_comp);
    float x_index_compute = uint32_to_float(x_index_compute_comp);
    for (uint32_t image_row = 0; image_row < in_image_rows_per_core * scale_h; ++image_row) {
        x_index = x_index_compute;
        for (uint32_t j = 0; j < in_w * scale_w; j++) {
            cb_reserve_back(out_cb_id, 4);
            cb_reserve_back(in_scalar_cb_id, 1);

            x = x_index < 0 ? 0 : x_index;
            y = y_index < read_offset ? read_offset : y_index;
            dx = x - int(x);
            dy = y - int(y);

            uint32_t x1 = int(x);
            uint32_t y1 = int(y);
            uint32_t x2 = min(x1 + 1, in_w - 1);
            uint32_t y2 = y1 + 1;
            if (is_last_row) {
                y2 = min(y2, in_image_rows_per_core);  // if last row, y2 should be in_image_rows_per_core
            }

            fill_four_val(get_write_ptr(in_scalar_cb_id),
                          float_to_bfloat16((1 - dx) * (1 - dy)),
                          float_to_bfloat16(dx * (1 - dy)),
                          float_to_bfloat16((1 - dx) * dy),
                          float_to_bfloat16(dx * dy));

            uint32_t l1_write_addr = get_write_ptr(out_cb_id);
            uint32_t l1_read_addr_temp = l1_read_addr + x1 * stick_nbytes + y1 * in_w * stick_nbytes;
            // 1st tile
            uint64_t src_noc_addr = get_noc_addr(l1_read_addr_temp);
            noc_async_read(src_noc_addr, l1_write_addr, stick_nbytes);
            l1_write_addr += stick_nbytes;

            // 2nd tile
            l1_read_addr_temp = l1_read_addr + y1 * in_w * stick_nbytes + x2 * stick_nbytes;
            src_noc_addr = get_noc_addr(l1_read_addr_temp);
            noc_async_read(src_noc_addr, l1_write_addr, stick_nbytes);
            l1_write_addr += stick_nbytes;

            // 3rd tile
            l1_read_addr_temp = l1_read_addr + y2 * in_w * stick_nbytes + x1 * stick_nbytes;
            src_noc_addr = get_noc_addr(l1_read_addr_temp);
            noc_async_read(src_noc_addr, l1_write_addr, stick_nbytes);
            l1_write_addr += stick_nbytes;

            // 4th tile
            l1_read_addr_temp = l1_read_addr + y2 * in_w * stick_nbytes + x2 * stick_nbytes;
            src_noc_addr = get_noc_addr(l1_read_addr_temp);
            noc_async_read(src_noc_addr, l1_write_addr, stick_nbytes);
            l1_write_addr += stick_nbytes;

            // push scaler and data into cb.
            noc_async_read_barrier();
            cb_push_back(out_cb_id, 4);
            cb_push_back(in_scalar_cb_id, 1);
            x_index += scale_w_inv;
        }
        y_index += scale_h_inv;
    }
}
