// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #include "dataflow_api.h"
#include <algorithm>
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "debug/dprint.h"

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
    uint32_t start_input_row_in_image_id = get_arg_val<uint32_t>(6);
    uint32_t in_h = get_arg_val<uint32_t>(7);

    constexpr bool src1_is_dram = false;
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(2);
    // constexpr uint32_t is_reader = get_compile_time_arg_val(2);
    constexpr uint32_t scale_h_inv_comp = get_compile_time_arg_val(3);
    constexpr uint32_t scale_w_inv_comp = get_compile_time_arg_val(4);
    constexpr uint32_t y_index_comp = get_compile_time_arg_val(5);
    constexpr uint32_t x_index_compute_comp = get_compile_time_arg_val(6);
    constexpr uint32_t is_reader = get_compile_time_arg_val(7);
    constexpr uint32_t blocks = get_compile_time_arg_val(8);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(9);
    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    uint32_t total_nsticks_to_process = in_w * scale_w;
    // Calculate the number of sticks to process per core by dividing the total number of sticks (in width direction)
    // by 2.
    uint32_t nsticks_to_process_on_core =
        (total_nsticks_to_process % 2) ? total_nsticks_to_process / 2 + 1 : total_nsticks_to_process / 2;
    // assuming shard begins with a new row. TODO: generalize?
    float scale_h_inv = uint32_to_float(scale_h_inv_comp);
    float scale_w_inv = uint32_to_float(scale_w_inv_comp);
    float x, x_index, y_index, dx, dy;
    y_index = uint32_to_float(y_index_comp);
    float x_index_compute = uint32_to_float(x_index_compute_comp);

    // If the current core is a writer core, adjust the x_index_compute to start from the correct position.

    if (!is_reader) {
        x_index_compute += scale_w_inv;
        // If the total number of sticks is odd, process one less stick.
        nsticks_to_process_on_core =
            (total_nsticks_to_process % 2) ? nsticks_to_process_on_core - 1 : nsticks_to_process_on_core;
    }
    int32_t accumulated_offset = 0;
    for (uint32_t image_row = 0; image_row < in_image_rows_per_core * scale_h; ++image_row) {
        x_index = x_index_compute;

        uint32_t y1 = int(y_index);
        uint32_t y2 = y1 + 1;

        // After haloing, the last row from the previous core (or a padding row)
        // Gets inserted into as the first (index 0) row for the current core
        // So the start_input_row_in_image_id corresponds to the row with index 1
        // for the current core

        int32_t in_batch_index_y1 = int32_t(y1) - 1 + start_input_row_in_image_id - accumulated_offset;
        int32_t in_batch_index_y2 = int32_t(y2) - 1 + start_input_row_in_image_id - accumulated_offset;

        dy = y_index - y1;

        // In no circumstance should the padding rows have weights greater than 0.5

        if (in_batch_index_y1 == -1) {
            // This would mean that in_batch_index_y1 (the "upper" row) corresponds to a padding row (specifically the
            // top padding row) Reduce the padding row's weight to 0 and put full weight on the row below it
            dy = 1;
        }

        if (in_batch_index_y2 == int(in_h)) {
            // This would would mean that in_batch_index_y2 (the "lower row") corresponds to a padding row(specifically
            // the bottom padding row)
            if (dy > 0.5) {  // Due to math behind bilinear upsampling, dy could never be exactly 0.5, so this check
                             // should be numerically safe
                // In this case, a padding row has weight higher than 0.5.
                // This means we have to skip the next 2 rows (lower padding of current image and upper padding of
                // previous image) The iteration that enters this case outputs the first row of a new image
                y1 += 2;
                y2 += 2;
                y_index += 2;
                dy = 1;
                accumulated_offset += 2 + in_h;
            } else {
                // In this case, a padding row has weight lower than 0.5
                // We should do no skipping, but just set weight to 0 for this row
                // The iteration that enters this case outputs the final few rows of the image
                dy = 0;
            }
        }
        for (uint32_t j = 0; j < nsticks_to_process_on_core; j++) {
            for (uint32_t i = 0; i < blocks; i++) {
                cb_reserve_back(out_cb_id, 4);
                cb_reserve_back(in_scalar_cb_id, 1);

                x = x_index < 0 ? 0 : x_index;
                dx = x - int(x);
                uint32_t x1 = int(x);
                uint32_t x2 = std::min(x1 + 1, in_w - 1);

                fill_four_val(
                    get_write_ptr(in_scalar_cb_id),
                    float_to_bfloat16((1 - dx) * (1 - dy)),
                    float_to_bfloat16(dx * (1 - dy)),
                    float_to_bfloat16((1 - dx) * dy),
                    float_to_bfloat16(dx * dy));

                uint32_t l1_write_addr = get_write_ptr(out_cb_id);
                uint32_t l1_read_addr_temp =
                    l1_read_addr + x1 * stick_nbytes + y1 * in_w * stick_nbytes + i * input_block_size_bytes;
                // 1st tile
                uint64_t src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 2nd tile
                l1_read_addr_temp =
                    l1_read_addr + y1 * in_w * stick_nbytes + x2 * stick_nbytes + i * input_block_size_bytes;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 3rd tile
                l1_read_addr_temp =
                    l1_read_addr + y2 * in_w * stick_nbytes + x1 * stick_nbytes + i * input_block_size_bytes;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 4th tile
                l1_read_addr_temp =
                    l1_read_addr + y2 * in_w * stick_nbytes + x2 * stick_nbytes + i * input_block_size_bytes;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // push scaler and data into cb.
                noc_async_read_barrier();
                cb_push_back(out_cb_id, 4);
                cb_push_back(in_scalar_cb_id, 1);
            }
            x_index += scale_w_inv * 2;
        }
        y_index += scale_h_inv;
    }
}
