// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t tile_height = 32;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t padded_X_size = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    const uint32_t n_block_reps = get_arg_val<uint32_t>(4);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(3) == 1;

    const uint32_t num_tiles_per_row = padded_X_size >> (FLOAT32_DTYPE ? 7 : 6);

    constexpr bool stick_size_is_power_of_two = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);

    const auto s = get_interleaved_addr_gen<dst_is_dram, stick_size_is_power_of_two>(
        dst_addr, unpadded_X_size, log_base_2_of_page_size);

    auto pop_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_wait_front(cb_id_out0, num_tiles_per_row);
            cb_pop_front(cb_id_out0, num_tiles_per_row);
        }
    };

    auto write_block = [&](uint32_t base_stick_id, uint32_t num_rows) {
        uint32_t padding_rows = (tile_height - num_rows) & 31;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_wait_front(cb_id_out0, num_tiles_per_row * has_rows);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t k = 0; k < num_rows; k++) {
            uint64_t dst_noc_addr = get_noc_addr(base_stick_id + k, s);

            // Write out tmp buffer
            noc_async_write(l1_read_addr, dst_noc_addr, unpadded_X_size);

            noc_async_write_barrier();
            l1_read_addr += padded_X_size;
        }
        cb_pop_front(cb_id_out0, num_tiles_per_row * has_rows);
    };

    uint32_t stick_id = start_stick_id;
    uint32_t rt_arg_idx = 5;
    uint32_t count = 1;
    constexpr int32_t n_mixed_idx = 1;
    constexpr int32_t n_pad_idx = 2;
    constexpr int32_t times_idx = 3;
    constexpr uint32_t repeat_ct_idx = 4;
    constexpr int32_t num_rt_idx = 5;

    for (uint32_t block_rep_idx = 0; block_rep_idx < n_block_reps; ++block_rep_idx) {
        const uint32_t repeat_count = get_arg_val<uint32_t>(rt_arg_idx + repeat_ct_idx);
        const uint32_t n_data = get_arg_val<uint32_t>(rt_arg_idx);  // number of full tile-rows
        const uint32_t n_mixed =
            get_arg_val<uint32_t>(rt_arg_idx + n_mixed_idx);  // number of rows in a partially filled tile-row
        const uint32_t n_pads = get_arg_val<uint32_t>(rt_arg_idx + n_pad_idx);  // number of padding tile-rows
        const uint32_t times =
            get_arg_val<uint32_t>(rt_arg_idx + times_idx);  // number of times the pattern of tile-rows repeats
        if (count == repeat_count) {
            rt_arg_idx = rt_arg_idx + num_rt_idx;
            count = 1;
        } else {
            count++;
        }

        for (uint32_t t = 0; t < times; ++t) {
            for (uint32_t y_t = 0; y_t < n_data; y_t++) {
                write_block(stick_id, tile_height);
                stick_id += tile_height;
            }

            write_block(stick_id, n_mixed);
            stick_id += n_mixed;

            pop_blocks(n_pads);
        }
    }
}
