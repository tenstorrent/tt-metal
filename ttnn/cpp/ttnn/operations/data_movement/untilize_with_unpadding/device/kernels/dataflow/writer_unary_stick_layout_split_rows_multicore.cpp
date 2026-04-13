// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t tile_height = 32;
    experimental::CircularBuffer cb_out0(cb_id_out0);
    experimental::Noc noc;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t padded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    const uint32_t n_block_reps = get_arg_val<uint32_t>(3);

    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t unpadded_X_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const uint32_t num_tiles_per_row = padded_X_size >> (FLOAT32_DTYPE ? 7 : 6);

    const auto s = TensorAccessor(dst_args, dst_addr, unpadded_X_size);

    auto pop_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_out0.wait_front(num_tiles_per_row);
            cb_out0.pop_front(num_tiles_per_row);
        }
    };

    auto write_block = [&](uint32_t base_stick_id, uint32_t num_rows) {
        uint32_t padding_rows = (tile_height - num_rows) & 31;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_out0.wait_front(num_tiles_per_row * has_rows);
        uint32_t l1_read_addr = cb_out0.get_read_ptr();
        for (uint32_t k = 0; k < num_rows; k++) {
            // Write out tmp buffer
            uint32_t src_offset = l1_read_addr - cb_out0.get_read_ptr();
            noc.async_write(
                cb_out0,
                s,
                unpadded_X_size,
                {.offset_bytes = src_offset},
                {.page_id = base_stick_id + k, .offset_bytes = 0});

            noc.async_write_barrier();
            l1_read_addr += padded_X_size;
        }
        cb_out0.pop_front(num_tiles_per_row * has_rows);
    };

    uint32_t stick_id = start_stick_id;
    uint32_t rt_arg_idx = 4;
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
