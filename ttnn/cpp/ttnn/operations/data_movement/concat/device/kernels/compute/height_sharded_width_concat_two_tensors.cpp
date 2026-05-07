// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/transpose_wh.h"
#include "experimental/circular_buffer.h"

template <uint32_t BatchSize = 1>
FORCE_INLINE void transpose(
    uint32_t cb_in_id, uint32_t cb_out_id, experimental::CircularBuffer& cb_in, experimental::CircularBuffer& cb_out) {
    cb_in.wait_front(BatchSize);

    tile_regs_acquire();
    tile_regs_wait();

    cb_out.reserve_back(BatchSize);

    transpose_wh_init_short(cb_in_id);
    for (uint32_t i = 0; i < BatchSize; i++) {
        transpose_wh_tile(cb_in_id, i, i);
        pack_tile(i, cb_out_id);
    }

    tile_regs_commit();
    tile_regs_release();

    cb_out.push_back(BatchSize);
    cb_in.pop_front(BatchSize);
}

void kernel_main() {
    constexpr uint32_t input0_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input0_transpose_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input1_transpose_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t concat_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t output_transpose_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(6);

    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(7);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(8);
    constexpr uint32_t input1_num_tiles_height = get_compile_time_arg_val(9);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(10);

    constexpr uint32_t tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t groups = get_compile_time_arg_val(12);
    constexpr uint32_t MAX_BATCH_SIZE = get_compile_time_arg_val(13);

    experimental::CircularBuffer input0_cb(input0_cb_id);
    experimental::CircularBuffer input1_cb(input1_cb_id);
    experimental::CircularBuffer input0_transpose_cb(input0_transpose_cb_id);
    experimental::CircularBuffer input1_transpose_cb(input1_transpose_cb_id);
    experimental::CircularBuffer concat_cb(concat_cb_id);
    experimental::CircularBuffer output_transpose_cb(output_transpose_cb_id);
    experimental::CircularBuffer output_cb(output_cb_id);

    transpose_wh_init(input0_cb_id, input0_transpose_cb_id);

    constexpr uint32_t output_num_tiles_width = input0_num_tiles_width + input1_num_tiles_width;

    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        reconfig_data_format_srca(input0_cb_id);
        pack_reconfig_data_format(input0_transpose_cb_id);
        if constexpr (input0_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<input0_num_tiles_width>(input0_cb_id, input0_transpose_cb_id, input0_cb, input0_transpose_cb);
        } else {
            for (uint32_t j = 0; j < input0_num_tiles_width; j++) {
                transpose(input0_cb_id, input0_transpose_cb_id, input0_cb, input0_transpose_cb);
            }
        }
        if constexpr (input1_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<input1_num_tiles_width>(input1_cb_id, input1_transpose_cb_id, input1_cb, input1_transpose_cb);
        } else {
            for (uint32_t j = 0; j < input1_num_tiles_width; j++) {
                transpose(input1_cb_id, input1_transpose_cb_id, input1_cb, input1_transpose_cb);
            }
        }

        reconfig_data_format_srca(concat_cb_id);
        pack_reconfig_data_format(output_transpose_cb_id);
        if constexpr (output_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<output_num_tiles_width>(concat_cb_id, output_transpose_cb_id, concat_cb, output_transpose_cb);
        } else {
            for (uint32_t j = 0; j < output_num_tiles_width; j++) {
                transpose(concat_cb_id, output_transpose_cb_id, concat_cb, output_transpose_cb);
            }
        }
    }
}
