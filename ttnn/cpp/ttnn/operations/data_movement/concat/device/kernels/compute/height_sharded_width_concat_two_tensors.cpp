// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"

template <uint32_t BatchSize = 1>
FORCE_INLINE void transpose(uint32_t cb_in_id, uint32_t cb_out_id, DataflowBuffer& dfb_in, DataflowBuffer& dfb_out) {
    dfb_in.wait_front(BatchSize);

    tile_regs_acquire();
    tile_regs_wait();

    dfb_out.reserve_back(BatchSize);

    transpose_init(cb_in_id);
    for (uint32_t i = 0; i < BatchSize; i++) {
        transpose_tile(cb_in_id, i, i);
        pack_tile(i, cb_out_id);
    }

    tile_regs_commit();
    tile_regs_release();

    dfb_out.push_back(BatchSize);
    dfb_in.pop_front(BatchSize);
}

void kernel_main() {
    constexpr uint32_t input0_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input0_transpose_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input1_transpose_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t concat_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t output_transpose_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t output_dfb_id = get_compile_time_arg_val(6);

    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(7);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(8);
    constexpr uint32_t input1_num_tiles_height = get_compile_time_arg_val(9);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(10);

    constexpr uint32_t tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t groups = get_compile_time_arg_val(12);
    constexpr uint32_t MAX_BATCH_SIZE = get_compile_time_arg_val(13);

    DataflowBuffer input0_dfb(input0_cb_id);
    DataflowBuffer input1_dfb(input1_cb_id);
    DataflowBuffer input0_transpose_dfb(input0_transpose_cb_id);
    DataflowBuffer input1_transpose_dfb(input1_transpose_cb_id);
    DataflowBuffer concat_dfb(concat_cb_id);
    DataflowBuffer output_transpose_dfb(output_transpose_cb_id);
    DataflowBuffer output_dfb(output_dfb_id);

    compute_kernel_hw_startup(input0_cb_id, input0_transpose_cb_id);
    transpose_init(input0_cb_id);

    constexpr uint32_t output_num_tiles_width = input0_num_tiles_width + input1_num_tiles_width;

    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        reconfig_data_format_srca(input0_cb_id);
        pack_reconfig_data_format(input0_transpose_cb_id);
        if constexpr (input0_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<input0_num_tiles_width>(input0_cb_id, input0_transpose_cb_id, input0_dfb, input0_transpose_dfb);
        } else {
            for (uint32_t j = 0; j < input0_num_tiles_width; j++) {
                transpose(input0_cb_id, input0_transpose_cb_id, input0_dfb, input0_transpose_dfb);
            }
        }
        if constexpr (input1_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<input1_num_tiles_width>(input1_cb_id, input1_transpose_cb_id, input1_dfb, input1_transpose_dfb);
        } else {
            for (uint32_t j = 0; j < input1_num_tiles_width; j++) {
                transpose(input1_cb_id, input1_transpose_cb_id, input1_dfb, input1_transpose_dfb);
            }
        }

        reconfig_data_format_srca(concat_cb_id);
        pack_reconfig_data_format(output_transpose_cb_id);
        if constexpr (output_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<output_num_tiles_width>(concat_cb_id, output_transpose_cb_id, concat_dfb, output_transpose_dfb);
        } else {
            for (uint32_t j = 0; j < output_num_tiles_width; j++) {
                transpose(concat_cb_id, output_transpose_cb_id, concat_dfb, output_transpose_dfb);
            }
        }
    }
}
