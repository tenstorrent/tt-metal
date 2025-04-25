// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t batch_size = get_compile_time_arg_val(0);
    constexpr uint32_t input_height = get_compile_time_arg_val(1);
    constexpr uint32_t input_width = get_compile_time_arg_val(2);
    constexpr uint32_t stride_height = get_compile_time_arg_val(3);
    constexpr uint32_t stride_width = get_compile_time_arg_val(4);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(5);
    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(6);
    constexpr uint32_t element_size = get_compile_time_arg_val(7);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(8);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block_id = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t nblocks_per_core = get_arg_val<uint32_t>(3);
    constexpr bool dst_is_dram = true;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;

    const InterleavedAddrGen<dst_is_dram> d = {.bank_base_address = dst_addr, .page_size = stick_nbytes};

    uint32_t OH = input_height / stride_height;
    uint32_t OW = input_width / stride_width;
    uint32_t patch_size = stride_height * stride_width;
    const uint32_t W_PAD = TILE_HEIGHT;
    const uint32_t C_PAD = TILE_WIDTH * element_size * ntiles_per_row;

    uint32_t tile_cols = (input_width + W_PAD - 1) / W_PAD;
    uint32_t tiles_per_batch = input_height * tile_cols;

    uint32_t end_block_id = start_block_id + num_blocks;
    for (uint32_t i = start_block_id; i < end_block_id; ++i) {
        cb_wait_front(cb_id_in1, ntiles_per_row);
        uint64_t l1_read_addr = get_read_ptr(cb_id_in1);

        int b = i / tiles_per_batch;
        int bh_index = i % tiles_per_batch;
        int h = bh_index / tile_cols;
        int tile_col = bh_index % tile_cols;

        int w_start = tile_col * W_PAD;
        int w_end = (w_start + W_PAD > input_width) ? input_width : w_start + W_PAD;
        for (int w_local = 0; w_start + w_local < w_end; ++w_local) {
            int w = w_start + w_local;
            uint64_t src = l1_read_addr + w_local * C_PAD;

            int oh = h / stride_height;
            int ow = w / stride_width;
            int kh = h % stride_height;
            int kw = w % stride_width;

            int dst_row = b * OH * OW + oh * OW + ow;
            int dst_col = (kh * stride_width + kw);

            uint64_t dst = dst_row * patch_size + dst_col;
            uint64_t dst_addr_ = get_noc_addr(dst, d);
            noc_async_write(src, dst_addr_, stick_nbytes);
        }
        noc_async_write_barrier();

        cb_pop_front(cb_id_in1, ntiles_per_row);
    }
}
