// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "dataflow_api.h"

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_tile_id = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_tile_id_with_transpose = get_arg_val<uint32_t>(4);

    // COMPILE TIME ARGS
    // WRITER COMPILE TIME ARGS
    constexpr bool block_size_is_one = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr uint32_t out_num_blocks_per_tensor = get_compile_time_arg_val(2);
    constexpr uint32_t out_num_c_per_block = get_compile_time_arg_val(3);
    constexpr uint32_t out_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t out_h_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t out_HtWt = get_compile_time_arg_val(6);
    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t cb_id_out1 = 1;  // same as cb_id_in1
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);
    const auto sq = TensorAccessor(q_args, q_tensor_addr, single_tile_size_bytes);
    const auto sk = TensorAccessor(k_args, k_tensor_addr, single_tile_size_bytes);
    const auto sv = TensorAccessor(v_args, v_tensor_addr, single_tile_size_bytes);

    uint32_t l1_read_addr_out0 = get_read_ptr(cb_id_out0);
    uint32_t l1_read_addr_out1 = get_read_ptr(cb_id_out1);
    uint32_t out_num_tiles_read_out0 = block_size;
    uint32_t out_num_tiles_read_out1 = block_size;
    uint32_t out_tensor_current_tile_id_along_c;
    uint32_t out_tensor_current_tile_id;

    // Create q head
    out_tensor_current_tile_id_along_c = out_tensor_tile_id;
    for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
        if constexpr (!block_size_is_one) {
            cb_wait_front(cb_id_out1, out_num_tiles_read_out1);
            out_num_tiles_read_out1 += block_size;
        }

        for (uint32_t c_dim_idx = 0; c_dim_idx < out_num_c_per_block; c_dim_idx++) {
            out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < out_w_tiles; w_dim++) {
                if constexpr (block_size_is_one) {
                    cb_wait_front(cb_id_out1, out_num_tiles_read_out1);
                    out_num_tiles_read_out1++;
                }

                noc_async_write_tile(out_tensor_current_tile_id, sq, l1_read_addr_out1);
                l1_read_addr_out1 += single_tile_size_bytes;
                out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += out_HtWt;
        }
    }

    // Create k head
    out_tensor_current_tile_id = out_tensor_tile_id_with_transpose;
    for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
        if constexpr (!block_size_is_one) {
            cb_wait_front(cb_id_out0, out_num_tiles_read_out0);
            out_num_tiles_read_out0 += block_size;
        }

        for (uint32_t c_dim_idx = 0; c_dim_idx < out_num_c_per_block; c_dim_idx++) {
            for (uint32_t w_dim = 0; w_dim < out_w_tiles; w_dim++) {
                if constexpr (block_size_is_one) {
                    cb_wait_front(cb_id_out0, out_num_tiles_read_out0);
                    out_num_tiles_read_out0++;
                }

                noc_async_write_tile(out_tensor_current_tile_id, sk, l1_read_addr_out0);
                l1_read_addr_out0 += single_tile_size_bytes;
                out_tensor_current_tile_id += out_h_tiles;
            }
        }
    }

    // Create v head
    out_tensor_current_tile_id_along_c = out_tensor_tile_id;
    for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
        if constexpr (!block_size_is_one) {
            cb_wait_front(cb_id_out1, out_num_tiles_read_out1);
            out_num_tiles_read_out1 += block_size;
        }

        for (uint32_t c_dim_idx = 0; c_dim_idx < out_num_c_per_block; c_dim_idx++) {
            out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < out_w_tiles; w_dim++) {
                if constexpr (block_size_is_one) {
                    cb_wait_front(cb_id_out1, out_num_tiles_read_out1);
                    out_num_tiles_read_out1++;
                }

                noc_async_write_tile(out_tensor_current_tile_id, sv, l1_read_addr_out1);
                l1_read_addr_out1 += single_tile_size_bytes;
                out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += out_HtWt;
        }
    }

    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, out_num_tiles_read_out0);
    cb_pop_front(cb_id_out1, out_num_tiles_read_out1);
}
