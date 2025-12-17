// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    // RUNTIME ARGS
    // READER
    uint32_t rt_args_idx = 0;
    // in1 tensor args
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // batch args
    const uint32_t batch = get_arg_val<uint32_t>(rt_args_idx++);

    // WRITER
    // out tensor args
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(2);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(3);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(6);
    // batch args
    constexpr uint32_t bcast_B = get_compile_time_arg_val(7);
    constexpr uint32_t KtNt = get_compile_time_arg_val(8);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(9);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(14);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_subblocks_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_num_subblocks_h = get_compile_time_arg_val(17);
    // batch args
    constexpr uint32_t MtNt = get_compile_time_arg_val(18);

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t one_tile = 1;
    // WRITER
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;

    constexpr auto in1_args = TensorAccessorArgs<19>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();

#ifdef IN1_SHARDED
    const uint32_t in1_num_tiles = batch * num_blocks * in1_block_h * in1_block_w;
    cb_reserve_back(cb_id_in1, in1_num_tiles);
    cb_push_back(cb_id_in1, in1_num_tiles);
#else
    const uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    const auto s1 = TensorAccessor(in1_args, in1_tensor_addr, in1_single_tile_size_bytes);
    uint32_t l1_write_addr_in1;
#endif  // IN1_SHARDED

#ifndef OUT_SHARDED
    const uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    const auto s = TensorAccessor(out_args, out_tensor_addr, output_single_tile_size_bytes);
#endif  // OUT_SHARDED

#if not defined IN1_SHARDED or not defined OUT_SHARDED
    for (uint32_t b = 0; b < batch; ++b) {
#ifndef IN1_SHARDED
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

#ifdef INTERMEDIATE_CB_READ
            constexpr uint32_t in1_intermediate_cb_index = tt::CBIndex::c_9;
            cb_reserve_back(in1_intermediate_cb_index, one_tile);
            uint32_t l1_write_addr_helper = get_write_ptr(in1_intermediate_cb_index);
#endif  // INTERMEDIATE_CB_READ

            l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; ++h) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; ++w) {
#ifndef INTERMEDIATE_CB_READ
                    noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
#else
                    noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_helper);
                    noc_async_read_barrier();
                    memcpy(
                        /*dst=*/reinterpret_cast<void*>(l1_write_addr_in1),
                        /*src=*/reinterpret_cast<const void*>(l1_write_addr_helper),
                        /*size=*/in1_single_tile_size_bytes);
#endif  // INTERMEDIATE_CB_READ
                    l1_write_addr_in1 += in1_single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            noc_async_read_barrier();

            cb_push_back(cb_id_in1, in1_block_num_tiles);
#ifdef INTERMEDIATE_CB_READ
            // Clean up helper CB
            cb_push_back(in1_intermediate_cb_index, one_tile);
            cb_wait_front(in1_intermediate_cb_index, one_tile);
            cb_pop_front(in1_intermediate_cb_index, one_tile);
#endif  // INTERMEDIATE_CB_READ
        }
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }
#endif  // IN1_SHARDED

#ifndef OUT_SHARDED
        // WRITER
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; ++sbh) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; ++sbw) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);

                        l1_read_addr += output_single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
#endif  // OUT_SHARDED
    }
#endif  // not defined IN1_SHARDED or not defined OUT_SHARDED

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out0, batch * out_num_subblocks_h * out_num_subblocks_w * out_subblock_w * out_subblock_h);
#endif  // OUT_SHARDED
}
