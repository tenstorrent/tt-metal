// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

void kernel_main() {
    // RUNTIME ARGS
    uint32_t rt_args_idx = 0;
    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // batch args
    const uint32_t batch = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in0_tensor_next_block_stride = get_compile_time_arg_val(2);
    // in0 block args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(3);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t last_ktile_w = get_compile_time_arg_val(6);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    // batch args
    constexpr uint32_t bcast_B = get_compile_time_arg_val(8);
    constexpr uint32_t MtKt = get_compile_time_arg_val(9);

    constexpr auto in0_args = TensorAccessorArgs<10>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t one_tile = 1;
#ifdef IN0_SHARDED
    const uint32_t in0_num_tiles = batch * num_blocks * in0_block_h * in0_block_w;
    cb_reserve_back(cb_id_in0, in0_num_tiles);
    cb_push_back(cb_id_in0, in0_num_tiles);
#else

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr const uint32_t in0_tile_hw = get_tile_hw(cb_id_in0);

    uint32_t l1_write_addr_in0;

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, in0_single_tile_size_bytes);

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);

#ifdef INTERMEDIATE_CB_READ
            constexpr uint32_t in0_intermediate_cb_index = tt::CBIndex::c_8;
            cb_reserve_back(in0_intermediate_cb_index, one_tile);
            uint32_t l1_write_addr_helper = get_write_ptr(in0_intermediate_cb_index);
#endif  // INTERMEDIATE_CB_READ

            l1_write_addr_in0 = get_write_ptr(cb_id_in0);

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; ++h) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; ++w) {
#ifndef INTERMEDIATE_CB_READ
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
#else
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_helper);
                    noc_async_read_barrier();
                    memcpy(
                        /*dst=*/reinterpret_cast<void*>(l1_write_addr_in0),
                        /*src=*/reinterpret_cast<const void*>(l1_write_addr_helper),
                        /*size=*/in0_single_tile_size_bytes);
#endif  // INTERMEDIATE_CB_READ

                    // Zero out padded regions for the very last tile
                    if constexpr (last_ktile_w > 0) {
                        if ((block == num_blocks - 1) && (w == in0_block_w - 1)) {
                            noc_async_read_barrier();
                            constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);
                            pad_last_ktile<in0_data_format, last_ktile_w>(l1_write_addr_in0);
                        }
                    }

                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            noc_async_read_barrier();

            cb_push_back(cb_id_in0, in0_block_num_tiles);

#ifdef INTERMEDIATE_CB_READ
            // Clean up helper CB
            cb_push_back(in0_intermediate_cb_index, one_tile);
            cb_wait_front(in0_intermediate_cb_index, one_tile);
            cb_pop_front(in0_intermediate_cb_index, one_tile);
#endif  // INTERMEDIATE_CB_READ
        }
        in0_tensor_start_tile_id += MtKt;
    }
#endif  // IN0_SHARDED
}
