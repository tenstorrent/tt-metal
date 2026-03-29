// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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
    constexpr uint32_t last_ktile_h = get_compile_time_arg_val(7);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
    // batch args
    constexpr uint32_t bcast_B = get_compile_time_arg_val(9);
    constexpr uint32_t MtKt = get_compile_time_arg_val(10);

    constexpr auto in0_args = TensorAccessorArgs<11>();

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t one_tile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);

#ifdef IN0_SHARDED
    const uint32_t in0_num_tiles = batch * num_blocks * in0_block_h * in0_block_w;
    cb_in0.reserve_back(in0_num_tiles);
    cb_in0.push_back(in0_num_tiles);
#else

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr const uint32_t in0_tile_hw = get_tile_hw(cb_id_in0);

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, in0_single_tile_size_bytes);

#ifdef INTERMEDIATE_CB_READ
    constexpr uint32_t in0_intermediate_cb_index = get_named_compile_time_arg_val("cb_in0_intermediate");
    experimental::CircularBuffer cb_helper(in0_intermediate_cb_index);
#endif

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_in0.reserve_back(in0_block_num_tiles);

#ifdef INTERMEDIATE_CB_READ
            cb_helper.reserve_back(one_tile);
#endif  // INTERMEDIATE_CB_READ

            uint32_t in0_write_offset = 0;

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; ++h) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; ++w) {
#ifndef INTERMEDIATE_CB_READ
                    noc.async_read(
                        s0,
                        cb_in0,
                        in0_single_tile_size_bytes,
                        {.page_id = in0_tensor_tile_id},
                        {.offset_bytes = in0_write_offset});
#else
                    noc.async_read(
                        s0,
                        cb_helper,
                        in0_single_tile_size_bytes,
                        {.page_id = in0_tensor_tile_id},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    memcpy(
                        /*dst=*/reinterpret_cast<void*>(cb_in0.get_write_ptr() + in0_write_offset),
                        /*src=*/reinterpret_cast<const void*>(cb_helper.get_write_ptr()),
                        /*size=*/in0_single_tile_size_bytes);
#endif  // INTERMEDIATE_CB_READ

                    // Zero out padded regions for the very last tile
                    if constexpr (last_ktile_w > 0) {
                        if ((block == num_blocks - 1) && (w == in0_block_w - 1)) {
                            noc.async_read_barrier();
                            constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);
                            pad_last_ktile<in0_data_format, last_ktile_w>(cb_in0.get_write_ptr() + in0_write_offset);
                        }
                    }
                    if constexpr (last_ktile_h > 0) {
                        if ((block == num_blocks - 1) && (w == in0_block_w - 1)) {
                            noc.async_read_barrier();
                            constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);
                            pad_last_transposed_ktile<in0_data_format, last_ktile_h>(
                                cb_in0.get_write_ptr() + in0_write_offset);
                        }
                    }

                    in0_write_offset += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            noc.async_read_barrier();

            cb_in0.push_back(in0_block_num_tiles);

#ifdef INTERMEDIATE_CB_READ
            // Clean up helper CB
            cb_helper.push_back(one_tile);
            cb_helper.wait_front(one_tile);
            cb_helper.pop_front(one_tile);
#endif  // INTERMEDIATE_CB_READ
        }
        in0_tensor_start_tile_id += MtKt;
    }
#endif  // IN0_SHARDED
}
