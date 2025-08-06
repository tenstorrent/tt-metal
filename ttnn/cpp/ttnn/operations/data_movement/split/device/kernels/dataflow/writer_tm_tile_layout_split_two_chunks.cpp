// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <array>

#include "dataflow_api.h"
#include "tensix_types.h"

// #define DEBUG

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t out_tensor_tile_id = get_arg_val<uint32_t>(0);
    uint32_t out0_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t out1_tensor_addr = get_arg_val<uint32_t>(2);
    bool out0_only = (bool)get_arg_val<uint32_t>(3);
    bool out1_only = (bool)get_arg_val<uint32_t>(4);

    // WRITER COMPILE TIME ARGS
    constexpr uint32_t out_num_tiles_per_tensor_y = get_compile_time_arg_val(0);
    constexpr uint32_t out_num_tiles_per_tensor_x = get_compile_time_arg_val(1);
    constexpr uint32_t z = get_compile_time_arg_val(2);
    constexpr uint32_t z_stride = get_compile_time_arg_val(3);
    constexpr uint32_t y_stride = get_compile_time_arg_val(4);
    constexpr auto out0_tensor_args = TensorAccessorArgs<5>();
    constexpr auto out1_tensor_args = TensorAccessorArgs<out0_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_out0 = 0;  // same as cb_id_in0
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(out0_tensor_args, out0_tensor_addr, single_tile_size_bytes);
    const auto s1 = TensorAccessor(out1_tensor_args, out1_tensor_addr, single_tile_size_bytes);

    if (!out1_only) {
        uint32_t z_stride_cum = 0;
        for (uint32_t k = 0; k < z; k++) {
            uint32_t y_stride_cum = 0;
            for (uint32_t j = 0; j < out_num_tiles_per_tensor_y; j++) {
                for (uint32_t i = 0; i < out_num_tiles_per_tensor_x; i++) {
                    uint32_t tile_id = y_stride_cum + z_stride_cum + i;
                    cb_wait_front(cb_id_out0, onetile);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                    noc_async_write_tile(tile_id + out_tensor_tile_id, s0, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_out0, onetile);
                }
                y_stride_cum += y_stride;
            }
            z_stride_cum += z_stride;
        }
    }
    if (!out0_only) {
        uint32_t z_stride_cum = 0;
        for (uint32_t k = 0; k < z; k++) {
            uint32_t y_stride_cum = 0;
            for (uint32_t j = 0; j < out_num_tiles_per_tensor_y; j++) {
                for (uint32_t i = 0; i < out_num_tiles_per_tensor_x; i++) {
                    uint32_t tile_id = y_stride_cum + z_stride_cum + i;
                    cb_wait_front(cb_id_out0, onetile);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                    noc_async_write_tile(tile_id + out_tensor_tile_id, s1, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_out0, onetile);
                }
                y_stride_cum += y_stride;
            }
            z_stride_cum += z_stride;
        }
    }
}
