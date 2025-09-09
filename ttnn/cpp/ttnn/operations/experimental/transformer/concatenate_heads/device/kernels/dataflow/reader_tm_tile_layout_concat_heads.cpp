// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_tile_id = get_arg_val<uint32_t>(1);

    // COMPILE TIME ARGS
    // READER COMPILE TIME ARGS
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_c = get_compile_time_arg_val(1);
    constexpr uint32_t in0_HtWt = get_compile_time_arg_val(2);
    constexpr auto in0_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_id_in0 = 0;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, single_tile_size_bytes);

    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t in0_tensor_current_tile_id = in0_tensor_tile_id;

    for (uint32_t c_dim = 0; c_dim < in0_c; c_dim++) {
        cb_reserve_back(cb_id_in0, in0_w_tiles);

        in0_tensor_current_tile_id = in0_tensor_tile_id;
        for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
            noc_async_read_tile(in0_tensor_current_tile_id, s0, l1_write_addr_in0);
            l1_write_addr_in0 += single_tile_size_bytes;
            in0_tensor_current_tile_id++;
        }
        in0_tensor_tile_id += in0_HtWt;
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, in0_w_tiles);
    }
}
