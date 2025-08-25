// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "dataflow_api.h"
#include "tensix_types.h"

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_tile_id = get_arg_val<uint32_t>(1);

    // COMPILE TIME ARGS
    // WRITER COMPILE TIME ARGS
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_c = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_id_out0 = 0;  // same as cb_id_in0
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    const auto s = TensorAccessor(out_args, out_tensor_addr, single_tile_size_bytes);

    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
    uint32_t out_num_tiles_read = in0_w_tiles;

    for (uint32_t c_dim = 0; c_dim < in0_c; c_dim++) {
        cb_wait_front(cb_id_out0, out_num_tiles_read);

        for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
            noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
            l1_read_addr += single_tile_size_bytes;
            out_tensor_tile_id++;
        }
        out_num_tiles_read += in0_w_tiles;
    }

    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, out_num_tiles_read);
}
