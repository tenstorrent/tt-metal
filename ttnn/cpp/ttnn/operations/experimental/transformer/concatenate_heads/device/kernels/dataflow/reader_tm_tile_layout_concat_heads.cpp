// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

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
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);

    CircularBuffer cb_in0(cb_id_in0);
    uint32_t l1_write_addr_in0 = cb_in0.get_write_ptr();
    uint32_t in0_tensor_current_tile_id = in0_tensor_tile_id;

    for (uint32_t c_dim = 0; c_dim < in0_c; c_dim++) {
        cb_in0.reserve_back(in0_w_tiles);

        in0_tensor_current_tile_id = in0_tensor_tile_id;
        for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
            noc.async_read(
                s0,
                CoreLocalMem<uint32_t>(l1_write_addr_in0),
                single_tile_size_bytes,
                {.page_id = in0_tensor_current_tile_id},
                {});
            l1_write_addr_in0 += single_tile_size_bytes;
            in0_tensor_current_tile_id++;
        }
        in0_tensor_tile_id += in0_HtWt;
        noc.async_read_barrier();
        cb_in0.push_back(in0_w_tiles);
    }
}
