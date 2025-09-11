// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_tile_id = get_arg_val<uint32_t>(3);
    uint32_t in1_tensor_tile_id = get_arg_val<uint32_t>(4);

    // COMPILE TIME ARGS
    // interleaved accessor args
    // READER COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_compile_time_arg_val(0);
    constexpr auto in0_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_qv = 1;  // cb for Q, V heads

    constexpr uint32_t onetile = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_qv);
    const DataFormat data_format = get_dataformat(cb_id_qv);

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, single_tile_size_bytes);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Q
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            cb_reserve_back(cb_id_qv, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_qv);
            noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_qv, onetile);
            in0_tensor_tile_id++;
        }
    }
}
