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
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t out_num_blocks_per_tensor = get_compile_time_arg_val(1);
    constexpr auto in0_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, single_tile_size_bytes);

    uint32_t cb_id;
    uint32_t l1_write_addr;
    uint32_t out_num_tensors = 3;
    for (uint32_t out_tensor = 0; out_tensor < out_num_tensors; out_tensor++) {
        // Q or V heads
        if (out_tensor == 0 or out_tensor == 2) {
            cb_id = cb_id_in1;
        }
        // V heads
        else if (out_tensor == 1) {
            cb_id = cb_id_in0;
        }

        l1_write_addr = get_write_ptr(cb_id);
        for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
            cb_reserve_back(cb_id, block_size);
            for (uint32_t i = 0; i < block_size; i++) {
                noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr);
                l1_write_addr += single_tile_size_bytes;
                in0_tensor_tile_id++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id, block_size);
        }
    }
}
