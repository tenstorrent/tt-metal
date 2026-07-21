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
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t out_num_blocks_per_tensor = get_compile_time_arg_val(1);
    constexpr auto in0_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);

    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in1(cb_id_in1);

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

        CircularBuffer cb_cur(cb_id);
        l1_write_addr = cb_cur.get_write_ptr();
        for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
            cb_cur.reserve_back(block_size);
            for (uint32_t i = 0; i < block_size; i++) {
                noc.async_read(
                    s0,
                    CoreLocalMem<uint32_t>(l1_write_addr),
                    single_tile_size_bytes,
                    {.page_id = in0_tensor_tile_id},
                    {});
                l1_write_addr += single_tile_size_bytes;
                in0_tensor_tile_id++;
            }
            noc.async_read_barrier();
            cb_cur.push_back(block_size);
        }
    }
}
