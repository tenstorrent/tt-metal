// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    Noc noc;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t input_num_blocks_w_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t input_num_blocks_h = get_arg_val<uint32_t>(3);
    uint32_t input_total_blocks_w = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in2 = 2;
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_id_in2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();

    constexpr uint32_t cb_id_in0 = 0;
    CircularBuffer cb_in0(cb_id_in0);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(src_args, src_addr);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t block_h_id = 0; block_h_id < input_num_blocks_h; block_h_id++) {
        uint32_t end_id = start_id + input_num_blocks_w_per_core;
        for (uint32_t i = start_id; i < end_id; i++) {
            cb_in0.reserve_back(onetile);
            uint32_t l1_write_addr = cb_in0.get_write_ptr();
            noc.async_read(
                s,
                CoreLocalMem<uint32_t>(l1_write_addr),
                get_tile_size(cb_id_in0),
                {.page_id = (block_h_id * input_total_blocks_w) + i},
                {});
            noc.async_read_barrier();
            cb_in0.push_back(onetile);
        }
    }
}
