// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // READER RUNTIME ARGS
    uint32_t num_blocks = get_arg(args::num_blocks);
    uint32_t in0_tensor_tile_id = get_arg(args::in0_tensor_tile_id);
    uint32_t in1_tensor_tile_id = get_arg(args::in1_tensor_tile_id);

    // COMPILE TIME ARGS
    // interleaved accessor args
    // READER COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_arg(args::q_num_tiles);

    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(tensor::input);

    DataflowBuffer dfb_qv(dfb::qv);  // dfb for Q, V heads
    uint32_t tile_bytes = dfb_qv.get_tile_size();

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Q
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            dfb_qv.reserve_back(onetile);
            uint32_t l1_write_addr = dfb_qv.get_write_ptr();
            noc.async_read(s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes, {.page_id = in0_tensor_tile_id}, {});
            noc.async_read_barrier();
            dfb_qv.push_back(onetile);
            in0_tensor_tile_id++;
        }
    }
}
