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
    uint32_t in0_tensor_tile_id = get_arg(args::in0_tensor_tile_id);

    // COMPILE TIME ARGS
    // READER COMPILE TIME ARGS
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t out_num_blocks_per_tensor = get_arg(args::out_num_blocks_per_tensor);

    DataflowBuffer dfb_in0(dfb::k);   // K heads, transposed by compute before the writer drains them
    DataflowBuffer dfb_in1(dfb::qv);  // Q and V heads, drained by the writer directly
    const uint32_t single_tile_size_bytes = dfb_in0.get_tile_size();
    const auto s0 = TensorAccessor(tensor::input);

    DataflowBuffer* dfb_cur;
    uint32_t l1_write_addr;
    uint32_t out_num_tensors = 3;
    for (uint32_t out_tensor = 0; out_tensor < out_num_tensors; out_tensor++) {
        // Q or V heads
        if (out_tensor == 0 or out_tensor == 2) {
            dfb_cur = &dfb_in1;
        }
        // V heads
        else if (out_tensor == 1) {
            dfb_cur = &dfb_in0;
        }

        l1_write_addr = dfb_cur->get_write_ptr();
        for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
            dfb_cur->reserve_back(block_size);
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
            dfb_cur->push_back(block_size);
        }
    }
}
