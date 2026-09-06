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

    constexpr uint32_t Wt = get_arg(args::Wt);

    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t noop = get_arg(args::noop);

    if (noop == 1) {
        return;  // Early exit, no work done
    }

    const auto s = TensorAccessor(tensor::src);

    DataflowBuffer dfb_in(dfb::in);

    const uint32_t tile_bytes = dfb_in.get_tile_size();

    // read a ublock of tiles from src to the DFB, and then push the ublock to unpacker
    uint32_t tile_id = start_tile_id;
    for (uint32_t row_num = 0; row_num < num_rows; ++row_num) {
        dfb_in.reserve_back(Wt);
        uint32_t l1_write_addr = dfb_in.get_write_ptr();
        for (uint32_t w = 0; w < Wt; ++w) {
            noc.async_read(s, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes, {.page_id = tile_id}, {});
            l1_write_addr += tile_bytes;
            tile_id++;
        }
        noc.async_read_barrier();
        dfb_in.push_back(Wt);
    }
}
