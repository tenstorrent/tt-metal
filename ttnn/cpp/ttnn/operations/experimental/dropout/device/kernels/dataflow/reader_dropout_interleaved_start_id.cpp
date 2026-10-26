// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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

    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(tensor::input);

    DataflowBuffer dfb_in0(dfb::in0);

// read a ublock of tiles from src to the DFB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb_in0.reserve_back(onetile);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        noc.async_read(s, CoreLocalMem<uint32_t>(l1_write_addr), dfb_in0.get_tile_size(), {.page_id = i}, {});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
    }
}
