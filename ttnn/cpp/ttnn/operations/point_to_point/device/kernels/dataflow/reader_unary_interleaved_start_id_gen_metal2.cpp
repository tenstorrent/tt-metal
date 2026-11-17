// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const uint32_t page_bytes = get_arg(args::page_bytes);
    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);

    // read a ublock of tiles from src to DFB, and then push the ublock to unpacker
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_in0.reserve_back(onetile);
        noc.async_read(s, dfb_in0, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
    }
}
