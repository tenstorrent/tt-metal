// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    std::uint32_t num_tiles = get_arg(args::num_tiles);
    std::uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_in0(dfb::in0);

#ifdef INPUT_SHARDED
    dfb_in0.reserve_back(num_tiles);
    dfb_in0.push_back(num_tiles);
#else
    // ublocks size defined in tiles
    constexpr std::uint32_t onetile = 1;
    Noc noc;
    const auto s = TensorAccessor(tensor::input);
    const std::uint32_t tile_bytes = dfb_in0.get_tile_size();

// read a ublock of tiles from src to the DFB, and then push the ublock to unpacker
#ifdef BACKWARDS
    std::uint32_t end_id = start_id - num_tiles;
    for (std::uint32_t i = start_id; i != end_id; --i) {
#else
    std::uint32_t end_id = start_id + num_tiles;
    for (std::uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb_in0.reserve_back(onetile);
        noc.async_read(s, dfb_in0, tile_bytes, {.page_id = i}, {});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
    }
#endif
}
