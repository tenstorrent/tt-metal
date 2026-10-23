// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
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

#ifdef INPUT_ROW_OFFSET
    {
        // The first input row to copy is read on device; it lands in the tile slot the first read reuses.
        constexpr auto Wt = get_arg(args::Wt);
        constexpr auto row_offset_page_bytes = get_arg(args::row_offset_page_bytes);
        const auto offset_gen = TensorAccessor(tensor::row_offset);
        dfb_in0.reserve_back(onetile);
        const std::uint32_t offset_l1 = dfb_in0.get_write_ptr();
#ifdef ARCH_QUASAR
        noc.async_read(offset_gen, dfb_in0, row_offset_page_bytes, {.page_id = 0}, {});
#else
        noc.async_read(offset_gen, CoreLocalMem<std::uint32_t>(offset_l1), row_offset_page_bytes, {.page_id = 0}, {});
#endif
        noc.async_read_barrier();
        const std::uint32_t row_offset = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(offset_l1)[0];
        start_id += (row_offset / 32) * Wt;
    }
#endif

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
