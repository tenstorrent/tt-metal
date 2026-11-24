// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Loads the 8 host-prepared constant tiles once (persistent), then streams the mixes
// token-tiles. See the program factory for the DFB layout.
void kernel_main() {
    const uint32_t num_token_tiles = get_arg(args::num_token_tiles);
    const uint32_t start_tile = get_arg(args::start_tile);  // this core's first mixes page

    constexpr uint32_t num_const_tiles = 8;

    Noc noc;
    DataflowBuffer dfb_c(dfb::consts);
    DataflowBuffer dfb_m(dfb::mixes);

    const uint32_t mixes_page = dfb_m.get_entry_size();
    const uint32_t consts_page = dfb_c.get_entry_size();

    const auto s_mixes = TensorAccessor(tensor::mixes);
    const auto s_consts = TensorAccessor(tensor::consts);

    // Constants: load once, never popped by compute -> resident for every token-tile.
    for (uint32_t i = 0; i < num_const_tiles; ++i) {
        dfb_c.reserve_back(1);
        noc.async_read(s_consts, dfb_c, consts_page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_c.push_back(1);
    }

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        dfb_m.reserve_back(1);
        noc.async_read(s_mixes, dfb_m, mixes_page, {.page_id = start_tile + t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_m.push_back(1);
    }
}
