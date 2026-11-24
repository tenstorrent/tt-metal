// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Sharded-input reader: mixes are already resident in this core's L1 (the mixes DFB is borrowed from
// the shard), so we only fetch the 8 constant tiles from DRAM and signal the mixes as ready.
void kernel_main() {
    const uint32_t num_token_tiles = get_arg(args::num_token_tiles);

    constexpr uint32_t num_const_tiles = 8;

    Noc noc;
    DataflowBuffer dfb_c(dfb::consts);
    DataflowBuffer dfb_m(dfb::mixes);

    const uint32_t consts_page = dfb_c.get_entry_size();
    const auto s_consts = TensorAccessor(tensor::consts);

    for (uint32_t i = 0; i < num_const_tiles; ++i) {
        dfb_c.reserve_back(1);
        noc.async_read(s_consts, dfb_c, consts_page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_c.push_back(1);
    }

    dfb_m.push_back(num_token_tiles);  // mixes already in L1 via the borrowed DFB
}
