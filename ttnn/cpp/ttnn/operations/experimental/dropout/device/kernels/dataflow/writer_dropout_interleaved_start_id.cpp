// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

    DataflowBuffer dfb_out(dfb::out);

#ifdef OUT_SHARDED
    dfb_out.wait_front(num_tiles);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(tensor::output);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb_out.wait_front(onetile);
        uint32_t l1_read_addr = dfb_out.get_read_ptr();
        noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), s, dfb_out.get_tile_size(), {}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
#endif
}
