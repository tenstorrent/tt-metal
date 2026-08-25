// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t num_tiles = get_arg(args::num_tiles);
#ifndef OUT_SHARDED
    // Unread when the output is sharded: tiles land in the resident output shard, no page writes.
    uint32_t start_id = get_arg(args::start_id);
#endif

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_out(dfb::out);

#ifndef OUT_SHARDED
    const auto s = TensorAccessor(tensor::dst);
#endif

#ifdef DECODE_MODE
    uint32_t cos_sin_offset = get_arg(args::cos_sin_offset);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t Wbytes = get_arg(args::Wbytes);

    DataflowBuffer dfb_untilized_cos(dfb::untilized_cos);
    DataflowBuffer dfb_untilized_cos_sync(dfb::untilized_cos_sync);
    DataflowBuffer dfb_untilized_sin(dfb::untilized_sin);
    DataflowBuffer dfb_untilized_sin_sync(dfb::untilized_sin_sync);

    dfb_untilized_sin.wait_front(Wt);
    dfb_untilized_sin_sync.reserve_back(Wt);
    uint32_t sin_l1_read_addr = dfb_untilized_sin.get_read_ptr() + cos_sin_offset;
    uint32_t sin_l1_write_addr = dfb_untilized_sin.get_read_ptr();
    noc.async_read(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(sin_l1_write_addr),
        Wbytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
         .noc_y = (uint32_t)my_y[noc.get_noc_id()],
         .addr = sin_l1_read_addr},
        {});
    noc.async_read_barrier();
    dfb_untilized_sin_sync.push_back(Wt);

    dfb_untilized_cos.wait_front(Wt);
    dfb_untilized_cos_sync.reserve_back(Wt);
    uint32_t cos_l1_read_addr = dfb_untilized_cos.get_read_ptr() + cos_sin_offset;
    uint32_t cos_l1_write_addr = dfb_untilized_cos.get_read_ptr();
    noc.async_read(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(cos_l1_write_addr),
        Wbytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
         .noc_y = (uint32_t)my_y[noc.get_noc_id()],
         .addr = cos_l1_read_addr},
        {});
    noc.async_read_barrier();
    dfb_untilized_cos_sync.push_back(Wt);
#endif

#ifdef OUT_SHARDED
    dfb_out.wait_front(num_tiles);
#else
    constexpr uint32_t out_tile_size = get_tile_size(dfb::out);
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out.wait_front(onetile);
        uint32_t l1_read_addr = dfb_out.get_read_ptr();

        noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), s, out_tile_size, {}, {.page_id = i});

        noc.async_write_barrier();

        dfb_out.pop_front(onetile);
    }
#endif
}
