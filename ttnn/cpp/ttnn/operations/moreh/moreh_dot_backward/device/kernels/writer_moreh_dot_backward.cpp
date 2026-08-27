// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    Noc noc;

    // input_grad (out0) and other_grad (out1) are optional outputs. Construction of both the
    // tensor accessor and the DFB sat under #ifdef HAS_*_GRAD.
    auto dfb_out0 = construct_nullable_dfb(dfb::out0);
    auto dfb_out1 = construct_nullable_dfb(dfb::out1);
    auto s0 = construct_nullable_tensor(tensor::s0);
    auto s1 = construct_nullable_tensor(tensor::s1);

    uint32_t out0_tile_bytes = 0;
    with_nullable_dfb(dfb_out0, [&](DataflowBuffer& dfb_out0) { out0_tile_bytes = dfb_out0.get_entry_size(); });
    uint32_t out1_tile_bytes = 0;
    with_nullable_dfb(dfb_out1, [&](DataflowBuffer& dfb_out1) { out1_tile_bytes = dfb_out1.get_entry_size(); });

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; i++) {
        with_nullable_token(tensor::s0, dfb::out0, [&](auto const&, DFBBindingToken const&) {
            dfb_out0.wait_front(onetile);
            noc.async_write(dfb_out0, s0, out0_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
            noc.async_write_barrier();
            dfb_out0.pop_front(onetile);
        });

        with_nullable_token(tensor::s1, dfb::out1, [&](auto const&, DFBBindingToken const&) {
            dfb_out1.wait_front(onetile);
            noc.async_write(dfb_out1, s1, out1_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
            noc.async_write_barrier();
            dfb_out1.pop_front(onetile);
        });
    }
}
