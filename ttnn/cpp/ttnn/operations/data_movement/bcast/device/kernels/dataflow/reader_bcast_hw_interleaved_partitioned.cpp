// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t HtWt = get_arg(args::HtWt);
    uint32_t base_start_id_HtWt = get_arg(args::base_start_id_HtWt);
    uint32_t curr_id_from_base = get_arg(args::curr_id_from_base);
    uint32_t bcast_id = get_arg(args::bcast_id);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    const uint32_t tile_bytes_0 = dfb_in0.get_tile_size();
    const uint32_t tile_bytes_1 = dfb_in1.get_tile_size();

    // src1 base address + layout arrive via the tensor binding (tensor::src1). src0 does too
    // (tensor::src0) in the interleaved config; when in0 is sharded it is resident and dfb::in0 borrows
    // the shard directly, so the reader just signals the resident tiles instead of reading them.
    const auto s1 = TensorAccessor(tensor::src1);
#ifndef IN0_SHARDED
    const auto s0 = TensorAccessor(tensor::src0);
#else
    dfb_in0.reserve_back(num_tiles);
    dfb_in0.push_back(num_tiles);
#endif

#ifdef BCAST_SCALAR
    dfb_in1.reserve_back(onetile);
    noc.async_read(s1, dfb_in1, tile_bytes_1, {.page_id = bcast_id, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb_in1.push_back(onetile);
#endif

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t curr_id = base_start_id_HtWt + curr_id_from_base;

#ifndef IN0_SHARDED
        dfb_in0.reserve_back(onetile);
        noc.async_read(s0, dfb_in0, tile_bytes_0, {.page_id = curr_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
#endif

        curr_id_from_base++;

#ifndef BCAST_SCALAR
        dfb_in1.reserve_back(onetile);
        noc.async_read(s1, dfb_in1, tile_bytes_1, {.page_id = bcast_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in1.push_back(onetile);

        if (curr_id_from_base == HtWt) {
            bcast_id++;
#else
        if (curr_id_from_base == HtWt) {
#endif
            base_start_id_HtWt += HtWt;
            curr_id_from_base = 0;
        }
    }
}
