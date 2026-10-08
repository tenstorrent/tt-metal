// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
    auto num_tiles = get_arg(args::num_tiles);
    auto start_id = get_arg(args::start_id);

    const auto s0 = TensorAccessor(tensor::src0);
    const auto s1 = TensorAccessor(tensor::src1);

    const Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    const auto in0_tile_bytes = dfb_in0.get_tile_size();
    const auto in1_tile_bytes = dfb_in1.get_tile_size();

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        dfb_in0.reserve_back(onetile);
        noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = i}, {.offset_bytes = 0});

        dfb_in1.reserve_back(onetile);
        noc.async_read(s1, dfb_in1, in1_tile_bytes, {.page_id = i}, {.offset_bytes = 0});

        noc.async_read_barrier();

        dfb_in0.push_back(onetile);
        dfb_in1.push_back(onetile);
    }
}
