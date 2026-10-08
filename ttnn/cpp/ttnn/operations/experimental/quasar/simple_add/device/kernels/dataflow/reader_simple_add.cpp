// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// simple_add reader: one thread reads tile i of A into dfb::in0 and tile i of
// B into dfb::in1, for i in [0, num_tiles), from DRAM-interleaved tensors. Explicit sync, as on Blackhole.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    const uint32_t in0_tile_bytes = dfb_in0.get_entry_size();
    const uint32_t in1_tile_bytes = dfb_in1.get_entry_size();

    const auto a = TensorAccessor(tensor::a);
    const auto b = TensorAccessor(tensor::b);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in0.reserve_back(1);
        dfb_in1.reserve_back(1);
        noc.async_read(a, dfb_in0, in0_tile_bytes, {.page_id = i}, {});
        noc.async_read(b, dfb_in1, in1_tile_bytes, {.page_id = i}, {});
        noc.async_read_barrier();
        dfb_in0.push_back(1);
        dfb_in1.push_back(1);
    }

    dfb_in0.finish();
    dfb_in1.finish();
}
