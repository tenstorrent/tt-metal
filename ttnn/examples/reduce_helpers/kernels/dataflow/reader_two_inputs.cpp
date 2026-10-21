// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <typename Accessor>
void read_tiles(Noc& noc, const Accessor& src, DataflowBuffer& dfb, uint32_t num_tiles) {
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        dfb.reserve_back(1);
        noc.async_read(src, dfb, dfb.get_tile_size(), {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(1);
    }
}

// Compute issues the call for a before the call for b, so a is read first.
void kernel_main() {
    Noc noc;
    DataflowBuffer in_a(dfb::in_a);
    DataflowBuffer in_b(dfb::in_b);
    read_tiles(noc, TensorAccessor(tensor::a), in_a, get_arg(args::a_tiles));
    read_tiles(noc, TensorAccessor(tensor::b), in_b, get_arg(args::b_tiles));
}
