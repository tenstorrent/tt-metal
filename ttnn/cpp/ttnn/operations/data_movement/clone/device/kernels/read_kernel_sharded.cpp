// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_tiles = get_arg(args::num_tiles);

    Noc noc;
    DataflowBuffer src_dfb(dfb::src);
    // The resident shard is this core's L1: walk it as a loopback NoC endpoint (same transfers as the legacy kernel).
    const LocalTensorAccessor<uint32_t> s(tensor::input);

    const uint32_t tile_size = src_dfb.get_tile_size();
    uint32_t local_l1_read_offset = 0;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        src_dfb.reserve_back(1);
        noc.async_read(s, src_dfb, tile_size, {.offset_bytes = local_l1_read_offset}, {.offset_bytes = 0});
        noc.async_read_barrier();

        src_dfb.push_back(1);
        local_l1_read_offset += tile_size;
    }
}
