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
    DataflowBuffer dst_dfb(dfb::dst);
    // The resident shard is this core's L1: walk it as a loopback NoC endpoint (same transfers as the legacy kernel).
    const LocalTensorAccessor<uint32_t> s(tensor::output);

    const uint32_t tile_size = dst_dfb.get_tile_size();
    uint32_t local_l1_write_offset = 0;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dst_dfb.wait_front(1);
        noc.async_write(dst_dfb, s, tile_size, {.offset_bytes = 0}, {.offset_bytes = local_l1_write_offset});
        noc.async_write_barrier();

        dst_dfb.pop_front(1);
        local_l1_write_offset += tile_size;
    }
}
