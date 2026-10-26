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
    auto stick_size = get_arg(args::stick_size);
    auto num_sticks = get_arg(args::num_sticks);

    Noc noc;
    DataflowBuffer src_dfb(dfb::src);
    // The resident shard is this core's L1: walk it as a loopback NoC endpoint (same transfers as the legacy kernel).
    const LocalTensorAccessor<uint32_t> s(tensor::input);
    uint32_t local_l1_read_offset = 0;

    for (uint32_t i = 0; i < num_sticks; ++i) {
        src_dfb.reserve_back(1);
        noc.async_read(s, src_dfb, stick_size, {.offset_bytes = local_l1_read_offset}, {.offset_bytes = 0});
        noc.async_read_barrier();

        src_dfb.push_back(1);
        local_l1_read_offset += stick_size;
    }
}
