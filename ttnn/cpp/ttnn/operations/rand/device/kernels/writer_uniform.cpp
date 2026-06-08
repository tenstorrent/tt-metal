// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Metal 2.0 writer: consumer of the rand DFB.
// The descriptor-era writer needed two CBs, a TensorAccessorArgs compile-time payload, and two
// dtype-specific paths (a manual fp32 -> bf16 narrowing into a scratch CB). With Metal 2.0 that
// all collapses: the DFB already holds the output dtype (the compute kernel packed it), and the
// output tensor is reached via a TensorAccessor binding. So this is a single dtype-agnostic copy
// of each DFB entry to its output page — no scratch buffer, no OUTPUT_DTYPE_* branches.
void kernel_main() {
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t end_id = start_id + num_tiles;

    TensorAccessor out(ta::output);
    DataflowBuffer rand_tiles(dfb::rand_tiles);
    const uint32_t entry_size = rand_tiles.get_entry_size();

    for (uint32_t i = start_id; i < end_id; ++i) {
        rand_tiles.wait_front(1);
        const uint64_t dst_noc_addr = out.get_noc_addr(i);
        noc_async_write(rand_tiles.get_read_ptr(), dst_noc_addr, entry_size);
        noc_async_write_barrier();
        rand_tiles.pop_front(1);
    }
}
