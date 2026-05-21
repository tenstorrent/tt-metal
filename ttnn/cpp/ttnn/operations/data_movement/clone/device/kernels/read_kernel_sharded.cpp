// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Clone tilized-sharded reader, ported to Metal 2.0.
//
// Host bindings expected (per CloneOperation::ProgramFactory's KernelSpec):
//   runtime_arguments_schema.named_runtime_args: { "input_buffer_addr", "num_tiles" }
//   dfb_bindings: { INPUT_DFB (PRODUCER, name="src_dfb") }
//
// Note: this kernel does NOT use TensorAccessor. The legacy form took the
// input shard's local L1 base address as an RTA; the Metal 2.0 port preserves
// that escape hatch (audit Q2 option c). TensorAccessor's bank_base_address
// is private, so this is the supported path until a public surface lands.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto input_buffer_addr = get_arg(args::input_buffer_addr);
    auto num_tiles = get_arg(args::num_tiles);

    DataflowBuffer src_dfb(dfb::src_dfb);

    const uint32_t tile_size = src_dfb.get_tile_size();
    uint64_t local_l1_read_addr = get_noc_addr(input_buffer_addr);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        src_dfb.reserve_back(1);
        uint32_t src_dfb_write_addr = src_dfb.get_write_ptr();

        noc_async_read(local_l1_read_addr, src_dfb_write_addr, tile_size);
        noc_async_read_barrier();

        src_dfb.push_back(1);
        local_l1_read_addr += tile_size;
    }
}
