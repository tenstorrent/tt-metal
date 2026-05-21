// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Clone tilized-sharded writer, ported to Metal 2.0.
//
// Host bindings expected (per CloneOperation::ProgramFactory's KernelSpec):
//   runtime_arguments_schema.named_runtime_args: { "output_buffer_addr", "num_tiles" }
//   dfb_bindings: { (INPUT_DFB or OUTPUT_DFB) (CONSUMER, name="dst_dfb") }
//
// Note: see read_kernel_sharded.cpp -- escape-hatch path (audit Q2 option c)
// preserving the legacy buffer-address RTA. No TensorAccessor here.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto output_buffer_addr = get_arg(args::output_buffer_addr);
    auto num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dst_dfb(dfb::dst_dfb);

    const uint32_t tile_size = dst_dfb.get_tile_size();
    uint64_t local_l1_write_addr = get_noc_addr(output_buffer_addr);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dst_dfb.wait_front(1);
        uint32_t dst_dfb_read_addr = dst_dfb.get_read_ptr();

        noc_async_write(dst_dfb_read_addr, local_l1_write_addr, tile_size);
        noc_async_write_barrier();

        dst_dfb.pop_front(1);
        local_l1_write_addr += tile_size;
    }
}
