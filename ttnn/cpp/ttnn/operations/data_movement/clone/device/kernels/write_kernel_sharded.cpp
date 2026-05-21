// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Clone tilized-sharded writer, ported to Metal 2.0.
//
// Host bindings expected (per CloneOperation::ProgramFactory's KernelSpec):
//   runtime_arguments_schema.named_runtime_args: { "num_tiles" }
//   dfb_bindings: { (INPUT_DFB or OUTPUT_DFB) (CONSUMER, name="dst_dfb") }
//   tensor_bindings: { OUTPUT_TENSOR (name="output") }
//
// The output shard's local L1 base address is sourced from the TensorAccessor's
// bank_base_address (auto-injected by the binding mechanism), substituting for
// the legacy buffer-address RTA.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dst_dfb(dfb::dst_dfb);
    const auto output_a = TensorAccessor(ta::output);

    const uint32_t tile_size = dst_dfb.get_tile_size();
    uint64_t local_l1_write_addr = get_noc_addr(static_cast<uint32_t>(output_a.bank_base_address));

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dst_dfb.wait_front(1);
        uint32_t dst_dfb_read_addr = dst_dfb.get_read_ptr();

        noc_async_write(dst_dfb_read_addr, local_l1_write_addr, tile_size);
        noc_async_write_barrier();

        dst_dfb.pop_front(1);
        local_l1_write_addr += tile_size;
    }
}
