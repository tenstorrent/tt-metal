// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Clone row-major-sharded reader, ported to Metal 2.0.
//
// Host bindings expected (per CloneOperation::ProgramFactory's KernelSpec):
//   runtime_arguments_schema.named_runtime_args: { "input_buffer_addr", "stick_size", "num_sticks" }
//   dfb_bindings: { INPUT_DFB (PRODUCER, name="src_dfb") }
//
// Note: see read_kernel_sharded.cpp -- escape-hatch path (audit Q2 option c).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto input_buffer_addr = get_arg(args::input_buffer_addr);
    auto stick_size = get_arg(args::stick_size);
    auto num_sticks = get_arg(args::num_sticks);

    DataflowBuffer src_dfb(dfb::src_dfb);

    uint64_t local_l1_read_addr = get_noc_addr(input_buffer_addr);

    for (uint32_t i = 0; i < num_sticks; ++i) {
        src_dfb.reserve_back(1);
        uint32_t src_dfb_write_addr = src_dfb.get_write_ptr();

        noc_async_read(local_l1_read_addr, src_dfb_write_addr, stick_size);
        noc_async_read_barrier();

        src_dfb.push_back(1);
        local_l1_read_addr += stick_size;
    }
}
