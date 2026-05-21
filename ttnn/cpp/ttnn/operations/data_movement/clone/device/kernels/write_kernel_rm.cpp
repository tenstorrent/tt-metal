// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Clone row-major-interleaved writer, ported to Metal 2.0.
//
// Host bindings expected (per CloneOperation::ProgramFactory's KernelSpec):
//   runtime_arguments_schema.named_runtime_args: { "stick_size", "num_sticks", "start_id" }
//   dfb_bindings: { (INPUT_DFB or OUTPUT_DFB) (CONSUMER, name="dst_dfb") }
//   tensor_bindings: { OUTPUT_TENSOR (name="output") }

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto stick_size = get_arg(args::stick_size);
    auto num_sticks = get_arg(args::num_sticks);
    auto start_id = get_arg(args::start_id);

    DataflowBuffer dst_dfb(dfb::dst_dfb);
    Noc noc;
    const auto s = TensorAccessor(ta::output);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dst_dfb.wait_front(1);
        noc.async_write(dst_dfb, s, stick_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dst_dfb.pop_front(1);
    }
}
