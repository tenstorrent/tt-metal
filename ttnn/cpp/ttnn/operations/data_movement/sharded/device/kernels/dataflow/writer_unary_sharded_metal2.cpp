// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_sharded.cpp.
//
// Forked because the legacy file is consumed by many ops still on the legacy
// positional-CTA path. During the bulk-port window, the legacy copy and this
// fork coexist; the legacy copy is deleted once the last unmigrated consumer
// ports. See the shared-dataflow-kernel Caution in
//   docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md
//
// Differences from the legacy writer:
//   - Named DFB binding (`dfb::output`) replaces the magic CB index CTA.
//   - Named RTA (`num_units`) replaces positional `get_arg_val<>`.
//
// Host bindings expected:
//   compile_time_arg_bindings: none
//   runtime_arguments_schema.named_runtime_args: { "num_units" }
//   dfb_bindings: { OUTPUT (CONSUMER, name="output") }
//
// The DFB for the output is host-declared as borrowed-memory
// (DataflowBufferSpec::borrowed_from = "output"); the borrow attach updates its
// backing L1 address from the corresponding TensorArg at run-time, so no
// `dst_addr` RTA is required.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_units = get_arg(args::num_units);

    DataflowBuffer cb_out(dfb::output);

    cb_out.wait_front(num_units);
}
