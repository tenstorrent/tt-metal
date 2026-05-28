// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_sharded.cpp.
//
// Consumes from a sharded (borrowed-memory) output DFB. No actual NoC traffic —
// the DFB is bound to the output tensor's L1-resident shard buffer.
//
// Bindings (named, from host KernelSpec):
//   dfb::out                 — DFB endpoint (CONSUMER)
//   args::num_units          — runtime arg (uint32_t)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_units = get_arg(args::num_units);

    DataflowBuffer cb_out(dfb::out);
    cb_out.wait_front(num_units);
}
