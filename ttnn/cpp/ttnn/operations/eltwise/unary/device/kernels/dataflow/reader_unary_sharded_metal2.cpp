// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of reader_unary_sharded.cpp.
//
// Reads num_tiles_per_core entries already resident in a sharded (borrowed-memory)
// DFB and signals their availability to the consumer via push_back. No actual NoC
// traffic — the DFB is bound to the input tensor's L1-resident shard buffer.
//
// Bindings (named, from host KernelSpec):
//   dfb::shard            — DFB endpoint (PRODUCER)
//   args::num_tiles_per_core — runtime arg (uint32_t)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_tiles_per_core = get_arg(args::num_tiles_per_core);

    DataflowBuffer dfb(dfb::shard);
    dfb.push_back(num_tiles_per_core);
}
