// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp. Zero-copy sharded
// path: the input DFB is built directly on the input shard's L1 buffer
// (DataflowBufferSpec::borrowed_from = input), so this reader only advances the DFB write pointer to
// hand the already-resident pages to the consumer — no NoC read, no TensorAccessor. Only the plumbing
// changes: the CB index compile-time arg becomes dfb::in and the per-core tile count becomes a named
// runtime arg. Forked rather than converted in place because the legacy file is instantiated by many
// factories still on the legacy positional-arg API.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg(args::num_tiles_per_core);

    // dfb::in — borrowed from the input shard buffer; the data is already resident, so signalling
    // the pages to the consumer is the reader's whole job
    DataflowBuffer dfb(dfb::in);
    dfb.push_back(num_tiles_per_core);
}
