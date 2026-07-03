// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) reader for binary_ng's no-broadcast ADD, fully-sharded case.
//
// For a fully-sharded input the source tiles are already L1-resident (the DFB borrows the input
// tensor's shard via DataflowBufferSpec::borrowed_from). There is no NoC read to perform: the
// reader simply publishes the resident shard's tiles into the DFB so the compute kernel can consume
// them. This is the DFB analog of the CB reader's `#if SRC_SHARDED` short-circuit in
// reader_interleaved_no_bcast.cpp (bulk reserve_back + push_back, no NoC traffic).
//
// Two input DFBs (in0, in1) are published. `num_tiles` is this core's shard tile count.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    // Borrowed shards are already resident in L1; publish them to the DFB FIFOs. Mirrors the CB
    // reader's sharded fast path (bulk reserve_back + push_back, no NoC traffic).
    dfb_in0.reserve_back(num_tiles);
    dfb_in0.push_back(num_tiles);

    dfb_in1.reserve_back(num_tiles);
    dfb_in1.push_back(num_tiles);
}
