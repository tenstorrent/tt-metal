// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) reader for the unary_lut op, fully-sharded input.
//
// UNARY analog of binary_ng's reader_sharded_no_bcast_dfb.cpp with a SINGLE input
// DFB (in0). The input shard is already L1-resident (the DFB borrows the input
// tensor's shard via DataflowBufferSpec::borrowed_from), so there is no NoC read:
// the reader simply publishes the resident shard's tiles into the DFB FIFO.
//
// `num_tiles` is this core's shard tile count.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in0(dfb::in0);

    // Borrowed shard is already resident in L1; publish it to the DFB FIFO. Mirrors
    // the CB reader's sharded fast path (bulk reserve_back + push_back, no NoC traffic).
    dfb_in0.reserve_back(num_tiles);
    dfb_in0.push_back(num_tiles);
}
