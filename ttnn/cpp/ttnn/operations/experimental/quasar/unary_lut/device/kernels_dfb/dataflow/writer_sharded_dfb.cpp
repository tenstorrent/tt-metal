// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) writer for the unary_lut op, fully-sharded output.
//
// UNARY analog of binary_ng's writer_sharded_no_bcast_dfb.cpp. The compute kernel
// packs directly into the output DFB, which borrows the output tensor's shard
// (DataflowBufferSpec::borrowed_from). There is no NoC write: the writer just
// drains the output DFB so its credits return to the compute producer.
//
// `num_tiles` is this core's shard tile count.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_out(dfb::out);

    // Output shard is written in place by the compute kernel; the writer only
    // consumes credits so the DFB ring drains (FIFO sync, no NoC traffic).
    dfb_out.wait_front(num_tiles);
    dfb_out.pop_front(num_tiles);
}
