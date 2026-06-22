// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) writer for binary_ng's no-broadcast ADD, fully-sharded output.
//
// For a fully-sharded output the compute kernel packs directly into the output DFB, which borrows
// the output tensor's shard (DataflowBufferSpec::borrowed_from). There is no NoC write to perform:
// the writer just drains the output DFB so its credits return to the compute producer. This is the
// DFB analog of the CB writer's `#if DST_SHARDED` skip (writer_interleaved_no_bcast.cpp — all
// noc.async_write calls are compiled out, leaving only FIFO sync).
//
// `num_tiles` is this core's shard tile count.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_out(dfb::out);

    // Output shard is written in place by the compute kernel; the writer only consumes credits so
    // the DFB ring drains. Mirrors the CB writer's DST_SHARDED no-op (FIFO sync, no NoC traffic).
    dfb_out.wait_front(num_tiles);
    dfb_out.pop_front(num_tiles);
}
