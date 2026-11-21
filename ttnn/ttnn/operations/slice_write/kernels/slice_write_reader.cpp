// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// slice_write reader: load_block. The ring dfb_src_shard is borrowed onto this core's resident shard of the
// source (zero-copy), so loading the block moves no data: the reader only publishes the shard's entries to
// the writer (explicit sync; the host opts the ring out of implicit sync). No NoC transfer is issued here.
//
// Reader hart p of R publishes entries p, p+R, ... of the src_shard_entries entries over the shard; on the
// STRIDED ring the writer hart that owns an entry is the one that consumes it.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t src_shard_entries = get_arg(args::src_shard_entries);

    DataflowBuffer dfb_src_shard(dfb::src_shard);
    for (uint32_t entry = get_my_thread_id(); entry < src_shard_entries; entry += R) {
        dfb_src_shard.reserve_back(1);
        dfb_src_shard.push_back(1);
    }
}
