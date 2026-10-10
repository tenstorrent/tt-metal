// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe counterpart of writer_l1.cpp (the GlobalCircularBuffer writer, which lives beside it).
//
// Drains the reader's staging ring into this core's PrefetcherPipe: each staged block becomes one pipe
// entry per receiver, receiver r getting the r-th column slice of every block row (write_strided). The
// pipe's entry size follows the tensor being delivered.
//
// Bindings:
//   dfb::staging - CONSUMER: the reader's staging ring, max_block_num_tiles entries per block
//   pipe::out    - sender of this core's PrefetcherPipe
//   dfb::sync    - PRODUCER: one entry pushed once the pipe is quiet, which lets the reader exit
// Common runtime varargs, each num_tensors long, in this order: per-receiver block size (the pipe entry
// size), block height in tile rows, coalesced_page_size, coalesced_num_pages. One receiver's slice of a
// block row is coalesced_num_pages writes of coalesced_page_size bytes.
//
// The pipe's destructor commits its cursors and drains its credit atomics; the reader, which shares this
// NOC, re-syncs the NoC counters at exit, so it is released only after that.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_layers = get_arg(args::num_layers);
    constexpr uint32_t num_tensors = get_arg(args::num_tensors);
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t max_block_num_tiles = get_arg(args::max_block_num_tiles);

    const Noc noc;
    DataflowBuffer staging(dfb::staging);
    DataflowBuffer sync(dfb::sync);

    {
        experimental::PrefetcherPipe pipe(pipe::out);

        for (uint32_t layer = 0; layer < num_layers; layer++) {
            for (uint32_t t = 0; t < num_tensors; t++) {
                const uint32_t entry_size = get_common_vararg(t);
                const uint32_t block_height_in_tiles = get_common_vararg(num_tensors + t);
                const uint32_t coalesced_page_size = get_common_vararg(2 * num_tensors + t);
                const uint32_t coalesced_num_pages = get_common_vararg(3 * num_tensors + t);

                if (pipe.get_entry_size() != entry_size) {
                    pipe.set_entry_size(entry_size);
                }

                for (uint32_t block = 0; block < num_blocks; ++block) {
                    staging.wait_front(max_block_num_tiles);
                    pipe.reserve_back(1);
                    pipe.write_strided(noc, staging, block_height_in_tiles, coalesced_num_pages, coalesced_page_size);
                    // Also what frees the staging slot: write_strided's source is that slot.
                    pipe.flush_writes(noc);
                    pipe.push_back(1, noc);
                    staging.pop_front(max_block_num_tiles);
                }

                if (t == num_tensors - 1) {
                    pipe.barrier();
                }
            }
        }
    }  // The pipe's destructor runs here: nothing of this kernel's is in flight on the NOC past this point.

    sync.reserve_back(1);
    sync.push_back(1);
}
