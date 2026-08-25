// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

constexpr uint32_t txns_per_device = get_arg(args::txns_per_device);
constexpr uint32_t num_producers = get_arg(args::num_producers);
constexpr uint32_t bytes_per_dma_txn = get_arg(args::bytes_per_dma_txn);
constexpr uint32_t in_shard_bytes = get_arg(args::in_shard_bytes);
constexpr uint32_t in_shard_tiles = get_arg(args::in_shard_tiles);
constexpr uint32_t out_shard_bytes = get_arg(args::out_shard_bytes);
constexpr uint32_t block_bytes = get_arg(args::block_bytes);

// Bytes this chunk may carry, starting `cursor` bytes into the device block.
// Three-way minimum: the packet cap, what is left of the current input shard
// (whose last one per block may be ragged), and what is left of the current
// output shard (uniform, because the block is a whole number of them).
constexpr uint32_t txn_bytes_at(uint32_t cursor) {
    const uint32_t in_end = std::min((cursor / in_shard_bytes + 1) * in_shard_bytes, block_bytes);
    const uint32_t out_left = out_shard_bytes - (cursor % out_shard_bytes);
    return std::min(bytes_per_dma_txn, std::min(in_end - cursor, out_left));
}

void kernel_main() {
    DataflowBuffer payload(dfb::payload);
    const auto input_tensor = TensorAccessor(tensor::input_tensor);
    Noc noc;

    ASSERT(payload.get_entry_size() == bytes_per_dma_txn);

    const uint32_t producer_idx = get_my_thread_id();

    // Walk every chunk; act on the ones this thread owns. STRIDED requires
    // producer i to fill entries i, i+P, i+2P, ... in order, which this does.
    // The skipped iterations are a few arithmetic ops -- cheaper than passing a
    // per-entry table.
    uint32_t cursor = 0;
    for (uint32_t entry = 0; entry < txns_per_device; ++entry) {
        const uint32_t size = txn_bytes_at(cursor);

        if (entry % num_producers == producer_idx) {
            // A chunk is a byte range inside one input shard: page at that
            // shard's start, plus an offset. Nothing is tile-quantised.
            const uint32_t shard = cursor / in_shard_bytes;
            const uint32_t page_id = shard * in_shard_tiles;
            const uint32_t offset_bytes = cursor - shard * in_shard_bytes;

            if (size == bytes_per_dma_txn) {
                // One read fills the whole entry. The overload issues a single
                // noc_async_read of get_entry_size() bytes from this address,
                // and the chunk is contiguous inside one shard. The ISR bumps
                // `posted` when it lands -- no reserve_back, no push_back, no
                // barrier, and the credit is tied to the read completing rather
                // than to the thread reaching a barrier.
                noc.async_read<NocOptions::TXN_ID>(
                    input_tensor, payload, {.page_id = page_id, .offset_bytes = offset_bytes});
            } else {
                // Short chunk. The implicit overload has no size argument -- it
                // would over-read past the shard, into a different core -- so
                // this one goes the explicit way. The read must stay untagged:
                // a tagged one would also bump `posted` through the ISR and
                // double-count against the push_back below.
                payload.reserve_back(1);
                noc.async_read(input_tensor, payload, size, {.page_id = page_id, .offset_bytes = offset_bytes});
                noc.async_read_barrier();
                payload.push_back(1);
            }
        }

        cursor += size;
    }
}
