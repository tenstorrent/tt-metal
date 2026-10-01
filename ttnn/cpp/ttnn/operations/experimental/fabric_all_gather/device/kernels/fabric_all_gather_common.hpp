// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Kernel helpers shared by the reader, sender and copy writer (terms: fabric_all_gather_chunk_walk.hpp).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric_all_gather_chunk_walk.hpp"

using namespace ttnn::operations::experimental::fabric_all_gather::chunk_walk;

// Read element 0 of a 1-element uint32 DRAM tensor through `landing_l1`. The host rewrites the tensor between trace
// replays at the same address, so the data cache must be invalidated after the read.
template <typename Accessor>
FORCE_INLINE uint32_t read_metadata_word(const Accessor& metadata, uint32_t landing_l1) {
    noc_async_read(metadata.get_noc_addr(0), landing_l1, sizeof(uint32_t));
    noc_async_read_barrier();
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(landing_l1);
}

// Common args 8..15, the same in every kernel: [8] prefix metadata address [9] active stripe pages (host) [10] stripes
// [11] stripe pages [12] ranks [13] slab [14] full extent [15] pages per slab.
template <bool kPrefixFromMetadata, typename PrefixArgs>
FORCE_INLINE ShardGeometry read_shard_geometry(uint32_t landing_l1, const PrefixArgs& prefix_args) {
    ShardGeometry shard{};
    shard.num_stripes = get_common_arg_val<uint32_t>(10);
    shard.stripe_pages = get_common_arg_val<uint32_t>(11);
    shard.num_ranks = get_common_arg_val<uint32_t>(12);
    if constexpr (kPrefixFromMetadata) {
        const auto prefix = TensorAccessor(prefix_args, get_common_arg_val<uint32_t>(8), sizeof(uint32_t));
        shard.active_stripe_pages = active_stripe_pages_from_prefix(
            read_metadata_word(prefix, landing_l1),
            get_common_arg_val<uint32_t>(13),
            get_common_arg_val<uint32_t>(14),
            get_common_arg_val<uint32_t>(15));
    } else {
        (void)landing_l1;
        (void)prefix_args;
        shard.active_stripe_pages = get_common_arg_val<uint32_t>(9);
    }
    return shard;
}

// First input page of the selected cache slot, from the reader's common args [3..7]: the host value, or (trace-safe
// path) slot = user * layers + layer with the user read from the batch index tensor.
template <bool kBatchFromMetadata, typename BatchArgs>
FORCE_INLINE uint32_t read_slot_base(uint32_t landing_l1, const BatchArgs& batch_args) {
    if constexpr (kBatchFromMetadata) {
        const auto batch_index = TensorAccessor(batch_args, get_common_arg_val<uint32_t>(3), sizeof(uint32_t));
        const uint32_t user = read_metadata_word(batch_index, landing_l1);
        return (user * get_common_arg_val<uint32_t>(4) + get_common_arg_val<uint32_t>(5)) *
               get_common_arg_val<uint32_t>(7);
    } else {
        (void)landing_l1;
        (void)batch_args;
        return get_common_arg_val<uint32_t>(6);
    }
}

// Non-interleaved input (e.g. the ND-sharded KV cache): its chunks are not contiguous, so the copy cores convert the
// own shard into this chip's (interleaved) output slot, and the link workers read it from there. The shard, flattened
// over stripes, is cut into blocks of `block_pages` pages; copy core c of n converts blocks c, c + n, ... in order, in
// runs of up to pages_per_chunk pages that are contiguous in the input (one read each), and signals each finished
// block. visit(stripe, first_page, num_pages, last_run_of_block); pages are stripe-local.
template <typename Accessor, typename Visit>
FORCE_INLINE void for_each_input_run(
    const ShardGeometry& shard,
    const Accessor& input,
    uint32_t slot_base,
    uint32_t page_bytes,
    uint32_t pages_per_chunk,
    uint32_t block_pages,
    uint32_t copy_index,
    uint32_t num_copy_cores,
    Visit&& visit) {
    const uint32_t stripe_pages = shard.active_stripe_pages;
    const uint32_t total = shard.num_stripes * stripe_pages;
    for (uint32_t block = copy_index; block * block_pages < total; block += num_copy_cores) {
        const uint32_t block_end = (block + 1) * block_pages < total ? (block + 1) * block_pages : total;
        for (uint32_t flat = block * block_pages; flat < block_end;) {
            const uint32_t stripe = flat / stripe_pages, page = flat % stripe_pages;
            const uint32_t base = slot_base + stripe * shard.stripe_pages;
            const uint32_t limit = (stripe + 1) * stripe_pages < block_end ? (stripe + 1) * stripe_pages : block_end;
            const uint64_t first_addr = input.get_noc_addr(base + page);
            uint32_t num_pages = 1;
            while (num_pages < pages_per_chunk && flat + num_pages < limit &&
                   input.get_noc_addr(base + page + num_pages) == first_addr + num_pages * page_bytes) {
                ++num_pages;
            }
            flat += num_pages;
            visit(stripe, page, num_pages, flat == block_end);
        }
    }
}

// Batches per copy-core block: the copy writer's full write barrier and its signals come once per block (QuietBox,
// 32k-row KV cache: 1 -> 2 batches 995 -> 964 us at 6 KiB; larger blocks delay the first chunk more than they save).
constexpr uint32_t kCopyBlockBatches = 2;

// Blocks copy core c must have finished before the page at flat index `flat` of the own shard is in the output.
FORCE_INLINE uint32_t blocks_needed(uint32_t flat, uint32_t block_pages, uint32_t copy_index, uint32_t num_copy_cores) {
    const uint32_t block = flat / block_pages;
    return block >= copy_index ? (block - copy_index) / num_copy_cores + 1 : 0;
}
