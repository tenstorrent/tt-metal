// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Kernel helpers shared by the reader, sender and local copy writer (terms: fabric_all_gather_chunk_walk.hpp).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "fabric_all_gather_chunk_walk.hpp"

using namespace ttnn::operations::experimental::high_bw_all_gather::chunk_walk;

// Read element 0 of a 1-element uint32 DRAM tensor through `landing_l1`. The host rewrites the tensor between trace
// replays at the same address, so the data cache must be invalidated after the read.
template <typename Accessor>
FORCE_INLINE uint32_t read_metadata_word(const Noc& noc, const Accessor& metadata, uint32_t landing_l1) {
    noc.async_read(metadata, CoreLocalMem<uint32_t>(landing_l1), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(landing_l1);
}

// The chip shard geometry block of the common args, the same in every kernel, starting at common arg kFirstArg (after
// the kernel's own common args): [+0] prefix tensor address [+1] valid pages per outer slice (host) [+2] outer slices
// [+3] pages per outer slice [+4] ranks [+5] KV slab [+6] full gathered length [+7] pages per KV slab.
template <bool kValidPrefixFromMetadata, uint32_t kFirstArg, typename PrefixArgs>
FORCE_INLINE ChipShardGeometry
read_chip_shard_geometry(const Noc& noc, uint32_t landing_l1, const PrefixArgs& prefix_args) {
    ChipShardGeometry geometry{};
    geometry.num_outer_slices = get_common_arg_val<uint32_t>(kFirstArg + 2);
    geometry.pages_per_outer_slice = get_common_arg_val<uint32_t>(kFirstArg + 3);
    geometry.num_ranks = get_common_arg_val<uint32_t>(kFirstArg + 4);
    if constexpr (kValidPrefixFromMetadata) {
        const auto prefix = TensorAccessor(prefix_args, get_common_arg_val<uint32_t>(kFirstArg), sizeof(uint32_t));
        geometry.valid_pages_per_outer_slice = valid_pages_from_prefix_start(
            read_metadata_word(noc, prefix, landing_l1),
            get_common_arg_val<uint32_t>(kFirstArg + 5),
            get_common_arg_val<uint32_t>(kFirstArg + 6),
            get_common_arg_val<uint32_t>(kFirstArg + 7));
    } else {
        (void)noc;
        (void)landing_l1;
        (void)prefix_args;
        geometry.valid_pages_per_outer_slice = get_common_arg_val<uint32_t>(kFirstArg + 1);
    }
    return geometry;
}

// Where the next CB batch starts in the chunk CB (2 * chunks_per_cb_batch chunk slots), the same on both sides of the
// CB. A CB batch stops at the end of the CB, so it stays contiguous in L1.
template <uint32_t chunks_per_cb_batch>
struct ChunkCbCursor {
    static constexpr uint32_t chunk_slots_in_cb = 2 * chunks_per_cb_batch;
    uint32_t chunk_offset = 0;

    // chunk slots the next CB batch can take
    FORCE_INLINE uint32_t capacity() const {
        return chunks_per_cb_batch < chunk_slots_in_cb - chunk_offset ? chunks_per_cb_batch
                                                                      : chunk_slots_in_cb - chunk_offset;
    }
    FORCE_INLINE void advance(uint32_t num_chunks) { chunk_offset = (chunk_offset + num_chunks) % chunk_slots_in_cb; }
};

// Input page of (outer_slice, page_in_slice) of this chip's shard, in the selected cache slot.
FORCE_INLINE uint32_t input_page_index(
    const ChipShardGeometry& geometry, uint32_t cache_slot_first_page, uint32_t outer_slice, uint32_t page_in_slice) {
    return cache_slot_first_page + outer_slice * geometry.pages_per_outer_slice + page_in_slice;
}

// Conversion blocks of local copy core c of n: blocks c, c + n, ... of conversion_block_pages flat pages each, over the
// valid pages of the shard (the last one can be short). visit(first_flat_page, end_flat_page).
template <typename Visit>
FORCE_INLINE void for_each_conversion_block(
    const ChipShardGeometry& geometry,
    uint32_t conversion_block_pages,
    uint32_t local_copy_core_index,
    uint32_t num_local_copy_cores,
    const Visit& visit) {
    const uint32_t valid_pages_in_shard = geometry.num_outer_slices * geometry.valid_pages_per_outer_slice;
    for (uint32_t conversion_block = local_copy_core_index;
         conversion_block * conversion_block_pages < valid_pages_in_shard;
         conversion_block += num_local_copy_cores) {
        const uint32_t block_end_page = (conversion_block + 1) * conversion_block_pages < valid_pages_in_shard
                                            ? (conversion_block + 1) * conversion_block_pages
                                            : valid_pages_in_shard;
        visit(conversion_block * conversion_block_pages, block_end_page);
    }
}

// First input page of the selected cache slot, from the reader's common args [3..7]: the host value, or (trace-safe
// path) cache slot = user * layers + layer with the user read from the cache slot tensor (input_batch_index_tensor).
template <bool kCacheSlotFromMetadata, typename CacheSlotArgs>
FORCE_INLINE uint32_t
read_cache_slot_first_page(const Noc& noc, uint32_t landing_l1, const CacheSlotArgs& cache_slot_args) {
    if constexpr (kCacheSlotFromMetadata) {
        const auto cache_slot_tensor =
            TensorAccessor(cache_slot_args, get_common_arg_val<uint32_t>(3), sizeof(uint32_t));
        const uint32_t user = read_metadata_word(noc, cache_slot_tensor, landing_l1);
        return (user * get_common_arg_val<uint32_t>(4) + get_common_arg_val<uint32_t>(5)) *
               get_common_arg_val<uint32_t>(7);
    } else {
        (void)noc;
        (void)landing_l1;
        (void)cache_slot_args;
        return get_common_arg_val<uint32_t>(6);
    }
}

// Non-interleaved input (e.g. the ND-sharded KV cache): its fabric chunks are not physically contiguous, so the local
// copy cores convert the own shard into this chip's (interleaved) output, and the link workers read it from there.
// The valid pages of the shard, numbered flat (flat_page = outer_slice * valid pages per slice + page_in_slice), are
// cut into conversion blocks of conversion_block_pages; local copy core c of n converts its blocks
// (for_each_conversion_block) in order, in contiguous input runs of up to pages_per_fabric_chunk pages (logically and
// physically consecutive: one read each), and signals each finished block. visit(outer_slice, page_in_slice,
// num_pages, last_run_of_block).
template <typename Accessor, typename Visit>
FORCE_INLINE void for_each_contiguous_input_run(
    const ChipShardGeometry& geometry,
    const Accessor& input,
    uint32_t cache_slot_first_page,
    uint32_t page_bytes,
    uint32_t pages_per_fabric_chunk,
    uint32_t conversion_block_pages,
    uint32_t local_copy_core_index,
    uint32_t num_local_copy_cores,
    const Visit& visit) {
    const uint32_t valid_pages = geometry.valid_pages_per_outer_slice;
    for_each_conversion_block(
        geometry,
        conversion_block_pages,
        local_copy_core_index,
        num_local_copy_cores,
        [&](uint32_t block_first_page, uint32_t block_end_page) {
            for (uint32_t flat_page = block_first_page; flat_page < block_end_page;) {
                const uint32_t outer_slice = flat_page / valid_pages, page_in_slice = flat_page % valid_pages;
                const uint32_t first_input_page =
                    input_page_index(geometry, cache_slot_first_page, outer_slice, page_in_slice);
                // a run stops at the end of the outer slice or of the block
                const uint32_t run_limit_page =
                    (outer_slice + 1) * valid_pages < block_end_page ? (outer_slice + 1) * valid_pages : block_end_page;
                const uint64_t first_page_addr = input.get_noc_addr(first_input_page);
                uint32_t num_pages = 1;
                while (num_pages < pages_per_fabric_chunk && flat_page + num_pages < run_limit_page &&
                       input.get_noc_addr(first_input_page + num_pages) == first_page_addr + num_pages * page_bytes) {
                    ++num_pages;
                }
                flat_page += num_pages;
                visit(outer_slice, page_in_slice, num_pages, flat_page == block_end_page);
            }
        });
}

// CB batches per conversion block: the local copy writer's full write barrier and its signals come once per block
// (QuietBox, 32k-row KV cache: 1 -> 2 batches 995 -> 964 us at 6 KiB; larger blocks delay the first chunk more than
// they save).
constexpr uint32_t kCbBatchesPerConversionBlock = 2;

// Conversion blocks local copy core c must have finished before the own shard's flat_page is in the output.
FORCE_INLINE uint32_t conversion_blocks_needed(
    uint32_t flat_page,
    uint32_t conversion_block_pages,
    uint32_t local_copy_core_index,
    uint32_t num_local_copy_cores) {
    const uint32_t conversion_block = flat_page / conversion_block_pages;
    return conversion_block >= local_copy_core_index
               ? (conversion_block - local_copy_core_index) / num_local_copy_cores + 1
               : 0;
}
