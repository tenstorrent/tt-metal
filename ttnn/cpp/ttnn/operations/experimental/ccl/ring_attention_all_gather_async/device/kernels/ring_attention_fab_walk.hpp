// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// fabric_all_gather's data path for the ring attention all-gather (RING_AG_FAB_PATH): a slice (one rank's shard of
// every input) is walked as fabric_all_gather walks a shard (fabric_all_gather_chunk_walk.hpp): per input, outer slice
// (batch/head) by outer slice, the first chunk of every owned page lane, then the second, ... A worker owns lanes
// first_lane, first_lane + lane_stride, ... (its link, every num_links-th lane), the same on every chip, so the worker
// that forwards a slice reads exactly the pages its upstream worker of the same link sent. The chunk CB holds two
// batches of chunks_per_cb_batch chunk slots; reader and writer cut batches at the same points (full, end of the CB,
// end of the slice).

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>
#include "cpp/ttnn/operations/experimental/fabric_all_gather/device/kernels/fabric_all_gather_chunk_walk.hpp"

namespace ring_attention_fab_walk {

using namespace ttnn::operations::experimental::fabric_all_gather::chunk_walk;

template <uint32_t num_inputs>
struct SliceGeometry {
    std::array<uint32_t, num_inputs> outer_slices;      // batch * heads
    std::array<uint32_t, num_inputs> valid_pages;       // valid prefix of every outer slice
    std::array<uint32_t, num_inputs> input_pages;       // pages per outer slice of the input (one rank's shard)
    std::array<uint32_t, num_inputs> output_pages;      // pages per outer slice of the output
    uint32_t first_lane, lane_stride, num_dram_banks, pages_per_chunk;

    ChipShardGeometry chunk_geometry(uint32_t input) const {
        return ChipShardGeometry{outer_slices[input], valid_pages[input], input_pages[input], 1};
    }
    // output page of (rank, outer slice, page in slice)
    uint32_t output_page(uint32_t input, uint32_t rank, uint32_t outer, uint32_t page) const {
        return outer * output_pages[input] + rank * input_pages[input] + page;
    }
    uint32_t chunks_per_slice() const {
        uint32_t chunks = 0;
        for (uint32_t input = 0; input < num_inputs; ++input) {
            chunks += count_fabric_chunks(
                chunk_geometry(input), num_dram_banks, pages_per_chunk, first_lane, lane_stride, kWholeChipShard);
        }
        return chunks;
    }
    // visit(std::integral_constant<uint32_t, input>, outer slice, first page in outer slice, pages): the input index is a
    // compile-time constant so the kernel addresses each input through its own typed (inlined) TensorAccessor.
    template <typename Visit>
    void for_each_chunk(Visit&& visit) const {
        for_each_chunk_impl(visit, std::make_integer_sequence<uint32_t, num_inputs>{});
    }

private:
    template <typename Visit, uint32_t... I>
    void for_each_chunk_impl(Visit& visit, std::integer_sequence<uint32_t, I...>) const {
        (for_each_input_chunk<I>(visit), ...);
    }
    template <uint32_t I, typename Visit>
    void for_each_input_chunk(Visit& visit) const {
        for_each_fabric_chunk(
            chunk_geometry(I),
            num_dram_banks,
            pages_per_chunk,
            first_lane,
            lane_stride,
            kWholeChipShard,
            [&](uint32_t outer, uint32_t page, uint32_t num_pages) {
                visit(std::integral_constant<uint32_t, I>{}, outer, page, num_pages);
            });
    }
};

// Batch bookkeeping shared by both sides: a batch starts at chunk slot `offset` and holds at most
// min(chunks_per_cb_batch, slots to the end of the CB) chunks, so it is contiguous in L1.
struct CbBatches {
    uint32_t chunk_slots, chunks_per_cb_batch, offset = 0, in_batch = 0;
    uint32_t capacity() const {
        return chunks_per_cb_batch < chunk_slots - offset ? chunks_per_cb_batch : chunk_slots - offset;
    }
    void close() {
        offset = (offset + in_batch) % chunk_slots;
        in_batch = 0;
    }
};

}  // namespace ring_attention_fab_walk
