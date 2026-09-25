// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "dispatch_fabric2d_placement.hpp"
#include "kernels/dataflow/dispatch_fabric2d_kernel_interface.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// One unit of work for one stream: either a forward, which passes a chunk received from upstream one hop
// further, or a share of this chip's own tokens for one destination chip. The share is split across the
// num_links streams going that way, or across every stream for the diametrically opposite chip, which is
// equally far in both directions.
//
// A token's destination depends on the data (indices -> dispatch table), so the reader builds the
// per-destination token list on device; this names only the destination and the share. The kernel runs
// all own assignments before any forward, which gives the upstream chip time to write the chunks the
// forwards consume.
struct Assignment {
    bool is_forward = false;
    uint32_t dst_chip_id = 0;  // !is_forward: fabric name of the destination chip
    uint32_t dst_row = 0;      // !is_forward: that chip's position on the dispatch axis
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// Work for every stream on one chip. `ring_chip_ids` holds the fabric chip id of each position on the
// dispatch axis, so this needs nothing from the mesh API.
std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_row, uint32_t num_links);

// Chunks a stream forwards, in the order the upstream chip emits them into this stream's section.
std::vector<dspf2d::ChunkDescriptor> forwarding_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links);

// Chunks this chip writes into the downstream chip's section for this stream, in emission order: its own
// destinations beyond the neighbour, furthest first, then the chunks it forwards. A chunk bound for the
// neighbour itself is written straight to that chip's output, so it is not listed.
std::vector<dspf2d::ChunkDescriptor> outgoing_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links);

// The section holds no per-chunk addresses, so a chunk is found by counting the chunks before it. What a
// chip writes and what its neighbour reads must therefore agree in identity and order, or the reader waits
// for a chunk that is never written. The chips exchange nothing at run time, so this checks it on the host.
void validate_chunk_agreement(uint32_t ring_extent, uint32_t num_links);

// Own assignments a stream carries: one per destination it reaches sooner in its own direction, plus its
// share of the diametrically opposite chip.
constexpr uint32_t own_assignments_per_stream(uint32_t ring_extent) { return ring_extent / 2; }

// Forward chunks a stream receives, which is also how many its upstream neighbour writes into this
// stream's section, so writer and reader agree on the count without exchanging it.
//
// A chunk is one (origin, destination) pair whose path passes through this chip and continues. Summing
// over upstream distance k the pairs from that origin reaching further than k gives
// sum_{k=1..m-1} (m-k) = m(m-1)/2 for m = extent/2. Each of these expands on device into
// experts_per_chip chunks.
constexpr uint32_t forward_chunks_per_stream(uint32_t ring_extent) {
    const uint32_t m = ring_extent / 2;
    return m * (m - 1) / 2;
}

// Pages one stream's fwd_section must hold. A chunk's length depends on expert_offsets, which is on
// device; reading it back while building the program would break trace capture, so this is an upper bound
// that does not depend on the data.
//
// One origin sends one destination at most seq_len_per_chip * min(num_experts_per_tok,
// experts_per_chip) tokens, because a token picks distinct experts and the destination hosts
// experts_per_chip of them. The destinations a section carries sit at distance 1..m-1. A destination at
// distance dd is fed by m-dd origins: exactly one is a full m away and splits across every stream, and
// the rest split across the num_links streams going that way. One spare page per chunk covers rounding
// when a chunk is split.
uint32_t fwd_pages_per_stream(
    uint32_t ring_extent,
    uint32_t num_links,
    uint32_t seq_len_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t experts_per_chip);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
