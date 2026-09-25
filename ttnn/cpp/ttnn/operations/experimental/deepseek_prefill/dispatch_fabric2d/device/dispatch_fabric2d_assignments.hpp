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

// One unit of work for one stream: either a forward, which pushes an incoming forwarding chunk one hop
// further whole, or this chip's own tokens for ONE destination chip, narrowed to a share of what it
// owes that chip -- halved between the routing planes, or split across every stream for the
// diametrically opposite chip, which is equally far either way.
//
// WHICH tokens those are is not here. A token's destination is data-dependent (indices -> dispatch
// table), so the reader builds the per-destination list on device and this only names the destination
// and the share. The order (all own assignments, then all forwards) is fixed in the kernel; it is
// load-bearing, because the own phase is the slack a forward has between the upstream chunk being
// written and this stream consuming it.
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

// Chunks this chip writes INTO the downstream chip's section for this stream, in emission order: its own
// destinations beyond the neighbour, furthest first, then the arrivals it passes on. A chunk bound for the
// neighbour itself is a final write to that chip's output and so is not here.
std::vector<dspf2d::ChunkDescriptor> outgoing_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links);

// The section is dense and carries no addresses, so a chunk is found only by counting the chunks before
// it: what a chip writes and what its neighbour reads must agree in identity AND order, or every chip on
// the axis ends up waiting for a chunk nobody wrote. Nothing is exchanged to establish that, so it is
// asserted here instead.
void validate_chunk_agreement(uint32_t ring_extent, uint32_t num_links);

// Own assignments a stream carries: one per destination it reaches sooner in its own direction, plus its
// share of the diametrically opposite chip.
constexpr uint32_t own_assignments_per_stream(uint32_t ring_extent) { return ring_extent / 2; }

// Forward chunks a stream receives, which is also how many its upstream neighbour emits into this
// stream's section -- so writer and reader agree on the count without exchanging anything.
//
// A chunk is one (origin, destination) pair whose path passes through this chip and continues; summing
// over upstream distance k the pairs from that origin reaching further than k gives
// sum_{k=1..m-1} (m-k) = m(m-1)/2 for m = extent/2. Each of these expands on device into
// experts_per_chip chunks.
constexpr uint32_t forward_chunks_per_stream(uint32_t ring_extent) {
    const uint32_t m = ring_extent / 2;
    return m * (m - 1) / 2;
}

// Pages one stream's forwarding section must hold. The host cannot know a chunk's length --
// expert_offsets lives on device and reading it back at build time would sync mid-build and defeat
// trace capture -- so this bounds the section data-independently.
//
// One origin sends one destination at most seq_len_per_chip * min(num_experts_per_tok,
// experts_per_chip) tokens, because a token picks DISTINCT experts and the destination hosts
// experts_per_chip of them. The destinations a section carries sit at distance 1..m-1; a destination at
// distance dd is fed by m-dd origins, of which exactly one is a full m away and so splits across every
// stream while the rest split between the two planes. One spare page per chunk covers integer slicing.
uint32_t fwd_pages_per_stream(
    uint32_t ring_extent,
    uint32_t num_links,
    uint32_t seq_len_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t experts_per_chip);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
