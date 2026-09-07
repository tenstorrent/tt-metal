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

// One unit of work for one stream, in execution order.
//
// A relay pushes an incoming forwarding chunk one hop further, whole. Otherwise the work is this chip's
// own tokens for ONE destination chip, narrowed to a fraction of what it owes that chip: halved between
// the routing planes for destinations nearer than the diametrically opposite chip, split across all
// streams for that chip, which is equally far in both directions.
//
// Which tokens those are is NOT here. A token's destination is data-dependent (indices -> dispatch
// table), so the reader builds the per-destination token list on device and this only names the
// destination and the share.
struct Assignment {
    bool is_relay = false;
    uint32_t relay_chunk = 0;  // is_relay: which chunk of this stream's region
    uint32_t dst_chip_id = 0;  // !is_relay: fabric name of the destination chip
    uint32_t dst_row = 0;      // !is_relay: that chip's position on the dispatch axis
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// Work for every stream on one chip. `ring_chip_ids` holds the fabric chip id of each position on the
// dispatch axis, so this needs nothing from the mesh API.
std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_row, uint32_t num_links);

// Chunks a stream forwards, in the order the upstream chip emits them into this stream's region.
std::vector<dspf2d::ChunkDescriptor> forwarding_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links);

// Chunks this chip writes INTO the downstream chip's region for this stream, in emission order: its own
// destinations beyond the neighbour, furthest first, then the arrivals it passes on. A chunk bound for the
// neighbour itself is a final write to that chip's output and so is not here.
std::vector<dspf2d::ChunkDescriptor> outgoing_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links);

// The region is dense and carries no addresses, so a chunk is found only by counting the chunks before
// it: what a chip writes and what its neighbour reads must agree in identity AND order, or every chip on
// the axis ends up waiting for a chunk nobody wrote. Nothing is exchanged to establish that, so it is
// asserted here instead.
void validate_chunk_agreement(uint32_t ring_extent, uint32_t num_links);

// Own assignments a stream carries: one per destination it reaches sooner in its own direction, plus its
// share of the diametrically opposite chip.
constexpr uint32_t own_assignments_per_stream(uint32_t ring_extent) { return ring_extent / 2; }

// Relay chunks a stream receives, which is also how many its upstream neighbour emits into this stream's
// share of the forwarding buffer. Equals (own forwarding) + (re-forwarded), so upstream writer and
// downstream reader agree on the chunk count without exchanging anything.
//
// A chunk arriving at C in one direction is one (origin, destination) pair whose path passes through C and
// continues. Summing over upstream distance k >= 1 the movements from that origin whose distance exceeds k
// gives sum_{k=1..m-1} (m-k) = m(m-1)/2 for m = extent/2. That is the worse of the two directions -- the
// one not carrying the diametrically-opposite chip needs only (m-1)(m-2)/2 -- and every stream is sized
// alike. Each of these expands on device into experts_per_chip chunks.
constexpr uint32_t relay_chunks_per_stream(uint32_t ring_extent) {
    const uint32_t m = ring_extent / 2;
    return m * (m - 1) / 2;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
