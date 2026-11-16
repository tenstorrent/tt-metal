// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "combine_fabric2d_placement.hpp"
#include "kernels/dataflow/combine_fabric2d_chunk.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// One unit of work for one stream, in execution order. An assignment names a run and a share of it, not a
// page range: run boundaries live in the control tensors and are only knowable on device.
struct Assignment {
    bool is_relay = false;
    uint32_t relay_chunk = 0;  // is_relay: which chunk of this stream's region
    uint32_t dst_chip_id = 0;  // !is_relay: fabric name of the destination chip
    uint32_t dst_dg_index = 0;  // !is_relay: that chip's index in the dispatch group, indexing expert_offsets
    uint32_t distance = 0;      // !is_relay: hops to that chip, which names the channel it is queued in
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// `ring_chip_ids` holds the fabric chip id of each row of the ring, so this needs nothing from the mesh API.
std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_dg_index, uint32_t num_links);

// In the order the upstream chip emits them into its region, which is how both sides agree on positions.
std::vector<combine_chunk::ChunkDescriptor> forwarding_chunks(
    StreamId stream, uint32_t my_dg_index, uint32_t ring_extent, uint32_t num_links);

// A relay chunk is one (source, destination) pair whose path passes through this chip and continues.
// Summing over upstream distance k >= 1 the movements from that source whose distance exceeds k gives
// sum_{k=1..m-1} (m-k) = m(m-1)/2 for m = extent/2. Every stream is sized to this, the worse of the two
// directions; the one not carrying the diametrically-opposite chip needs only (m-1)(m-2)/2.
constexpr uint32_t relay_chunks_per_stream(uint32_t ring_extent) {
    const uint32_t m = ring_extent / 2;
    return m * (m - 1) / 2;
}

// The forwarding region is one channel per MOVEMENT CLASS: the (hops upstream, hops downstream) pair a
// page is in, which for an extent-8 ring is (-1,+3) (-1,+2) (-1,+1) (-2,+2) (-2,+1) (-3,+1). Two
// properties follow from splitting by class rather than anything coarser, and both are load-bearing:
//
//   DRAIN. A class with dst == +1 is delivered to the next chip's OUTPUT, so draining it needs no space
//   anywhere, and every other class moves one hop into a class nearer +1. The graph is a forest of paths,
//   never a cycle, so a blocked chip always has a legal move and capacity is not a correctness parameter.
//
//   ORDER. Each class at a chip is fed by exactly one class on the chip before it, so a channel has a
//   single producer and stays a dense FIFO, which is what lets both sides derive page positions from the
//   same generator with nothing exchanged.

// The (src, dst) hop offsets of every channel, in channel order, which is also chunk order.
std::vector<std::pair<int32_t, int32_t>> forwarding_classes(uint32_t dg_size);

inline uint32_t num_vcs(uint32_t dg_size) { return relay_chunks_per_stream(dg_size); }

// The channel a page in channel `vc` lands in on the next chip, or VC_NONE when it is delivered there.
uint32_t vc_after_hop(uint32_t vc, uint32_t dg_size);

// The channel this chip's own tokens for a destination `distance` hops away land in downstream, or
// VC_NONE at distance 1, which is a final write.
uint32_t vc_for_own(uint32_t distance, uint32_t dg_size);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
