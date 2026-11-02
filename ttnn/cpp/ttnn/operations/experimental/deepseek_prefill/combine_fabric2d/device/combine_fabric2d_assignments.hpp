// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "combine_fabric2d_placement.hpp"
// Only the chunk layout, NOT the whole kernel interface: the overlapped fork includes this header, and
// its interface declares same-named helpers over the shared input struct that ADL would find too.
#include "kernels/dataflow/combine_fabric2d_chunk.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// One unit of work for one stream, in execution order.
//
// A relay pushes an incoming forwarding chunk one hop further, whole. Otherwise the work is this chip's own
// tokens for one destination chip, narrowed to a fraction of each run: halved between the routing planes for
// destinations nearer than the opposite chip, split across all streams for the opposite chip, which is
// equally far in both directions.
//
// Where a run starts and how long it is are NOT here. Run boundaries live in the caller's control tensors
// and are only knowable on device, so an assignment names the run (`dst_dg_index`) and the share
// (`split_idx`/`split_count`) and the kernel resolves both to page ranges.
struct Assignment {
    bool is_relay = false;
    uint32_t relay_chunk = 0;  // is_relay: which chunk of this stream's region
    uint32_t dst_chip_id = 0;  // !is_relay: fabric name of the destination chip
    uint32_t dst_dg_index = 0;  // !is_relay: that chip's index in the dispatch group, indexing expert_offsets
    uint32_t distance = 0;      // !is_relay: hops to that chip, which names the channel it is queued in
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// Work for every stream on `coord`. `ring_chip_ids` holds the fabric chip id of each row of the ring, so
// this needs nothing from the mesh API.
std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_dg_index, uint32_t num_links);

// Chunks a stream forwards, in the order the upstream chip emits them into its region.
std::vector<combine_chunk::ChunkDescriptor> forwarding_chunks(
    StreamId stream, uint32_t my_dg_index, uint32_t ring_extent, uint32_t num_links);



// Relay chunks a stream receives, which is also how many its upstream neighbour emits into this stream's
// share of the forwarding buffer. Equals (own forwarding) + (re-forwarded), so upstream writer and
// downstream reader agree on the chunk count without exchanging anything.
//
// A chunk arriving at C in one direction is one (source, destination) pair whose path passes through C and
// continues. Summing over upstream distance k >= 1 the movements from that source whose distance exceeds k
// gives sum_{k=1..m-1} (m-k) = m(m-1)/2 for m = extent/2. That is the worse of the two directions — the one
// not carrying the diametrically-opposite chip needs only (m-1)(m-2)/2 — and every stream is sized alike.
constexpr uint32_t relay_chunks_per_stream(uint32_t ring_extent) {
    const uint32_t m = ring_extent / 2;
    return m * (m - 1) / 2;
}

// ---------------------------------------------------------------------------------------------------
// Virtual channels.
//
// A forwarding region is not one queue but one per MOVEMENT CLASS: the (hops upstream, hops downstream)
// pair a page is in. For an extent-8 ring that is six -- (-1,+3) (-1,+2) (-1,+1) (-2,+2) (-2,+1) (-3,+1) --
// which is exactly what chunks_in_forwarder_ref_frame already enumerates, so a channel index IS an index
// into that list.
//
// Two properties make this work, and both are why the split is by class rather than by anything coarser:
//
//   DRAIN. A class with dst == +1 is delivered to the next chip's OUTPUT, so draining it needs no space
//   anywhere. Every other class moves one hop and becomes a class nearer +1, so every chain ends in a
//   deliverable one within half_extent - 1 hops: (-1,+3) -> (-2,+2) -> (-3,+1) -> out. The graph is a
//   forest of paths, never a cycle, so a blocked chip always has a legal move and capacity stops being a
//   correctness parameter. Merge classes and that stops holding -- lump every dst > +1 together and
//   (-1,+3) -> (-2,+2) lands back in the same channel, which wraps the ring.
//
//   ORDER. Each class at a chip is fed by exactly one class on the chip before it, so a channel has a
//   single producer and stays a dense FIFO. That is what lets both sides keep deriving page positions
//   from the same generator with nothing exchanged.
// ---------------------------------------------------------------------------------------------------

// The (src, dst) hop offsets of every channel, in channel order.
std::vector<std::pair<int32_t, int32_t>> forwarding_classes(uint32_t dg_size);

inline uint32_t num_vcs(uint32_t dg_size) { return relay_chunks_per_stream(dg_size); }

// The channel a page in channel `vc` lands in on the next chip, or VC_NONE when it is delivered there.
uint32_t vc_after_hop(uint32_t vc, uint32_t dg_size);

// The channel this chip's own tokens for a destination `distance` hops away land in downstream. Only
// distance >= 2 is forwarded at all; distance 1 is a final write and has no channel.
uint32_t vc_for_own(uint32_t distance, uint32_t dg_size);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
