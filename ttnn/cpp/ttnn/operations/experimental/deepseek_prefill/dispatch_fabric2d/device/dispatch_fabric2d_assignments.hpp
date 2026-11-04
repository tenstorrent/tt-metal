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

// One unit of work for one stream: either a forward, which passes a descriptor received from upstream one hop
// further, or a share of this chip's own tokens for one destination chip. The share is split across the
// num_links streams going that way, or across every stream for the diametrically opposite chip, which is
// equally far in both directions.
//
// A token's destination depends on the data (indices -> dispatch table), so the reader builds the
// per-destination token list on device; this names only the destination and the share. The kernel runs
// all own assignments before any forward, which gives the upstream chip time to write the descriptors the
// forwards consume.
struct Assignment {
    bool is_forward = false;
    uint32_t dst_chip_id = 0;  // !is_forward: fabric chip id of the destination chip
    uint32_t dst_pos = 0;      // !is_forward: that chip's position on the dispatch axis
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// Work for every stream on one chip. `ring_chip_ids` holds the fabric chip id of each position on the
// dispatch axis, so this needs nothing from the mesh API.
std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_pos, uint32_t num_links);

// Descriptors a stream forwards, in the order the upstream chip emits them into this stream's fwd_section.
std::vector<dspf2d::ChunkDescriptor> forwarding_descriptors(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links);

// Descriptors this chip writes into the downstream chip's fwd_section for this stream, in emission order:
// its own destinations beyond the downstream chip, furthest first, then the descriptors it forwards. A
// descriptor bound for the downstream chip itself is written straight to that chip's output, so it is not listed.
std::vector<dspf2d::ChunkDescriptor> outgoing_descriptors(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links);

// The chunks the upstream chip on `stream` writes straight into `my_pos`'s output, one per origin 1 .. extent/2 hops
// upstream: its own share for this chip, then every chunk it forwards whose destination is this chip. Built from
// generate_assignments and forwarding_descriptors for the upstream chip, the lists its program is built from, so the
// receive count the reader derives from them is what the upstream sender sends.
std::vector<dspf2d::ChunkDescriptor> incoming_final_descriptors(
    const std::vector<uint32_t>& ring_chip_ids, StreamId stream, uint32_t my_pos, uint32_t num_links);

// A fwd_section holds no per-chunk addresses, so a chunk is found by counting the chunks before it. What
// a chip writes and what the downstream chip reads must therefore agree in descriptor identity and order,
// or the reader waits for a chunk that is never written. The chips exchange nothing at run time, so this checks
// it on the host.
void validate_descriptor_agreement(uint32_t ring_extent, uint32_t num_links);

// Own assignments a stream carries: one per destination it reaches sooner in its own direction, plus its
// share of the diametrically opposite chip.
constexpr uint32_t own_assignments_per_stream(uint32_t ring_extent) { return ring_extent / 2; }

// Forward descriptors a stream receives, which is also how many the upstream chip writes into this stream's
// fwd_section, so writer and reader agree on the count without exchanging it.
//
// A descriptor is one (origin, destination) pair whose path passes through this chip and continues. Summing
// over upstream distance k the pairs from that origin reaching further than k gives
// sum_{k=1..m-1} (m-k) = m(m-1)/2 for m = extent/2. Each of these expands on device into
// experts_per_chip chunks.
constexpr uint32_t forward_descriptors_per_stream(uint32_t ring_extent) {
    const uint32_t m = ring_extent / 2;
    return m * (m - 1) / 2;
}

// ---------------------------------------------------------------------------------------------------
// Virtual channels.
//
// A forwarding region is not one queue but one per MOVEMENT CLASS: the (hops upstream, hops downstream)
// pair a page is in. That is the same set the forward descriptors enumerate -- a descriptor is an
// (origin, destination) pair crossing this chip -- so a channel index IS an index into that list and
// there are forward_descriptors_per_stream of them.
//
// Two properties make a channel per class safe at ANY depth, which is what lets the region shrink:
//
//   DRAIN. A class with dst == +1 is delivered to the next chip's OUTPUT, so draining it needs no space
//   anywhere. Every other class moves one hop and becomes a class nearer +1, so every chain ends in a
//   deliverable one within half_extent - 1 hops: (-1,+3) -> (-2,+2) -> (-3,+1) -> out. The graph is a
//   forest of paths, never a cycle, so a blocked chip always has a legal move and capacity stops being a
//   correctness parameter. Merge classes and that stops holding -- lump every dst > +1 together and
//   (-1,+3) -> (-2,+2) lands back in the same channel, which wraps the ring.
//
//   ORDER. Each class at a chip is fed by exactly one class on the chip before it, so a channel has a
//   single producer and stays a dense FIFO. Both chips derive page positions from the same running count
//   with nothing exchanged, exactly as the single flat section used to.
// ---------------------------------------------------------------------------------------------------

constexpr uint32_t num_vcs(uint32_t ring_extent) { return forward_descriptors_per_stream(ring_extent); }

// The (src, dst) hop offsets of every channel, in channel order.
std::vector<std::pair<int32_t, int32_t>> forwarding_classes(uint32_t ring_extent);

// The channel a page in channel `vc` lands in on the next chip, or VC_NONE when it is delivered there.
uint32_t vc_after_hop(uint32_t vc, uint32_t ring_extent);

// The channel this chip's own tokens for a destination `distance` hops away land in downstream. Only
// distance >= 2 is forwarded at all; distance 1 is a final write and has no channel.
uint32_t vc_for_own(uint32_t distance, uint32_t ring_extent);

// A stream's fwd_section is its slice of the DRAM forwarding buffer: fwd_pages_per_stream pages starting
// at page stream * fwd_pages_per_stream, divided into num_vcs channels of fwd_vc_pages each. Each page is
// one token plus its fwd_meta. The upstream chip's sender writes it and this chip's reader reads it.
//
// Nothing here is derived from how much traffic a launch carries. It used to be: the region was one flat
// queue that had to hold a whole launch, so it was sized from a data-independent upper bound on the chunks
// crossing a chip. Credit return made that obsolete -- a producer now waits for a page rather than assuming
// one -- and a channel is just a pipeline buffer, so its depth is given rather than computed.
constexpr uint32_t fwd_pages_per_stream(uint32_t ring_extent, uint32_t fwd_vc_pages) {
    return num_vcs(ring_extent) * fwd_vc_pages;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
