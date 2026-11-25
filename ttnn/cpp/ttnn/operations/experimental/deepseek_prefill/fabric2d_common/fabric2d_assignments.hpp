// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The ring schedule shared by dispatch_fabric2d and combine_fabric2d.
//
// The two ops move data in opposite senses -- dispatch sends tokens to the chips hosting their experts,
// combine sends the expert outputs back -- but on the ring each is the same all-to-all: every chip owes
// every other chip one run, sent the shorter way round (both ways for the opposite chip). The op's direction
// changes which tensor a run is read from and written to, not which chip sends to which, and both ops name a
// chip by its position on the ring axis, so who goes to whom, on which stream, in which order and through
// which channel is one computation. Only how a chip finds the tokens it owes another (data-dependent, on
// device) and what the kernels do with a chunk differ, and none of that is here.

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::fabric2d {

// A stream is one routing plane travelled in one direction along the ring axis, encoded as
// link * 2 + (clockwise ? 0 : 1) -- the encoding both ops' placements and kernels use.
using StreamId = uint32_t;

// "Not queued anywhere" -- a page delivered to the next chip's output rather than forwarded again. The
// same value as each op's kernel-side VC_NONE, which is what the host writes into the channel tables.
constexpr uint32_t VC_NONE = 0xFFFFFFFFu;

// One share of one (origin, destination) run, as positions on the ring axis.
struct RingChunk {
    uint32_t origin_pos = 0;
    uint32_t dst_pos = 0;
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// The share `stream` carries of `origin_pos`'s run for the chip `distance` hops downstream. The diametrically
// opposite chip is equally far in both directions, so its run is split across every stream; a nearer one is
// reached in one direction only, so its run is split across the num_links streams going that way.
inline RingChunk stream_chunk(
    StreamId stream, uint32_t origin_pos, uint32_t distance, uint32_t ring_extent, uint32_t num_links) {
    const bool opposite = distance == ring_extent / 2;
    const uint32_t step = (stream % 2) == 0 ? distance : ring_extent - distance;
    return RingChunk{
        .origin_pos = origin_pos,
        .dst_pos = (origin_pos + step) % ring_extent,
        .split_idx = opposite ? stream : stream / 2,
        .split_count = opposite ? num_links * 2 : num_links};
}

// One unit of work for one stream, in execution order: either a forward, which passes a chunk received from
// upstream one hop further, or a share of this chip's own run for one destination chip. An assignment names
// a run and a share of it, not a page range: run lengths are data-dependent and only knowable on device.
struct Assignment {
    bool is_forward = false;
    uint32_t forward_chunk = 0;  // is_forward: which chunk of this stream's forwarding region
    uint32_t dst_chip_id = 0;    // !is_forward: fabric chip id of the destination chip
    uint32_t dst_pos = 0;        // !is_forward: that chip's position on the ring axis
    uint32_t distance = 0;       // !is_forward: hops to that chip, which names the channel it is queued in
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// Own assignments a stream carries: one per destination it reaches sooner in its own direction, plus its
// share of the diametrically opposite chip.
constexpr uint32_t own_assignments_per_stream(uint32_t ring_extent) { return ring_extent / 2; }

// Forward chunks a stream receives, which is also how many the upstream chip writes into this stream's
// region, so writer and reader agree on the count without exchanging it.
//
// A chunk is one (origin, destination) pair whose path passes through this chip and continues. Summing over
// upstream distance k the pairs from that origin reaching further than k gives sum_{k=1..m-1} (m-k) =
// m(m-1)/2 for m = extent/2.
constexpr uint32_t forward_chunks_per_stream(uint32_t ring_extent) {
    const uint32_t m = ring_extent / 2;
    return m * (m - 1) / 2;
}

// The forwarding region is one channel per MOVEMENT CLASS: the (hops upstream, hops downstream) pair a page
// is in, which for an extent-8 ring is (-1,+3) (-1,+2) (-1,+1) (-2,+2) (-2,+1) (-3,+1). That is the same set
// the forward chunks enumerate -- a chunk is an (origin, destination) pair crossing this chip -- so a channel
// index IS an index into that list. Two properties follow from splitting by class rather than anything
// coarser, and both are load-bearing:
//
//   DRAIN. A class with dst == +1 is delivered to the next chip's OUTPUT, so draining it needs no space
//   anywhere, and every other class moves one hop into a class nearer +1. The graph is a forest of paths,
//   never a cycle, so a blocked chip always has a legal move and capacity is not a correctness parameter.
//
//   ORDER. Each class at a chip is fed by exactly one class on the chip before it, so a channel has a
//   single producer and stays a dense FIFO, which is what lets both chips derive page positions from the
//   same running count with nothing exchanged.
constexpr uint32_t num_vcs(uint32_t ring_extent) { return forward_chunks_per_stream(ring_extent); }

// The (src, dst) hop offsets of every channel, in channel order, which is also forward chunk order: origins
// upstream are negative, destinations downstream are positive. A chunk addressed to the forwarding chip itself
// is delivered, not forwarded, so the nearest destination is 1 and the furthest origin is -(m - 1). The offsets
// are the same on every chip, so this takes only the extent.
//
// The order is the order the upstream chip writes the chunks, which the forwarder must match: the region is
// dense and holds no per-chunk addresses, so a chunk is found only by counting those before it. Upstream
// writes its own destinations furthest first (the whole origin == -1 group) before any chunk it forwards, so
// origins run outwards from -1.
inline std::vector<std::pair<int32_t, int32_t>> forwarding_classes(uint32_t ring_extent) {
    const int32_t m = static_cast<int32_t>(ring_extent / 2);
    std::vector<std::pair<int32_t, int32_t>> classes;
    classes.reserve(forward_chunks_per_stream(ring_extent));
    for (int32_t origin = -1; origin > -m; origin--) {
        for (int32_t dst = origin + m; dst >= 1; dst--) {
            classes.emplace_back(origin, dst);
        }
    }
    return classes;
}

// The channel a page in channel `vc` lands in on the next chip, or VC_NONE when it is delivered there.
inline uint32_t vc_after_hop(uint32_t vc, uint32_t ring_extent) {
    const auto classes = forwarding_classes(ring_extent);
    const auto [src, dst] = classes[vc];
    if (dst == 1) {
        return VC_NONE;  // delivered on the next chip, not queued there
    }
    // One hop on: the origin is one further behind and the destination one nearer.
    for (uint32_t v = 0; v < classes.size(); v++) {
        if (classes[v].first == src - 1 && classes[v].second == dst - 1) {
            return v;
        }
    }
    TT_FATAL(false, "fabric2d: class ({},{}) has no successor channel", src, dst);
    return VC_NONE;
}

// The channel this chip's own tokens for a destination `distance` hops away land in downstream, or
// VC_NONE at distance 1, which is a final write.
inline uint32_t vc_for_own(uint32_t distance, uint32_t ring_extent) {
    if (distance <= 1) {
        return VC_NONE;  // our neighbour's own output; never queued
    }
    const auto classes = forwarding_classes(ring_extent);
    for (uint32_t v = 0; v < classes.size(); v++) {
        if (classes[v].first == -1 && classes[v].second == static_cast<int32_t>(distance) - 1) {
            return v;
        }
    }
    TT_FATAL(false, "fabric2d: own distance {} has no channel", distance);
    return VC_NONE;
}

// The chunks a stream forwards, in the order the upstream chip emits them into this stream's region, which is
// how both sides agree on positions. Channel v and chunk v are the same (origin, destination) offsets, which is
// what lets the reader map an arriving chunk to its channel by index alone.
inline std::vector<RingChunk> forwarding_chunks(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links) {
    std::vector<RingChunk> chunks;
    chunks.reserve(forward_chunks_per_stream(ring_extent));
    for (const auto& [origin, dst] : forwarding_classes(ring_extent)) {
        // Upstream is against the stream's direction, so a clockwise stream steps back and a counter-clockwise
        // one forward.
        const uint32_t back = static_cast<uint32_t>(-origin);
        const uint32_t origin_pos = ((stream % 2) == 0 ? my_pos + ring_extent - back : my_pos + back) % ring_extent;
        chunks.push_back(stream_chunk(stream, origin_pos, static_cast<uint32_t>(dst - origin), ring_extent, num_links));
    }
    return chunks;
}

namespace detail {

// Every remote destination must be claimed exactly once, in whole. How many tokens go to each is
// data-dependent and unknown here, so the check is on the FRACTIONS: for each destination the claimed
// shares must partition [0,1), which for split_idx in [0,split_count) means the indices are distinct and
// there are split_count of them.
inline void validate_coverage(const std::map<StreamId, std::vector<Assignment>>& per_stream, uint32_t ring_extent) {
    std::map<uint32_t, std::set<uint32_t>> claimed;
    std::map<uint32_t, uint32_t> split_count;
    for (const auto& [stream, list] : per_stream) {
        for (const auto& a : list) {
            if (a.is_forward) {
                continue;
            }
            TT_FATAL(
                a.split_idx < a.split_count,
                "fabric2d: share {} is out of range for a {}-way split",
                a.split_idx,
                a.split_count);
            TT_FATAL(
                claimed[a.dst_pos].insert(a.split_idx).second,
                "fabric2d: two streams both claim share {} of {} for position {}",
                a.split_idx,
                a.split_count,
                a.dst_pos);
            auto& want = split_count[a.dst_pos];
            TT_FATAL(
                want == 0 || want == a.split_count,
                "fabric2d: position {} is split {} ways by one stream and {} ways by another; the shares would "
                "not tile what this chip owes it",
                a.dst_pos,
                want,
                a.split_count);
            want = a.split_count;
        }
    }
    TT_FATAL(
        claimed.size() == ring_extent - 1,
        "fabric2d: streams cover {} of the {} remote chips on the ring axis",
        claimed.size(),
        ring_extent - 1);
    for (const auto& [pos, shares] : claimed) {
        TT_FATAL(
            shares.size() == split_count.at(pos),
            "fabric2d: position {} has {} of its {} shares claimed, so part of what this chip owes it would never "
            "be sent",
            pos,
            shares.size(),
            split_count.at(pos));
    }
}

}  // namespace detail

// Work for every stream on one chip. `ring_chip_ids` holds the fabric chip id of each position on the ring
// axis, so this needs nothing from the mesh API.
inline std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_pos, uint32_t num_links) {
    const uint32_t extent = static_cast<uint32_t>(ring_chip_ids.size());
    const uint32_t m = own_assignments_per_stream(extent);
    TT_FATAL(extent >= 4 && extent % 2 == 0, "fabric2d: ring extent {} must be even and at least 4", extent);
    TT_FATAL(my_pos < extent, "fabric2d: position {} is outside a {}-chip ring", my_pos, extent);

    std::map<StreamId, std::vector<Assignment>> per_stream;
    for (StreamId stream = 0; stream < num_links * 2; stream++) {
        auto& list = per_stream[stream];
        // Furthest destination first, then the forwards. Furthest first is the order the upstream chip writes
        // its own chunks into the downstream region (see forwarding_classes). Doing our own work first is what
        // puts DISTANCE between a forward and the upstream chunk feeding it: the upstream emits that chunk at
        // the top of its pass, we consume it at the bottom of ours, and the whole own phase is the slack the
        // ring has to absorb jitter with.
        //
        // Emission order is unaffected: a forward for our neighbour and the nearest own assignment are both
        // final writes, so neither puts a chunk downstream, and the forwarding chunks come off in the same
        // sequence either way.
        for (uint32_t distance = m; distance >= 1; distance--) {
            const RingChunk chunk = stream_chunk(stream, my_pos, distance, extent, num_links);
            list.push_back(Assignment{
                .dst_chip_id = ring_chip_ids[chunk.dst_pos],
                .dst_pos = chunk.dst_pos,
                .distance = distance,
                .split_idx = chunk.split_idx,
                .split_count = chunk.split_count});
        }
        for (uint32_t c = 0; c < forward_chunks_per_stream(extent); c++) {
            list.push_back(Assignment{.is_forward = true, .forward_chunk = c});
        }
    }
    detail::validate_coverage(per_stream, extent);
    return per_stream;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::fabric2d
