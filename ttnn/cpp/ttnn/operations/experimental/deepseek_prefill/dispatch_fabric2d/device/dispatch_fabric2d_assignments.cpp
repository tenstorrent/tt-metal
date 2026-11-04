// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_assignments.hpp"

#include <algorithm>
#include <set>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

namespace {

// Every remote destination must be claimed exactly once, in whole. How many tokens go to each is
// data-dependent and unknown here, so the check is on the fractions: for each destination the claimed
// shares must partition [0,1), which for split_idx in [0,split_count) means the indices are distinct and
// there are split_count of them.
void validate_coverage(const std::map<StreamId, std::vector<Assignment>>& per_stream, uint32_t ring_extent) {
    std::map<uint32_t, std::set<uint32_t>> claimed;
    std::map<uint32_t, uint32_t> split_count;
    for (const auto& [stream, list] : per_stream) {
        for (const auto& a : list) {
            if (a.is_forward) {
                continue;
            }
            TT_FATAL(
                a.split_idx < a.split_count,
                "dispatch_fabric2d: share {} is out of range for a {}-way split",
                a.split_idx,
                a.split_count);
            TT_FATAL(
                claimed[a.dst_pos].insert(a.split_idx).second,
                "dispatch_fabric2d: two streams both claim share {} of {} for position {}",
                a.split_idx,
                a.split_count,
                a.dst_pos);
            auto& want = split_count[a.dst_pos];
            TT_FATAL(
                want == 0 || want == a.split_count,
                "dispatch_fabric2d: position {} is split {} ways by one stream and {} ways by another; the shares "
                "would not tile what this chip owes it",
                a.dst_pos,
                want,
                a.split_count);
            want = a.split_count;
        }
    }
    TT_FATAL(
        claimed.size() == ring_extent - 1,
        "dispatch_fabric2d: streams cover {} of the {} remote chips on the dispatch axis",
        claimed.size(),
        ring_extent - 1);
    for (const auto& [pos, shares] : claimed) {
        TT_FATAL(
            shares.size() == split_count.at(pos),
            "dispatch_fabric2d: position {} has {} of its {} shares claimed, so part of what this chip owes it "
            "would never be sent",
            pos,
            shares.size(),
            split_count.at(pos));
    }
}

// Every descriptor a chip forwards, as (origin, destination) hop offsets from that chip along the stream's
// direction: origins upstream are negative, destinations downstream are positive. A descriptor addressed to the
// forwarding chip itself is delivered, not forwarded, so the nearest destination is 1 and the furthest
// origin is -(m - 1). The offsets are the same on every chip, so this takes only the extent.
//
// The order is the order the upstream chip writes the descriptors (see validate_descriptor_agreement). Upstream
// writes its own destinations furthest first (the origin == -1 group) before any descriptor it forwards, so
// origins run outwards from -1.
std::vector<std::pair<int32_t, int32_t>> descriptors_in_forwarder_ref_frame(uint32_t ring_extent) {
    const int32_t m = static_cast<int32_t>(ring_extent / 2);
    std::vector<std::pair<int32_t, int32_t>> descriptors;
    descriptors.reserve(forward_descriptors_per_stream(ring_extent));
    for (int32_t origin = -1; origin > -m; origin--) {
        for (int32_t dst = origin + m; dst >= 1; dst--) {
            descriptors.emplace_back(origin, dst);
        }
    }
    return descriptors;
}

}  // namespace

std::vector<dspf2d::ChunkDescriptor> forwarding_descriptors(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links) {
    const bool is_cw = (stream % 2) == 0;
    const uint32_t link = stream / 2;
    const uint32_t m = ring_extent / 2;
    const int32_t travel = is_cw ? 1 : -1;
    const int32_t extent = static_cast<int32_t>(ring_extent);

    std::vector<dspf2d::ChunkDescriptor> descriptors;
    for (const auto& [origin, dst] : descriptors_in_forwarder_ref_frame(ring_extent)) {
        // A counter-clockwise stream mirrors the offsets through 0; adding this chip's position then gives a
        // position on the dispatch axis.
        const uint32_t distance = static_cast<uint32_t>(dst - origin);
        descriptors.push_back(dspf2d::ChunkDescriptor{
            .origin_pos = static_cast<uint32_t>((static_cast<int32_t>(my_pos) + travel * origin + extent) % extent),
            .dst_pos = static_cast<uint32_t>((static_cast<int32_t>(my_pos) + travel * dst + extent) % extent),
            .split_idx = distance == m ? stream : link,
            .split_count = distance == m ? stream_count(num_links) : num_links});
    }
    return descriptors;
}

// A channel IS a forward descriptor's movement class, so this is the descriptor enumeration itself rather
// than a parallel list: descriptor d of a stream and channel d are the same (origin, destination) offsets,
// which is what lets the reader map an arriving descriptor to its channel by index alone.
std::vector<std::pair<int32_t, int32_t>> forwarding_classes(uint32_t ring_extent) {
    return descriptors_in_forwarder_ref_frame(ring_extent);
}

uint32_t vc_after_hop(uint32_t vc, uint32_t ring_extent) {
    const auto classes = forwarding_classes(ring_extent);
    const auto [src, dst] = classes[vc];
    if (dst == 1) {
        return dspf2d::VC_NONE;  // delivered on the next chip, not queued there
    }
    // One hop on: the origin is one further behind and the destination one nearer.
    for (uint32_t v = 0; v < classes.size(); v++) {
        if (classes[v].first == src - 1 && classes[v].second == dst - 1) {
            return v;
        }
    }
    TT_FATAL(false, "dispatch_fabric2d: class ({},{}) has no successor channel", src, dst);
    return dspf2d::VC_NONE;
}

uint32_t vc_for_own(uint32_t distance, uint32_t ring_extent) {
    if (distance <= 1) {
        return dspf2d::VC_NONE;  // our neighbour's own output; never queued
    }
    const auto classes = forwarding_classes(ring_extent);
    for (uint32_t v = 0; v < classes.size(); v++) {
        if (classes[v].first == -1 && classes[v].second == static_cast<int32_t>(distance) - 1) {
            return v;
        }
    }
    TT_FATAL(false, "dispatch_fabric2d: own distance {} has no channel", distance);
    return dspf2d::VC_NONE;
}

std::vector<dspf2d::ChunkDescriptor> outgoing_descriptors(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links) {
    const bool is_cw = (stream % 2) == 0;
    const uint32_t link = stream / 2;
    const uint32_t m = ring_extent / 2;
    const int32_t travel = is_cw ? 1 : -1;
    const int32_t extent = static_cast<int32_t>(ring_extent);
    const auto pos_at = [&](int32_t offset) {
        return static_cast<uint32_t>((static_cast<int32_t>(my_pos) + travel * offset + extent) % extent);
    };
    const uint32_t downstream_pos = pos_at(1);

    std::vector<dspf2d::ChunkDescriptor> descriptors;
    for (uint32_t d = m; d >= 2; d--) {
        descriptors.push_back(dspf2d::ChunkDescriptor{
            .origin_pos = my_pos,
            .dst_pos = pos_at(static_cast<int32_t>(d)),
            .split_idx = d == m ? stream : link,
            .split_count = d == m ? stream_count(num_links) : num_links});
    }
    for (const auto& descriptor : forwarding_descriptors(stream, my_pos, ring_extent, num_links)) {
        if (descriptor.dst_pos != downstream_pos) {
            descriptors.push_back(descriptor);
        }
    }
    return descriptors;
}

void validate_descriptor_agreement(uint32_t ring_extent, uint32_t num_links) {
    for (uint32_t pos = 0; pos < ring_extent; pos++) {
        for (uint32_t stream = 0; stream < stream_count(num_links); stream++) {
            const int32_t travel = (stream % 2) == 0 ? 1 : -1;
            const uint32_t downstream_pos = static_cast<uint32_t>(
                (static_cast<int32_t>(pos) + travel + static_cast<int32_t>(ring_extent)) %
                static_cast<int32_t>(ring_extent));
            const auto written = outgoing_descriptors(stream, pos, ring_extent, num_links);
            const auto expected = forwarding_descriptors(stream, downstream_pos, ring_extent, num_links);
            TT_FATAL(
                written.size() == expected.size(),
                "dispatch_fabric2d: position {} stream {} writes {} descriptors into position {}, which reads {}",
                pos,
                stream,
                written.size(),
                downstream_pos,
                expected.size());
            for (size_t i = 0; i < written.size(); i++) {
                TT_FATAL(
                    written[i].origin_pos == expected[i].origin_pos && written[i].dst_pos == expected[i].dst_pos &&
                        written[i].split_idx == expected[i].split_idx &&
                        written[i].split_count == expected[i].split_count,
                    "dispatch_fabric2d: position {} stream {} writes descriptor {} as (origin {}, dst {}, share {}/{}) "
                    "but position {} reads it as (origin {}, dst {}, share {}/{})",
                    pos,
                    stream,
                    i,
                    written[i].origin_pos,
                    written[i].dst_pos,
                    written[i].split_idx,
                    written[i].split_count,
                    downstream_pos,
                    expected[i].origin_pos,
                    expected[i].dst_pos,
                    expected[i].split_idx,
                    expected[i].split_count);
            }
        }
    }
}

std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_pos, uint32_t num_links) {
    const uint32_t extent = static_cast<uint32_t>(ring_chip_ids.size());
    const uint32_t m = own_assignments_per_stream(extent);
    TT_FATAL(extent >= 4 && extent % 2 == 0, "dispatch_fabric2d: axis extent {} must be even and at least 4", extent);
    TT_FATAL(my_pos < extent, "dispatch_fabric2d: position {} is outside a {}-chip axis", my_pos, extent);

    std::map<StreamId, std::vector<Assignment>> per_stream;
    for (uint32_t link = 0; link < num_links; link++) {
        for (bool is_cw : {true, false}) {
            const StreamId stream = make_stream_id(link, is_cw);
            auto& list = per_stream[stream];

            auto own = [&](uint32_t distance, uint32_t split_idx, uint32_t split_count) {
                const uint32_t pos = (my_pos + (is_cw ? distance : extent - distance)) % extent;
                list.push_back(Assignment{
                    .dst_chip_id = ring_chip_ids[pos],
                    .dst_pos = pos,
                    .split_idx = split_idx,
                    .split_count = split_count});
            };

            // Furthest destination first, the order outgoing_descriptors writes them into the downstream fwd_section.
            for (uint32_t j = 1; j <= m; j++) {
                const uint32_t distance = m - j + 1;
                if (distance == m) {
                    own(m, stream, stream_count(num_links));  // the opposite chip, shared by every stream
                } else {
                    own(distance, link, num_links);
                }
            }
            const uint32_t forwards = forward_descriptors_per_stream(extent);
            for (uint32_t c = 0; c < forwards; c++) {
                list.push_back(Assignment{.is_forward = true});
            }
            TT_FATAL(
                list.size() == m + forwards,
                "dispatch_fabric2d: stream {} got {} work items, expected {} own + {} forward",
                stream,
                list.size(),
                m,
                forwards);
        }
    }
    validate_coverage(per_stream, extent);
    return per_stream;
}

std::vector<dspf2d::ChunkDescriptor> incoming_final_descriptors(
    const std::vector<uint32_t>& ring_chip_ids, StreamId stream, uint32_t my_pos, uint32_t num_links) {
    const uint32_t extent = static_cast<uint32_t>(ring_chip_ids.size());
    const int32_t travel = (stream % 2) == 0 ? 1 : -1;
    const uint32_t upstream_pos = static_cast<uint32_t>(
        (static_cast<int32_t>(my_pos) - travel + static_cast<int32_t>(extent)) % static_cast<int32_t>(extent));

    std::vector<dspf2d::ChunkDescriptor> finals;
    // The upstream chip's own share for this chip: its direct write, never staged in a fwd_section.
    const auto upstream_work = generate_assignments(ring_chip_ids, upstream_pos, num_links);
    for (const auto& a : upstream_work.at(stream)) {
        if (!a.is_forward && a.dst_pos == my_pos) {
            finals.push_back(dspf2d::ChunkDescriptor{
                .origin_pos = upstream_pos, .dst_pos = my_pos, .split_idx = a.split_idx, .split_count = a.split_count});
        }
    }
    // What it forwards for this chip: the descriptors its reader delivers rather than passing on.
    for (const auto& d : forwarding_descriptors(stream, upstream_pos, extent, num_links)) {
        if (d.dst_pos == my_pos) {
            finals.push_back(d);
        }
    }
    TT_FATAL(
        finals.size() == own_assignments_per_stream(extent),
        "dispatch_fabric2d: position {} stream {} receives {} final-write chunks, expected one per origin up to {} "
        "hops upstream",
        my_pos,
        stream,
        finals.size(),
        own_assignments_per_stream(extent));
    return finals;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
