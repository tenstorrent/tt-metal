// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_assignments.hpp"

#include <set>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// Every remote destination's every run must be claimed exactly once, in whole. Token counts are unknown
// here, so the check is on the FRACTIONS: for each destination the claimed shares must partition [0,1),
// which for split_idx in [0,split_count) means the indices are distinct and there are split_count of them.
void validate_coverage(const std::map<StreamId, std::vector<Assignment>>& per_stream, uint32_t ring_extent) {
    std::map<uint32_t, std::set<uint32_t>> claimed;
    std::map<uint32_t, uint32_t> split_count;
    for (const auto& [stream, list] : per_stream) {
        for (const auto& a : list) {
            if (a.is_relay) {
                continue;
            }
            TT_FATAL(
                a.split_idx < a.split_count,
                "combine_fabric2d: share {} is out of range for a {}-way split",
                a.split_idx,
                a.split_count);
            TT_FATAL(
                claimed[a.dst_dg_index].insert(a.split_idx).second,
                "combine_fabric2d: two streams both claim share {} of {} for dispatch-group index {}",
                a.split_idx,
                a.split_count,
                a.dst_dg_index);
            auto& want = split_count[a.dst_dg_index];
            TT_FATAL(
                want == 0 || want == a.split_count,
                "combine_fabric2d: dispatch-group index {} is split {} ways by one stream and {} ways by another; the "
                "shares would "
                "not tile the run",
                a.dst_dg_index,
                want,
                a.split_count);
            want = a.split_count;
        }
    }
    TT_FATAL(
        claimed.size() == ring_extent - 1,
        "combine_fabric2d: streams cover {} of the {} remote chips on the ring",
        claimed.size(),
        ring_extent - 1);
    for (const auto& [dg_index, shares] : claimed) {
        TT_FATAL(
            shares.size() == split_count.at(dg_index),
            "combine_fabric2d: dispatch-group index {} has {} of its {} shares claimed, so part of every run to that "
            "chip would "
            "never be sent",
            dg_index,
            shares.size(),
            split_count.at(dg_index));
    }
}

}  // namespace

namespace {

// Every chunk a chip forwards, as (source, destination) hop offsets from that chip along the stream's own
// direction: sources upstream, so negative, destinations downstream, so positive. A chunk whose destination
// is the forwarder itself is delivered rather than forwarded and so is not here, which is why the nearest
// destination is 1 and the furthest source is -(dg_size/2 - 1). Offsets are the same on every chip, so this
// takes only the size of the dispatch group, which is assumed even.
//
// The order is the order the upstream chip writes the chunks, which the forwarder must match: the region is
// dense and holds no per-chunk addresses, so a chunk is found only by walking those before it. Upstream
// emits its own destinations furthest first (the whole src == -1 group) before any chunk it is itself
// relaying, so sources run outwards from -1.
std::vector<std::pair<int32_t, int32_t>> chunks_in_forwarder_ref_frame(uint32_t dg_size) {
    const int32_t half = static_cast<int32_t>(dg_size / 2);
    std::vector<std::pair<int32_t, int32_t>> chunks;
    chunks.reserve(relay_chunks_per_stream(dg_size));
    for (int32_t src = -1; src > -half; src--) {
        for (int32_t dst = src + half; dst >= 1; dst--) {
            chunks.emplace_back(src, dst);
        }
    }
    return chunks;
}

}  // namespace

std::vector<cmbf2d::ChunkDescriptor> forwarding_chunks(
    StreamId stream, uint32_t my_dg_index, uint32_t ring_extent, uint32_t num_links) {
    const bool is_cw = (stream % 2) == 0;
    const uint32_t link = stream / 2;
    const uint32_t m = ring_extent / 2;
    const int32_t travel = is_cw ? 1 : -1;

    std::vector<cmbf2d::ChunkDescriptor> chunks;
    for (const auto& [src, dst] : chunks_in_forwarder_ref_frame(ring_extent)) {
        // A counter-clockwise stream mirrors the offsets through 0; then both land on a dispatch-group index
        // by adding where this chip sits on the ring.
        const uint32_t distance = static_cast<uint32_t>(dst - src);
        chunks.push_back(cmbf2d::ChunkDescriptor{
            .origin_dg_index = static_cast<uint32_t>(
                (static_cast<int32_t>(my_dg_index) + travel * src + static_cast<int32_t>(ring_extent)) % ring_extent),
            .dst_dg_index = static_cast<uint32_t>(
                (static_cast<int32_t>(my_dg_index) + travel * dst + static_cast<int32_t>(ring_extent)) % ring_extent),
            .split_idx = distance == m ? stream : link,
            .split_count = distance == m ? stream_count(num_links) : num_links});
    }
    return chunks;
}

std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_dg_index, uint32_t num_links) {
    const uint32_t extent = static_cast<uint32_t>(ring_chip_ids.size());
    const uint32_t m = extent / 2;
    TT_FATAL(extent >= 3 && extent % 2 == 0, "combine_fabric2d: ring extent {} must be even and at least 3", extent);
    TT_FATAL(
        my_dg_index < extent,
        "combine_fabric2d: dispatch-group index {} is outside a {}-chip ring",
        my_dg_index,
        extent);

    std::map<StreamId, std::vector<Assignment>> per_stream;
    for (uint32_t link = 0; link < num_links; link++) {
        for (bool is_cw : {true, false}) {
            const StreamId stream = make_stream_id(link, is_cw);
            auto& list = per_stream[stream];

            auto own = [&](uint32_t distance, uint32_t split_idx, uint32_t split_count) {
                const uint32_t dg_index = (my_dg_index + (is_cw ? distance : extent - distance)) % extent;
                list.push_back(Assignment{
                    .dst_chip_id = ring_chip_ids[dg_index],
                    .dst_dg_index = dg_index,
                    .split_idx = split_idx,
                    .split_count = split_count});
            };

            // Furthest destination first, with relays interleaved so downstream streams get work early.
            // After the j-th own assignment the cumulative relay count is the triangular number
            // T(j-2) = (j-2)(j-1)/2, which for m=4 gives own4 own3 own2 relay0 own1 relay1 relay2 relay3..5
            // (own by distance, so own1 is the neighbour, which is delivered rather than forwarded).
            uint32_t relays = 0;
            for (uint32_t j = 1; j <= m; j++) {
                const uint32_t distance = m - j + 1;
                if (distance == m) {
                    own(m, stream, stream_count(num_links));  // the opposite chip, shared by every stream
                } else {
                    own(distance, link, num_links);
                }
                const uint32_t want = (j >= 2) ? (j - 2) * (j - 1) / 2 : 0;
                for (; relays < want && relays < relay_chunks_per_stream(extent); relays++) {
                    list.push_back(Assignment{.is_relay = true, .relay_chunk = relays});
                }
            }
            for (; relays < relay_chunks_per_stream(extent); relays++) {
                list.push_back(Assignment{.is_relay = true, .relay_chunk = relays});
            }
            TT_FATAL(
                list.size() == m + relay_chunks_per_stream(extent),
                "combine_fabric2d: stream {} got {} work items, expected {} own + {} relay",
                stream,
                list.size(),
                m,
                relay_chunks_per_stream(extent));
        }
    }
    validate_coverage(per_stream, extent);
    return per_stream;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
