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
                claimed[a.dst_row].insert(a.split_idx).second,
                "combine_fabric2d: two streams both claim share {} of {} for row {}",
                a.split_idx,
                a.split_count,
                a.dst_row);
            auto& want = split_count[a.dst_row];
            TT_FATAL(
                want == 0 || want == a.split_count,
                "combine_fabric2d: row {} is split {} ways by one stream and {} ways by another; the shares would "
                "not tile the run",
                a.dst_row,
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
    for (const auto& [row, shares] : claimed) {
        TT_FATAL(
            shares.size() == split_count.at(row),
            "combine_fabric2d: row {} has {} of its {} shares claimed, so part of every run to that chip would "
            "never be sent",
            row,
            shares.size(),
            split_count.at(row));
    }
}

}  // namespace

namespace {

// A chunk's (origin, destination) as hop offsets along the stream's own direction, in emission order. The
// list a chip emits is its own forwarding destinations, furthest first, followed by the arrivals it passes
// on — which is the same list one hop older with the arrivals for its neighbour removed. That recursion has
// a fixed point, so the offsets are the same for every chip and need no simulation.
std::vector<std::pair<int32_t, int32_t>> emitted_offsets(uint32_t m) {
    std::vector<std::pair<int32_t, int32_t>> emitted;
    emitted.reserve(relay_chunks_per_stream(2 * m));
    for (uint32_t hop = 0; hop + 2 <= m; hop++) {
        for (uint32_t dst = m - hop; dst >= 2; dst--) {
            emitted.emplace_back(-static_cast<int32_t>(hop), static_cast<int32_t>(dst));
        }
    }
    return emitted;
}

std::vector<ChunkDescriptor> chunks_at_offsets(
    const std::vector<std::pair<int32_t, int32_t>>& offsets,
    StreamId stream,
    uint32_t my_row,
    uint32_t extent,
    uint32_t num_links,
    int32_t hop_shift) {
    const bool is_cw = (stream % 2) == 0;
    const uint32_t link = stream / 2;
    const uint32_t m = extent / 2;
    const int32_t travel = is_cw ? 1 : -1;

    std::vector<ChunkDescriptor> chunks;
    chunks.reserve(offsets.size());
    for (const auto& [origin_off, dst_off] : offsets) {
        const int32_t origin = origin_off + hop_shift;
        const int32_t dst = dst_off + hop_shift;
        const uint32_t distance = static_cast<uint32_t>(dst - origin);
        chunks.push_back(ChunkDescriptor{
            .origin_row = static_cast<uint32_t>(
                (static_cast<int32_t>(my_row) + travel * origin + static_cast<int32_t>(extent)) % extent),
            .dst_row = static_cast<uint32_t>(
                (static_cast<int32_t>(my_row) + travel * dst + static_cast<int32_t>(extent)) % extent),
            .split_idx = distance == m ? stream : link,
            .split_count = distance == m ? stream_count(num_links) : num_links});
    }
    return chunks;
}

}  // namespace

std::vector<ChunkDescriptor> incoming_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links) {
    return chunks_at_offsets(emitted_offsets(ring_extent / 2), stream, my_row, ring_extent, num_links, -1);
}

std::vector<ChunkDescriptor> outgoing_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links) {
    return chunks_at_offsets(emitted_offsets(ring_extent / 2), stream, my_row, ring_extent, num_links, 0);
}

std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_row, uint32_t num_links) {
    const uint32_t extent = static_cast<uint32_t>(ring_chip_ids.size());
    const uint32_t m = extent / 2;
    TT_FATAL(extent >= 3 && extent % 2 == 0, "combine_fabric2d: ring extent {} must be even and at least 3", extent);
    TT_FATAL(my_row < extent, "combine_fabric2d: row {} is outside a {}-chip ring", my_row, extent);

    std::map<StreamId, std::vector<Assignment>> per_stream;
    for (uint32_t link = 0; link < num_links; link++) {
        for (bool is_cw : {true, false}) {
            const StreamId stream = make_stream_id(link, is_cw);
            auto& list = per_stream[stream];

            auto own = [&](uint32_t distance, uint32_t split_idx, uint32_t split_count) {
                const uint32_t row = (my_row + (is_cw ? distance : extent - distance)) % extent;
                list.push_back(Assignment{
                    .dst_chip_id = ring_chip_ids[row],
                    .dst_row = row,
                    .split_idx = split_idx,
                    .split_count = split_count});
            };

            // Furthest destination first, with relays interleaved so downstream streams get work early.
            // After the j-th own assignment the cumulative relay count is the triangular number
            // T(j-2) = (j-2)(j-1)/2, which for m=4 gives own3 own2 relay0 own1 relay1 relay2 own0 relay3..5.
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
