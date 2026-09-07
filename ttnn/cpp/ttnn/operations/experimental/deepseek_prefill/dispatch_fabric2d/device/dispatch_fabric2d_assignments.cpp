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
// data-dependent and unknown here, so the check is on the FRACTIONS: for each destination the claimed
// shares must partition [0,1), which for split_idx in [0,split_count) means the indices are distinct and
// there are split_count of them.
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
                "dispatch_fabric2d: share {} is out of range for a {}-way split",
                a.split_idx,
                a.split_count);
            TT_FATAL(
                claimed[a.dst_row].insert(a.split_idx).second,
                "dispatch_fabric2d: two streams both claim share {} of {} for row {}",
                a.split_idx,
                a.split_count,
                a.dst_row);
            auto& want = split_count[a.dst_row];
            TT_FATAL(
                want == 0 || want == a.split_count,
                "dispatch_fabric2d: row {} is split {} ways by one stream and {} ways by another; the shares "
                "would not tile what this chip owes it",
                a.dst_row,
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
    for (const auto& [row, shares] : claimed) {
        TT_FATAL(
            shares.size() == split_count.at(row),
            "dispatch_fabric2d: row {} has {} of its {} shares claimed, so part of what this chip owes it "
            "would never be sent",
            row,
            shares.size(),
            split_count.at(row));
    }
}

// Every chunk a chip forwards, as (origin, destination) hop offsets from that chip along the stream's own
// direction: origins upstream so negative, destinations downstream so positive. A chunk whose destination
// is the forwarder itself is delivered rather than forwarded and so is not here, which is why the nearest
// destination is 1 and the furthest origin is -(m - 1). Offsets are the same on every chip, so this takes
// only the extent.
//
// The order is the order the upstream chip writes the chunks, which the forwarder must match: the region
// is dense and holds no per-chunk addresses, so a chunk is found only by walking those before it. Upstream
// emits its own destinations furthest first (the whole origin == -1 group) before any chunk it is itself
// relaying, so origins run outwards from -1.
std::vector<std::pair<int32_t, int32_t>> chunks_in_forwarder_ref_frame(uint32_t ring_extent) {
    const int32_t m = static_cast<int32_t>(ring_extent / 2);
    std::vector<std::pair<int32_t, int32_t>> chunks;
    chunks.reserve(relay_chunks_per_stream(ring_extent));
    for (int32_t origin = -1; origin > -m; origin--) {
        for (int32_t dst = origin + m; dst >= 1; dst--) {
            chunks.emplace_back(origin, dst);
        }
    }
    return chunks;
}

}  // namespace

std::vector<dspf2d::ChunkDescriptor> forwarding_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links) {
    const bool is_cw = (stream % 2) == 0;
    const uint32_t link = stream / 2;
    const uint32_t m = ring_extent / 2;
    const int32_t travel = is_cw ? 1 : -1;
    const int32_t extent = static_cast<int32_t>(ring_extent);

    std::vector<dspf2d::ChunkDescriptor> chunks;
    for (const auto& [origin, dst] : chunks_in_forwarder_ref_frame(ring_extent)) {
        // A counter-clockwise stream mirrors the offsets through 0; then both land on a position on the
        // dispatch axis by adding where this chip sits.
        const uint32_t distance = static_cast<uint32_t>(dst - origin);
        chunks.push_back(dspf2d::ChunkDescriptor{
            .origin_row = static_cast<uint32_t>((static_cast<int32_t>(my_row) + travel * origin + extent) % extent),
            .dst_row = static_cast<uint32_t>((static_cast<int32_t>(my_row) + travel * dst + extent) % extent),
            .split_idx = distance == m ? stream : link,
            .split_count = distance == m ? stream_count(num_links) : num_links});
    }
    return chunks;
}

uint32_t fwd_pages_per_stream(
    uint32_t ring_extent,
    uint32_t num_links,
    uint32_t seq_len_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t experts_per_chip) {
    const uint32_t m = ring_extent / 2;
    const uint32_t per_pair = seq_len_per_chip * std::min(num_experts_per_tok, experts_per_chip);
    const uint32_t sc = stream_count(num_links);
    const auto div_up = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

    uint32_t pages = 0;
    for (uint32_t dd = 1; dd < m; dd++) {
        const uint32_t origins = m - dd;
        pages += (origins - 1) * div_up(per_pair, num_links);
        pages += div_up(per_pair, sc);
    }
    return pages + relay_chunks_per_stream(ring_extent) * experts_per_chip;
}

std::vector<dspf2d::ChunkDescriptor> outgoing_chunks(
    StreamId stream, uint32_t my_row, uint32_t ring_extent, uint32_t num_links) {
    const bool is_cw = (stream % 2) == 0;
    const uint32_t link = stream / 2;
    const uint32_t m = ring_extent / 2;
    const int32_t travel = is_cw ? 1 : -1;
    const int32_t extent = static_cast<int32_t>(ring_extent);
    const auto row_at = [&](int32_t offset) {
        return static_cast<uint32_t>((static_cast<int32_t>(my_row) + travel * offset + extent) % extent);
    };
    const uint32_t nbr_row = row_at(1);

    std::vector<dspf2d::ChunkDescriptor> chunks;
    for (uint32_t d = m; d >= 2; d--) {
        chunks.push_back(dspf2d::ChunkDescriptor{
            .origin_row = my_row,
            .dst_row = row_at(static_cast<int32_t>(d)),
            .split_idx = d == m ? stream : link,
            .split_count = d == m ? stream_count(num_links) : num_links});
    }
    for (const auto& chunk : forwarding_chunks(stream, my_row, ring_extent, num_links)) {
        if (chunk.dst_row != nbr_row) {
            chunks.push_back(chunk);
        }
    }
    return chunks;
}

void validate_chunk_agreement(uint32_t ring_extent, uint32_t num_links) {
    for (uint32_t row = 0; row < ring_extent; row++) {
        for (uint32_t stream = 0; stream < stream_count(num_links); stream++) {
            const int32_t travel = (stream % 2) == 0 ? 1 : -1;
            const uint32_t nbr_row = static_cast<uint32_t>(
                (static_cast<int32_t>(row) + travel + static_cast<int32_t>(ring_extent)) %
                static_cast<int32_t>(ring_extent));
            const auto written = outgoing_chunks(stream, row, ring_extent, num_links);
            const auto expected = forwarding_chunks(stream, nbr_row, ring_extent, num_links);
            TT_FATAL(
                written.size() == expected.size(),
                "dispatch_fabric2d: row {} stream {} writes {} chunks into row {}, which reads {}",
                row,
                stream,
                written.size(),
                nbr_row,
                expected.size());
            for (size_t i = 0; i < written.size(); i++) {
                TT_FATAL(
                    written[i].origin_row == expected[i].origin_row && written[i].dst_row == expected[i].dst_row &&
                        written[i].split_idx == expected[i].split_idx &&
                        written[i].split_count == expected[i].split_count,
                    "dispatch_fabric2d: row {} stream {} writes chunk {} as (origin {}, dst {}, share {}/{}) but "
                    "row {} reads it as (origin {}, dst {}, share {}/{})",
                    row,
                    stream,
                    i,
                    written[i].origin_row,
                    written[i].dst_row,
                    written[i].split_idx,
                    written[i].split_count,
                    nbr_row,
                    expected[i].origin_row,
                    expected[i].dst_row,
                    expected[i].split_idx,
                    expected[i].split_count);
            }
        }
    }
}

std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_row, uint32_t num_links) {
    const uint32_t extent = static_cast<uint32_t>(ring_chip_ids.size());
    const uint32_t m = own_assignments_per_stream(extent);
    TT_FATAL(extent >= 4 && extent % 2 == 0, "dispatch_fabric2d: axis extent {} must be even and at least 4", extent);
    TT_FATAL(my_row < extent, "dispatch_fabric2d: row {} is outside a {}-chip axis", my_row, extent);

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

            // Furthest destination first, and all own assignments before any relay. Emission order is
            // what the downstream chip walks its region by, and upstream emits its own destinations
            // before anything it relays; the own phase is also the slack a relay has between the
            // upstream chunk being written and this stream consuming it.
            for (uint32_t j = 1; j <= m; j++) {
                const uint32_t distance = m - j + 1;
                if (distance == m) {
                    own(m, stream, stream_count(num_links));  // the opposite chip, shared by every stream
                } else {
                    own(distance, link, num_links);
                }
            }
            const uint32_t relays = relay_chunks_per_stream(extent);
            for (uint32_t c = 0; c < relays; c++) {
                list.push_back(Assignment{.is_relay = true, .relay_chunk = c});
            }
            TT_FATAL(
                list.size() == m + relays,
                "dispatch_fabric2d: stream {} got {} work items, expected {} own + {} relay",
                stream,
                list.size(),
                m,
                relays);
        }
    }
    validate_coverage(per_stream, extent);
    return per_stream;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
