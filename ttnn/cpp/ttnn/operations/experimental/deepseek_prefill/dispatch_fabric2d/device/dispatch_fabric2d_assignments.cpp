// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_assignments.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

namespace {

dspf2d::ChunkDescriptor to_descriptor(const fabric2d::RingChunk& c) {
    return dspf2d::ChunkDescriptor{
        .origin_pos = c.origin_pos, .dst_pos = c.dst_pos, .split_idx = c.split_idx, .split_count = c.split_count};
}

// The position `hops` chips downstream of `pos` on `stream`.
uint32_t downstream_of(StreamId stream, uint32_t pos, uint32_t hops, uint32_t ring_extent) {
    return ((stream % 2) == 0 ? pos + hops : pos + ring_extent - hops) % ring_extent;
}

}  // namespace

std::vector<dspf2d::ChunkDescriptor> forwarding_descriptors(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links) {
    std::vector<dspf2d::ChunkDescriptor> descriptors;
    for (const auto& chunk : fabric2d::forwarding_chunks(stream, my_pos, ring_extent, num_links)) {
        descriptors.push_back(to_descriptor(chunk));
    }
    return descriptors;
}

std::vector<dspf2d::ChunkDescriptor> outgoing_descriptors(
    StreamId stream, uint32_t my_pos, uint32_t ring_extent, uint32_t num_links) {
    const uint32_t downstream_pos = downstream_of(stream, my_pos, 1, ring_extent);

    std::vector<dspf2d::ChunkDescriptor> descriptors;
    for (uint32_t d = own_assignments_per_stream(ring_extent); d >= 2; d--) {
        descriptors.push_back(to_descriptor(fabric2d::stream_chunk(stream, my_pos, d, ring_extent, num_links)));
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
            const uint32_t downstream_pos = downstream_of(stream, pos, 1, ring_extent);
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

std::vector<dspf2d::ChunkDescriptor> incoming_final_descriptors(
    const std::vector<uint32_t>& ring_chip_ids, StreamId stream, uint32_t my_pos, uint32_t num_links) {
    const uint32_t extent = static_cast<uint32_t>(ring_chip_ids.size());
    // One hop upstream is extent - 1 hops downstream.
    const uint32_t upstream_pos = downstream_of(stream, my_pos, extent - 1, extent);

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
