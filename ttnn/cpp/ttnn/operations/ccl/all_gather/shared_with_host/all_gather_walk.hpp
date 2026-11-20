// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The all_gather walk geometry and slot layout. The host factories size the CB slots and the packet cap with it; the
// kernels walk with it (device/kernels/all_gather_common.hpp).
//
// Page indexing. The kernels walk this device's stripe of the output in units of unit_bytes and pack the runs that are
// contiguous in the output into packets of up to kMaxChunksPerPacket chunks. A walk element is one output page, or one
// unit when a stripe row is not whole output pages; workers split the elements into contiguous slices.
//   unit:    unit_bytes, the granularity every cut lands on. Local unit u is global unit
//            (u / stripe_units) * num_devices * stripe_units + stripe * stripe_units + u % stripe_units, which lives in
//            output page (global unit) / out_page_units.
//   element: what the walk steps over, element_units units: one output page or one unit.
//   stripe:  the stripe_units units this device contributes to each output row.
//   lane:    local elements `step` apart whose output pages are out_page_stride apart (the output's
//            contiguous_page_stride()), i.e. adjacent in memory.

#include <algorithm>
#include <cstdint>

namespace ttnn::operations::ccl::all_gather_walk {

// The geometry of one device's contribution; counts are in units of unit_bytes.
struct WalkGeo {
    uint32_t unit_bytes;        // a multiple of the NOC write alignment and every source's read one
    uint32_t in_page_units;     // input page footprint
    uint32_t out_page_units;    // output page footprint, ceil(page content / unit_bytes)
    uint32_t stripe_units;      // units this device contributes to one output row
    uint32_t num_devices;       // devices in the collective
    uint32_t element_units;     // units per walk element: out_page_units when a stripe row is whole pages, else 1
    uint32_t max_packet_bytes;  // fabric packet payload limit, a multiple of unit_bytes
    bool dst_cross;             // an output page's footprint is its bank pitch, so a run may continue into the next
    bool src_cross;             // the same for input pages
};

// Compile-time args the kernels read the geometry from: the WalkGeo fields in declaration order, then the packet cap.
inline constexpr uint32_t kWalkCtArgCount = 10;

struct WalkShape {
    uint32_t step;    // element step inside a run
    uint32_t height;  // elements per lane before the walk moves to the next lane
    bool cross_row;   // runs continue across output rows
};

// If num_devices * (stripe_units / out_page_units) divides out_page_stride, step = out_page_stride / num_devices steps
// whole rows onto the next page of the same bank, so a run is not clipped at the row end (cross_row). Otherwise step =
// out_page_stride (or 1 for unit elements) and a run stops at the row end, where the other devices' stripes start; the
// column is then capped at one row of its lane so that most columns are a single run. A column is at most a packet
// (but at least one element).
constexpr WalkShape walk_shape(const WalkGeo& g, uint32_t out_page_stride) {
    const bool pages = g.element_units == g.out_page_units;
    const bool cross_row = pages && out_page_stride % (g.num_devices * (g.stripe_units / g.out_page_units)) == 0;
    const uint32_t step = cross_row ? out_page_stride / g.num_devices : (pages ? out_page_stride : 1);
    uint32_t height = g.max_packet_bytes / (g.element_units * g.unit_bytes);
    if (pages && !cross_row) {
        const uint32_t row_lane = (g.stripe_units / g.element_units + step - 1) / step;
        height = std::min(height, row_lane);
    }
    return {step, std::max<uint32_t>(height, 1), cross_row};
}

// Chunks one fabric scatter packet carries at most. The kernels check it against NOC_SCATTER_WRITE_MAX_CHUNKS.
inline constexpr uint32_t kMaxChunksPerPacket = 4;

// The packet builder's payload cap: max_packet_bytes, or less where every chunk is short, so a slot spends its L1 and
// its reads in flight on more packets instead of on payload room no packet uses. A chunk is at most one walk column,
// with unit elements also at most the stripe row (stripe_units), and never longer than the destination is contiguous:
// one page where a page never continues into the next, else the output's out_contig_pages contiguous pages.
// Correctness never depends on this bound: a packet that reaches the cap just ends there.
constexpr uint32_t packet_cap(const WalkGeo& g, uint32_t out_page_stride, uint32_t out_contig_pages) {
    const WalkShape s = walk_shape(g, out_page_stride);
    uint64_t chunk_bytes = uint64_t{s.height} * g.element_units * g.unit_bytes;
    if (g.element_units != g.out_page_units) {
        chunk_bytes = std::min<uint64_t>(chunk_bytes, uint64_t{g.stripe_units} * g.unit_bytes);
    }
    uint64_t contig = uint64_t{g.out_page_units} * g.unit_bytes;
    if (g.dst_cross && (g.element_units == g.out_page_units || out_page_stride == 1)) {
        contig *= out_contig_pages;
    }
    return static_cast<uint32_t>(
        std::min<uint64_t>(g.max_packet_bytes, kMaxChunksPerPacket * std::min(chunk_bytes, contig)));
}

// ---- slot layout ----

// A slot is one CB page. Slots per worker: one filling while the other drains.
inline constexpr uint32_t kNumSlots = 2;

// Written by the reader after the slot's packets. units: units of one iteration. iter_end: this is the iteration's last
// slot, so the writer signals after it without re-deriving the schedule.
struct SlotInfo {
    uint16_t packets;
    uint16_t iter_end;
    uint32_t units;
};

// One packet as the reader built it, in fabric addresses (the form the fabric header setters take). A packet of one
// chunk is a plain unicast write, more is a scatter write. The writer turns it into the fabric command.
struct PacketRecord {
    uint64_t dst[kMaxChunksPerPacket];
    uint16_t chunk_bytes[kMaxChunksPerPacket];
    uint16_t num_chunks;
    uint16_t payload_bytes;
};

// Own-NOC start of each chunk of a packet, for the reader's local copy.
inline constexpr uint32_t kOwnAddrBytes = kMaxChunksPerPacket * sizeof(uint64_t);

// [SlotInfo][records][own-NOC address sets | pad to unit][unicast headers][scatter headers][payloads of packet_cap],
// one of each per packet. header_bytes is 0 for a writer that keeps no headers in the slot. Every offset from the
// headers on is a multiple of unit_bytes and so is slot_bytes, so payload L1 addresses are congruent with their
// unit-aligned sources (CB bases are DRAM-aligned, a multiple of unit_bytes). The records and own-NOC addresses belong
// to the reader while it holds the slot; the headers belong to the writer for the whole op, and the reader never
// writes them.
struct SlotLayout {
    uint32_t packets, rec_off, own_off, hdr_off, hdr_stride, pay_off, pay_stride, slot_bytes;

    // L1 addresses of packet j's record, own-NOC addresses, headers and payload in the slot at `slot`.
    constexpr uint32_t record(uint32_t slot, uint32_t j) const { return slot + rec_off + j * sizeof(PacketRecord); }
    constexpr uint32_t own_addrs(uint32_t slot, uint32_t j) const { return slot + own_off + j * kOwnAddrBytes; }
    constexpr uint32_t unicast_header(uint32_t slot, uint32_t j) const { return slot + hdr_off + j * hdr_stride; }
    constexpr uint32_t scatter_header(uint32_t slot, uint32_t j) const { return unicast_header(slot, packets + j); }
    constexpr uint32_t payload(uint32_t slot, uint32_t j) const { return slot + pay_off + j * pay_stride; }
};

constexpr uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// `packets` per slot; header_bytes: the fabric packet header; packet_cap: a multiple of unit_bytes.
constexpr SlotLayout slot_offsets(uint32_t packets, uint32_t unit_bytes, uint32_t header_bytes, uint32_t packet_cap) {
    // Records and own-NOC addresses hold uint64_t addresses, and SlotInfo is 8 bytes, so both start 8-aligned.
    const uint32_t rec_off = sizeof(SlotInfo);
    const uint32_t own_off = rec_off + packets * sizeof(PacketRecord);
    const uint32_t hdr_off = align_up(own_off + packets * kOwnAddrBytes, unit_bytes);
    const uint32_t hdr_stride = align_up(header_bytes, unit_bytes);
    const uint32_t pay_off = hdr_off + 2 * packets * hdr_stride;
    return {packets, rec_off, own_off, hdr_off, hdr_stride, pay_off, packet_cap, pay_off + packets * packet_cap};
}

}  // namespace ttnn::operations::ccl::all_gather_walk
