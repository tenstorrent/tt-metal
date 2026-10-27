// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The all_gather walk geometry and slot layout. The host factories size the CB slots and the packet cap with it; the
// kernels walk with it (device/kernels/all_gather_common.hpp).
//
// Symbols (WalkGeo fields): A unit_bytes, Uo out_page_units, S stripe_units, N num_devices, E element_units, L the
// output's contiguous_page_stride().
//   unit:    A bytes, the cut quantum. Local unit u is global unit g = (u / S) * N * S + stripe * S + u % S, which
//            lives in output page g / Uo at byte (g % Uo) * A.
//   element: what the walk steps over: E units, one output page (E == Uo) or one unit (E == 1).
//   stripe:  the S units this device contributes to each output row.
//   lane:    local elements lambda apart whose output pages are L apart, i.e. adjacent in memory.

#include <algorithm>
#include <cstdint>

namespace ttnn::operations::ccl::all_gather_walk {

// The geometry of one device's contribution; counts are in units of A bytes.
struct WalkGeo {
    uint32_t unit_bytes;        // the cut quantum: a multiple of the NOC write alignment and every source's read one
    uint32_t in_page_units;     // input page footprint
    uint32_t out_page_units;    // output page footprint, ceil(page content / unit_bytes)
    uint32_t stripe_units;      // units this device contributes to one output row
    uint32_t num_devices;       // devices in the collective
    uint32_t element_units;     // units per walk element: out_page_units when a stripe row is whole pages, else 1
    uint32_t max_packet_bytes;  // fabric packet payload limit, a multiple of unit_bytes
    bool dst_cross;             // an output page's footprint is its bank pitch, so a run may continue into the next
    bool src_cross;             // the same for input pages
};

struct WalkShape {
    uint32_t lambda;         // element step inside a run
    uint32_t column_height;  // elements per lane before the walk moves to the next lane
    bool cross_row;          // runs continue across output rows
};

// If N * (S / Uo) divides L, lambda = L / N steps whole rows onto the next page of the same bank, so a run is not
// clipped at the row end (cross_row; BH, one bf16 tile per device per row, 2-26 MB: 5% faster on geomean over line /
// ring and algorithms, up to 25% for ring multicast). Otherwise lambda = L (or 1 for unit elements) and a run stops at
// the row end, where the other devices' stripes start; the column is then capped at one row of its lane so that most
// columns are a single run. A column is at most a packet (but at least one element).
constexpr WalkShape walk_shape(const WalkGeo& g, uint32_t L) {
    const bool pages = g.element_units == g.out_page_units;
    const bool cross_row = pages && L % (g.num_devices * (g.stripe_units / g.out_page_units)) == 0;
    const uint32_t lambda = cross_row ? L / g.num_devices : (pages ? L : 1);
    uint32_t h = g.max_packet_bytes / (g.element_units * g.unit_bytes);
    if (pages && !cross_row) {
        const uint32_t row_lane = (g.stripe_units / g.element_units + lambda - 1) / lambda;
        h = std::min(h, row_lane);
    }
    return {lambda, std::max<uint32_t>(h, 1), cross_row};
}

// Segments one fabric scatter packet carries. The kernels check it against NOC_SCATTER_WRITE_MAX_CHUNKS.
inline constexpr uint32_t kSegmentsPerPacket = 4;

// The packet builder's payload cap: max_packet_bytes, or less where every segment is short, so a slot spends its L1 and
// its reads in flight on more packets instead of on payload room no packet uses (BH: 10-13% with L1 outputs, 3-8% on
// row-major rows). A segment is at most one walk column, with unit elements also at most the stripe row (S units), and
// never longer than the destination is contiguous: one page where a page never continues into the next, else the
// output's contiguous run of out_run_pages pages. Correctness never depends on this bound: a packet that reaches the
// cap just ends there.
constexpr uint32_t packet_cap(const WalkGeo& g, uint32_t L, uint32_t out_run_pages) {
    const WalkShape s = walk_shape(g, L);
    uint64_t segment = uint64_t{s.column_height} * g.element_units * g.unit_bytes;
    if (g.element_units != g.out_page_units) {
        segment = std::min<uint64_t>(segment, uint64_t{g.stripe_units} * g.unit_bytes);
    }
    uint64_t contig = uint64_t{g.out_page_units} * g.unit_bytes;
    if (g.dst_cross && (g.element_units == g.out_page_units || L == 1)) {
        contig *= out_run_pages;
    }
    return static_cast<uint32_t>(
        std::min<uint64_t>(g.max_packet_bytes, kSegmentsPerPacket * std::min(segment, contig)));
}

// ---- slot layout (one CB page) ----

// CB pages per worker: one slot filling while the other drains.
inline constexpr uint32_t kNumSlots = 2;

// Written by the reader after the slot's packets. units: A-units of one iteration. iter_end: this is the iteration's
// last slot, so the writer signals after it without re-deriving the schedule.
struct SlotInfo {
    uint16_t packets;
    uint16_t iter_end;
    uint32_t units;
};

// One packet as the reader built it, in fabric-frame addresses. A packet of one segment is a plain unicast write, more
// is a scatter write. The writer turns it into the fabric command.
struct PacketRecord {
    uint64_t dst[kSegmentsPerPacket];
    uint16_t bytes[kSegmentsPerPacket];
    uint16_t segments;
    uint16_t payload_bytes;
};

// Own-NOC start of each segment of a packet, for the reader's local copy.
inline constexpr uint32_t kLocalAddrBytes = kSegmentsPerPacket * sizeof(uint64_t);

// [SlotInfo][k records][k local-address sets | pad to A][k unicast headers][k scatter headers][k payloads of `cap`].
// header_bytes is 0 for a writer that keeps no headers in the slot. Every offset from the headers on is a multiple of A
// and so is slot_bytes, so payload L1 addresses are congruent with their A-aligned sources (CB bases are DRAM-aligned,
// a multiple of A). The records and local addresses belong to the reader while it holds the slot; the headers belong
// to the writer for the whole op, and the reader never writes them.
struct SlotLayout {
    uint32_t k, rec_off, loc_off, hdr_off, hdr_stride, pay_off, pay_stride, slot_bytes;

    // L1 addresses of packet j's record, local addresses, headers and payload in the slot at `slot`.
    constexpr uint32_t record(uint32_t slot, uint32_t j) const { return slot + rec_off + j * sizeof(PacketRecord); }
    constexpr uint32_t local(uint32_t slot, uint32_t j) const { return slot + loc_off + j * kLocalAddrBytes; }
    constexpr uint32_t unicast_header(uint32_t slot, uint32_t j) const { return slot + hdr_off + j * hdr_stride; }
    constexpr uint32_t scatter_header(uint32_t slot, uint32_t j) const { return unicast_header(slot, k + j); }
    constexpr uint32_t payload(uint32_t slot, uint32_t j) const { return slot + pay_off + j * pay_stride; }
};

constexpr uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// k packets per slot; header_bytes: the fabric packet header; cap: packet_cap, a multiple of A.
constexpr SlotLayout slot_offsets(uint32_t k, uint32_t A, uint32_t header_bytes, uint32_t cap) {
    // Records and local addresses hold uint64_t addresses, and SlotInfo is 8 bytes, so both start 8-aligned.
    const uint32_t rec_off = sizeof(SlotInfo);
    const uint32_t loc_off = rec_off + k * sizeof(PacketRecord);
    const uint32_t hdr_off = align_up(loc_off + k * kLocalAddrBytes, A);
    const uint32_t hdr_stride = align_up(header_bytes, A);
    const uint32_t pay_off = hdr_off + 2 * k * hdr_stride;
    return {k, rec_off, loc_off, hdr_off, hdr_stride, pay_off, cap, pay_off + k * cap};
}

}  // namespace ttnn::operations::ccl::all_gather_walk
