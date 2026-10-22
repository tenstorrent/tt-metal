// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/fabric_edm_packet_header.hpp"

#include <cstdint>

#include "walk.hpp"

// Packetizer: dst runs -> packets in CB slots [C fabric headers | pad | payload]. The payload is the packet's
// segments in header order, so units = payload bytes / A. Every L1 offset is a multiple of A, which is a multiple
// of every source's read alignment, so a read's source and L1 addresses are congruent mod that alignment.
//
// Ring: uint32_t acquire();  // L1 address of the next slot to fill
//       uint8_t trid();      // read transaction id of that slot
//       void commit();       // the slot is complete; its reads may still be in flight
//       void drain();        // push every committed slot
// Io:   void read(uint64_t src, uint32_t l1, uint32_t bytes, uint8_t trid);
//       void write_headers(uint32_t slot, const Seg*, uint32_t nseg, uint32_t bytes);
// slot_io.hpp has the device versions; host tests record instead.

struct Seg {
    uint64_t dst;
    uint32_t bytes;
};

// One packet: <= P bytes (P cut to a multiple of A) and <= NOC_SCATTER_WRITE_MAX_CHUNKS segments.
// A run continues the last segment when it starts where that one ends.
template <class Ring, class Io>
struct Packetizer {
    Packetizer(Ring& ring, Io& io, uint32_t A, uint32_t P, uint32_t payload_off) :
        ring_(ring), io_(io), A_(A), P_(P - P % A), payload_off_(payload_off) {}

    template <class Src>
    FORCE_INLINE void add(const DstRun& r, Src& src) {
        uint64_t dst = r.addr;
        src.for_each(r, [&](const Piece& p) {
            uint64_t from = p.src;
            uint32_t units = p.units;
            uint32_t readable = p.readable_bytes;
            while (units) {
                if (used_ == 0) {
                    slot_ = ring_.acquire();
                }
                const uint32_t room = (P_ - used_) / A_;
                const bool merge = nseg_ != 0 && dst == segs_[nseg_ - 1].dst + segs_[nseg_ - 1].bytes;
                if (room == 0 || (!merge && nseg_ == NOC_SCATTER_WRITE_MAX_CHUNKS)) {
                    close();
                    continue;
                }
                const uint32_t n = std::min(units, room);
                const uint32_t bytes = n * A_;
                const uint32_t rd = std::min(bytes, readable);
                if (rd) {
                    io_.read(from, slot_ + payload_off_ + used_, rd, ring_.trid());
                }
                if (merge) {
                    segs_[nseg_ - 1].bytes += bytes;
                } else {
                    segs_[nseg_++] = {dst, bytes};
                }
                used_ += bytes;
                from += bytes;
                dst += bytes;
                units -= n;
                readable -= rd;
            }
        });
    }

    // Sends nothing for an empty packet: the pump counts units, so it expects no slot for it.
    FORCE_INLINE void close() {
        if (used_ == 0) {
            return;
        }
        ASSERT(nseg_ != 0 && nseg_ <= NOC_SCATTER_WRITE_MAX_CHUNKS && used_ <= P_ && used_ % A_ == 0);
        io_.write_headers(slot_, segs_, nseg_, used_);
        ring_.commit();
        used_ = 0;
        nseg_ = 0;
    }

private:
    Ring& ring_;
    Io& io_;
    uint32_t A_, P_, payload_off_;
    uint32_t slot_ = 0, used_ = 0, nseg_ = 0;
    Seg segs_[NOC_SCATTER_WRITE_MAX_CHUNKS];
};

// Header codec. One segment is a plain unicast write, 2..4 a scatter write. The to_noc_* setters also
// move each address to the EDM's NOC.
template <class Hdr>
FORCE_INLINE void encode_segments(volatile Hdr* hdr, const Seg* segs, uint32_t nseg, uint32_t bytes) {
    if (nseg == 1) {
        hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{segs[0].dst}, bytes);
        return;
    }
    uint64_t addrs[NOC_SCATTER_WRITE_MAX_CHUNKS];
    uint16_t sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1];
    for (uint32_t i = 0; i < nseg; ++i) {
        addrs[i] = segs[i].dst;
        if (i + 1 < nseg) {
            sizes[i] = static_cast<uint16_t>(segs[i].bytes);
        }
    }
    hdr->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader(addrs, sizes, static_cast<uint8_t>(nseg)), bytes);
}

// Calls f(payload_off, addr, len) per segment of an encoded header, in payload order.
template <class Hdr, class F>
FORCE_INLINE void for_each_segment(volatile Hdr* hdr, F&& f) {
    const uint32_t bytes = hdr->payload_size_bytes;
    if (hdr->noc_send_type == tt::tt_fabric::NOC_UNICAST_WRITE) {
        f(0u, static_cast<uint64_t>(hdr->command_fields.unicast_write.noc_address), bytes);
        return;
    }
    volatile auto& s = hdr->command_fields.unicast_scatter_write;
    const uint32_t last = s.chunk_count - 1;
    uint32_t off = 0;
    for (uint32_t i = 0; i < last; ++i) {
        const uint32_t len = s.chunk_size[i];
        f(off, static_cast<uint64_t>(s.noc_address[i]), len);
        off += len;
    }
    f(off, static_cast<uint64_t>(s.noc_address[last]), bytes - off);
}
