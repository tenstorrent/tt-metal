// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#endif

#include <cstdint>

#include "walk.hpp"

// Packetizer: dst runs -> packets in CB slots [pad | C fabric headers | payload]. The payload is the packet's
// segments in header order. Every L1 offset is a multiple of A, which is a multiple of every source's read
// alignment, so a read's source and L1 addresses are congruent mod that alignment.
//
// Ring: uint32_t acquire();                        // L1 address of the next slot to fill
//       uint8_t trid();                            // read transaction id of that slot
//       void commit(const Seg*, uint32_t nseg);    // the slot is complete; its reads may still be in flight
//       void drain();                              // push every committed slot
// Io:   static constexpr uint32_t kReadsPerPoll;   // reads a packet may issue between throttle() calls
//       void start(uint8_t trid);                  // a packet opens; its reads use trid
//       void read(uint64_t src, uint32_t l1, uint32_t bytes, uint8_t trid);
//       void throttle(uint8_t trid);               // waits until another kReadsPerPoll reads fit
//       void write_headers(uint32_t slot, const Seg*, uint32_t nseg, uint32_t bytes);
// slot_io.hpp has the device versions; host tests record instead.

struct Seg {
    uint64_t dst;
    uint32_t bytes;
};

// One packet: <= P bytes (P cut to a multiple of A) and <= NOC_SCATTER_WRITE_MAX_CHUNKS segments.
// A piece continues the last segment when it starts where that one ends. Byte counts, never units: no divides.
// add() works on a local copy of the open packet's state, so the per-piece path keeps it in registers; opening
// and finishing a packet are out of line (once per packet), which keeps the hot loop small.
template <class Ring, class Io>
class Packetizer {
public:
    Packetizer(Ring& ring, Io& io, uint32_t A, uint32_t P, uint32_t payload_off) :
        ring_(ring), io_(io), P_(P - P % A), payload_off_(payload_off) {}

    template <class Src>
    FORCE_INLINE void add(const DstRun& r, Src& src) {
        State s = s_;
        uint64_t dst = r.addr;
        src.for_each(r, [&](const Piece& p) __attribute__((always_inline)) { put(s, p, dst); });
        s_ = s;
    }

    // add() for every run the walk yields, calling before(run) first. One state copy for the whole walk.
    template <class W, class Src, class Before>
    FORCE_INLINE void add_all(W& walk, Src& src, Before&& before) {
        State s = s_;
        DstRun r;
        while (walk.next(r)) {
            before(r);
            uint64_t dst = r.addr;
            src.for_each(r, [&](const Piece& p) __attribute__((always_inline)) { put(s, p, dst); });
        }
        s_ = s;
    }

    // Sends nothing for an empty packet: the pump counts units, so it expects no slot for it.
    FORCE_INLINE void close() {
        if (s_.used != 0) {
            finish(s_.l1, s_.nseg, s_.used);
            s_ = State{};
        }
    }

private:
    struct State {
        uint32_t l1 = 0, used = 0, nseg = 0, reads = 0;
        uint64_t end = ~uint64_t{0};  // where the last segment ends; never a piece's start when nseg == 0
        uint8_t trid = 0;
    };

    FORCE_INLINE void put(State& s, const Piece& p, uint64_t& dst) {
        uint64_t from = p.src;
        uint32_t bytes = p.bytes;
        uint32_t readable = p.readable_bytes;
        while (true) {
            if (s.used == 0) [[unlikely]] {
                s.l1 = open();
                s.trid = ring_.trid();
            }
            const bool merge = dst == s.end;
            if (!merge && s.nseg == NOC_SCATTER_WRITE_MAX_CHUNKS) [[unlikely]] {
                finish(s.l1, s.nseg, s.used);
                s = State{};
                continue;
            }
            const uint32_t room = P_ - s.used;
            const uint32_t n = bytes < room ? bytes : room;
            const uint32_t rd = n < readable ? n : readable;
            if (rd) [[likely]] {
                if (++s.reads == Io::kReadsPerPoll) [[unlikely]] {
                    io_.throttle(s.trid);
                    s.reads = 0;
                }
                io_.read(from, s.l1 + s.used, rd, s.trid);
            }
            if (merge) [[unlikely]] {
                segs_[s.nseg - 1].bytes += n;
            } else {
                segs_[s.nseg++] = {dst, n};
            }
            s.used += n;
            dst += n;
            s.end = dst;
            if (s.used == P_) [[unlikely]] {
                finish(s.l1, s.nseg, s.used);
                s = State{};
            }
            if (n == bytes) [[likely]] {
                return;
            }
            from += n;
            bytes -= n;
            readable -= rd;
        }
    }

    // Returns the payload's L1 address.
    __attribute__((noinline)) uint32_t open() {
        const uint32_t slot = ring_.acquire();
        io_.start(ring_.trid());
        return slot + payload_off_;
    }

    __attribute__((noinline)) void finish(uint32_t l1, uint32_t nseg, uint32_t used) {
        ASSERT(nseg != 0 && nseg <= NOC_SCATTER_WRITE_MAX_CHUNKS && used <= P_);
        io_.write_headers(l1 - payload_off_, segs_, nseg, used);
        ring_.commit(segs_, nseg);
    }

    Ring& ring_;
    Io& io_;
    const uint32_t P_, payload_off_;
    State s_;
    Seg segs_[NOC_SCATTER_WRITE_MAX_CHUNKS];
};

// A builder address (our NOC's coordinates) as the EDM's NOC sees it: the same on BH (virtual coordinates) or
// when we share the EDM's NOC; mirrored on WH otherwise.
FORCE_INLINE uint64_t edm_noc_addr(uint64_t addr) {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
    if constexpr (noc_index == tt::tt_fabric::edm_to_local_chip_noc) {
        return addr;
    } else {
        const auto c = get_noc_address_components(addr);
        return safe_get_noc_addr(c.first.x, c.first.y, c.second, tt::tt_fabric::edm_to_local_chip_noc);
    }
#else
    return addr;
#endif
}

// Header codec. One segment is a plain unicast write, 2..4 a scatter write. Writes only the fields the EDM reads
// for that type (the route was set once per slot).
template <class Hdr>
FORCE_INLINE void encode_segments(volatile Hdr* hdr, const Seg* segs, uint32_t nseg, uint32_t bytes) {
    hdr->payload_size_bytes = static_cast<uint16_t>(bytes);
    if (nseg == 1) {
        hdr->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
        hdr->command_fields.unicast_write.noc_address = edm_noc_addr(segs[0].dst);
        return;
    }
    hdr->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    volatile auto& s = hdr->command_fields.unicast_scatter_write;
    s.chunk_count = static_cast<uint8_t>(nseg);
    s.chunk_encoding = tt::tt_fabric::expand_encoding_to_all_chunks(
        tt::tt_fabric::NocScatterWriteChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE);
    for (uint32_t i = 0; i < nseg; ++i) {
        s.noc_address[i] = edm_noc_addr(segs[i].dst);
        if (i + 1 < nseg) {
            s.chunk_size[i] = static_cast<uint16_t>(segs[i].bytes);
        }
    }
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
