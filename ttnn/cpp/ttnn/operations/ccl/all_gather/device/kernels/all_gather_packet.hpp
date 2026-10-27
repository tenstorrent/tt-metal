// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Packet builder: walk runs -> packets of <= cap bytes (packet_cap) and <= NOC_SCATTER_WRITE_MAX_CHUNKS segments, cut
// on the unit. The reader builds each packet's record (the fabric's own scatter command) in its slot; the writer
// encodes it into the slot's header with the official setters. Compiled on host by the kernel-lib sim.

#include <cstdint>
#include <new>

#include "hostdev/fabric_edm_packet_header.hpp"

#include "ttnn/operations/ccl/all_gather/shared_with_host/all_gather_walk.hpp"

namespace ag {

// noc_address[0, chunk_count) are fabric-frame (all_gather_sync.hpp address rule); chunk_size[0, chunk_count - 1) (the
// last chunk is the rest). chunk_count == 1 is a plain unicast write (the scatter setter needs >= 2 chunks).
struct PacketRecord {
    tt::tt_fabric::NocUnicastScatterCommandHeader cmd;
    uint16_t payload_bytes;
};
static_assert(sizeof(PacketRecord) == kPacketRecordBytes, "host slot_offsets() assumes this record size");
static_assert(NOC_SCATTER_WRITE_MAX_CHUNKS == kSegmentsPerPacket, "host packet_cap() and slot_offsets() assume this");

// Calls f(payload_off, dst[i], bytes) per segment i of a record, in payload order. dst: the record's own addresses
// (fabric frame), or the per-segment own-NOC addresses PacketBuilder::next reported.
template <class F>
void for_each_segment(const PacketRecord& rec, const uint64_t* dst, F&& f) {
    const uint32_t last = rec.cmd.chunk_count - 1;
    uint32_t off = 0;
    for (uint32_t i = 0; i < last; ++i) {
        f(off, dst[i], uint32_t{rec.cmd.chunk_size[i]});
        off += rec.cmd.chunk_size[i];
    }
    f(off, dst[last], rec.payload_bytes - off);
}

// Packs the runs a walk yields, in order, into packets. W: ag::Walk. Src: ag::InputPieces or ag::RelayPieces.
// A piece that starts where the last segment ends extends it, so a run split into several input pieces, or runs that
// happen to abut, cost one segment. A packet never spans walk iterations: next() returns 0 once the walk is exhausted,
// and the caller re-inits the walk for the next iteration.
template <class W, class Src>
class PacketBuilder {
public:
    PacketBuilder(W& walk, Src& src, uint32_t cap) : walk_(walk), src_(src), cap_(cap) {}

    // Fills one packet: read(src_noc_addr, payload_off, bytes) for each read into this packet's payload
    // (payload_off a multiple of A), then builds *rec in place. Returns the payload bytes, 0 when the walk is
    // exhausted (rec untouched). local, if given, receives each segment's own-NOC address (DstRun::local), for a
    // local copy of the packet.
    template <class Read>
    AG_INLINE uint32_t next(PacketRecord* rec, Read&& read, uint64_t* local = nullptr) {
        uint64_t addrs[NOC_SCATTER_WRITE_MAX_CHUNKS];
        uint16_t sizes[NOC_SCATTER_WRITE_MAX_CHUNKS];
        uint32_t nseg = 0, used = 0;
        while (used < cap_ && (have_ || pull())) {
            const bool merge = nseg != 0 && dst_ == addrs[nseg - 1] + sizes[nseg - 1];
            if (!merge && nseg == NOC_SCATTER_WRITE_MAX_CHUNKS) {
                break;
            }
            // Both are multiples of A, so every cut is on A.
            const uint32_t n = std::min(cur_.bytes, cap_ - used);
            const uint32_t rd = std::min(n, cur_.readable_bytes);
            if (rd != 0) {
                read(cur_.src, used, rd);
            }
            if (merge) {
                // Contiguous in the fabric frame is the same bank and page run, so contiguous locally too.
                sizes[nseg - 1] += n;
            } else {
                addrs[nseg] = dst_;
                sizes[nseg] = n;
                if (local != nullptr) {
                    local[nseg] = local_;
                }
                ++nseg;
            }
            used += n;
            dst_ += n;
            local_ += n;
            cur_.src += n;
            cur_.bytes -= n;
            cur_.readable_bytes -= rd;
            have_ = cur_.bytes != 0;
        }
        if (used != 0) {
            new (rec) PacketRecord{
                tt::tt_fabric::NocUnicastScatterCommandHeader(addrs, sizes, static_cast<uint8_t>(nseg)),
                static_cast<uint16_t>(used)};
        }
        return used;
    }

private:
    // Loads the next piece (and the next run when this one is done); false when the walk is exhausted.
    AG_INLINE bool pull() {
        while (!src_.next(cur_)) {
            DstRun r;
            if (!walk_.next(r)) {
                return false;
            }
            src_.start(r);
            run_dst_ = r.addr;
            run_local_ = r.local;
        }
        dst_ = run_dst_;
        local_ = run_local_;
        run_dst_ += cur_.bytes;
        run_local_ += cur_.bytes;
        have_ = true;
        return true;
    }

    W& walk_;
    Src& src_;
    const uint32_t cap_;
    Piece cur_{};                         // the unsent rest of the current piece, valid while have_
    uint64_t dst_ = 0;                    // where cur_ lands
    uint64_t run_dst_ = 0;                // where the run's next piece lands
    uint64_t local_ = 0, run_local_ = 0;  // the same two places on the caller's own NOC
    bool have_ = false;
};

}  // namespace ag
