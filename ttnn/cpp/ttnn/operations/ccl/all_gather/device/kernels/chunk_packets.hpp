// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Include this after your fabric api: the scatter header type and the segment limit come from there.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

#include <cstdint>

////////////////////////////////////////////////////////////////
// Packing runs into packets.
//
// Nothing here knows which op is running, or where the packets go. A sender handles that.
//
// Glossary:
//   segment        -- one run inside a packet. A packet holds at most 4 of them
//                     (NOC_SCATTER_WRITE_MAX_CHUNKS), and each one costs the receiver a NOC write.
//   payload        -- bytes a packet may carry. The fabric maximum, in bytes, because that is what
//                     the packet header counts. The host thinks in chunks (see chunk_plan.hpp).
//   packet_chunks  -- chunks a packet really carries: min(payload / chunk, 4 runs). Short runs, not
//                     the payload, are usually what ends a packet.
//   entry          -- one CB page, in chunks. Always a whole number of packets, because the caller
//                     has to flush before it pops an entry: an entry that is not a whole number of
//                     packets cuts one at every boundary, and a cut packet wastes a fabric slot.
//
// Add runs until 4 segments or the payload runs out, then send. A run that does not fit starts the
// next packet rather than splitting: splitting would fill the tail but cost an extra segment, i.e.
// an extra NOC write at the receiver. One segment goes as a plain write instead of a scatter write,
// which costs the receiver one NOC command rather than one per segment.
////////////////////////////////////////////////////////////////

// A Sender gives the packer two calls, and owns the routes and headers behind them:
//   void write_one(uint32_t l1_addr, uint64_t dst, uint32_t bytes);
//   void write_scatter(uint32_t l1_addr, NocUnicastScatterCommandHeader& header, uint32_t payload);
template <uint32_t chunk_size, uint32_t payload, typename Sender>
class Packer {
public:
    Packer(const Noc& noc, Sender& sender) :
        noc{noc}, sender{sender}, header({}, {}), segments{0}, queued{0}, start_l1_addr{0} {}

    ~Packer() {
        ASSERT(segments == 0);  // outstanding segments! flush() not called correctly
    }

    // Precondition: a packet has one payload from start_l1_addr, so its segments must be contiguous
    // in L1. A packed CB gives that; a CB with gaps would need a send() here.
    FORCE_INLINE void add_run(uint32_t l1_addr, uint64_t dst, uint32_t bytes) {
        ASSERT(segments == 0 || l1_addr == start_l1_addr + queued);
        if constexpr (oversized) {
            // A chunk bigger than a packet cannot be accumulated, so it goes as whole packets and a
            // remainder. Such a chunk always walks as a run of one.
            while (bytes > payload) {
                send();
                push(l1_addr, dst, payload);
                send();
                l1_addr += payload;
                dst += payload;
                bytes -= payload;
            }
        }
        if (segments == max_segments || queued + bytes > payload) {
            send();
        }
        push(l1_addr, dst, bytes);
    }

    // Call this before popping a CB entry: a queued packet still points into it.
    void flush() {
        send();
        noc.async_writes_flushed();
    }

private:
    static constexpr uint32_t max_segments = NOC_SCATTER_WRITE_MAX_CHUNKS;
    static constexpr bool oversized = chunk_size > payload;
    static_assert(payload <= 0xFFFF, "NocUnicastScatterCommandHeader::chunk_size is uint16_t");
    static_assert(NOC_SCATTER_WRITE_MIN_CHUNKS == 2, "send() covers the too-few-segments case with one write");

    FORCE_INLINE void push(uint32_t l1_addr, uint64_t dst, uint32_t bytes) {
        if (segments == 0) {
            start_l1_addr = l1_addr;
        }
        // Only the first max_segments-1 sizes travel; the last one is implied by the payload size.
        if (segments < max_segments - 1) {
            header.chunk_size[segments] = static_cast<uint16_t>(bytes);
        }
        header.noc_address[segments++] = dst;
        queued += bytes;
    }

    void send() {
        if (segments == 0) {
            return;
        }
        noc.async_writes_flushed();
        if (segments == 1) {
            sender.write_one(start_l1_addr, header.noc_address[0], queued);
        } else {
            header.chunk_count = segments;
            sender.write_scatter(start_l1_addr, header, queued);
        }
        segments = 0;
        queued = 0;
    }

    const Noc& noc;
    Sender& sender;
    NocUnicastScatterCommandHeader header;
    uint8_t segments;        // segments queued for the current packet
    uint32_t queued;         // bytes queued for the current packet
    uint32_t start_l1_addr;  // start of the queued segments, contiguous in L1
};
