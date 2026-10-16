// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/dataflow/dataflow_api.h"
#elif !defined(FORCE_INLINE) || !defined(ASSERT)
// Host build (unit tests): include tests/ttnn/unit_tests/gtests/accessor/host_kernel_shim.hpp first (with
// HOST_KERNEL_SHIM_KEEP_MACROS). No fallbacks here, so the shim's throwing ASSERT can never be shadowed by a
// no-op one that an earlier include defined.
#error "host build: define FORCE_INLINE and ASSERT (include host_kernel_shim.hpp) before this header"
#endif

#include <cstdint>

#include "gather_rules.hpp"  // gather_scatter_{max,min}_chunks

////////////////////////////////////////////////////////////////
// Packet packing, shared by both all_gather algorithms. Host-safe: no fabric includes, so a host test
// can drive it with a recording Sink.
//
// A segment is a run of chunks contiguous at the destination. Segments are packed into one packet
// until either the payload or the scatter-chunk count runs out; a one-segment packet goes out as a
// unicast write, which costs each receiving ERISC a single NoC command instead of one per segment.
//
// Sink interface (see HopSink / McastSink in gather_fabric.hpp, and a host mock in the unit test):
//   using ScatterSlots = ...;  // NocUnicastScatterCommandHeader on device, GatherScatterSlots on host
//   void unicast_write(uint32_t l1, uint64_t dst, uint32_t bytes);
//   void scatter_write(uint32_t l1, const ScatterSlots& slots, uint32_t payload);
//       slots.noc_address[0..chunk_count), slots.chunk_size[0..chunk_count-1) -- the last size is
//       implied by payload. The packer fills the slots in place, so a packet costs no copy.
//   void flush();  // the NoC flush that must precede popping a CB entry a packet pointed into
// Each write is responsible for its own pre-send NoC flush (the packet header slot is reused).
////////////////////////////////////////////////////////////////

// Host stand-in for NocUnicastScatterCommandHeader: the fields the packer writes, same names.
struct GatherScatterSlots {
    uint64_t noc_address[gather_scatter_max_chunks];
    uint16_t chunk_size[gather_scatter_max_chunks - 1];
    uint8_t chunk_count = 0;
};

template <uint32_t packet_size, uint32_t chunk_size, typename Sink>
class PacketPacker {
public:
    explicit PacketPacker(Sink& sink) : sink_{sink}, slots_{{}, {}}, chunk_count_{0}, payload_{0}, start_l1_addr_{0} {}

    ~PacketPacker() {
        ASSERT(chunk_count_ == 0);  // outstanding segments! flush_packet_and_wait() not called correctly
    }

    // A segment that does not fit starts a new packet rather than spilling into this one: splitting it
    // would fill the tail but cost an extra scatter chunk, i.e. an extra NoC write at each receiver.
    //
    // Precondition: a packet has one payload from start_l1_addr_, so its segments must be contiguous
    // in L1. The packed CB gives that; a CB with gaps would need a send() here.
    FORCE_INLINE void queue_segment(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        if constexpr (oversized) {
            // A chunk bigger than a packet always walks as a run of one (max_xfer() is 1 when
            // the chunk exceeds the packet), so each segment is exactly one chunk, sent as a train of
            // full packets and a tail -- the same packets the old unicast while-loop sent, but with
            // the tail sent here rather than at the next segment or flush.
            ASSERT(bytes == chunk_size);
            for (uint32_t packet = 0; packet < packets_per_chunk; ++packet) {
                sink_.unicast_write(
                    l1_addr, remote_noc_addr, (packet < packets_per_chunk - 1) ? packet_size : last_payload_size);
                l1_addr += packet_size;
                remote_noc_addr += packet_size;
            }
        } else {
            ASSERT(chunk_count_ == 0 || l1_addr == start_l1_addr_ + payload_);
            if (chunk_count_ == max_chunks || payload_ + bytes > packet_size) {
                send();
            }
            push(l1_addr, remote_noc_addr, bytes);
        }
    }

    // Call this before popping a CB entry: a queued packet still points into it.
    void flush_packet_and_wait() {
        if constexpr (!oversized) {
            send();
        }
        sink_.flush();
    }

private:
    static constexpr uint32_t max_chunks = gather_scatter_max_chunks;
    // A chunk larger than a packet cannot be accumulated, so it is split across packets instead.
    static constexpr bool oversized = chunk_size > packet_size;
    static constexpr uint32_t packets_per_chunk = (chunk_size + packet_size - 1) / packet_size;  // div_up
    static constexpr uint32_t last_payload_size = chunk_size - ((packets_per_chunk - 1) * packet_size);
    static_assert(packet_size > 0 && chunk_size > 0, "packet and chunk sizes must be nonzero");
    static_assert(packet_size <= 0xFFFF, "NocUnicastScatterCommandHeader::chunk_size is uint16_t");
    static_assert(gather_scatter_min_chunks == 2, "send() covers the too-few-chunks case with one unicast write");

    FORCE_INLINE void push(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        if (chunk_count_ == 0) {
            start_l1_addr_ = l1_addr;
        }
        // Only the first max_chunks-1 sizes travel; the last one is implied by the payload size.
        if (chunk_count_ < max_chunks - 1) {
            slots_.chunk_size[chunk_count_] = static_cast<uint16_t>(bytes);
        }
        slots_.noc_address[chunk_count_++] = remote_noc_addr;
        payload_ += bytes;
    }

    FORCE_INLINE void send() {
        if (chunk_count_ == 0) {
            return;
        }
        if (chunk_count_ == 1) {
            sink_.unicast_write(start_l1_addr_, slots_.noc_address[0], payload_);
        } else {
            slots_.chunk_count = chunk_count_;
            sink_.scatter_write(start_l1_addr_, slots_, payload_);
        }
        chunk_count_ = 0;
        payload_ = 0;
    }

    Sink& sink_;
    typename Sink::ScatterSlots slots_;  // segment list of the current packet
    uint8_t chunk_count_;                // segments queued for the current packet
    uint32_t payload_;                   // bytes queued for the current packet
    uint32_t start_l1_addr_;             // start of the queued segments, contiguous in L1
};
