// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

#include "packetizer.hpp"
#include "route.hpp"
#include "slot_ring.hpp"

// Builder side of the Packetizer: the device Io and the compile-time args both builders share.

// Named compile-time args from all_gather_program.cpp::create_kernels: the slot CB plus the walk geometry.
struct BuilderCt : SlotCt {
    static constexpr uint32_t P = get_named_compile_time_arg_val("P");
    static constexpr uint32_t inflight = get_named_compile_time_arg_val("inflight");
    static constexpr uint32_t Ui = get_named_compile_time_arg_val("Ui");
    static constexpr uint32_t Uo = get_named_compile_time_arg_val("Uo");
    static constexpr uint32_t S = get_named_compile_time_arg_val("S");
    static constexpr uint32_t N = get_named_compile_time_arg_val("N");
    static constexpr uint32_t E = get_named_compile_time_arg_val("E");
    static constexpr bool dst_cross = get_named_compile_time_arg_val("dst_cross") != 0;
    static constexpr bool src_cross = get_named_compile_time_arg_val("src_cross") != 0;

    static_assert(P >= A && P % A == 0 && P <= NOC_MAX_BURST_SIZE, "a packet is whole units and one NOC burst");
};

// NOC reads into slots, and C headers per slot (route.hpp's layout).
template <class Hdr, uint32_t C, uint32_t H, uint32_t payload_off>
struct NocIo {
    // A trid counts at most NOC_MAX_TRANSACTION_ID_COUNT outstanding reads; a packet of small pieces can issue
    // more. A trid is fresh (nothing outstanding) when its slot opens, so check only every kReadsPerPoll reads.
    static constexpr uint32_t kReadsPerPoll = (NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2;

    // The trid is sticky on the read command buffer.
    FORCE_INLINE void start(uint8_t trid) { noc_async_read_set_trid(trid); }

    FORCE_INLINE void throttle(uint8_t trid) {
        while (noc_available_transactions(noc_index, trid) < kReadsPerPoll) {
        }
    }

    FORCE_INLINE void read(uint64_t src, uint32_t l1, uint32_t bytes, uint8_t) {
        noc_async_read<NOC_MAX_BURST_SIZE>(src, l1, bytes);
    }

    FORCE_INLINE void write_headers(uint32_t slot, const Seg* segs, uint32_t nseg, uint32_t bytes) {
        for (uint32_t c = 0; c < C; ++c) {
            encode_segments(
                reinterpret_cast<volatile tt_l1_ptr Hdr*>(slot + slot_header_offset(c, C, H, payload_off)),
                segs,
                nseg,
                bytes);
        }
    }

    // Waits for every read and hands the read command buffer back with trid 0.
    void finish() {
        noc_async_read_set_trid(0);
        noc_async_read_barrier();
    }
};

// The builder's slot pipeline, all from BuilderCt.
using SlotRing = CbSlotRing<BuilderCt>;
using SlotIo = NocIo<PACKET_HEADER_TYPE, BuilderCt::C, BuilderCt::hdr_bytes, BuilderCt::payload_off>;
using SlotPacketizer = Packetizer<SlotRing, SlotIo>;
