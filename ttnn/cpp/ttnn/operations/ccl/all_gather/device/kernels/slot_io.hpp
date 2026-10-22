// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

#include "packetizer.hpp"

// Builder side of the Packetizer: the device Ring and Io, and the compile-time args both builders share.

// Named compile-time args from all_gather_program.cpp::create_kernels.
struct BuilderCt {
    static constexpr uint32_t cb = get_named_compile_time_arg_val("cb");
    static constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    static constexpr uint32_t slot_bytes = get_named_compile_time_arg_val("slot_bytes");
    static constexpr uint32_t payload_off = get_named_compile_time_arg_val("payload_off");
    static constexpr uint32_t hdr_bytes = get_named_compile_time_arg_val("hdr_bytes");
    static constexpr uint32_t C = get_named_compile_time_arg_val("C");
    static constexpr uint32_t A = get_named_compile_time_arg_val("A");
    static constexpr uint32_t P = get_named_compile_time_arg_val("P");
    static constexpr uint32_t inflight = get_named_compile_time_arg_val("inflight");
    static constexpr uint32_t Ui = get_named_compile_time_arg_val("Ui");
    static constexpr uint32_t Uo = get_named_compile_time_arg_val("Uo");
    static constexpr uint32_t S = get_named_compile_time_arg_val("S");
    static constexpr uint32_t N = get_named_compile_time_arg_val("N");
    static constexpr uint32_t E = get_named_compile_time_arg_val("E");
    static constexpr bool dst_cross = get_named_compile_time_arg_val("dst_cross") != 0;
    static constexpr bool src_cross = get_named_compile_time_arg_val("src_cross") != 0;
    static constexpr bool cross_row = get_named_compile_time_arg_val("cross_row") != 0;

    static_assert(noc_index == 0, "slot headers hold NOC0 addresses until to_noc_* converts them");
    static_assert(sizeof(PACKET_HEADER_TYPE) == hdr_bytes, "host and kernel disagree on the header size");
    static_assert(P >= A && P % A == 0 && P <= NOC_MAX_BURST_SIZE, "a packet is whole units and one NOC burst");
};

// The CB as a ring of slots, with up to `inflight` committed slots still reading. A committed slot is pushed
// only once its reads land, so the CB write pointer stays on the oldest of them. Slot i reads with trid 1 + i.
class CbSlotRing {
public:
    CbSlotRing(uint32_t cb, uint32_t slots, uint32_t slot_bytes, uint32_t inflight) :
        cb_(cb), slots_(slots), slot_bytes_(slot_bytes), inflight_(inflight), base_(get_write_ptr(cb)) {}

    FORCE_INLINE uint32_t acquire() {
        // Room for the committed slots and this one; the pump may hold the rest.
        cb_reserve_back(cb_, pending_ + 1);
        return base_ + current() * slot_bytes_;
    }

    FORCE_INLINE uint8_t trid() const { return 1 + current(); }

    FORCE_INLINE void commit() {
        if (++pending_ > inflight_) {
            retire();
        }
    }

    FORCE_INLINE void drain() {
        while (pending_) {
            retire();
        }
    }

private:
    FORCE_INLINE uint32_t current() const {
        const uint32_t i = head_ + pending_;
        return i < slots_ ? i : i - slots_;
    }

    // The barrier also orders the header stores before the push.
    FORCE_INLINE void retire() {
        noc_async_read_barrier_with_trid(1 + head_);
        cb_push_back(cb_, 1);
        head_ = head_ + 1 == slots_ ? 0 : head_ + 1;
        --pending_;
    }

    const uint32_t cb_, slots_, slot_bytes_, inflight_, base_;
    uint32_t head_ = 0, pending_ = 0;
};

// NOC reads into slots, and C headers per slot, H bytes apart. The kernel must reset the read trid to 0 at exit.
template <class Hdr>
struct NocIo {
    uint32_t C, H;
    uint8_t trid_ = 0;

    FORCE_INLINE void read(uint64_t src, uint32_t l1, uint32_t bytes, uint8_t trid) {
        // The trid is sticky on the read command buffer, so set it only when it changes.
        if (trid != trid_) {
            noc_async_read_set_trid(trid);
            trid_ = trid;
        }
        // A trid counts at most NOC_MAX_TRANSACTION_ID_COUNT outstanding reads; a packet of small pieces can
        // issue more.
        while (noc_available_transactions(noc_index, trid) < (NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2) {
        }
        noc_async_read<NOC_MAX_BURST_SIZE>(src, l1, bytes);
    }

    FORCE_INLINE void write_headers(uint32_t slot, const Seg* segs, uint32_t nseg, uint32_t bytes) {
        for (uint32_t c = 0; c < C; ++c) {
            encode_segments(reinterpret_cast<volatile tt_l1_ptr Hdr*>(slot + c * H), segs, nseg, bytes);
        }
    }
};
