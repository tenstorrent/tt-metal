// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

#include "packetizer.hpp"
#include "route.hpp"

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

    static_assert(sizeof(PACKET_HEADER_TYPE) == hdr_bytes, "host and kernel disagree on the header size");
    static_assert(P >= A && P % A == 0 && P <= NOC_MAX_BURST_SIZE, "a packet is whole units and one NOC burst");
};

// The CB as a ring of slots; slot i reads with trid 1 + i. A slot is pushed as soon as it is complete, its reads
// still in flight: the pump waits for its trid before sending (pump.hpp). With local copy on (multicast), a slot
// is instead held until its reads land, written to its segments' addresses on this device (our NOC's addresses,
// before the header's EDM conversion) and then pushed; at most `inflight` slots are held. Those writes read the
// slot after the push; the pump only reads it too, and we are its only writer, so we flush them before refilling
// it.
template <uint32_t cb, uint32_t slots, uint32_t slot_bytes, uint32_t payload_off, uint32_t inflight>
class CbSlotRing {
public:
    CbSlotRing() : base_(get_write_ptr(cb)), received_(*get_cb_tiles_received_ptr(cb)) {}

    FORCE_INLINE void set_local_copy(bool on) {
        if (copy_ && !on) {
            drain();
        }
        copy_ = on;
    }

    FORCE_INLINE uint32_t acquire() {
        if (free_ == 0) [[unlikely]] {
            wait_free();
        }
        --free_;
        if (shadow_[fill_].nseg != 0) [[unlikely]] {
            flush_copy();
        }
        return base_ + fill_ * slot_bytes;
    }

    FORCE_INLINE uint8_t trid() const { return 1 + fill_; }

    FORCE_INLINE void commit(const Seg* segs, uint32_t nseg) {
        if (copy_) [[unlikely]] {
            commit_copy(segs, nseg);
            return;
        }
        advance();
        // Order the header stores before the push.
        invalidate_l1_cache();
        push();
    }

    // Pushes every held slot.
    __attribute__((noinline)) void drain() {
        while (pending_) {
            retire();
        }
    }

private:
    struct Shadow {
        Seg segs[NOC_SCATTER_WRITE_MAX_CHUNKS];
        uint32_t nseg = 0;
    };

    FORCE_INLINE void advance() { fill_ = fill_ + 1 == slots ? 0 : fill_ + 1; }

    // cb_push_back without the read-modify-write of the stream register: we are its only writer. Slot addresses
    // come from fill_, so the CB's own write pointer is never used.
    FORCE_INLINE void push() { *get_cb_tiles_received_ptr(cb) = ++received_; }

    __attribute__((noinline, cold)) void commit_copy(const Seg* segs, uint32_t nseg) {
        Shadow& sh = shadow_[fill_];
        sh.nseg = nseg;
        for (uint32_t k = 0; k < nseg; ++k) {
            sh.segs[k] = segs[k];
        }
        advance();
        if (++pending_ > inflight) {
            retire();
        }
    }

    __attribute__((noinline, cold)) void flush_copy() {
        noc_async_writes_flushed();
        shadow_[fill_].nseg = 0;
    }

    // Waits for a slot the pump has released; held slots are not in the CB yet but are not free either.
    __attribute__((noinline, cold)) void wait_free() {
        do {
            invalidate_l1_cache();
            const uint16_t received = *get_cb_tiles_received_ptr(cb);
            const uint16_t acked = *get_cb_tiles_acked_ptr(cb);
            free_ = slots - static_cast<uint16_t>(received - acked) - pending_;
        } while (static_cast<int32_t>(free_) <= 0);
    }

    // The barrier also orders the header stores before the push.
    FORCE_INLINE void retire() {
        const uint32_t h = fill_ >= pending_ ? fill_ - pending_ : fill_ + slots - pending_;
        noc_async_read_barrier_with_trid(1 + h);
        const Shadow& sh = shadow_[h];
        uint32_t src = base_ + h * slot_bytes + payload_off;
        for (uint32_t k = 0; k < sh.nseg; ++k) {
            noc_async_write<NOC_MAX_BURST_SIZE>(src, sh.segs[k].dst, sh.segs[k].bytes);
            src += sh.segs[k].bytes;
        }
        push();
        --pending_;
    }

    const uint32_t base_;
    uint32_t received_;
    uint32_t fill_ = 0, pending_ = 0, free_ = 0;
    bool copy_ = false;
    Shadow shadow_[slots];
};

// NOC reads into slots, and C headers per slot (route.hpp's layout). The kernel must reset the read trid to 0 at
// exit.
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
};

// The builder's slot pipeline, all from BuilderCt.
using SlotRing =
    CbSlotRing<BuilderCt::cb, BuilderCt::slots, BuilderCt::slot_bytes, BuilderCt::payload_off, BuilderCt::inflight>;
using SlotIo = NocIo<PACKET_HEADER_TYPE, BuilderCt::C, BuilderCt::hdr_bytes, BuilderCt::payload_off>;
using SlotPacketizer = Packetizer<SlotRing, SlotIo>;
