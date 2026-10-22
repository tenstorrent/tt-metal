// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host tests (test_all_gather_kernel_lib.cpp) compile this file with their own CB and NOC calls.
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/dataflow/dataflow_api.h"
#endif

#include <cstdint>

#include "packetizer.hpp"

// The builder's CB as a ring of slots (the Packetizer's Ring); slot i reads with trid 1 + i. A slot is pushed as soon
// as it is complete, its reads still in flight: the pump waits for its trid before sending (pump.hpp). With local
// copy (multicast), a slot is instead held until its reads land, written to its segments' addresses on this device
// (our NOC's addresses, before the header's EDM conversion) and then pushed; at most Ct::inflight slots are held.
// The pump only reads a slot, so those writes are flushed before the slot is refilled.
template <class Ct>
class CbSlotRing {
    static constexpr uint32_t slots = Ct::slots;

public:
    explicit CbSlotRing(bool local_copy = false) :
        base_(get_write_ptr(Ct::cb)), received_(*get_cb_tiles_received_ptr(Ct::cb)), copy_(local_copy) {}

    FORCE_INLINE uint32_t acquire() {
        if (free_ == 0) [[unlikely]] {
            wait_free();
        }
        --free_;
        if (shadow_[fill_].nseg != 0) [[unlikely]] {
            flush_copy();
        }
        return base_ + fill_ * Ct::slot_bytes;
    }

    FORCE_INLINE uint8_t trid() const { return 1 + fill_; }

    FORCE_INLINE void commit(const Seg* segs, uint32_t nseg) {
        if (copy_) [[unlikely]] {
            commit_copy(segs, nseg);
            return;
        }
        advance();
        // The pump reads our trid's outstanding count: the last read must be counted before it can see the slot.
        while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
        }
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
    FORCE_INLINE void push() { *get_cb_tiles_received_ptr(Ct::cb) = ++received_; }

    __attribute__((noinline, cold)) void commit_copy(const Seg* segs, uint32_t nseg) {
        Shadow& sh = shadow_[fill_];
        sh.nseg = nseg;
        for (uint32_t k = 0; k < nseg; ++k) {
            sh.segs[k] = segs[k];
        }
        advance();
        if (++pending_ > Ct::inflight) {
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
            const uint16_t received = *get_cb_tiles_received_ptr(Ct::cb);
            const uint16_t acked = *get_cb_tiles_acked_ptr(Ct::cb);
            free_ = slots - static_cast<uint16_t>(received - acked) - pending_;
        } while (static_cast<int32_t>(free_) <= 0);
    }

    // The barrier also orders the header stores before the push.
    FORCE_INLINE void retire() {
        const uint32_t h = fill_ >= pending_ ? fill_ - pending_ : fill_ + slots - pending_;
        noc_async_read_barrier_with_trid(1 + h);
        const Shadow& sh = shadow_[h];
        uint32_t src = base_ + h * Ct::slot_bytes + Ct::payload_off;
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
    const bool copy_;
    Shadow shadow_[slots];
};
