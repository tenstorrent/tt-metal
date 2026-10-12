// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Decoupled slot protocol for the vsa_sdpa streaming kernel (VSA_DECOUPLE).
//
// The leader's stream ring is a CACHE, not a barrier: its slot gate waits for the k-th slowest consumer (not the
// slowest), and a consumer that falls further behind fetches the blocks it missed from DRAM itself. The log ring
// holds a whole pass of entries (pass_len block entries + 1 sentinel), so a lagging worker always finds the entry
// of the arrival it is at; the leader may run at most one pass ahead of the slowest worker (the "log gate").
//
// Safety of a pull from the leader's slot (worker side): the leader issues fetch m = n + stream_depth (which
// refills the slot of arrival n) only after it has published arrival m - kFetchLag. So if arrival
// n + stream_depth - kFetchLag - kMargin is NOT yet published in this worker's log, the refill cannot have been
// issued yet and the slot still holds block n. The check is made when the pull is issued (else the block is
// fetched from DRAM directly) and again once the pull has landed (a refill that raced the pull is repaired by a
// DRAM re-read). kMargin arrivals of slack cover the NoC skew between the leader's publish write and its refill
// landing. The data is bit-identical from either source and the window partition never depends on the source,
// so the kernel stays deterministic.
#pragma once

#include <stdint.h>

namespace vsa_dec {

#ifndef VSA_FETCH_LAG
#define VSA_FETCH_LAG 4
#endif
constexpr uint32_t kFetchLag = VSA_FETCH_LAG;  // == the leader reader's kFetchLag (blocks fetched ahead of publishing)
#ifndef VSA_DEC_MARGIN
#define VSA_DEC_MARGIN 2
#endif
constexpr uint32_t kMargin = VSA_DEC_MARGIN;  // extra arrivals of slack (publish write vs. refill landing skew); a
                                              // larger margin turns late (blocking) repairs into early async fallbacks
constexpr uint32_t kEntryWords = 4;

struct View {
    uint32_t log_l1 = 0;        // the log ring base (same CB base on every core of the group)
    uint32_t log_depth = 0;     // entries in the ring (>= pass_len + 1)
    uint32_t pass_len = 1;      // block arrivals per pass (real blocks)
    uint32_t total = 0;         // block arrivals over all passes
    uint32_t stream_depth = 0;  // slots in the stream ring

    // global log number of block arrival j: every pass adds pass_len block entries and one sentinel
    uint32_t log_of_arrival(uint32_t j) const {
        const uint32_t p = j / pass_len;
        return p * (pass_len + 1) + (j - p * pass_len);
    }
    // has log entry L been published? (its ring slot's seq word is L + 1, or larger if the ring wrapped past it)
    bool published(uint32_t L) const {
        invalidate_l1_cache();
        const uint32_t seq =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(log_l1 + (L % log_depth) * kEntryWords * 4 + 8);
        return seq >= L + 1;
    }
    // may the leader already be refilling the slot of arrival n?
    bool slot_unsafe(uint32_t n) const {
        const uint32_t j = n + stream_depth - kFetchLag - kMargin;
        if (j >= total) {
            return false;  // no fetch will ever refill it
        }
        return published(log_of_arrival(j));
    }
};

}  // namespace vsa_dec
