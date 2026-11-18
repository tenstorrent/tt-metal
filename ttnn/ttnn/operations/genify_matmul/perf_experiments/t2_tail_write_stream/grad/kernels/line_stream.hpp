// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Refinement 16: the R1a split line streams (host R1A_SPLIT_LINES). One operand's line channel on ONE data-movement
// RISC -- the A row lines on the writer (BRISC / NoC1), the B column lines on the reader (NCRISC / NoC0) -- so the two
// lines no longer serialize on one RISC (the R1a reader received A(kb) before it could ack B(kb)), with ROTATING
// injectors: round r (the walk's global K-block sequence; every core of a line walks the same nominal sequence, so
// their operand rings are in lockstep and round r lands at the same slot address everywhere) is read from DRAM and
// multicast by the line core whose turn it is (`mc.should_send(r)`, span = the line's length); every other core acks
// its reserved ring slot and waits. Before, ONE fixed core per line issued every tile read of its line (text encoder /
// Qwen3.8 M = 512: 1920 bfp4 tile reads per B line at ~35 RISC cycles each, 50 of a 97 us data-movement-only walk) and
// its whole line waited on that serial chain each K-block.
//
// Staging: an injector reads its turn into its own one-K-block staging buffer (`stage_l1`, a scratch CB this RISC
// alone owns), not into the ring slot, which frees only when compute pops the round DEPTH earlier (a depth-2 ring left
// the read no slack: the issue sat on the critical path every round). At its turn it reserves the round's slot like
// every receiver and multicasts staging -> slot, the payload looping back into its own slot (the sender is in its
// line's rectangle); the send fences its writes, so staging is free on return.
// Issue pacing (WINDOW): a turn's reads go out in K-row chunks spread evenly over the WINDOW rounds before it, one
// chunk at the start of each round, before this RISC waits for that round's slot (compute holds it for most of a round,
// so the chunk hides there); round 0 is issued alone, at round 0. In steady state a line then has about one K-block of
// reads issued per round -- what it consumes -- and the soonest rounds are never queued behind later ones. Measured on
// Qwen3.8 M = 512 (B: 240 bfp4 tile reads per K-block at ~35 RISC cycles each, the whole walk ~87 % of DRAM bandwidth):
// issuing every turn as soon as staging allowed put the whole walk's operand on the wire at once (round 0 landed after
// ~40 us, round 1 after ~29); issuing a whole turn one round ahead delayed that round's ack by the ~6 us issue.
// Chunk reads carry transaction id TRID and are retired by it alone, at the send.
//
// Contract: the operand CB is exactly DEPTH K-blocks of BLOCK_TILES pages and this RISC is its only producer; rounds
// are called in order 0, 1, .. total - 1; `read(r, l1, k_lo, k_hi)` issues K rows [k_lo, k_hi) of round r's K-block
// (K_ROWS nominal rows; the functor skips rows past a ragged K-block's valid width) into the slot-shaped buffer `l1`.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace gm_line {

template <
    uint32_t CB,
    uint32_t BLOCK_TILES,
    uint32_t BLOCK_BYTES,
    uint32_t DEPTH,
    uint32_t SPAN,
    uint32_t TRID,
    uint32_t K_ROWS,
    uint32_t WINDOW,
    uint32_t STARTUP_SHARE = 1>
class Stream {
public:
    static_assert(DEPTH >= 1 && SPAN >= 1 && TRID >= 1 && K_ROWS >= 1 && WINDOW >= 1);

    // `phase`: this core's first turn (< SPAN), or SPAN when it never sends.
    FORCE_INLINE Stream(uint32_t total, uint32_t phase, uint32_t stage_l1) :
        ring_base_(total != 0 ? get_write_ptr(CB) : 0),
        stage_(stage_l1),
        total_(total),
        pending_(phase < SPAN ? phase : total) {}

    FORCE_INLINE uint32_t slot(uint32_t r) const { return ring_base_ + (r % DEPTH) * BLOCK_BYTES; }

    // One round (the next one): send it on this core's turn, else ack the reserved slot and wait for it.
    template <typename Sender, typename Receiver, typename ReadFn>
    FORCE_INLINE void round(Noc noc, Sender& sender, Receiver& receiver, ReadFn& read, uint32_t bytes) {
        const uint32_t r = seq_++;
#ifdef GM_ABLATE_DM
        // Perf tooling (/perf-measure): the line's reads and multicasts are stubbed (reserve / push only).
        cb_reserve_back(CB, BLOCK_TILES);
        cb_push_back(CB, BLOCK_TILES);
        return;
#endif
        if (pending_ < total_ && done_ < K_ROWS && pending_ - r <= WINDOW) {
            // This round's share of the pending turn: the rest over the rounds left before it (all of it on the turn).
            // At round 0 a later turn takes only the share it would over one more round, so round 0's own read is
            // not queued behind whole later turns (STARTUP_SHARE).
            const uint32_t rounds_left = pending_ == r ? 1 : pending_ - r + (r == 0 ? STARTUP_SHARE : 0);
            issue(noc, read, done_ + (K_ROWS - done_ + rounds_left - 1) / rounds_left);
        }
        cb_reserve_back(CB, BLOCK_TILES);
        if (r == pending_) {
            {
                MaybeDeviceZoneScope("line_send");
                dataflow_kernel_lib::async_read_barrier_with_trid(noc, TRID);
                sender->send(stage_, slot(r), bytes);
            }
            cb_push_back(CB, BLOCK_TILES);
            pending_ += SPAN;
            done_ = 0;  // staging is free again
        } else {
            MaybeDeviceZoneScope("line_recv");
            receiver->receive(r);
            cb_push_back(CB, BLOCK_TILES);
        }
    }

private:
    template <typename ReadFn>
    FORCE_INLINE void issue(Noc noc, ReadFn& read, uint32_t k_hi) {
        MaybeDeviceZoneScope("line_issue");
        dataflow_kernel_lib::set_read_trid(noc, TRID);
        read(pending_, stage_, done_, k_hi);
        dataflow_kernel_lib::set_read_trid(noc, 0);
        done_ = k_hi;
    }

    uint32_t ring_base_;
    uint32_t stage_;
    uint32_t total_;
    uint32_t seq_ = 0;   // rounds walked
    uint32_t pending_;   // this core's next turn (>= seq_), or >= total_
    uint32_t done_ = 0;  // K rows of pending_ issued so far
};

// `line.round(...)` when ON (a template, so a regime without the stream never instantiates it on its absent faces).
template <bool ON, typename Line, typename Sender, typename Receiver, typename ReadFn>
FORCE_INLINE void round_if(Line& line, Noc noc, Sender& sender, Receiver& receiver, ReadFn& read, uint32_t bytes) {
    if constexpr (ON) {
        line.round(noc, sender, receiver, read, bytes);
    }
}

// This core's first turn on a rotating channel (SPAN when it never sends).
template <uint32_t SPAN, typename Mc>
FORCE_INLINE uint32_t first_turn(const Mc& mc) {
    for (uint32_t r = 0; r < SPAN; ++r) {
        if (mc.should_send(r)) {
            return r;
        }
    }
    return SPAN;
}

}  // namespace gm_line
