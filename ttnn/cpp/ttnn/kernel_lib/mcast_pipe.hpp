// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// mcast_pipe — `Pipe`: a two-sided NoC-multicast + semaphore-handshake helper.
// =============================================================================
//
// Wraps the recurring dataflow block:
//
//   set up a source L1 region -> multicast a block to a receiver rectangle ->
//   handshake with the receivers (sem set / wait / inc) -> flush before reuse.
//
// `Pipe` exists on BOTH sides of the channel: the sender core calls `send()` /
// `send_signal()`, each receiver core calls `receive()` / `receive_signal()`.
// They are two faces of the same channel.
//
// Built on the object API (`Noc`, `Semaphore<>`) — NOT the legacy free functions.
//
// All style choices are decided by the on-device bake-off (helper_design/mcast_pipe/
// style_bakeoff.md), not by argument:
//   * F1 fence       -> async_writes_flushed (SENT), NOT barrier  (flush −27% vs barrier)
//   * F2 staging     -> level flag (VALID/INVALID) default         (flag −29% vs counter)
//   * F4 linking     -> linked data+flag pair + flush (default)    (linked −36% vs unlinked)
//   * flag reset     -> receiver clears BEFORE acking (clear-before-ack, H11)
//   * data->flag     -> data then flag, same Noc / VC-4 (INV4) — the flag proves arrival
//
// Internal dual/tri-paths (constexpr predicates, NOT a config blob the caller navigates):
//   * F3 loopback    -> EXCLUDE_SRC (default) | INCLUDE_SRC (sender in rect, needs own copy)
//                       | degenerate (num_dests==1 self-only) -> local copy guard (else hangs)
//   * F4 linking     -> LINKED (default) | unlinked + barrier-between (LINK=false)
//   * Staging::Counter forces the fence to async_atomic_barrier (a write flush HANGS the
//     non-posted multicast atomic — bake-off F2).
//
// Use-case knobs (the only caller-facing template choices):
//   * STAGING       (default Flag)   — Counter only for monotone / streaming protocols.
//   * PRE_HANDSHAKE (default true)   — false when each receiver reserves a fresh CB slot.
//   * MCAST         (default EXCLUDE_SRC) — INCLUDE_SRC for sender-in-rect loopback (F3).
//   * LINK          (default true)   — false where a barrier is structurally required
//                                       between data and flag (e.g. sdpa read_k).
//
// Preconditions (INV9): single sender per receiver; semaphores created on the union of
// sender+receiver cores; the landing address `dst_l1` is identical across all receivers.
//
// Scoped OUT (raw API this round): rotating-sender / role-flip same-core (R6); streaming
// chunked send of a not-yet-complete block (R4); preprogram-state mcast set-state; ring /
// fabric CCL. The object API auto-chunks a *fully-ready* block > NOC_MAX_BURST_SIZE.
// =============================================================================

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

namespace dataflow_kernel_lib {

// -----------------------------------------------------------------------------
// Staging style (F2). Flag is the baked-in default (fastest). Counter is exposed
// ONLY for protocols that genuinely need a monotone, reset-free counter (census C3:
// layernorm phase-2 streaming, deepseek_prefill).
// -----------------------------------------------------------------------------
enum class Staging { Flag, Counter };

// -----------------------------------------------------------------------------
// A multicast destination, in NoC (virtual) coordinates.
//   SENDER side  : {rect corners, num_dests} = the receiver rectangle.
//   RECEIVER side: {sender_x, sender_y, sender_x, sender_y, 1} = points back at the
//                  sender (the target of the R->S "consumed" ack). Construct with the
//                  `single_core` factory below.
//
// num_dests is the count the handshake accounts for. The Pipe never re-derives it from
// the rectangle area because EXCLUDE_SRC vs INCLUDE_SRC changes the count by one (INV8);
// the caller (which knows its geometry) passes the value that matches MCAST.
// -----------------------------------------------------------------------------
struct McastRect {
    uint32_t x0{};
    uint32_t y0{};
    uint32_t x1{};
    uint32_t y1{};
    uint32_t num_dests{};

    // Receiver-side helper: a degenerate 1x1 rect pointing at the sender core.
    static constexpr McastRect single_core(uint32_t x, uint32_t y) { return McastRect{x, y, x, y, 1}; }
};

// -----------------------------------------------------------------------------
// Pipe — the two-sided channel.
// -----------------------------------------------------------------------------
template <
    Noc::McastMode MCAST = Noc::McastMode::EXCLUDE_SRC,  // F3 loopback mode (constexpr)
    Staging STAGING = Staging::Flag,                     // F2 use-case knob
    bool PRE_HANDSHAKE = true,                           // H2 use-case knob
    bool LINK = true>                                    // F4 linking (constexpr)
class Pipe {
public:
    Pipe(const Noc& noc, const McastRect& dest, Semaphore<> data_ready, Semaphore<> consumed) :
        noc_(noc), dest_(dest), data_ready_(data_ready), consumed_(consumed) {}

    // ===== DATA channel (a block + a ready-flag) =====
    // send() is atomic and absorbs ALL FOUR guards (callers cannot reorder or skip them):
    //   [if PRE_HANDSHAKE] wait(consumed)  — gate the mcast on receivers having drained (H2)
    //   mcast data                          — object API auto-chunks a ready block > burst
    //   raise flag                          — data-before-flag, same VC (INV4); reset owned (H11)
    //   fence                               — flush (F1); atomic-barrier on the counter path
    void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
        if constexpr (MCAST == Noc::McastMode::INCLUDE_SRC) {
            // Degenerate self-only rect (INCLUDE_SRC counts self, so num_dests==1 means NO remote
            // core): loopback data + flag mcast are unspecified (may hang, H5). Collapse to a local
            // copy and skip the handshake/fence — there is no remote receiver to signal or wait on.
            if (dest_.num_dests <= 1) {
                local_copy_(src_l1, dst_l1, size);
                return;
            }
        }
        if constexpr (PRE_HANDSHAKE) {
            consumed_.wait(dest_.num_dests);
            consumed_.set(0);
        }
        send_data_(src_l1, dst_l1, size);
        raise_flag_();
        fence_();
    }

    // receive(): wait data-ready, clear the flag (clear-before-ack, H11), [ack slot-free].
    // On return the block is in the receiver's dst L1, bit-exact (flag arrival => data
    // arrival via INV4). What the caller does with the dst is its own business (the Pipe
    // never owned the downstream CB consume).
    void receive() {
        if constexpr (PRE_HANDSHAKE) {
            // tell the sender "my dest is free / I am ready" (remote atomic inc on its counter)
            consumed_.up(noc_, dest_.x0, dest_.y0, 1);
        }
        if constexpr (STAGING == Staging::Counter) {
            data_ready_.wait_min(++round_);
        } else {
            data_ready_.wait(VALID);
            data_ready_.set(INVALID);  // clear this round's flag; next receive()'s ack follows (H11)
        }
    }

    // ===== CONTROL channel (a pure flag, no data block, R2) =====
    // Broadcast a control flag. `value` carries a payload for value-carrying flags
    // (e.g. moe_gpt token counts); defaults to VALID for a plain doorbell.
    void send_signal(uint32_t value = VALID) {
        raise_flag_(value);
        fence_();
    }

    // Wait the control flag and RETURN its value. Symmetric with send_signal().
    //   * Staging::Flag    — a plain doorbell: returns VALID once the flag arrives, then clears.
    //   * Staging::Counter — returns the monotone round number reached.
    // Value-carrying-flag payload extraction (moe_gpt: token counts packed in the sem word) is
    // a migration-time concern for that op — it reads its own sem cell directly, since it owns
    // the cell address. Kept out of the generic doorbell path to keep receive()/receive_signal()
    // dead simple.
    uint32_t receive_signal() {
        if constexpr (STAGING == Staging::Counter) {
            data_ready_.wait_min(++round_);
            return round_;
        } else {
            data_ready_.wait(VALID);
            data_ready_.set(INVALID);
            return VALID;
        }
    }

private:
    // ---- data multicast (degenerate self-only already short-circuited in send()) ----
    void send_data_(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
        const uint64_t dst_noc =
            ::get_noc_multicast_addr(dest_.x0, dest_.y0, dest_.x1, dest_.y1, dst_l1, noc_.get_noc_id());
        if constexpr (MCAST == Noc::McastMode::INCLUDE_SRC) {
            noc_async_write_multicast_loopback_src(
                src_l1, dst_noc, size, dest_.num_dests, /*linked=*/LINK, noc_.get_noc_id());
        } else {
            noc_async_write_multicast(src_l1, dst_noc, size, dest_.num_dests, /*linked=*/LINK, noc_.get_noc_id());
        }
        if constexpr (!LINK) {
            // Unlinked fallback: a barrier must separate data and flag (H10 / F4 unlinked arm).
            noc_.async_write_barrier();
        }
    }

    // ---- raise the data-ready flag (or a control flag with a payload) ----
    void raise_flag_(uint32_t value = VALID) {
        if constexpr (STAGING == Staging::Counter) {
            data_ready_.inc_multicast(noc_, dest_.x0, dest_.y0, dest_.x1, dest_.y1, value, dest_.num_dests);
        } else {
            data_ready_.set(value);
            data_ready_.set_multicast<MCAST>(
                noc_, dest_.x0, dest_.y0, dest_.x1, dest_.y1, dest_.num_dests, /*linked=*/false);
        }
    }

    // ---- post-send fence (F1) ----
    void fence_() {
        if constexpr (STAGING == Staging::Counter) {
            // inc_multicast is a NON-POSTED multicast atomic: it expects num_dests ACKs that
            // must be drained, so a write flush is NOT sufficient — an atomic barrier is forced.
            noc_.async_writes_flushed();
            noc_.async_atomic_barrier();
        } else {
            noc_.async_writes_flushed();  // SENT — source L1 safe to reuse (H1). Flag proves arrival (INV4).
        }
    }

    // ---- local L1 self-copy (degenerate loopback guard, F3) ----
    void local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
        if (src_l1 == dst_l1) {
            return;  // src == dst: nothing to copy (source polymorphism, R4)
        }
        const uint64_t self_src =
            get_noc_addr(my_x[noc_.get_noc_id()], my_y[noc_.get_noc_id()], src_l1, noc_.get_noc_id());
        noc_async_read(self_src, dst_l1, size);
        noc_.async_read_barrier();
    }

    Noc noc_;
    McastRect dest_;
    Semaphore<> data_ready_;
    Semaphore<> consumed_;
    uint32_t round_ = 0;  // monotone round counter for Staging::Counter
};

}  // namespace dataflow_kernel_lib
