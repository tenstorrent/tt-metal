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
// -----------------------------------------------------------------------------
// TWO COUNTS, NOT A MODE KNOB (Round 2)
// -----------------------------------------------------------------------------
// The caller does NOT pick a multicast mode. It states two quantities and the Pipe
// infers everything else:
//   * `McastRect` is PURE GEOMETRY — its area is the data-mcast destination population
//     (the `num_dsts` the NoC broadcasts to; the hardware fills the whole bounding box).
//   * `num_active_cores` is the HANDSHAKE participant count — how many cores actually ACK
//     the R->S "consumed" semaphore. In the canonical case it equals the rect area; it is
//     SMALLER when the broadcast box contains cores that receive the data but do not
//     participate in the handshake (e.g. conv-1D weights — the round-1 single-`num_dests`
//     API could not express this and HUNG).
//
// The EXCLUDE_SRC vs INCLUDE_SRC (loopback) choice is INFERRED AT RUNTIME from whether the
// sender's own core lies inside the rect — `my_x`/`my_y` are read in the Pipe's `noc_`
// index space, the same space the rect is expressed in (IR1). No caller input, no knob.
//   * sender OUTSIDE rect            -> plain multicast        (EXCLUDE_SRC)
//   * sender INSIDE rect             -> loopback multicast     (INCLUDE_SRC, gets own copy)
//   * num_active_cores == 1 & in rect-> self-only: local copy  (loopback to 1 is unspecified)
//
// All style choices are decided by the on-device bake-off (helper_design/mcast_pipe/
// style_bakeoff.md), not by argument:
//   * F1 fence       -> async_writes_flushed (SENT), NOT barrier  (flush −27% vs barrier)
//   * F2 staging     -> level flag (VALID/INVALID) default         (flag −29% vs counter)
//   * F4 linking     -> linked data+flag pair + flush (default)    (linked −36% vs unlinked)
//   * flag reset     -> receiver clears BEFORE acking (clear-before-ack, H11)
//   * data->flag     -> data then flag, same Noc / VC-4 (INV4) — the flag proves arrival
//
// Internal dual-paths (predicates, NOT a config blob the caller navigates):
//   * loopback       -> EXCLUDE_SRC | INCLUDE_SRC | self-only-local-copy (inferred, above)
//   * F4 linking     -> LINKED (default) | unlinked + barrier-between (LINK=false)
//   * Staging::Counter forces the fence to async_atomic_barrier (a write flush HANGS the
//     non-posted multicast atomic — bake-off F2).
//
// Use-case knobs (the only caller-facing template choices):
//   * STAGING       (default Flag)   — Counter only for monotone / streaming protocols.
//   * PRE_HANDSHAKE (default true)   — false when each receiver reserves a fresh CB slot.
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
// A multicast destination rectangle, in NoC (virtual) coordinates. PURE GEOMETRY:
// the rect area `(x1-x0+1)*(y1-y0+1)` IS the data-mcast destination population (what the
// NoC broadcasts to — the hardware fills the whole bounding box). The handshake ACK count
// is passed separately to the Pipe ctor (`num_active_cores`), because the broadcast box can
// contain cores that receive the data but do not participate in the handshake.
//
//   SENDER side  : the receiver rectangle.
//   RECEIVER side: {sender_x, sender_y} 1x1 — points back at the sender (the target of the
//                  R->S "consumed" ack). Construct with the `single_core` factory below.
// -----------------------------------------------------------------------------
struct McastRect {
    uint32_t x0{};
    uint32_t y0{};
    uint32_t x1{};
    uint32_t y1{};

    // Receiver-side helper: a degenerate 1x1 rect pointing at the sender core.
    static constexpr McastRect single_core(uint32_t x, uint32_t y) { return McastRect{x, y, x, y}; }

    // Data-mcast destination population (the NoC `num_dsts`).
    constexpr uint32_t area() const { return (x1 - x0 + 1) * (y1 - y0 + 1); }
};

// -----------------------------------------------------------------------------
// Pipe — the two-sided channel.
// -----------------------------------------------------------------------------
template <
    Staging STAGING = Staging::Flag,  // F2 use-case knob
    bool PRE_HANDSHAKE = true,        // H2 use-case knob
    bool LINK = true>                 // F4 linking (constexpr)
class Pipe {
public:
    // `dest`             — receiver rectangle (geometry only; area = data-mcast population).
    // `num_active_cores` — handshake participant count (R->S ACKs the sender waits for, and
    //                      the degenerate self-only trigger). On the RECEIVER side this is
    //                      unused (receivers never multicast); pass 1 by convention.
    Pipe(
        const Noc& noc,
        const McastRect& dest,
        uint32_t num_active_cores,
        Semaphore<> data_ready,
        Semaphore<> consumed) :
        noc_(noc), dest_(dest), num_active_cores_(num_active_cores), data_ready_(data_ready), consumed_(consumed) {}

    // ===== DATA channel (a block + a ready-flag) =====
    // send() is atomic and absorbs ALL FOUR guards (callers cannot reorder or skip them):
    //   [if PRE_HANDSHAKE] wait(consumed)  — gate the mcast on receivers having drained (H2)
    //   mcast data                          — object API auto-chunks a ready block > burst
    //   raise flag                          — data-before-flag, same VC (INV4); reset owned (H11)
    //   fence                               — flush (F1); atomic-barrier on the counter path
    void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
        // Self-only degenerate: the sender is the only active core (it must be in the rect for
        // this to mean "no remote receiver"). Loopback data+flag mcast to 1 dest is unspecified
        // (may hang, H5) — collapse to a local copy and skip the handshake/fence.
        if (num_active_cores_ <= 1 && sender_in_rect_()) {
            local_copy_(src_l1, dst_l1, size);
            return;
        }
        if constexpr (PRE_HANDSHAKE) {
            consumed_.wait(num_active_cores_);
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
    // ---- is the sender's own core inside the receiver rect? (IR1: compare in noc_'s space) ----
    // Drives the loopback (INCLUDE_SRC) vs plain (EXCLUDE_SRC) decision. `my_x`/`my_y` are the
    // core's own NoC coords for this noc index — the same coordinate space the rect uses.
    bool sender_in_rect_() const {
        const uint32_t mx = my_x[noc_.get_noc_id()];
        const uint32_t my = my_y[noc_.get_noc_id()];
        return mx >= dest_.x0 && mx <= dest_.x1 && my >= dest_.y0 && my <= dest_.y1;
    }

    // ---- data multicast (degenerate self-only already short-circuited in send()) ----
    // num_dsts is the rect AREA (geometric broadcast population), independent of the handshake
    // ACK count. Loopback variant when the sender is in the rect (it needs its own copy too).
    void send_data_(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
        const uint64_t dst_noc =
            ::get_noc_multicast_addr(dest_.x0, dest_.y0, dest_.x1, dest_.y1, dst_l1, noc_.get_noc_id());
        const uint32_t num_dsts = dest_.area();
        if (sender_in_rect_()) {
            noc_async_write_multicast_loopback_src(src_l1, dst_noc, size, num_dsts, /*linked=*/LINK, noc_.get_noc_id());
        } else {
            noc_async_write_multicast(src_l1, dst_noc, size, num_dsts, /*linked=*/LINK, noc_.get_noc_id());
        }
        if constexpr (!LINK) {
            // Unlinked fallback: a barrier must separate data and flag (H10 / F4 unlinked arm).
            noc_.async_write_barrier();
        }
    }

    // ---- raise the data-ready flag (or a control flag with a payload) ----
    void raise_flag_(uint32_t value = VALID) {
        const uint32_t num_dsts = dest_.area();
        if constexpr (STAGING == Staging::Counter) {
            data_ready_.inc_multicast(noc_, dest_.x0, dest_.y0, dest_.x1, dest_.y1, value, num_dsts);
        } else {
            data_ready_.set(value);
            // Loopback vs plain inferred at runtime, same as the data mcast (INV4: same path).
            if (sender_in_rect_()) {
                data_ready_.set_multicast<Noc::McastMode::INCLUDE_SRC>(
                    noc_, dest_.x0, dest_.y0, dest_.x1, dest_.y1, num_dsts, /*linked=*/false);
            } else {
                data_ready_.set_multicast<Noc::McastMode::EXCLUDE_SRC>(
                    noc_, dest_.x0, dest_.y0, dest_.x1, dest_.y1, num_dsts, /*linked=*/false);
            }
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

    // ---- local L1 self-copy (degenerate self-only guard) ----
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
    uint32_t num_active_cores_;
    Semaphore<> data_ready_;
    Semaphore<> consumed_;
    uint32_t round_ = 0;  // monotone round counter for Staging::Counter
};

}  // namespace dataflow_kernel_lib
