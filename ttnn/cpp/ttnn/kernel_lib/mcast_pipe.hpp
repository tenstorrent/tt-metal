// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// mcast_pipe — `SenderPipe` / `ReceiverPipe`: a NoC-multicast + semaphore-handshake helper.
// =============================================================================
//
// Wraps the recurring dataflow block:
//
//   set up a source L1 region -> multicast a block to a receiver rectangle ->
//   handshake with the receivers (sem set / wait / inc) -> flush before reuse.
//
// The channel has TWO faces, materialized as TWO objects (Round 4):
//   * the SENDER core constructs a `SenderPipe` and calls `send()` / `send_signal()`;
//   * each RECEIVER core constructs a `ReceiverPipe` and calls `receive(sx, sy)` /
//     `receive_signal()`.
// They are NOT the same type. The asymmetry is real: a receiver never multicasts, so it
// has no use for the broadcast rectangle or the recipient count — it only needs the two
// semaphores and, at `receive()` time, the sender's core coords (the target of its ack).
// Forcing both onto one object meant the receiver carried dead `dest`/`num_active` args
// ("pass 1 by convention") — the split deletes them.
//
// Built ENTIRELY on the object API (`Noc`, `Semaphore<>`, Unicast/MulticastEndpoint) — NO
// legacy free functions. Every NoC transaction (data mcast, self-copy read, and the flag
// mcast) goes through the `Noc` object; the multicast NoC-address math lives in
// `MulticastEndpoint`, not an open-coded `get_noc_multicast_addr` (Round 4 / noc 2.0).
//
// -----------------------------------------------------------------------------
// GEOMETRY + RECIPIENT COUNT (sender side only)
// -----------------------------------------------------------------------------
// The caller does NOT pick a multicast mode. It states two quantities and the SenderPipe
// infers everything else:
//   * `McastRect` is PURE GEOMETRY — the broadcast bounding box; it is NOT a transfer count. It is
//     the ONLY runtime ctor arg (its virtual coords vary per sender core under one kernel binary).
//   * `NUM_ACTIVE_RECEIVER_CORES` (a TEMPLATE param — compile-time, core-uniform) is the RECIPIENT
//     count: the number of cores the data is multicast to, NOT counting the sender's own in-place
//     copy. It equals the EXCLUDE_SRC NoC `num_dests` AND the R->S ACK count — exactly the value every
//     production factory already computes (matmul's `in0_mcast_num_dests`, etc.), so a shared sender
//     kernel passes the same value across topologies (1D in-rect / 2D out-of-rect) with no host edit.
//     The SenderPipe derives the mcast population from it per inferred mode, so the caller never
//     reasons about mcast mode:
//        EXCLUDE_SRC            : mcast_dests = NUM_ACTIVE_RECEIVER_CORES         (ack count = same)
//        INCLUDE_SRC (loopback) : mcast_dests = NUM_ACTIVE_RECEIVER_CORES + 1     (+1 = the self-copy;
//                                 the self-copy does not ack, so the ACK count stays = NUM_ACTIVE...)
//     (For a future op where ACKs < recipients — conv-1D weights — a third count would be
//     needed; out of scope.)
//
// The EXCLUDE_SRC vs INCLUDE_SRC (loopback) choice is INFERRED per send(), no caller input:
//   loopback (INCLUDE_SRC) iff `sender_in_rect_() && src != dst`. `my_x`/`my_y` are read
//   in the Pipe's `noc_` index space, the same space the rect uses (IR1).
//   (N = NUM_ACTIVE_RECEIVER_CORES, the recipient count.)
//   * sender OUTSIDE box                  -> plain multicast (EXCLUDE_SRC), mcast_dests = N
//   * sender INSIDE box, src != dst       -> loopback multicast (INCLUDE_SRC): the sender is itself
//       a recipient (conv-WS self-gather) -> mcast_dests = N + 1 (its self-copy is the +1).
//   * sender INSIDE box, src == dst       -> plain multicast (EXCLUDE_SRC), mcast_dests = N:
//       its copy is already in place (matmul in0, R6 extract path); never self-overwrite. N here is
//       the OTHER cores (the in-box sender is not counted in N — N is recipients, not box area).
//   * N == 0 (no receiver cores) -> self-only local copy if in box (loopback to just self hangs,
//       H5); else nothing. Skip handshake/fence.
//   The flag mcast rides the SAME mode as the data mcast of that send() (INV4 single path);
//   send_signal() has no data so it is always EXCLUDE_SRC.
//   OUT of inference reach: a sender that needs the loopback FLAG with src == dst (R6 role-flip
//   extract path: flag INCLUDE, data EXCLUDE) — stays raw this round.
//
// All style choices are decided by the on-device bake-off (helper_design/mcast_pipe/
// style_bakeoff.md), not by argument:
//   * F1 fence       -> async_writes_flushed (SENT), NOT barrier  (flush −27% vs barrier)
//   * F2 staging     -> level flag (VALID/INVALID) default         (flag −29% vs counter)
//   * F4 linking     -> linked data+flag pair + flush (always)     (linked −36% vs unlinked)
//   * flag reset     -> receiver clears BEFORE acking (clear-before-ack, H11)
//   * data->flag     -> data then flag, same Noc / VC-4 (INV4) — the flag proves arrival
//
// Internal dual-paths (predicates, NOT a config blob the caller navigates):
//   * loopback       -> EXCLUDE_SRC | INCLUDE_SRC | self-only-local-copy (inferred, above)
//   * Staging::Counter forces the fence to async_atomic_barrier (a write flush HANGS the
//     non-posted multicast atomic — bake-off F2).
//
// F4 linking is NOT a knob (Round 4): every migrated call site links, so the data mcast is
// ALWAYS issued linked to the following flag mcast (the linked pair enforces data-before-flag
// without a barrier, −36%). The unlinked + barrier-between arm had no shipping consumer: its one
// cited candidate (sdpa_decode read_k) was assumed to need a barrier between data and flag, but on
// inspection it merely uses a conservative one — sdpa's own chain_link.hpp links the same broadcast,
// so read_k can link too and would gain the −36%. If a FUTURE kernel must genuinely fence between
// data and flag, re-introduce the arm as a refinement then; don't carry the knob speculatively now.
//
// TEMPLATE params (all compile-time + core-uniform; the caller-facing choices):
//   SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID,
//              STAGING = Flag, PRE_HANDSHAKE = true, INITIAL_READY = VALID>(noc, dest)
//   ReceiverPipe<DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING = Flag, PRE_HANDSHAKE = true>(noc)
//   * STAGING       (default Flag)   — Counter only for monotone / streaming protocols.
//   * PRE_HANDSHAKE (default true)   — false when each receiver reserves a fresh CB slot.
//   * INITIAL_READY (default VALID)  — sender's local "ready" value (Flag only); INVALID for a
//                                       signal sender that starts cleared (sharded-LN phase-1).
//
// -----------------------------------------------------------------------------
// SEMAPHORE LIFECYCLE OWNED BY THE PIPE (Round 4)
// -----------------------------------------------------------------------------
// The semaphore *IDs* are TEMPLATE params (compile-time in every kernel), not pre-built
// `Semaphore<>` objects. Each Pipe instantiates its `Semaphore<>` internally. A Pipe kernel-inits
// a cell ONLY when this core establishes a happens-before edge to every other writer of that cell
// before they write it — otherwise the init races and must come from the HOST:
//   * ReceiverPipe inits its `data_ready` = INVALID (Staging::Flag). SAFE: the receiver writes it
//                  before its own ack, and the sender (the only other writer) is gated behind that
//                  ack — strict happens-before.
//   * SenderPipe   pre-sets ONLY its OWN local `data_ready` = `INITIAL_READY` (default VALID — the
//                  value it broadcasts; Staging::Flag). SAFE: the sender is the sole writer of its
//                  own cell before the first send. `INITIAL_READY = INVALID` for a signal sender.
//   * SenderPipe DOES NOT init `consumed`. RACY if it did: receivers increment the sender's counter
//                  remotely with NO happens-before relative to the sender's ctor (a receiver can ack
//                  before the sender core even runs), so a ctor `set(0)` would clobber an early ack
//                  and HANG. Its initial 0 MUST come from host `CreateSemaphore(..., 0)`.
// HOST-side `CreateSemaphore` on the union of sender+receiver cores allocates the IDs AND owns the
// initial value of any cell written by a remote core (the `consumed` counter); the Pipe owns only
// the race-free local inits above.
//
// Preconditions (INV9): single sender per receiver; semaphores created on the union of
// sender+receiver cores; the landing address `dst_l1` is identical across all receivers.
//
// Scoped OUT (raw API this round): rotating-sender / role-flip same-core flag-INCLUDE arm (R6);
// streaming chunked send of a not-yet-complete block (R4); preprogram-state mcast set-state;
// ring / fabric CCL. The object API auto-chunks a *fully-ready* block > NOC_MAX_BURST_SIZE.
//
// Implementation of the member functions lives in the matching `mcast_pipe.inl`, included at
// the bottom of this header.
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
// A multicast destination rectangle, in NoC (virtual) coordinates. PURE GEOMETRY: the
// broadcast bounding box. The transfer/ACK count is the separate `NUM_ACTIVE_RECEIVER_CORES`
// template param — NOT the area. SENDER side only (the receiver does not multicast).
//
// CORNER ORDERING IS OWNED HERE (not the caller). The mcast hardware walks the rect from
// `start` in the NoC's routing direction up to `end`, so `start` must be the corner the
// routing reaches FIRST: the low corner on NoC0 (+x/+y), the high corner on NoC1 (-x/-y).
// `McastRect` stores the four numbers in WHATEVER order it was constructed with — canonical
// top-left→bottom-right, or already swapped (some hosts std::swap for NOC_1) — and
// `start_end_for_noc()` always re-derives the routing-correct (start,end) for the live NoC.
// So however the rect is initialized, the mcast APIs always receive the corners in good order.
//
// Pure geometry + all-constexpr, so the full definition stays inline in this header.
// -----------------------------------------------------------------------------
struct McastRect {
    uint32_t x0{};
    uint32_t y0{};
    uint32_t x1{};
    uint32_t y1{};

    // Order-agnostic normalized bounds (tolerate either input ordering).
    constexpr uint32_t xlo() const { return x0 < x1 ? x0 : x1; }
    constexpr uint32_t xhi() const { return x0 < x1 ? x1 : x0; }
    constexpr uint32_t ylo() const { return y0 < y1 ? y0 : y1; }
    constexpr uint32_t yhi() const { return y0 < y1 ? y1 : y0; }

    // Routing-correct (start_x, start_y, end_x, end_y) for the mcast APIs on `noc_id`.
    // NoC0 → start = low corner; NoC1 → start = high corner (the per-NoC swap the host used to
    // do with std::swap on NOC_1 — now owned by the rect, applied as a full diagonal-corner swap
    // to match the host's CoreCoord-pair swap).
    struct Bounds {
        uint32_t sx, sy, ex, ey;
    };
    constexpr Bounds start_end_for_noc(uint8_t noc_id) const {
        return noc_id == 1 ? Bounds{xhi(), yhi(), xlo(), ylo()} : Bounds{xlo(), ylo(), xhi(), yhi()};
    }
};

// =============================================================================
// SenderPipe — the broadcasting face of the channel.
// =============================================================================
// All compile-time-known, core-uniform values are TEMPLATE params (every migrated kernel sources
// them from `get_compile_time_arg_val`, and they are identical across all cores running the kernel):
//   * NUM_ACTIVE_RECEIVER_CORES — FULL recipient count, INCLUDING the sender when it is one of the
//                                 recipients. The Pipe derives ack_count and mcast_dests from it.
//   * DATA_READY_SEM_ID         — S->R "data is ready" flag id.
//   * CONSUMED_SEM_ID           — R->S "dest drained" counter id (used iff PRE_HANDSHAKE).
//   * STAGING / PRE_HANDSHAKE   — the use-case knobs.
//   * INITIAL_READY             — value the ctor pre-sets the sender's LOCAL data-ready cell to
//                                 (Flag staging only). DEFAULT `VALID` (5/6 migrated data senders set
//                                 VALID before the loop). A signal sender that must start INVALID
//                                 passes `INITIAL_READY = INVALID` (sharded-LN phase-1). Folding it in
//                                 lets call sites drop their manual `<flag_sem>.set(VALID)` line.
// The ONLY runtime ctor input is the `McastRect` — its (virtual) coords vary per sender core under
// one compiled binary (e.g. each row-sender in a 2D matmul targets a different row), so it is set
// per-core via runtime args and CANNOT be a template param. `send()`/`send_signal()` payload + the
// receiver's sender coords are runtime for the same reason (CB pointers; rotating senders).
template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING = Staging::Flag,  // F2 use-case knob
    bool PRE_HANDSHAKE = true,        // H2 use-case knob
    uint32_t INITIAL_READY = VALID>   // local "ready" value (Flag only); INVALID for a signal sender
class SenderPipe {
public:
    // `dest` — receiver rectangle (geometry only). The only runtime ctor arg (see above).
    explicit SenderPipe(const Noc& noc, const McastRect& dest);

    // ===== DATA channel (a block + a ready-flag) =====
    // send() is atomic and absorbs ALL FOUR guards (callers cannot reorder or skip them):
    //   [if PRE_HANDSHAKE] wait(consumed)  — gate the mcast on receivers having drained (H2)
    //   mcast data                          — object API auto-chunks a ready block > burst
    //   raise flag                          — data-before-flag, same VC (INV4); reset owned (H11)
    //   fence                               — flush (F1); atomic-barrier on the counter path
    void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    // ===== CONTROL channel (a pure flag, no data block, R2) =====
    // Broadcast a control flag. `value` carries a payload for value-carrying flags
    // (e.g. moe_gpt token counts); defaults to VALID for a plain doorbell. Always EXCLUDE_SRC
    // (no data accompanies it), so the destination population is the recipient count.
    void send_signal(uint32_t value = VALID);

private:
    // ---- is the sender's own core inside the receiver rect? (IR1: compare in noc_'s space) ----
    bool sender_in_rect_() const;

    // ---- data multicast via the Noc object (noc 2.0) ----
    void send_data_(uint32_t src_l1, uint32_t dst_l1, uint32_t size, bool loopback, uint32_t mcast_dests);

    // ---- raise the data-ready flag (or a control flag with a payload) ----
    // `loopback` mirrors the data mcast of the same send() (INV4: same path); send_signal()
    // has no data, so its flag is always EXCLUDE_SRC.
    void raise_flag_(uint32_t value, bool loopback, uint32_t mcast_dests);

    // ---- post-send fence (F1) ----
    void fence_();

    // ---- local L1 self-copy (degenerate self-only guard) via the Noc object (noc 2.0) ----
    void local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    Noc noc_;
    McastRect dest_;
    Semaphore<> data_ready_;
    Semaphore<> consumed_;
};

// =============================================================================
// ReceiverPipe — the listening face of the channel. No rectangle, no recipient count.
// =============================================================================
// Sem ids + use-case knobs are TEMPLATE params (compile-time, core-uniform — same as SenderPipe).
//   * DATA_READY_SEM_ID — S->R "data is ready" flag id (this core waits on it).
//   * CONSUMED_SEM_ID   — R->S "dest drained" counter id (this core increments it on the sender
//                         remotely; the id supplies the shared L1 offset).
// The only runtime input is the sender's coords, passed to receive() (they vary per receiver in 2D
// and rotate per block in R6 — must be runtime).
template <
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING = Staging::Flag,  // F2 use-case knob (must match the SenderPipe's)
    bool PRE_HANDSHAKE = true>        // H2 use-case knob (must match the SenderPipe's)
class ReceiverPipe {
public:
    explicit ReceiverPipe(const Noc& noc);

    // receive(): [ack the sender], wait data-ready, clear the flag (clear-before-ack, H11).
    // `sender_x`/`sender_y` are the SENDER core's NoC coords — the target of the R->S
    // "consumed" ack. On return the block is in the receiver's dst L1, bit-exact (flag arrival
    // => data arrival via INV4). What the caller does with the dst is its own business.
    void receive(uint32_t sender_x, uint32_t sender_y);

    // Wait the control flag and RETURN its value. Symmetric with SenderPipe::send_signal().
    //   * Staging::Flag    — a plain doorbell: returns VALID once the flag arrives, then clears.
    //   * Staging::Counter — returns the monotone round number reached.
    // Value-carrying-flag payload extraction (moe_gpt: token counts packed in the sem word) is
    // a migration-time concern for that op — it reads its own sem cell directly, since it owns
    // the cell address. Kept out of the generic doorbell path to keep receive_signal() simple.
    uint32_t receive_signal();

private:
    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumed_;
    uint32_t round_ = 0;  // monotone round counter for Staging::Counter
};

}  // namespace dataflow_kernel_lib

#include "mcast_pipe.inl"
