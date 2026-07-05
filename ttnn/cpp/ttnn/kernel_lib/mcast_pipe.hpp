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
//   stage a source L1 region -> multicast a block to a receiver rectangle ->
//   signal the receivers that the data is ready -> flush before reusing the source.
//
// The channel has TWO faces, materialized as TWO objects:
//   * the SENDER core constructs a `SenderPipe` and calls `send()` / `send_signal()`;
//   * each RECEIVER core constructs a `ReceiverPipe` and calls `receive()` / `receive_signal()`.
// They are NOT the same type: a receiver never multicasts, so it has no use for the broadcast
// rectangle or the recipient count — it needs only the two semaphores and the sender core coords it
// acks (handed to its constructor and kept).
//
// -----------------------------------------------------------------------------
// SEMAPHORE LIFECYCLE OWNED BY THE PIPE
// -----------------------------------------------------------------------------
// The semaphore *IDs* are template params; each Pipe constructs its `Semaphore<>` internally. A Pipe
// kernel-inits a cell ONLY when this core has a happens-before edge to every other writer of that cell
// before they write it — otherwise the init races and the initial value must come from the HOST:
//   * ReceiverPipe inits its own `data_ready` = INVALID (Flag signal). SAFE: the receiver writes it
//                  before its own ack, and the sender — the only other writer — is gated behind that
//                  ack.
//   * SenderPipe   does NOT init `data_ready` in the ctor. send() asserts VALID locally right before it
//                  broadcasts the flag (so a core that also receives on this cell sends a fresh VALID,
//                  never the stale INVALID its last receive left behind) — a ctor set would be
//                  redundant with that. Under ROTATING_SENDER, send() also resets the cell to INVALID
//                  after the flag is flushed, so this core's next RECEIVER turn waits for a real VALID
//                  instead of returning on the stale VALID its own send left behind.
//   * SenderPipe does NOT init `consumer_ready`. That counter is incremented by REMOTE receivers with
//                  no happens-before relative to the sender's ctor (a receiver can ack before the
//                  sender core even runs), so a ctor `set(0)` would clobber an early ack and hang. Its
//                  initial 0 MUST come from host `CreateSemaphore(..., 0)`.
// HOST-side `CreateSemaphore` on the union of sender+receiver cores allocates the IDs and owns the
// initial value of any cell a remote core writes (`consumer_ready`); the Pipe owns only the race-free
// local inits above.
//
// Preconditions: single sender per receiver; semaphores created on the union of sender+receiver cores;
// the landing address `dst_l1` is identical across all receivers.
// =============================================================================

#pragma once

// Caller-facing API version — the staleness key for the apply-dm-helper rollout ledger
// (helper_design/mcast_pipe/migration/ledger.json). BUMP THIS (and only this) whenever a
// re-materialization changes the caller-facing API (renamed/removed type, moved param, changed
// count/flag semantics — anything that forces a call site rewrite); leave it for internal-only
// changes.
#define MCAST_PIPE_API_VERSION 9

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

namespace dataflow_kernel_lib {

// -----------------------------------------------------------------------------
// How the sender tells the receivers the data is ready (the sender->receiver data-ready signal).
//   * Flag (default, fastest): a level flag set to VALID/INVALID. Pick this for the common case — one
//     handshake per round, with the receiver free to reset the flag between rounds.
//   * Counter: a monotone, reset-free counter. Pick this ONLY for tight multi-phase streaming, where
//     the sender would otherwise stall each round waiting for the receiver to reset the flag.
// -----------------------------------------------------------------------------
enum class DataReadySignal { Flag, Counter };

// Sentinel for the CONSUMER_READY_SEM_ID template param: "no consumer-ready semaphore". The default
// when PRE_HANDSHAKE is false (the receiver→sender readiness ack is not used), so the no-handshake
// caller omits the arg entirely. A real CTA semaphore id is small and dense, so this reserved value is
// never a live id; a `static_assert` rejects PRE_HANDSHAKE=true paired with this sentinel.
static constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

// Sentinel for the SenderPipe `consumer_ack_count` ctor arg: "the ack count equals the EXCLUDE-source
// mcast fan-out" (the dense case — every core the broadcast lands on also acks). The default, so dense
// callers omit the arg entirely and let the rect carry both the fan-out and the ack count. A divergent
// caller (the mcast box holds inactive cores that receive but never ack) passes its own smaller ack
// count to override.
static constexpr uint32_t ACK_EQUALS_FANOUT = 0xFFFFFFFFu;

// -----------------------------------------------------------------------------
// A multicast destination rectangle, in NoC (virtual) coordinates. PURE GEOMETRY: the broadcast
// bounding box. Sender side only (the receiver does not multicast).
//
// Templated on the NoC id. The mcast hardware walks the rect from `start` in the NoC's routing
// direction up to `end`, so `start` must be the corner the routing reaches FIRST: the low corner on
// NoC0 (+x/+y), the high corner on NoC1 (-x/-y). The NoC id is compile-time, so the constructor
// computes — ONCE — both the routing-correct (start,end) for `NOC_ID` and the normalized box; the
// per-send corner comparison and per-NoC swap are hoisted out of the hot path. Callers may pass the
// four corners in ANY order (canonical top-left→bottom-right or already swapped) — the normalization
// tolerates either, so the mcast APIs always receive the corners in routing order.
// PRECONDITION: `NOC_ID` must match the `Noc` the `SenderPipe` runs on.
// -----------------------------------------------------------------------------
template <uint8_t NOC_ID = noc_index>
struct McastRect {
    // Routing-correct (start_x, start_y, end_x, end_y) for the mcast APIs on NOC_ID.
    struct Bounds {
        uint32_t sx, sy, ex, ey;
    };

    // Coords may arrive in either ordering; normalize + precompute the routing corners ONCE.
    constexpr McastRect(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) :
        xlo_(x0 < x1 ? x0 : x1),
        xhi_(x0 < x1 ? x1 : x0),
        ylo_(y0 < y1 ? y0 : y1),
        yhi_(y0 < y1 ? y1 : y0),
        // NoC0 -> start = low corner; NoC1 -> start = high corner (matches the host's per-NoC corner
        // swap). Decided at construction, not per send().
        start_end_(NOC_ID == 1 ? Bounds{xhi_, yhi_, xlo_, ylo_} : Bounds{xlo_, ylo_, xhi_, yhi_}) {}

    // Precomputed routing-correct (start,end) for NOC_ID — a field read, no comparison.
    constexpr const Bounds& bounds() const { return start_end_; }

    // Normalized box (for the sender-in-rect containment test).
    constexpr uint32_t xlo() const { return xlo_; }
    constexpr uint32_t xhi() const { return xhi_; }
    constexpr uint32_t ylo() const { return ylo_; }
    constexpr uint32_t yhi() const { return yhi_; }

    // Bounding-box area = the INCLUDE-source mcast fan-out (the cores the broadcast lands on). COUNT
    // USE ONLY — never the loopback-mode test (that stays `in_rect_ && src!=dst`). Computed on the
    // normalized corners, so it is order-independent like the containment test. Runtime (the corners
    // are runtime), so the fan-out it feeds is runtime too.
    constexpr uint32_t area() const { return (xhi_ - xlo_ + 1) * (yhi_ - ylo_ + 1); }

private:
    uint32_t xlo_, xhi_, ylo_, yhi_;
    Bounds start_end_;
};

// =============================================================================
// SenderPipe — the broadcasting face of the channel.
// =============================================================================
// All compile-time-known, core-uniform values are TEMPLATE params:
//   * NOC_ID                     — compile-time NoC id; must match the `noc` arg (ctor ASSERTs this).
//                                  Folds my_x/my_y and the rect's routing corners to constants.
//   * DATA_READY_SEM_ID          — sender->receiver "data is ready" flag id.
//   * PRE_HANDSHAKE              — gate each send on receivers having drained (use-case knob, default).
//   * CONSUMER_READY_SEM_ID      — receiver->sender "my dest is free" counter id. Used ONLY when
//                                  PRE_HANDSHAKE; defaults to UNUSED_SEM_ID so the no-handshake caller
//                                  omits it. A static_assert rejects PRE_HANDSHAKE without a real id.
//   * DATA_READY_SIGNAL          — Flag (default) | Counter (use-case knob).
//   * ROTATING_SENDER            — rotating-sender mode (default false). When a core SENDS on some
//                                  rounds and RECEIVES on others over the SAME data_ready cell, send()
//                                  resets the cell to INVALID after the broadcast is flushed, so this
//                                  core's next receiver turn waits for a fresh VALID instead of its own
//                                  stale one. Flag signal only; the rarest knob, last.
// Runtime ctor inputs:
//   * the `McastRect` — its virtual coords vary per sender core under one compiled binary (each sender
//                       targets a different receiver rectangle), so it is set per-core via runtime
//                       args; its area gives the runtime fan-out.
//   * consumer_ack_count — the handshake (consumer-ready) wait count. Defaults to ACK_EQUALS_FANOUT (=
//                          the EXCLUDE fan-out the rect derives), so dense callers omit it. A caller
//                          whose mcast box holds inactive/noop cores (they receive but never ack) passes
//                          its own smaller ack count. Used only under PRE_HANDSHAKE; cached in the ctor.
template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE = true,
    uint32_t CONSUMER_READY_SEM_ID = UNUSED_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag,
    bool ROTATING_SENDER = false>
class SenderPipe {
    static_assert(
        !PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID,
        "PRE_HANDSHAKE=true requires a real CONSUMER_READY_SEM_ID (the receiver->sender readiness ack). "
        "Pass it, or set PRE_HANDSHAKE=false for a fire-and-forget broadcast.");

public:
    // `dest` — receiver rectangle (geometry only); its area gives the runtime mcast fan-out.
    // `consumer_ack_count` — the handshake wait count (used only under PRE_HANDSHAKE). Defaults to
    // ACK_EQUALS_FANOUT, meaning "= the EXCLUDE fan-out the rect derives" (the dense case). Both are
    // runtime ctor args; everything they feed is precomputed ONCE here so send() does no arithmetic.
    explicit SenderPipe(const Noc& noc, const McastRect<NOC_ID>& dest, uint32_t consumer_ack_count = ACK_EQUALS_FANOUT);

    // ===== DATA channel (a block + a ready signal) =====
    // send() is atomic and absorbs ALL FOUR guards (callers cannot reorder or skip them):
    //   [if PRE_HANDSHAKE] wait(consumer_ready)  — gate the mcast on receivers having drained
    //   mcast data                                — object API auto-chunks a ready block > burst
    //   signal ready                              — data-before-signal, same VC; reset owned by receiver
    //   fence                                     — flush; atomic-barrier on the Counter path
    void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    // ===== CONTROL channel (a pure ready signal, no data block) =====
    // Broadcast a plain readiness signal (a doorbell). Always plain (EXCLUDE-source) — no data
    // accompanies it. Pairs with ReceiverPipe::receive_signal().
    void send_signal();

private:
    // ---- data multicast via the Noc object ----
    void send_data_(uint32_t src_l1, uint32_t dst_l1, uint32_t size, bool loopback, uint32_t mcast_dests);

    // ---- signal the receivers the data is ready ----
    // `loopback` matches the data mcast of the same send(): when send() included the sender's own core
    // as a receiver, the signal must reach it too. send_signal() carries no data, so it never loops back.
    void signal_ready_(bool loopback, uint32_t mcast_dests);

    // ---- post-send fence ----
    void fence_();

    // ---- local L1 self-copy (degenerate self-only guard) via the Noc object ----
    void local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    Noc noc_;
    McastRect<NOC_ID> dest_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    bool in_rect_;             // is this sender's own core inside the receiver rect? computed once in the ctor
    bool degenerate_;          // self-only box (no receivers) -> send() does a local copy
    uint32_t num_dests_excl_;  // EXCLUDE-source mcast fan-out  = area - (in_rect?1:0)
    uint32_t num_dests_incl_;  // INCLUDE-source (loopback) fan-out = num_dests_excl_ + 1
    uint32_t ack_count_;       // consumer-ready handshake wait count (PRE_HANDSHAKE only)
};

// =============================================================================
// ReceiverPipe — the listening face of the channel. No rectangle, no recipient count.
// =============================================================================
// Sem ids + use-case knobs are TEMPLATE params (compile-time, core-uniform — same as SenderPipe).
//   * DATA_READY_SEM_ID      — sender->receiver "data is ready" flag id (this core waits on it).
//   * PRE_HANDSHAKE          — ack the sender before waiting (use-case knob, default); must match the
//                              SenderPipe's.
//   * CONSUMER_READY_SEM_ID  — receiver->sender "my dest is free" counter id (this core increments it
//                              on the sender remotely; the id supplies the shared L1 offset). Used ONLY
//                              when PRE_HANDSHAKE; defaults to UNUSED_SEM_ID so the no-handshake caller
//                              omits it. A static_assert rejects PRE_HANDSHAKE without a real id.
//   * DATA_READY_SIGNAL      — must match the SenderPipe's.
//   * NUM_SENDERS            — how many sender coord pairs this receiver keeps (1 for a fixed sender,
//                              SPAN for a rotating line where a different core sends each round).
//
// The sender coords are handed to the CONSTRUCTOR as an array and KEPT in the object — mirroring
// SenderPipe, which is handed its McastRect and keeps it. The pipe never touches runtime args: the
// arg-aware layer (McastArgs::receiver()) reads the RT block and builds the array; a by-hand caller
// builds it from its own coords. receive(round) then acks/listens to the round-th stored sender (round
// defaults to 0 — the only entry for a fixed receiver).
template <
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE = true,
    uint32_t CONSUMER_READY_SEM_ID = UNUSED_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag,
    uint32_t NUM_SENDERS = 1>
class ReceiverPipe {
    static_assert(
        !PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID,
        "PRE_HANDSHAKE=true requires a real CONSUMER_READY_SEM_ID (the receiver->sender readiness ack). "
        "Pass it, or set PRE_HANDSHAKE=false to wait the data-ready signal without acking.");
    static_assert(NUM_SENDERS >= 1, "ReceiverPipe needs at least one sender coord pair.");

public:
    // `sender_coords` — NUM_SENDERS (x,y) pairs laid out [x0, y0, x1, y1, ...], the sender(s) this
    // receiver acks/listens to (virtual NoC coords). COPIED into the object at construction; the pipe
    // never reads runtime args itself. McastArgs::receiver() fills this from the RT block; a by-hand
    // caller passes its own.
    explicit ReceiverPipe(const Noc& noc, const uint32_t (&sender_coords)[2 * NUM_SENDERS]);

    // receive(round): ack the round-th stored sender, wait data-ready, clear the flag (clear-before-ack).
    // A fixed receiver calls receive() (round 0); a rotating one passes the round to pick that round's
    // sender out of the stored list. Nothing is passed per call — the coords are already in the object.
    // On return the block is in the receiver's dst L1, bit-exact (signal arrival => data arrival).
    void receive(uint32_t round = 0);

    // Wait the control signal. Symmetric with SenderPipe::send_signal(). (Does not use the coords.)
    //   * Flag    — a plain doorbell: returns once the signal arrives, then clears it.
    //   * Counter — returns the monotone round number reached.
    uint32_t receive_signal();

private:
    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    uint32_t coords_[2 * NUM_SENDERS];  // sender coord pairs [x0,y0,...], copied at construction
    uint32_t round_ = 0;                // monotone round counter for DataReadySignal::Counter
};

// =============================================================================
// McastArgs — the KERNEL counterpart of host::Mcast1D / host::Mcast2D.
// =============================================================================
// The host (ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp) emits the mcast wire; McastArgs self-parses
// it the nice way — like TensorAccessorArgs<BASE> — so the call site stops hand-indexing CT slots,
// stops fishing coords out of RT, and stops re-spelling the SenderPipe/ReceiverPipe template.
//
// ONE decoder owns BOTH arg lists: it is templated on the CT base AND the RT base (and, for the
// rotating wire, the line span). Row vs column vs single-rect is a pure HOST concern (which cores
// send, what rect); the decoder is shape-agnostic, so the SAME McastArgs serves a row-family, a
// column-family, and a single-sender->rect.
//
//   CT block (5 words):  [ active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags ]
//        flags bit0 = pre_handshake (gate on the receiver->sender readiness ack)
//        flags bit1 = data-ready signal (0 = Flag, 1 = Counter)
//   RT block, fixed (SPAN == 0), 4 words:
//        sender   -> dest rect corners (virtual, NOC-ordered)
//        receiver -> [ sender_x, sender_y, 0, 0 ]
//   RT block, rotating (SPAN > 0), 4 + 2*SPAN words:
//        every core -> [ full-line rect corners, s0_x, s0_y, ... s{SPAN-1}_x, s{SPAN-1}_y ]
//
// The pipe *behaviour* (pre_handshake, data-ready signal, dense/divergent ack, rotating) is NOT a
// call-site knob any more — the host computes each and rides it on the wire, and McastArgs feeds them
// into the pipe template. So `sender(noc)` / `receiver(noc)` take nothing but the Noc, and McastArgs is
// the ONLY place that touches runtime args: sender() reads the dest rect off RT and hands it to a
// SenderPipe (which keeps it); receiver() reads the sender coord(s) off RT and hands them to a
// ReceiverPipe (which keeps them). The two pipes are symmetric — each is constructed with its coords
// and stores them; neither reads runtime args itself. The rotating vs fixed RT layout is the one thing
// the caller still spells, as the SPAN template param (it sizes the RT block); SPAN > 0 alone selects
// rotating.

// The one mcast-args decoder. Chainable in BOTH arg lists, exactly like TensorAccessorArgs:
//   McastArgs<a.next_compile_time_args_offset(), a.next_runtime_args_offset()> picks up right after a
//   previous family `a` in CT and RT alike — no hand-indexed slots. SPAN defaults to 0 (fixed
//   sender, 4-word RT); a non-zero SPAN selects the rotating wire (4 + 2*SPAN RT words) AND the
//   rotating-sender pipe behaviour. The CT reads are constexpr (valid non-type template args); the RT
//   reads are lazy (get_arg_val at access), so the object holds only its template bases and stays
//   constexpr-constructible for the chain.
template <uint32_t CT_BASE, uint32_t RT_BASE, uint32_t SPAN = 0>
struct McastArgs {
    // ---- CT (self-parsed) ----
    static constexpr uint32_t active = get_compile_time_arg_val(CT_BASE + 0);
    static constexpr uint32_t data_ready = get_compile_time_arg_val(CT_BASE + 1);
    static constexpr uint32_t consumer_ready = get_compile_time_arg_val(CT_BASE + 2);
    static constexpr uint32_t num_active = get_compile_time_arg_val(CT_BASE + 3);
    static constexpr uint32_t flags = get_compile_time_arg_val(CT_BASE + 4);

    // Pipe behaviour lifted off the flags word (host-computed): the caller never spells these.
    static constexpr bool pre_handshake = (flags & 0x1u) != 0u;
    static constexpr DataReadySignal signal =
        ((flags >> 1) & 0x1u) != 0u ? DataReadySignal::Counter : DataReadySignal::Flag;
    static constexpr bool rotating = SPAN > 0;

    // Sender coord pairs this family carries: 1 for a fixed sender, SPAN for a rotating line.
    static constexpr uint32_t num_senders = SPAN == 0 ? 1u : SPAN;

    static constexpr uint32_t next_compile_time_args_offset() { return CT_BASE + 5; }
    static constexpr uint32_t num_runtime_args() { return SPAN == 0 ? 4u : (4u + 2u * SPAN); }
    static constexpr uint32_t next_runtime_args_offset() { return RT_BASE + num_runtime_args(); }

    // ---- pipe construction: NO behaviour knobs; everything comes from the wire ----
    // sender(): reads its dest rect off RT (McastRect normalizes + NOC-orders the corners itself) and
    // builds the correctly-typed SenderPipe. NOC_ID stays a template default (noc_index) for the rare
    // writer-NoC sender. Guard the send() itself on `active` for an inactive family (no receivers).
    template <uint8_t NOC_ID = noc_index>
    SenderPipe<NOC_ID, data_ready, pre_handshake, consumer_ready, signal, rotating> sender(const Noc& noc) const {
        return SenderPipe<NOC_ID, data_ready, pre_handshake, consumer_ready, signal, rotating>(
            noc, rect<NOC_ID>(), num_active);
    }

    // receiver(): read the sender coords off the RT block HERE (the arg-aware layer — the pipe never
    // touches runtime args) and hand them to a ReceiverPipe that keeps them. FIXED: the one pair at
    // RT_BASE+0/+1. ROTATING: SPAN pairs, one per round, past the rect. The call site then just calls
    // receive() (fixed) / receive(round) (rotating) — no coords passed.
    ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders> receiver(const Noc& noc) const {
        uint32_t coords[2 * num_senders];
        if constexpr (SPAN == 0) {
            coords[0] = sender_x();
            coords[1] = sender_y();
        } else {
            for (uint32_t i = 0; i < SPAN; ++i) {
                coords[2 * i + 0] = sender_x(i);
                coords[2 * i + 1] = sender_y(i);
            }
        }
        return ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders>(noc, coords);
    }

    // ---- RT coord accessors (escape hatches) ----
    // The happy path never needs these (sender()/receiver() read RT internally); they exist for the
    // rare kernel that needs a coord for something else.
    // Sender view: the dest rectangle (fixed: receivers only; rotating: the full line incl. self).
    template <uint8_t NOC_ID = noc_index>
    McastRect<NOC_ID> rect() const {
        return McastRect<NOC_ID>(
            get_arg_val<uint32_t>(RT_BASE + 0),
            get_arg_val<uint32_t>(RT_BASE + 1),
            get_arg_val<uint32_t>(RT_BASE + 2),
            get_arg_val<uint32_t>(RT_BASE + 3));
    }
    // Receiver view, FIXED: the sender's coords (the target of this receiver's readiness ack).
    uint32_t sender_x() const { return get_arg_val<uint32_t>(RT_BASE + 0); }
    uint32_t sender_y() const { return get_arg_val<uint32_t>(RT_BASE + 1); }
    // Receiver view, ROTATING: the coords of the sender broadcasting on `round`, round in [0, SPAN).
    uint32_t sender_x(uint32_t round) const { return get_arg_val<uint32_t>(RT_BASE + 4 + 2 * round + 0); }
    uint32_t sender_y(uint32_t round) const { return get_arg_val<uint32_t>(RT_BASE + 4 + 2 * round + 1); }
};

}  // namespace dataflow_kernel_lib

#include "mcast_pipe.inl"
