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
//   * each RECEIVER core constructs a `ReceiverPipe` and calls `receive(sx, sy)` /
//     `receive_signal()`.
// They are NOT the same type: a receiver never multicasts, so it has no use for the broadcast
// rectangle or the recipient count — it needs only the two semaphores and, at `receive()` time,
// the sender's core coords (the target of its readiness ack).
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
//   * SenderPipe   sets its own local `data_ready` cell = VALID in the ctor (Flag signal). SAFE: the
//                  sender is the sole writer of its own cell before the first send. send() re-asserts
//                  VALID each call, so a core that also receives on this cell broadcasts a fresh VALID
//                  instead of the stale INVALID its last receive left behind.
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
#define MCAST_PIPE_API_VERSION 8

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
// caller (the mcast box holds inactive cores that receive but never ack — conv width-sharded,
// dram-sharded, conv-1D weights) passes its own smaller ack count to override.
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
//   * DATA_READY_SIGNAL          — Flag (default) | Counter (use-case knob); the rarest knob, last.
// Runtime ctor inputs:
//   * the `McastRect` — its virtual coords vary per sender core under one compiled binary (e.g. each
//                       row-sender in a 2D matmul targets a different row), so it is set per-core via
//                       runtime args; its area gives the runtime fan-out.
//   * consumer_ack_count — the handshake (consumer-ready) wait count. Defaults to ACK_EQUALS_FANOUT (=
//                          the EXCLUDE fan-out the rect derives), so dense callers omit it. A caller
//                          whose mcast box holds inactive/noop cores (they receive but never ack) passes
//                          its own smaller ack count. Used only under PRE_HANDSHAKE; cached in the ctor.
template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE = true,
    uint32_t CONSUMER_READY_SEM_ID = UNUSED_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag>
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
//   * DATA_READY_SIGNAL      — must match the SenderPipe's; the rarest knob, last.
// The only runtime input is the sender's coords, passed to receive() (they change per block when the
// sender role rotates — must be runtime).
template <
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE = true,
    uint32_t CONSUMER_READY_SEM_ID = UNUSED_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag>
class ReceiverPipe {
    static_assert(
        !PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID,
        "PRE_HANDSHAKE=true requires a real CONSUMER_READY_SEM_ID (the receiver->sender readiness ack). "
        "Pass it, or set PRE_HANDSHAKE=false to wait the data-ready signal without acking.");

public:
    explicit ReceiverPipe(const Noc& noc);

    // receive(): [ack the sender], wait data-ready, clear the flag (clear-before-ack).
    // `sender_x`/`sender_y` are the SENDER core's NoC coords — the target of the receiver->sender
    // readiness ack. On return the block is in the receiver's dst L1, bit-exact (signal arrival =>
    // data arrival). What the caller does with the dst is its own business.
    void receive(uint32_t sender_x, uint32_t sender_y);

    // Wait the control signal. Symmetric with SenderPipe::send_signal().
    //   * Flag    — a plain doorbell: returns once the signal arrives, then clears it.
    //   * Counter — returns the monotone round number reached.
    uint32_t receive_signal();

private:
    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    uint32_t round_ = 0;  // monotone round counter for DataReadySignal::Counter
};

}  // namespace dataflow_kernel_lib

#include "mcast_pipe.inl"
