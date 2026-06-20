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
//   * SenderPipe   sets its own local `data_ready` cell = VALID once in the ctor (Flag signal). SAFE:
//                  the sender is the sole writer of its own cell before the first send. This is the
//                  value the per-send mcast broadcasts (`set_multicast` reads the local cell as its
//                  source), so it is set once and reused — never re-set per send.
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
#define MCAST_PIPE_API_VERSION 6

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

// -----------------------------------------------------------------------------
// A multicast destination rectangle, in NoC (virtual) coordinates. PURE GEOMETRY: the broadcast
// bounding box. The transfer/ack count is the separate `NUM_ACTIVE_RECEIVER_CORES` template param, NOT
// the area. Sender side only (the receiver does not multicast).
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

private:
    uint32_t xlo_, xhi_, ylo_, yhi_;
    Bounds start_end_;
};

// =============================================================================
// SenderPipe — the broadcasting face of the channel.
// =============================================================================
// All compile-time-known, core-uniform values are TEMPLATE params:
//   * NOC_ID                     — compile-time NoC id; must match the `noc` arg. Folds my_x/my_y and
//                                  the rect's routing corners to constants.
//   * DATA_READY_SEM_ID          — sender->receiver "data is ready" flag id.
//   * CONSUMER_READY_SEM_ID      — receiver->sender "my dest is free" counter id (used iff PRE_HANDSHAKE).
//   * NUM_ACTIVE_RECEIVER_CORES  — the RECIPIENT count (see header). The Pipe derives the ack count and
//                                  mcast_dests from it.
//   * DATA_READY_SIGNAL          — Flag (default) | Counter (use-case knob).
//   * PRE_HANDSHAKE              — gate each send on receivers having drained (use-case knob).
// The ONLY runtime ctor input is the `McastRect` — its virtual coords vary per sender core under one
// compiled binary (e.g. each row-sender in a 2D matmul targets a different row), so it is set per-core
// via runtime args. `send()`/`send_signal()` payload + the receiver's sender coords are runtime for
// the same reason (CB pointers; rotating senders).
template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMER_READY_SEM_ID,
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag,
    bool PRE_HANDSHAKE = true>
class SenderPipe {
public:
    // `dest` — receiver rectangle (geometry only). The only runtime ctor arg (see above).
    explicit SenderPipe(const Noc& noc, const McastRect<NOC_ID>& dest) :
        noc_(noc), dest_(dest), data_ready_(DATA_READY_SEM_ID), consumer_ready_(CONSUMER_READY_SEM_ID) {
        // `consumer_ready` is NOT kernel-initialized: remote receivers increment it with no
        // happens-before relative to this ctor, so a ctor set(0) would clobber an early ack and hang.
        // Its initial 0 comes from host `CreateSemaphore(..., 0)`.
        //
        // The sender's OWN local data-ready cell IS owned by this ctor (only the sender writes it
        // before the first send). For the Flag signal, `set_multicast` broadcasts this local cell as
        // its source, and that source is always VALID, so it is set ONCE here and reused every send —
        // never re-set per send. The no-loopback path relies on this (its EXCLUDE-source mcast does not
        // touch the sender's own cell); the loopback path overwrites it harmlessly with the same VALID.
        if constexpr (DATA_READY_SIGNAL == DataReadySignal::Flag) {
            data_ready_.set(VALID);
        }
    }

    // ===== DATA channel (a block + a ready signal) =====
    // send() is atomic and absorbs ALL FOUR guards (callers cannot reorder or skip them):
    //   [if PRE_HANDSHAKE] wait(consumer_ready)  — gate the mcast on receivers having drained
    //   mcast data                                — object API auto-chunks a ready block > burst
    //   signal ready                              — data-before-signal, same VC; reset owned by receiver
    //   fence                                     — flush; atomic-barrier on the Counter path
    void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
        // Degenerate: no receiver cores. If the sender is in its own box and lands a copy elsewhere, do
        // a local copy (a loopback to just self may hang); else nothing.
        if (NUM_ACTIVE_RECEIVER_CORES == 0) {
            if (sender_in_rect_()) {
                local_copy_(src_l1, dst_l1, size);
            }
            return;
        }
        if constexpr (PRE_HANDSHAKE) {
            consumer_ready_.wait(NUM_ACTIVE_RECEIVER_CORES);
            consumer_ready_.set(0);
        }
        // Loopback iff the sender is in the box AND lands its own copy somewhere other than its source
        // (src == dst means the copy is already in place; never self-overwrite in place).
        const bool loopback = sender_in_rect_() && src_l1 != dst_l1;
        // NUM_ACTIVE_RECEIVER_CORES is the recipient count (= EXCLUDE-source num_dests = ack count). The
        // loopback path adds +1 for the sender's own self-copy, which never acks, so the consumer_ready
        // wait above stays on NUM_ACTIVE_RECEIVER_CORES.
        const uint32_t mcast_dests = loopback ? NUM_ACTIVE_RECEIVER_CORES + 1 : NUM_ACTIVE_RECEIVER_CORES;
        send_data_(src_l1, dst_l1, size, loopback, mcast_dests);
        signal_ready_(loopback, mcast_dests);  // the signal rides the same mode as the data
        fence_();
    }

    // ===== CONTROL channel (a pure ready signal, no data block) =====
    // Broadcast a plain readiness signal (a doorbell). Always plain (EXCLUDE-source) — no data
    // accompanies it. Pairs with ReceiverPipe::receive_signal().
    void send_signal() {
        if (NUM_ACTIVE_RECEIVER_CORES == 0) {
            return;  // nobody to signal
        }
        signal_ready_(/*loopback=*/false, NUM_ACTIVE_RECEIVER_CORES);
        fence_();
    }

private:
    // ---- is the sender's own core inside the receiver rect? (compare in NOC_ID's space) ----
    bool sender_in_rect_() const {
        const uint32_t mx = my_x[NOC_ID];
        const uint32_t my = my_y[NOC_ID];
        return mx >= dest_.xlo() && mx <= dest_.xhi() && my >= dest_.ylo() && my <= dest_.yhi();
    }

    // ---- data multicast via the Noc object ----
    void send_data_(uint32_t src_l1, uint32_t dst_l1, uint32_t size, bool loopback, uint32_t mcast_dests) {
        const auto& r = dest_.bounds();  // routing-correct start/end (precomputed in the rect's ctor)
        UnicastEndpoint src_ep;
        MulticastEndpoint dst_ep;
        const typename noc_traits_t<UnicastEndpoint>::src_args_type src_args{.addr = src_l1};
        const typename noc_traits_t<MulticastEndpoint>::dst_args_mcast_type dst_args{r.sx, r.sy, r.ex, r.ey, dst_l1};
        // Data is always linked to the following signal mcast (signal_ready_ issues the signal with
        // linked=false to terminate the chain). The linked pair enforces data-before-signal without a
        // barrier.
        if (loopback) {
            noc_.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                src_ep, dst_ep, size, mcast_dests, src_args, dst_args, /*linked=*/true);
        } else {
            noc_.async_write_multicast<NocOptions::DEFAULT>(
                src_ep, dst_ep, size, mcast_dests, src_args, dst_args, /*linked=*/true);
        }
    }

    // ---- signal the receivers the data is ready ----
    // `loopback` mirrors the data mcast of the same send(); send_signal() has no data, so it is always
    // plain (EXCLUDE-source). The Flag path broadcasts the sender's persistent local VALID cell (set
    // once in the ctor) — no per-send local set.
    void signal_ready_(bool loopback, uint32_t mcast_dests) {
        const auto& r = dest_.bounds();  // routing-correct start/end (precomputed in the rect's ctor)
        if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
            data_ready_.inc_multicast(noc_, r.sx, r.sy, r.ex, r.ey, /*value=*/1, mcast_dests);  // monotone +1
        } else if (loopback) {
            data_ready_.set_multicast<NocOptions::MCAST_INCL_SRC>(
                noc_, r.sx, r.sy, r.ex, r.ey, mcast_dests, /*linked=*/false);
        } else {
            data_ready_.set_multicast<NocOptions::DEFAULT>(noc_, r.sx, r.sy, r.ex, r.ey, mcast_dests, /*linked=*/false);
        }
    }

    // ---- post-send fence ----
    void fence_() {
        if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
            // inc_multicast is a NON-POSTED multicast atomic: it expects num_dests acks that must be
            // drained, so a write flush is not sufficient — an atomic barrier is forced.
            noc_.async_writes_flushed();
            noc_.async_atomic_barrier();
        } else {
            noc_.async_writes_flushed();  // SENT — source L1 safe to reuse. The signal proves arrival.
        }
    }

    // ---- local L1 self-copy (degenerate self-only guard) via the Noc object ----
    void local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
        if (src_l1 == dst_l1) {
            return;  // src == dst: nothing to copy
        }
        UnicastEndpoint src_ep, dst_ep;
        const uint32_t mx = my_x[NOC_ID];
        const uint32_t my = my_y[NOC_ID];
        noc_.async_read(
            src_ep,
            dst_ep,
            size,
            typename noc_traits_t<UnicastEndpoint>::src_args_type{mx, my, src_l1},
            typename noc_traits_t<UnicastEndpoint>::dst_args_type{0, 0, dst_l1});
        noc_.async_read_barrier();
    }

    Noc noc_;
    McastRect<NOC_ID> dest_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
};

// =============================================================================
// ReceiverPipe — the listening face of the channel. No rectangle, no recipient count.
// =============================================================================
// Sem ids + use-case knobs are TEMPLATE params (compile-time, core-uniform — same as SenderPipe).
//   * DATA_READY_SEM_ID      — sender->receiver "data is ready" flag id (this core waits on it).
//   * CONSUMER_READY_SEM_ID  — receiver->sender "my dest is free" counter id (this core increments it
//                              on the sender remotely; the id supplies the shared L1 offset).
//   * DATA_READY_SIGNAL / PRE_HANDSHAKE — must match the SenderPipe's.
// The only runtime input is the sender's coords, passed to receive() (they vary per receiver in 2D and
// rotate per block when the sender role rotates — must be runtime).
template <
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag,
    bool PRE_HANDSHAKE = true>
class ReceiverPipe {
public:
    explicit ReceiverPipe(const Noc& noc) :
        noc_(noc), data_ready_(DATA_READY_SEM_ID), consumer_ready_(CONSUMER_READY_SEM_ID) {
        // Init the flag THIS side waits on. The Counter signal needs no reset/init (monotone).
        if constexpr (DATA_READY_SIGNAL == DataReadySignal::Flag) {
            data_ready_.set(INVALID);
        }
    }

    // receive(): [ack the sender], wait data-ready, clear the flag (clear-before-ack).
    // `sender_x`/`sender_y` are the SENDER core's NoC coords — the target of the receiver->sender
    // readiness ack. On return the block is in the receiver's dst L1, bit-exact (signal arrival =>
    // data arrival). What the caller does with the dst is its own business.
    void receive(uint32_t sender_x, uint32_t sender_y) {
        if constexpr (PRE_HANDSHAKE) {
            // tell the sender "my dest is free / I am ready" (remote atomic inc on its counter)
            consumer_ready_.up(noc_, sender_x, sender_y, 1);
        }
        if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
            data_ready_.wait_min(++round_);
        } else {
            data_ready_.wait(VALID);
            data_ready_.set(INVALID);  // clear this round's flag; next receive()'s ack follows
        }
    }

    // Wait the control signal. Symmetric with SenderPipe::send_signal().
    //   * Flag    — a plain doorbell: returns once the signal arrives, then clears it.
    //   * Counter — returns the monotone round number reached.
    uint32_t receive_signal() {
        if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
            data_ready_.wait_min(++round_);
            return round_;
        } else {
            data_ready_.wait(VALID);
            data_ready_.set(INVALID);
            return VALID;
        }
    }

private:
    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    uint32_t round_ = 0;  // monotone round counter for DataReadySignal::Counter
};

}  // namespace dataflow_kernel_lib
