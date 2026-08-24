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
#define MCAST_PIPE_API_VERSION 15

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
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

// Whether send() must make the payload source safe to reuse before returning.
//   * Guard (default): send() waits until the linked data+signal writes have departed, so the caller
//     may immediately overwrite/reuse src_l1.
//   * CallerManaged: skip that remote-only SENT fence. The caller guarantees src_l1 remains unchanged
//     until a later NoC completion point (for example, an async write barrier at the CB reuse boundary).
// This policy never weakens completion required for a real sender loopback, a rotating Flag cell reset,
// or a Counter signal's multicast atomic acknowledgements.
enum class SourceL1Guard { Guard, CallerManaged };

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

    constexpr McastRect() : McastRect(0, 0, 0, 0) {}

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
    constexpr uint32_t area() const {
        uint32_t width = xhi_ - xlo_ + 1;
#if defined(ARCH_BLACKHOLE)
        // Blackhole virtual worker coordinates skip NoC columns 8 and 9. They are occupied by
        // non-worker cores and do not contribute multicast acknowledgements.
        width -= static_cast<uint32_t>(xlo_ <= 8 && 8 <= xhi_);
        width -= static_cast<uint32_t>(xlo_ <= 9 && 9 <= xhi_);
#endif
        return width * (yhi_ - ylo_ + 1);
    }

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
    bool ROTATING_SENDER = false,
    uint32_t MAX_RECTS = 1>
class SenderPipe {
    static_assert(
        !PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID,
        "PRE_HANDSHAKE=true requires a real CONSUMER_READY_SEM_ID (the receiver->sender readiness ack). "
        "Pass it, or set PRE_HANDSHAKE=false for a fire-and-forget broadcast.");
    static_assert(MAX_RECTS >= 1, "SenderPipe needs storage for at least one multicast rectangle.");

public:
    // `dest` — receiver rectangle (geometry only); its area gives the runtime mcast fan-out.
    // `consumer_ack_count` — the handshake wait count (used only under PRE_HANDSHAKE). Defaults to
    // ACK_EQUALS_FANOUT, meaning "= the EXCLUDE fan-out the rect derives" (the dense case). Both are
    // runtime ctor args; everything they feed is precomputed ONCE here so send() does no arithmetic.
    explicit SenderPipe(const Noc& noc, const McastRect<NOC_ID>& dest, uint32_t consumer_ack_count = ACK_EQUALS_FANOUT);

    // Multi-rectangle form used by McastFamily. `dest_coords` is MAX_RECTS packed
    // [x0,y0,x1,y1] records in stable runtime-argument storage; only the first `num_rects` are live.
    explicit SenderPipe(const Noc& noc, const uint32_t* dest_coords, uint32_t num_rects, uint32_t consumer_ack_count);

    // ===== DATA channel (a block + a ready signal) =====
    // By default send() is atomic and absorbs ALL FOUR guards:
    //   [if PRE_HANDSHAKE] wait(consumer_ready)  — gate the mcast on receivers having drained
    //   mcast data                                — object API auto-chunks a ready block > burst
    //   signal ready                              — data-before-signal, same VC; reset owned by receiver
    //   fence                                     — loopback ACKED, otherwise SENT; atomic-barrier on Counter
    // SOURCE_GUARD=CallerManaged skips only the remote-only SENT source-lifetime fence. The caller
    // must keep src_l1 unchanged until a later NoC completion point. Loopback, rotating-Flag, and
    // Counter correctness fences remain owned by the pipe.
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    // ===== CONTROL channel (a pure ready signal, no data block) =====
    // Broadcast a readiness signal (a doorbell, optionally carrying a small non-zero Flag value).
    // PRE_HANDSHAKE waits for and resets the configured consumer acknowledgements first, exactly as
    // send() does. The signal itself is always EXCLUDE-source because no data accompanies it. Typed
    // values are a Flag-only capability; Counter remains a monotone +1 event channel and requires
    // the default VALID argument. Pairs with ReceiverPipe::receive_signal().
    void send_signal(uint32_t value = VALID);

    // Whether this core belongs to the fixed receiver rectangle. Rotating protocols whose ordered
    // sender set extends beyond that rectangle use this to keep outside senders sender-only on the
    // other rounds, without duplicating the rectangle-containment calculation at the call site.
private:
    // ---- data multicast via the Noc object ----
    FORCE_INLINE void send_data_(
        uint32_t rect_index, uint32_t src_l1, uint32_t dst_l1, uint32_t size, bool loopback, uint32_t mcast_dests);

    // ---- signal the receivers the data is ready ----
    // `loopback` matches the data mcast of the same send(): when send() included the sender's own core
    // as a receiver, the signal must reach it too. send_signal() carries no data, so it never loops back.
    FORCE_INLINE void signal_ready_(uint32_t rect_index, bool loopback, uint32_t mcast_dests, uint32_t value = VALID);

    // ---- post-send fence ----
    template <SourceL1Guard SOURCE_GUARD>
    FORCE_INLINE void fence_(bool loopback);

    // ---- local L1 self-copy (degenerate self-only guard) via the Noc object ----
    void local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    void initialize_geometry_(uint32_t consumer_ack_count);

    Noc noc_;
    McastRect<NOC_ID> dests_[MAX_RECTS];
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    bool in_rect_[MAX_RECTS];
    uint32_t num_dests_excl_[MAX_RECTS];
    uint32_t num_dests_incl_[MAX_RECTS];
    uint32_t num_rects_ = 0;
    uint32_t total_num_dests_excl_ = 0;
    bool degenerate_ = true;
    uint32_t ack_count_;
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
// The sender coords are handed to the CONSTRUCTOR as a non-owning pointer and KEPT as a view in the
// object. Their storage MUST outlive the pipe. McastArgs::receiver() points directly into the stable
// kernel runtime-argument block; a by-hand caller keeps its own coord array alive while using the pipe.
// receive(round) then acks/listens to the round-th sender (round defaults to 0 — the only entry for a
// fixed receiver).
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
    // receiver acks/listens to (virtual NoC coords). Retained as a NON-OWNING view: storage must
    // outlive the pipe. McastArgs::receiver() supplies a pointer into the stable RT-argument block;
    // a by-hand caller supplies storage whose scope covers every pipe use.
    explicit ReceiverPipe(const Noc& noc, const uint32_t* sender_coords);

    // Handle receiver readiness when enabled, then wait for data from the sender selected by `round`.
    // Sender selection repeats every NUM_SENDERS rounds, so callers pass the absolute work round.
    void receive(uint32_t dst_l1, uint32_t size_bytes, uint32_t round = 0);

    // Wait for the next control signal. The pipe advances its internal absolute round and wraps sender
    // selection every NUM_SENDERS events. Returns the Flag value or consumed Counter event count.
    uint32_t receive_signal();

private:
    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    const uint32_t* coords_;  // non-owning sender coord pairs [x0,y0,...]; storage outlives this pipe
    uint32_t round_ = 0;      // monotone round counter for DataReadySignal::Counter
};

// =============================================================================
// McastArgs — the KERNEL counterpart of host::Mcast1D / host::Mcast2D.
// =============================================================================
// The host (ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp) emits the mcast wire; McastArgs self-parses
// it the nice way — like TensorAccessorArgs<BASE> — so the call site stops hand-indexing CT slots,
// stops fishing coords out of RT, and stops re-spelling the SenderPipe/ReceiverPipe template.
// Use can_send()/can_receive() to select the permitted pipe face. Call should_send(round) to select
// the sender for an absolute work round, and pass that same round to rotating receivers. Guard sender
// work with `has_receivers` when a family may have no receivers.
//
// ONE decoder owns BOTH arg lists: it is templated on the CT base AND the RT base. Row vs column vs
// single-rect is a pure HOST concern (which cores send, what rect); the decoder is shape-agnostic, so
// the SAME McastArgs serves a row-family, a column-family, and a single-sender->rect.
//
//   CT block, compact present (tag 1, 7 words):
//                        [ 1, has_receivers, data_ready_sem_id, consumer_ready_sem_id, ack_count,
//                          flags, rotating_span ]
//   CT block, extended family (tag 2, 7 words):
//                        [ 2, has_receivers, data_ready_sem_id, consumer_ready_sem_id, flags,
//                          rotating_span, max_rectangles ]
//   CT block, absent (1 word): [ present = 0 ]
//        flags bit0 = pre_handshake (gate on the receiver->sender readiness ack)
//        flags bit1 = data-ready signal (0 = Flag, 1 = Counter)
//        rotating_span = 0 for fixed sender; sender count for rotating mode
//   RT block, fixed (rotating_span == 0), 6 words:
//        sender   -> dest rect corners (virtual, NOC-ordered)
//        receiver -> [ sender_x, sender_y, 0, 0 ]
//        followed by [role flags, sender phase]
//   RT block, rotating (rotating_span > 0), 6 + 2*rotating_span words:
//        every core -> [ full-line rect corners, s0_x, s0_y, ... ]
//        followed by [role flags, sender phase]
//   RT block, extended family:
//        [4*max_rectangles packed rectangle words, 2*num_senders sender-coordinate words,
//         4 transport-topology words, role flags, sender phase, rectangle count, ack count, transport]
//        The host pads each group's rectangle list to the family maximum, so every core keeps one
//        compile-time-decoded boundary while its live rectangle count and handshake count remain
//        per-group runtime values.
//
// The pipe *behaviour* (pre_handshake, data-ready signal, dense/divergent ack, rotating) is NOT a
// call-site knob any more — the host computes each and rides it on the wire, and McastArgs feeds them
// into the pipe template. So `sender(noc)` / `receiver(noc)` take nothing but the Noc, and McastArgs is
// the ONLY place that touches runtime args: sender() reads the dest rect off RT and hands it to a
// SenderPipe (which keeps it); receiver() hands a stable view of the sender coord(s) in RT directly to
// ReceiverPipe. Neither pipe fetches args by index in its hot methods. The host-emitted rotating_span
// is the single source of truth for fixed/rotating mode, receiver type, and RT block size.

// The one mcast-args decoder. Chainable in BOTH arg lists, exactly like TensorAccessorArgs:
//   McastArgs<a.next_compile_time_args_offset(), a.next_runtime_args_offset()> picks up right after a
//   previous family `a` in CT and RT alike — no hand-indexed slots. Its first CT word is a presence
//   tag. A false tag consumes no payload CT/RT words; tag 1 selects the compact ordinary wire and tag
//   2 selects the extended family wire. A zero rotating_span selects fixed-sender behavior; a non-zero
//   value selects rotating-sender behavior. The CT reads are constexpr
//   (valid non-type template args); the RT reads are lazy (get_arg_val at access), so the object holds
//   only its template bases and stays constexpr-constructible for the chain.
namespace detail {

template <uint32_t>
static constexpr bool dependent_false = false;

template <bool PRESENT, uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgsImpl;

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgsImpl<true, CT_BASE, RT_BASE> {
    constexpr McastArgsImpl() = default;
    static constexpr bool active = true;
    static constexpr uint32_t wire_tag = get_compile_time_arg_val(CT_BASE);
    static constexpr bool extended = wire_tag == 2u;
    static_assert(wire_tag == 1u || wire_tag == 2u, "McastArgs: unknown present wire tag");

    // Use `has_receivers` to guard sender work. The per-core role metadata reports which pipe faces
    // this kernel instance may construct and its phase within the repeating sender rotation.
    static constexpr uint32_t has_receivers = get_compile_time_arg_val(CT_BASE + 1);
    static constexpr uint32_t data_ready = get_compile_time_arg_val(CT_BASE + 2);
    static constexpr uint32_t consumer_ready = get_compile_time_arg_val(CT_BASE + 3);
    static constexpr uint32_t legacy_ack_count = extended ? 0u : get_compile_time_arg_val(CT_BASE + 4);
    static constexpr uint32_t flags = get_compile_time_arg_val(CT_BASE + (extended ? 4u : 5u));
    static constexpr uint32_t rotating_span = get_compile_time_arg_val(CT_BASE + (extended ? 5u : 6u));
    static constexpr uint32_t max_rectangles = extended ? get_compile_time_arg_val(CT_BASE + 6u) : 1u;

    // Pipe behaviour lifted off the flags word (host-computed): the caller never spells these.
    static constexpr bool pre_handshake = (flags & 0x1u) != 0u;
    static constexpr DataReadySignal signal =
        ((flags >> 1) & 0x1u) != 0u ? DataReadySignal::Counter : DataReadySignal::Flag;
    static constexpr bool rotating = rotating_span > 0;

    // Sender coord pairs this family carries: 1 for a fixed sender, rotating_span otherwise.
    static constexpr uint32_t num_senders = rotating ? rotating_span : 1u;

    // Concrete pipe vocabulary for mixed-role kernels that must keep one or both faces in optional
    // storage. Callers should not need an unevaluated sender()/receiver() call merely to recover the
    // type that this argument block already determines.
    template <uint8_t NOC_ID>
    using SenderPipeFor = dataflow_kernel_lib::
        SenderPipe<NOC_ID, data_ready, pre_handshake, consumer_ready, signal, rotating, max_rectangles>;
    using SenderPipe = SenderPipeFor<noc_index>;
    using ReceiverPipe =
        dataflow_kernel_lib::ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders>;

    // TODO: Share these CT/RT argument layouts and counts with the host helpers. The topology payload
    // is followed by [role flags, sender phase], where a non-sender's phase is UINT32_MAX.
    // Offsets for chaining the next argument decoder.
    static constexpr uint32_t next_compile_time_args_offset() { return CT_BASE + 7; }
    static constexpr uint32_t sender_coord_runtime_args() { return rotating ? rotating_span : 1u; }
    static constexpr uint32_t sender_coords_runtime_offset() {
        return extended ? 4u * max_rectangles : (rotating ? 4u : 0u);
    }
    static constexpr uint32_t topology_runtime_args() {
        if constexpr (extended) {
            return 4u * max_rectangles + 2u * sender_coord_runtime_args() + 4u;
        }
        return rotating ? (4u + 2u * rotating_span) : 4u;
    }
    static constexpr uint32_t num_runtime_args() { return topology_runtime_args() + (extended ? 5u : 2u); }
    static constexpr uint32_t next_runtime_args_offset() { return RT_BASE + num_runtime_args(); }

    // ---- pipe construction: NO behaviour knobs; everything comes from the wire ----
    // Use these role queries only when one kernel is dispatched across a heterogeneous set of cores
    // (sender-only, receiver-only, both, or neither) and must decide which pipe faces to construct.
    // When every dispatched core has a known role, construct that pipe face directly; sender() and
    // receiver() assert that the runtime role metadata permits it.
    bool can_send() const { return (get_arg_val<uint32_t>(RT_BASE + topology_runtime_args()) & 0x1u) != 0u; }
    bool can_receive() const { return (get_arg_val<uint32_t>(RT_BASE + topology_runtime_args()) & 0x2u) != 0u; }
    // This is a recurring sender phase in [0, num_senders), not an absolute work round.
    uint32_t sender_round() const { return get_arg_val<uint32_t>(RT_BASE + topology_runtime_args() + 1u); }
    uint32_t rectangle_count() const {
        if constexpr (extended) {
            return get_arg_val<uint32_t>(RT_BASE + topology_runtime_args() + 2u);
        }
        return 1u;
    }
    uint32_t ack_count() const {
        if constexpr (extended) {
            return get_arg_val<uint32_t>(RT_BASE + topology_runtime_args() + 3u);
        }
        return legacy_ack_count;
    }
    uint32_t transport() const {
        if constexpr (extended) {
            return get_arg_val<uint32_t>(RT_BASE + topology_runtime_args() + 4u);
        }
        return 0u;
    }
    static constexpr uint32_t sender_index(uint32_t round) { return round % num_senders; }
    bool should_send(uint32_t round) const { return can_send() && sender_index(round) == sender_round(); }

    // Construct the sender pipe. Guard send() with `has_receivers` when the family may be inactive.
    template <uint8_t NOC_ID = noc_index>
    SenderPipeFor<NOC_ID> sender(const Noc& noc) const {
        ASSERT(can_send());
        if constexpr (extended) {
            const uint32_t* rectangles = reinterpret_cast<const uint32_t*>(get_arg_addr(RT_BASE));
            return SenderPipeFor<NOC_ID>(noc, rectangles, rectangle_count(), ack_count());
        }
        return SenderPipeFor<NOC_ID>(noc, rect<NOC_ID>(), legacy_ack_count);
    }

    // receiver(): hand ReceiverPipe a non-owning view directly into the stable RT block. FIXED: the
    // one pair starts at RT_BASE+0. ROTATING: rotating_span pairs, one per round, start past the rect
    // at RT_BASE+4. Pass the absolute work round to receive(round); sender selection wraps every
    // num_senders rounds.
    ReceiverPipe receiver(const Noc& noc) const {
        ASSERT(can_receive());
        const uint32_t* coords =
            reinterpret_cast<const uint32_t*>(get_arg_addr(RT_BASE + sender_coords_runtime_offset()));
        return ReceiverPipe(noc, coords);
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
    uint32_t sender_x() const { return get_arg_val<uint32_t>(RT_BASE + sender_coords_runtime_offset() + 0u); }
    uint32_t sender_y() const { return get_arg_val<uint32_t>(RT_BASE + sender_coords_runtime_offset() + 1u); }
    // Receiver view, ROTATING: the sender broadcasting on `round`, round in [0, rotating_span).
    uint32_t sender_x(uint32_t round) const {
        return get_arg_val<uint32_t>(RT_BASE + sender_coords_runtime_offset() + 2u * round + 0u);
    }
    uint32_t sender_y(uint32_t round) const {
        return get_arg_val<uint32_t>(RT_BASE + sender_coords_runtime_offset() + 2u * round + 1u);
    }
};

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgsImpl<false, CT_BASE, RT_BASE> {
    constexpr McastArgsImpl() = default;
    static constexpr bool active = false;
    static constexpr uint32_t has_receivers = 0;
    static constexpr uint32_t next_compile_time_args_offset() { return CT_BASE + 1; }
    static constexpr uint32_t topology_runtime_args() { return 0; }
    static constexpr uint32_t num_runtime_args() { return 0; }
    static constexpr uint32_t next_runtime_args_offset() { return RT_BASE; }
    bool can_send() const { return false; }
    bool can_receive() const { return false; }
    bool should_send(uint32_t) const { return false; }

    template <uint8_t NOC_ID = noc_index>
    void sender(const Noc&) const {
        static_assert(dependent_false<NOC_ID>, "McastArgs::sender() cannot be used when the presence tag is false");
    }

    template <bool PRESENT = active>
    void receiver(const Noc&) const {
        static_assert(PRESENT, "McastArgs::receiver() cannot be used when the presence tag is false");
    }
};

}  // namespace detail

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgs : detail::McastArgsImpl<(get_compile_time_arg_val(CT_BASE) != 0), CT_BASE, RT_BASE> {};

}  // namespace dataflow_kernel_lib

#include "mcast_pipe.inl"
