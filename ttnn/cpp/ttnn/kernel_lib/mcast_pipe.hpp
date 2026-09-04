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
//   signal the receivers that the data is ready.
//
// Sender cores use `SenderPipe`; receiver cores use `ReceiverPipe`.
//
// Example kernel call sites:
//
//   constexpr auto mcast = McastArgs</*CT=*/next_ct_arg, /*RT=*/next_rt_arg>();
//   Noc noc;
//
//   // Sender side
//   auto sender = mcast.sender(noc);
//   for (...) {
//       sender.send(src_l1, dst_l1, size);
//   }
//
//   // Receiver side
//   auto receiver = mcast.receiver(noc);
//   for (...) {
//       receiver.receive(round);
//   }
//
// Preconditions: one active sender per round; semaphores are initialized to INVALID on every
// participating core; the landing address `dst_l1` is identical across all receivers.
// =============================================================================

#pragma once

// Caller-facing API version — the staleness key for the apply-dm-helper rollout ledger
// (helper_design/mcast_pipe/migration/ledger.json). BUMP THIS (and only this) whenever a
// re-materialization changes the caller-facing API (renamed/removed type, moved param, changed
// count/flag semantics — anything that forces a call site rewrite); leave it for internal-only
// changes.
#define MCAST_PIPE_API_VERSION 16

#include <optional>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "hostdevcommon/common_values.hpp"

namespace dataflow_kernel_lib {

// -----------------------------------------------------------------------------
// Data-ready signaling mode.
//   * Flag: a level signal reset between events.
//   * Counter: a monotonic event counter.
// -----------------------------------------------------------------------------
enum class DataReadySignal { Flag, Counter };

// Source L1 protection policy.
//   * Guard: source L1 may be reused when send() returns.
//   * CallerManaged: the caller protects source L1 until a later NoC completion point.
enum class SourceL1Guard { Guard, CallerManaged };

// Indicates that no consumer-ready semaphore is configured.
static constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

// Uses the multicast fan-out as the consumer acknowledgment count.
static constexpr uint32_t ACK_EQUALS_FANOUT = 0xFFFFFFFFu;

// -----------------------------------------------------------------------------
// A multicast destination rectangle, in NoC (virtual) coordinates. PURE GEOMETRY: the broadcast
// bounding box. Sender side only.
//
// Templated on the NoC id. The mcast hardware walks the rect from `start` in the NoC's routing
// direction up to `end`, so `start` must be the corner the routing reaches FIRST: the low corner on
// NoC0 (+x/+y), the high corner on NoC1 (-x/-y). Callers may pass the four corners in ANY order
// (canonical top-left→bottom-right or already swapped) — the normalization
// tolerates either, so the mcast APIs always receive the corners in routing order.
// PRECONDITION: `NOC_ID` must match the `Noc` the `SenderPipe` runs on.
// -----------------------------------------------------------------------------
template <uint8_t NOC_ID = noc_index>
struct McastRect {
    struct Bounds {
        uint32_t sx, sy, ex, ey;
    };

    constexpr McastRect(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) :
        xlo_(x0 < x1 ? x0 : x1),
        xhi_(x0 < x1 ? x1 : x0),
        ylo_(y0 < y1 ? y0 : y1),
        yhi_(y0 < y1 ? y1 : y0),
        start_end_(NOC_ID == 1 ? Bounds{xhi_, yhi_, xlo_, ylo_} : Bounds{xlo_, ylo_, xhi_, yhi_}) {}

    constexpr const Bounds& bounds() const { return start_end_; }

    // Normalized rectangle bounds.
    constexpr uint32_t xlo() const { return xlo_; }
    constexpr uint32_t xhi() const { return xhi_; }
    constexpr uint32_t ylo() const { return ylo_; }
    constexpr uint32_t yhi() const { return yhi_; }

    // Returns the number of worker cores in the rectangle, including a sender located within it.
    constexpr uint32_t area() const {
        uint32_t width = xhi_ - xlo_ + 1;
#if defined(ARCH_BLACKHOLE)
        // TODO: Clean this up before opening a ready PR.
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
//   * NOC_ID                     — compile-time NoC id; must match the `noc` argument.
//   * DATA_READY_SEM_ID          — sender-to-receiver data-ready semaphore id.
//   * PRE_HANDSHAKE              — wait for receiver readiness before sending data or a signal.
//   * CONSUMER_READY_SEM_ID      — receiver-to-sender readiness semaphore id; required with PRE_HANDSHAKE.
//   * DATA_READY_SIGNAL          — Flag (default) or Counter.
//   * ROTATING_SENDER            — whether this core sends on some rounds and receives on others.
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
    // `dest` is the receiver rectangle.
    // `consumer_ack_count` is the number of receiver acknowledgments to wait for; it defaults to the receiver count.
    explicit SenderPipe(const Noc& noc, const McastRect<NOC_ID>& dest, uint32_t consumer_ack_count = ACK_EQUALS_FANOUT);

    // ===== DATA channel (a block + a ready signal) =====
    // send() handles receiver readiness when enabled, data multicast, ready signaling, and source L1 protection.
    // With SOURCE_GUARD=CallerManaged, the caller provides that protection, so send() may return before the NoC
    // finishes reading source L1.
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    // ===== CONTROL channel (a signal with no data block) =====
    // Handle receiver readiness when enabled, then broadcast a control signal.
    // Flag sends `value`; Counter records one event. Pairs with ReceiverPipe::receive_signal(round).
    void send_signal(uint32_t value = VALID);

private:
    // ---- data multicast via the Noc object ----
    FORCE_INLINE void send_data_(uint32_t src_l1, uint32_t dst_l1, uint32_t size, bool loopback, uint32_t mcast_dests);

    // ---- signal the receivers the data is ready ----
    FORCE_INLINE void signal_ready_(bool loopback, uint32_t mcast_dests, uint32_t value = VALID);

    // ---- post-send fence ----
    template <SourceL1Guard SOURCE_GUARD>
    FORCE_INLINE void fence_(bool loopback);

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
// ReceiverPipe — the listening face of the channel.
// =============================================================================
//   * DATA_READY_SEM_ID      — sender-to-receiver data-ready semaphore id.
//   * PRE_HANDSHAKE          — signal receiver readiness before waiting; must match the SenderPipe's.
//   * CONSUMER_READY_SEM_ID  — receiver-to-sender readiness semaphore id; required with PRE_HANDSHAKE.
//   * DATA_READY_SIGNAL      — must match the SenderPipe's.
//   * NUM_SENDERS            — number of stored sender coordinate pairs.
//
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
    // `sender_coords` contains NUM_SENDERS virtual NoC coordinate pairs and must outlive the pipe.
    explicit ReceiverPipe(const Noc& noc, const uint32_t* sender_coords);

    // Handle receiver readiness, then wait for data from the sender selected by the absolute work round.
    void receive(uint32_t round = 0);

    // Handle receiver readiness when enabled, then wait for a control signal.
    // Returns the Flag value or round + 1 for Counter. Pairs with SenderPipe::send_signal().
    uint32_t receive_signal(uint32_t round = 0);

private:
    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    const uint32_t* coords_;  // non-owning sender coord pairs [x0,y0,...]; storage outlives this pipe
};

// =============================================================================
// McastArgs — the KERNEL counterpart of host::Mcast1D / host::Mcast2D.
// =============================================================================
// Construct McastArgs with the starting offsets of the host helper's compile-time and runtime arguments.
// Sender kernels call sender(noc), receiver kernels call receiver(noc), and rotating receivers pass the
// absolute work round to receive(round). A present sender must still call send() when `has_receivers`
// is false because that is the degenerate local-copy case; absence is represented by `active == false`.
//
// can_send() and can_receive() report this core's roles. sender_index(round) maps an absolute work
// round to a rotating phase, and should_send(round) reports whether that phase belongs to this core.
//
// Use next_compile_time_args_offset() and next_runtime_args_offset() to place the next argument
// decoder after this one.
namespace detail {

template <uint32_t>
static constexpr bool dependent_false = false;

// These sentinels only make the optional pipe surface well-formed for an absent
// tagged block. The inactive specialization always returns empty optionals, so
// neither sentinel represents a constructed multicast pipe.
struct InactiveSenderPipe {
    template <SourceL1Guard = SourceL1Guard::Guard>
    FORCE_INLINE void send(uint32_t, uint32_t, uint32_t) {}
    FORCE_INLINE void send_signal(uint32_t = VALID) {}
};

struct InactiveReceiverPipe {
    FORCE_INLINE void receive(uint32_t = 0) {}
    FORCE_INLINE uint32_t receive_signal(uint32_t = 0) { return 0; }
};

template <bool PRESENT, uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgsImpl;

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgsImpl<true, CT_BASE, RT_BASE> {
    constexpr McastArgsImpl() = default;
    static constexpr bool active = true;

    // `has_receivers` is the legacy wire name for remote fan-out. Do not use it to suppress sender
    // work: a present zero-fan-out sender still calls send() to perform a degenerate local copy. The
    // per-core role metadata reports which pipe faces this kernel instance may construct and its phase.
    static constexpr uint32_t has_receivers = get_compile_time_arg_val(CT_BASE + 1);
    static constexpr uint32_t data_ready = get_compile_time_arg_val(CT_BASE + 2);
    static constexpr uint32_t consumer_ready = get_compile_time_arg_val(CT_BASE + 3);
    static constexpr uint32_t ack_count = get_compile_time_arg_val(CT_BASE + 4);
    static constexpr uint32_t flags = get_compile_time_arg_val(CT_BASE + 5);
    static constexpr uint32_t rotating_span = get_compile_time_arg_val(CT_BASE + 6);

    // Pipe behaviour lifted off the flags word (host-computed): the caller never spells these.
    static constexpr bool pre_handshake = (flags & 0x1u) != 0u;
    static constexpr DataReadySignal signal =
        ((flags >> 1) & 0x1u) != 0u ? DataReadySignal::Counter : DataReadySignal::Flag;
    static constexpr bool rotating = rotating_span > 0;

    // Sender coord pairs this family carries: 1 for a fixed sender, rotating_span otherwise.
    static constexpr uint32_t num_senders = rotating ? rotating_span : 1u;

    // Concrete pipe types determined by this argument block.
    using SenderPipe =
        dataflow_kernel_lib::SenderPipe<noc_index, data_ready, pre_handshake, consumer_ready, signal, rotating>;
    using ReceiverPipe =
        dataflow_kernel_lib::ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders>;

    // TODO: Share these CT/RT argument layouts and counts with the host helpers. The topology payload
    // is followed by [role flags, sender phase], where a non-sender's phase is UINT32_MAX.
    // Offsets for chaining the next argument decoder.
    static constexpr uint32_t next_compile_time_args_offset() { return CT_BASE + 7; }
    static constexpr uint32_t next_runtime_args_offset() {
        return RT_BASE + (rotating ? (6u + 2u * rotating_span) : 6u);
    }

    // ---- pipe construction: NO behaviour knobs; everything comes from the wire ----
    // Use these role queries only when one kernel is dispatched across a heterogeneous set of cores
    // (sender-only, receiver-only, both, or neither) and must decide which pipe faces to construct.
    // When every dispatched core has a known role, construct that pipe face directly; sender() and
    // receiver() assert that the runtime role metadata permits it.
    bool can_send() const { return (get_arg_val<uint32_t>(next_runtime_args_offset() - 2u) & 0x1u) != 0u; }
    bool can_receive() const { return (get_arg_val<uint32_t>(next_runtime_args_offset() - 2u) & 0x2u) != 0u; }
    static constexpr uint32_t sender_index(uint32_t round) { return round % num_senders; }
    bool should_send(uint32_t round) const;

    // Construct the sender pipe on a sender-role core. Use optional_sender() when a shared kernel
    // binary may run on non-sender roles or receive an absent tagged helper block.
    SenderPipe sender(const Noc& noc) const;
    std::optional<SenderPipe> optional_sender(const Noc& noc) const;

    // Construct the receiver pipe. Pass the absolute work round to receive() or receive_signal().
    ReceiverPipe receiver(const Noc& noc) const;
    std::optional<ReceiverPipe> optional_receiver(const Noc& noc) const;

    // Receiver view, FIXED: the sender's coords (the target of this receiver's readiness ack).
    uint32_t sender_x() const { return get_arg_val<uint32_t>(RT_BASE + 0); }
    uint32_t sender_y() const { return get_arg_val<uint32_t>(RT_BASE + 1); }

private:
    McastRect<> rect() const;
};

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgsImpl<false, CT_BASE, RT_BASE> {
    constexpr McastArgsImpl() = default;
    static constexpr bool active = false;
    static constexpr uint32_t has_receivers = 0;
    static constexpr uint32_t num_senders = 0;
    using SenderPipe = InactiveSenderPipe;
    using ReceiverPipe = InactiveReceiverPipe;
    static constexpr uint32_t next_compile_time_args_offset() { return CT_BASE + 1; }
    static constexpr uint32_t next_runtime_args_offset() { return RT_BASE; }
    bool can_send() const { return false; }
    bool can_receive() const { return false; }
    bool should_send(uint32_t) const { return false; }

    void sender(const Noc&) const {
        static_assert(dependent_false<CT_BASE>, "No multicast on this core; a sender pipe cannot be built");
    }

    void receiver(const Noc&) const {
        static_assert(dependent_false<CT_BASE>, "No multicast on this core; a receiver pipe cannot be built");
    }

    std::optional<SenderPipe> optional_sender(const Noc&) const { return std::nullopt; }
    std::optional<ReceiverPipe> optional_receiver(const Noc&) const { return std::nullopt; }
};

}  // namespace detail

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgs : detail::McastArgsImpl<(get_compile_time_arg_val(CT_BASE) != 0), CT_BASE, RT_BASE> {};

}  // namespace dataflow_kernel_lib

#include "mcast_pipe.inl"
