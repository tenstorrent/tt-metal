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
//   stage a source L1 region -> multicast a block to an exact receiver set ->
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
#define MCAST_PIPE_API_VERSION 17

#include <optional>
#include "ttnn/cpp/ttnn/kernel_lib/mcast_common.hpp"

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
//   * Guard: source L1 may be reused when send() or send_signal() returns.
//   * CallerManaged: the caller protects source L1 until a later NoC completion point.
enum class SourceL1Guard { Guard, CallerManaged };

// Indicates that no consumer-ready semaphore is configured.
static constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

// Uses the multicast fan-out as the consumer acknowledgment count.
static constexpr uint32_t ACK_EQUALS_FANOUT = 0xFFFFFFFFu;

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
    bool ROTATING_SENDER = false,
    SenderTransferMode TRANSFER_MODE = SenderTransferMode::TransferModeUnknown,
    uint32_t MAX_RECTS = 1>
class SenderPipe {
    static_assert(MAX_RECTS >= 1 && MAX_RECTS <= MAX_MCAST_RECTANGLES, "Multicast supports one to three rectangles");
    static_assert(
        mcast_wire::concrete(TRANSFER_MODE) || TRANSFER_MODE == SenderTransferMode::TransferModeUnknown,
        "SenderPipe requires a concrete transfer mode or TransferModeUnknown");
    static_assert(
        !PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID,
        "PRE_HANDSHAKE=true requires a real CONSUMER_READY_SEM_ID (the receiver->sender readiness ack). "
        "Pass it, or set PRE_HANDSHAKE=false for a fire-and-forget broadcast.");

public:
    using RuntimeArguments = SenderRuntimeArgumentsFor<MAX_RECTS>;

    // Capture prepared values once. Argument storage may be changed or destroyed after construction.
    explicit SenderPipe(const Noc& noc, const RuntimeArguments& runtime_args);

    // ===== DATA channel (a block + a ready signal) =====
    // send() handles receiver readiness when enabled, data multicast, ready signaling, and source L1 protection.
    // With SOURCE_GUARD=CallerManaged, the caller provides that protection, so send() may return before the NoC
    // finishes reading source L1.
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    // ===== CONTROL channel (a signal with no data block) =====
    // Handle receiver readiness when enabled, then broadcast a control signal.
    // Flag sends `value`; Counter records one event. Pairs with ReceiverPipe::receive_signal(round).
    // CallerManaged requires the caller to preserve the local Flag semaphore value until the NoC
    // has read it. Rotating Flag cleanup and Counter atomic completion remain protected in either policy.
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    void send_signal(uint32_t value = VALID);

private:
    template <SenderTransferMode MODE>
    FORCE_INLINE void send_rectangle_(
        const RectangleRuntimeArguments& rectangle, uint32_t src_l1, uint32_t dst_l1, uint32_t size);
    FORCE_INLINE void send_data_(
        const RectangleRuntimeArguments& rectangle,
        bool loopback,
        uint32_t src_l1,
        uint32_t dst_l1,
        uint32_t size,
        uint32_t mcast_dests);
    FORCE_INLINE void signal_ready_(
        const RectangleRuntimeArguments& rectangle, bool loopback, uint32_t mcast_dests, uint32_t value = VALID);
    template <SourceL1Guard SOURCE_GUARD>
    FORCE_INLINE void fence_(bool loopback);
    void local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    RuntimeArguments args_;
    // Sender membership selects the payload fence even when src == dst skips a local write.
    bool loopback_ = false;
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
// McastArgs — the KERNEL counterpart of host::McastFamily (including Mcast1D / Mcast2D wrappers).
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
    template <SourceL1Guard = SourceL1Guard::Guard>
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
    static constexpr uint32_t has_receivers = get_compile_time_arg_val(CT_BASE + mcast_wire::HAS_RECEIVERS);
    static constexpr uint32_t data_ready = get_compile_time_arg_val(CT_BASE + mcast_wire::DATA_READY);
    static constexpr uint32_t consumer_ready = get_compile_time_arg_val(CT_BASE + mcast_wire::CONSUMER_READY);
    static constexpr uint32_t ack_count = get_compile_time_arg_val(CT_BASE + mcast_wire::ACK_COUNT);
    static constexpr uint32_t flags = get_compile_time_arg_val(CT_BASE + mcast_wire::FLAGS);
    static constexpr uint32_t rotating_span = get_compile_time_arg_val(CT_BASE + mcast_wire::ROTATING_SPAN);

    // Pipe behaviour lifted off the flags word (host-computed): the caller never spells these.
    static constexpr bool pre_handshake = (flags & mcast_wire::PRE_HANDSHAKE) != 0u;
    static constexpr DataReadySignal signal =
        (flags & mcast_wire::COUNTER_SIGNAL) != 0u ? DataReadySignal::Counter : DataReadySignal::Flag;
    static constexpr bool rotating = rotating_span > 0;

    // Sender coord pairs this family carries: 1 for a fixed sender, rotating_span otherwise.
    static constexpr uint32_t num_senders = rotating ? rotating_span : 1u;

    static constexpr SenderTransferMode transfer_mode =
        static_cast<SenderTransferMode>(get_compile_time_arg_val(CT_BASE + mcast_wire::TRANSFER_MODE));
    static_assert(
        mcast_wire::concrete(transfer_mode) || transfer_mode == SenderTransferMode::TransferModeUnknown,
        "Invalid sender transfer mode");
    static constexpr uint32_t remote_count = get_compile_time_arg_val(CT_BASE + mcast_wire::REMOTE_COUNT);
    static constexpr uint32_t loopback_count = get_compile_time_arg_val(CT_BASE + mcast_wire::LOOPBACK_COUNT);
    static constexpr uint8_t sender_noc = (flags & mcast_wire::NOC1) ? 1 : 0;

    static constexpr uint32_t rectangle_capacity = get_compile_time_arg_val(CT_BASE + mcast_wire::RECTANGLE_CAPACITY);
    static_assert(
        rectangle_capacity >= 1 && rectangle_capacity <= MAX_MCAST_RECTANGLES,
        "Multicast supports one to three rectangles");
    using SenderPipe = dataflow_kernel_lib::SenderPipe<
        noc_index,
        data_ready,
        pre_handshake,
        consumer_ready,
        signal,
        rotating,
        transfer_mode,
        rectangle_capacity>;
    using ReceiverPipe =
        dataflow_kernel_lib::ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders>;

    static constexpr uint32_t next_compile_time_args_offset() { return CT_BASE + mcast_wire::CT_WORDS; }
    static constexpr uint32_t next_runtime_args_offset() {
        return RT_BASE + mcast_wire::runtime_words(rotating_span, rectangle_capacity);
    }

    // ---- pipe construction: NO behaviour knobs; everything comes from the wire ----
    // Use these role queries only when one kernel is dispatched across a heterogeneous set of cores
    // (sender-only, receiver-only, both, or neither) and must decide which pipe faces to construct.
    // When every dispatched core has a known role, construct that pipe face directly; sender() and
    // receiver() assert that the runtime role metadata permits it.
    bool can_send() const {
        return (get_arg_val<uint32_t>(next_runtime_args_offset() - mcast_wire::ROLE_FROM_END) & mcast_wire::CAN_SEND) !=
               0u;
    }
    bool can_receive() const {
        return (get_arg_val<uint32_t>(next_runtime_args_offset() - mcast_wire::ROLE_FROM_END) &
                mcast_wire::CAN_RECEIVE) != 0u;
    }
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
    uint32_t sender_x() const { return get_arg_val<uint32_t>(RT_BASE + mcast_wire::FIXED_SENDER_X); }
    uint32_t sender_y() const { return get_arg_val<uint32_t>(RT_BASE + mcast_wire::FIXED_SENDER_Y); }

private:
    typename SenderPipe::RuntimeArguments sender_runtime_arguments() const;
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
struct McastArgs : detail::McastArgsImpl<(get_compile_time_arg_val(CT_BASE) == mcast_wire::FAMILY), CT_BASE, RT_BASE> {
    static_assert(
        get_compile_time_arg_val(CT_BASE) == mcast_wire::ABSENT ||
            get_compile_time_arg_val(CT_BASE) == mcast_wire::FAMILY,
        "Unsupported multicast wire tag; rebuild host and kernels for the unified family format");
};

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl"
