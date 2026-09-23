// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/chain_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_pipe.hpp"

namespace dataflow_kernel_lib {

// =============================================================================
// McastArgs — the KERNEL counterpart of host::McastFamily (including Mcast1D / Mcast2D wrappers).
// =============================================================================
// Construct McastArgs with the starting offsets of the host helper's compile-time and runtime arguments.
// Sender kernels call sender(noc), receiver kernels call receiver(noc), and rotating receivers pass the
// absolute work round to their receive operation. The returned pipe resolves to hardware multicast or
// chain unicast from the family's wire arguments. A present sender must still call send() when
// `has_receivers` is false because that is the degenerate local-copy case; absence is represented by
// `active == false`.
//
// Regular single-rectangle receiver sets always use TransferMode::Multicast. If at least one group in a
// family has an irregular receiver set, the host's TransferMode selects either multiple
// hardware multicasts or chain links for the whole family. Hardware-only kernels call receive(round).
// Kernels which may use either transport call receive_and_forward(dst_l1, size_bytes, round): chain
// receivers relay the payload, while hardware multicast receivers ignore the destination and size.
//
// can_send() and can_receive() report this core's roles. sender_index(round) maps an absolute work
// round to a rotating phase, and should_send(round) reports whether that phase belongs to this core.
// Use next_compile_time_args_offset() and next_runtime_args_offset() to place the next argument decoder.
//
// Example kernel call sites:
//
//   constexpr auto mcast = McastArgs</*CT=*/next_ct_arg, /*RT=*/next_rt_arg>();
//   Noc noc;
//
//   // Sender side: identical for hardware multicast and chain unicast.
//   auto sender = mcast.sender(noc);
//   sender.send(src_l1, dst_l1, size_bytes);
//
//   // Most common receiver side: regular receiver sets and irregular receiver
//   // sets using the host's default TransferMode::Multicast.
//   auto receiver = mcast.receiver(noc);
//   receiver.receive(round);
//
//   // If the host may opt into TransferMode::ChainUnicast for irregular
//   // receiver sets, use the transport-compatible operation. It forwards on a
//   // chain and behaves like receive(round) for hardware multicast.
//   auto configurable_receiver = mcast.receiver(noc);
//   configurable_receiver.receive_and_forward(dst_l1, size_bytes, round);
namespace detail {

template <auto>
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
    template <SourceL1Guard = SourceL1Guard::Guard>
    FORCE_INLINE void receive_and_forward(uint32_t, uint32_t, uint32_t = 0) {}
    FORCE_INLINE uint32_t receive_signal(uint32_t = 0) { return 0; }
};

template <
    bool PRESENT,
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
struct McastArgsImpl;

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
struct McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE> {
    constexpr McastArgsImpl() = default;
    static constexpr bool active = true;

    // `has_receivers` indicates remote fan-out. Do not use it to suppress sender
    // work: a present zero-fan-out sender still calls send() to perform a degenerate local copy. The
    // per-core role metadata reports which pipe faces this kernel instance may construct and its phase.
    static constexpr uint32_t has_receivers = METADATA.has_remote_receivers;
    static constexpr auto data_ready = DATA_READY;
    static constexpr auto consumer_ready = CONSUMER_READY;
    static constexpr uint32_t ack_count = METADATA.ack_count;
    static constexpr uint32_t flags = METADATA.flags;
    static constexpr uint32_t rotating_span = METADATA.rotating_span;

    // Pipe behaviour lifted off the flags word (host-computed): the caller never spells these.
    static constexpr auto transfer_mode = mcast_wire::transfer_mode(flags);
    static constexpr auto signal_source = SIGNAL_SOURCE;
    static_assert(
        transfer_mode == TransferMode::Multicast || transfer_mode == TransferMode::ChainUnicast,
        "Invalid family multicast mode");
    static constexpr bool pre_handshake = (flags & mcast_wire::PRE_HANDSHAKE) != 0u;
    static constexpr DataReadySignal signal =
        (flags & mcast_wire::COUNTER_SIGNAL) != 0u ? DataReadySignal::Counter : DataReadySignal::Flag;
    static constexpr bool rotating = rotating_span > 0;

    // Sender coord pairs this family carries: 1 for a fixed sender, rotating_span otherwise.
    static constexpr uint32_t num_senders = rotating ? rotating_span : 1u;

    static constexpr SenderMcastMode sender_mcast_mode = METADATA.sender_mcast_mode;
    static_assert(
        mcast_wire::concrete(sender_mcast_mode) || sender_mcast_mode == SenderMcastMode::Unknown,
        "Invalid sender multicast mode");
    static constexpr uint32_t remote_count = METADATA.uniform_remote_count;
    static constexpr uint32_t loopback_count = METADATA.uniform_loopback_count;
    static constexpr uint8_t sender_noc = (flags & mcast_wire::NOC1) ? 1 : 0;

    static constexpr uint32_t rectangle_capacity = METADATA.rectangle_capacity;
    static_assert(
        transfer_mode == TransferMode::ChainUnicast
            ? rectangle_capacity == 0
            : rectangle_capacity >= 1 && rectangle_capacity <= MAX_MCAST_RECTANGLES,
        "Multicast needs one to three rectangles; chain unicast has no rectangle storage");
    static_assert(
        transfer_mode == TransferMode::Multicast || (pre_handshake && !rotating),
        "Chain-unicast families require fixed senders and receiver handshakes");
    // Use these role queries only when one kernel is dispatched across a heterogeneous set of cores
    // (sender-only, receiver-only, both, or neither) and must decide which pipe faces to construct.
    // When every dispatched core has a known role, construct that pipe face directly; sender() and
    // receiver() assert that the runtime role metadata permits it.
    bool can_send() const {
        return (Runtime::read(
                    mcast_wire::roles_offset(rotating_span, rectangle_capacity, transfer_mode) + mcast_wire::ROLES) &
                mcast_wire::CAN_SEND) != 0u;
    }
    bool can_receive() const {
        return (Runtime::read(
                    mcast_wire::roles_offset(rotating_span, rectangle_capacity, transfer_mode) + mcast_wire::ROLES) &
                mcast_wire::CAN_RECEIVE) != 0u;
    }
    static constexpr uint32_t sender_index(uint32_t round) { return round % num_senders; }
    bool should_send(uint32_t round) const;

    auto sender(const Noc& noc) const;
    auto optional_sender(const Noc& noc) const;
    auto receiver(const Noc& noc) const;
    auto optional_receiver(const Noc& noc) const;

    uint32_t sender_x() const {
        return Runtime::read(mcast_wire::sender_coords_offset(rotating_span) + mcast_wire::SENDER_X);
    }
    uint32_t sender_y() const {
        return Runtime::read(mcast_wire::sender_coords_offset(rotating_span) + mcast_wire::SENDER_Y);
    }

private:
    // A declaration-only fallback capacity keeps this unused decoder well-formed for chain-only layouts.
    SenderRuntimeArgumentsFor<rectangle_capacity ? rectangle_capacity : 1> sender_runtime_arguments() const;
    ChainRuntimeArguments chain_runtime_arguments() const;
};

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
struct McastArgsImpl<false, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE> {
    constexpr McastArgsImpl() = default;
    static constexpr bool active = false;
    static constexpr uint32_t has_receivers = 0;
    static constexpr uint32_t num_senders = 0;
    bool can_send() const { return false; }
    bool can_receive() const { return false; }
    bool should_send(uint32_t) const { return false; }

    void sender(const Noc&) const {
        static_assert(dependent_false<METADATA>, "No multicast on this core; a sender pipe cannot be built");
    }

    void receiver(const Noc&) const {
        static_assert(dependent_false<METADATA>, "No multicast on this core; a receiver pipe cannot be built");
    }

    std::optional<InactiveSenderPipe> optional_sender(const Noc&) const { return std::nullopt; }
    std::optional<InactiveReceiverPipe> optional_receiver(const Noc&) const { return std::nullopt; }
};

// Positional access stays isolated here; the spec frontend supplies its native vararg view.
template <uint32_t RT_BASE>
struct PositionalMcastRuntime {
    static uint32_t read(uint32_t offset) { return get_arg_val<uint32_t>(RT_BASE + offset); }
    static const uint32_t* coordinates(uint32_t offset) {
        return reinterpret_cast<const uint32_t*>(get_arg_addr(RT_BASE + offset));
    }
};

template <uint32_t CT_BASE>
constexpr mcast_wire::FamilyMetadata positional_mcast_metadata() {
    if constexpr (get_compile_time_arg_val(CT_BASE + mcast_wire::TAG) == mcast_wire::ABSENT) {
        return {};
    } else {
        return {
            get_compile_time_arg_val(CT_BASE + mcast_wire::ROTATING_SPAN),
            get_compile_time_arg_val(CT_BASE + mcast_wire::RECTANGLE_CAPACITY),
            get_compile_time_arg_val(CT_BASE + mcast_wire::ACK_COUNT),
            get_compile_time_arg_val(CT_BASE + mcast_wire::REMOTE_COUNT),
            get_compile_time_arg_val(CT_BASE + mcast_wire::LOOPBACK_COUNT),
            static_cast<SenderMcastMode>(get_compile_time_arg_val(CT_BASE + mcast_wire::SENDER_MCAST_MODE)),
            get_compile_time_arg_val(CT_BASE + mcast_wire::HAS_RECEIVERS) != 0u,
            get_compile_time_arg_val(CT_BASE + mcast_wire::FLAGS)};
    }
}

template <uint32_t CT_BASE, uint32_t ROLE>
constexpr uint32_t positional_mcast_semaphore() {
    if constexpr (get_compile_time_arg_val(CT_BASE + mcast_wire::TAG) == mcast_wire::ABSENT) {
        return UNUSED_SEM_ID;
    } else if constexpr (
        ROLE == mcast_wire::SIGNAL_SOURCE && mcast_wire::transfer_mode(get_compile_time_arg_val(
                                                 CT_BASE + mcast_wire::FLAGS)) != TransferMode::ChainUnicast) {
        return UNUSED_SEM_ID;
    } else {
        return get_compile_time_arg_val(CT_BASE + ROLE);
    }
}

}  // namespace detail

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgs : detail::McastArgsImpl<
                       (get_compile_time_arg_val(CT_BASE + mcast_wire::TAG) == mcast_wire::FAMILY),
                       detail::positional_mcast_metadata<CT_BASE>(),
                       detail::PositionalMcastRuntime<RT_BASE>,
                       detail::positional_mcast_semaphore<CT_BASE, mcast_wire::DATA_READY>(),
                       detail::positional_mcast_semaphore<CT_BASE, mcast_wire::CONSUMER_READY>(),
                       detail::positional_mcast_semaphore<CT_BASE, mcast_wire::SIGNAL_SOURCE>()> {
    static_assert(
        get_compile_time_arg_val(CT_BASE + mcast_wire::TAG) == mcast_wire::ABSENT ||
            get_compile_time_arg_val(CT_BASE + mcast_wire::TAG) == mcast_wire::FAMILY,
        "Unsupported multicast wire tag; rebuild host and kernels for the unified family format");
    static constexpr uint32_t next_compile_time_args_offset() {
        constexpr auto metadata = detail::positional_mcast_metadata<CT_BASE>();
        return CT_BASE + (get_compile_time_arg_val(CT_BASE + mcast_wire::TAG) == mcast_wire::ABSENT
                              ? mcast_wire::ABSENT_CT_WORDS
                              : mcast_wire::compile_time_words(mcast_wire::transfer_mode(metadata.flags)));
    }
    static constexpr uint32_t next_runtime_args_offset() {
        constexpr auto metadata = detail::positional_mcast_metadata<CT_BASE>();
        return RT_BASE + (get_compile_time_arg_val(CT_BASE + mcast_wire::TAG) == mcast_wire::ABSENT
                              ? 0
                              : mcast_wire::runtime_words(
                                    metadata.rotating_span,
                                    metadata.rectangle_capacity,
                                    mcast_wire::transfer_mode(metadata.flags)));
    }
};

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.inl"
