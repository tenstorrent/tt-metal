// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/cpp/ttnn/kernel_lib/chain_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

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
// Regular single-rectangle receiver sets always use McastMode::Multicast. If at least one group in a
// family has an irregular receiver set, the host's IrregularReceiverSetMode selects either multiple
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
//   // sets using the host's default IrregularReceiverSetMode::MultipleMcast.
//   auto receiver = mcast.receiver(noc);
//   receiver.receive(round);
//
//   // If the host may opt into IrregularReceiverSetMode::ChainLink for irregular
//   // receiver sets, use the transport-compatible operation. It forwards on a
//   // chain and behaves like receive(round) for hardware multicast.
//   auto configurable_receiver = mcast.receiver(noc);
//   configurable_receiver.receive_and_forward(dst_l1, size_bytes, round);
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
    FORCE_INLINE void receive_and_forward(uint32_t, uint32_t, uint32_t = 0) {}
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
    static constexpr auto mcast_mode = mcast_wire::mcast_mode(flags);
    static_assert(
        mcast_mode == McastMode::Multicast || mcast_mode == McastMode::ChainUnicast, "Invalid family multicast mode");
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
        mcast_mode == McastMode::ChainUnicast ? rectangle_capacity == 0
                                              : rectangle_capacity >= 1 && rectangle_capacity <= MAX_MCAST_RECTANGLES,
        "Multicast needs one to three rectangles; chain unicast has no rectangle storage");
    static_assert(
        mcast_mode == McastMode::Multicast || (pre_handshake && !rotating),
        "Chain-unicast families require fixed senders and receiver handshakes");
    static constexpr uint32_t next_compile_time_args_offset() { return CT_BASE + mcast_wire::CT_WORDS; }
    static constexpr uint32_t next_runtime_args_offset() {
        return RT_BASE + mcast_wire::runtime_words(rotating_span, rectangle_capacity, mcast_mode);
    }

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

    auto sender(const Noc& noc) const;
    auto optional_sender(const Noc& noc) const;
    auto receiver(const Noc& noc) const;
    auto optional_receiver(const Noc& noc) const;

    uint32_t sender_x() const { return get_arg_val<uint32_t>(RT_BASE + mcast_wire::FIXED_SENDER_X); }
    uint32_t sender_y() const { return get_arg_val<uint32_t>(RT_BASE + mcast_wire::FIXED_SENDER_Y); }

private:
    // A declaration-only fallback capacity keeps this unused decoder well-formed for chain-only layouts.
    SenderRuntimeArgumentsFor<rectangle_capacity ? rectangle_capacity : 1> sender_runtime_arguments() const;
    ChainRuntimeArguments chain_runtime_arguments() const;
};

template <uint32_t CT_BASE, uint32_t RT_BASE>
struct McastArgsImpl<false, CT_BASE, RT_BASE> {
    constexpr McastArgsImpl() = default;
    static constexpr bool active = false;
    static constexpr uint32_t has_receivers = 0;
    static constexpr uint32_t num_senders = 0;
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

    std::optional<InactiveSenderPipe> optional_sender(const Noc&) const { return std::nullopt; }
    std::optional<InactiveReceiverPipe> optional_receiver(const Noc&) const { return std::nullopt; }
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

#include "ttnn/cpp/ttnn/kernel_lib/mcast_args.inl"
