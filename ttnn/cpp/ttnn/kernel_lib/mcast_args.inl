// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file mcast_args.inl
 * @brief Out-of-line definitions for McastArgs.
 *
 * This file should only be included by mcast_args.hpp.
 */

namespace dataflow_kernel_lib::detail {

template <uint32_t CT_BASE, uint32_t RT_BASE>
bool McastArgsImpl<true, CT_BASE, RT_BASE>::should_send(uint32_t round) const {
    return can_send() &&
           sender_index(round) == get_arg_val<uint32_t>(next_runtime_args_offset() - mcast_wire::PHASE_FROM_END);
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
auto McastArgsImpl<true, CT_BASE, RT_BASE>::sender(const Noc& noc) const {
    ASSERT(can_send());
    static_assert(sender_noc == noc_index, "Host multicast NoC does not match sending kernel NoC");
    if constexpr (mcast_mode == McastMode::Multicast) {
        return dataflow_kernel_lib::SenderPipe<
            noc_index,
            data_ready,
            pre_handshake,
            consumer_ready,
            signal,
            rotating,
            transfer_mode,
            rectangle_capacity>(noc, sender_runtime_arguments());
    } else {
        return dataflow_kernel_lib::ChainSenderPipe<noc_index, data_ready, consumer_ready, signal>(
            noc, chain_runtime_arguments());
    }
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
auto McastArgsImpl<true, CT_BASE, RT_BASE>::optional_sender(const Noc& noc) const {
    if (!can_send()) {
        return std::optional<decltype(sender(noc))>{};
    }
    return std::optional<decltype(sender(noc))>{sender(noc)};
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
auto McastArgsImpl<true, CT_BASE, RT_BASE>::receiver(const Noc& noc) const {
    ASSERT(can_receive());
    if constexpr (mcast_mode != McastMode::Multicast) {
        // Relays issue payload writes, so they need the family NoC like a sender does.
        static_assert(sender_noc == noc_index, "Host family NoC does not match forwarding receiver kernel NoC");
    }
    if constexpr (mcast_mode == McastMode::ChainUnicast) {
        return dataflow_kernel_lib::ChainReceiverPipe<noc_index, data_ready, consumer_ready, signal>(
            noc, chain_runtime_arguments());
    } else {
        const uint32_t* coords =
            reinterpret_cast<const uint32_t*>(get_arg_addr(RT_BASE + mcast_wire::sender_coords_offset(rotating_span)));
        return dataflow_kernel_lib::ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders>(
            noc, coords);
    }
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
auto McastArgsImpl<true, CT_BASE, RT_BASE>::optional_receiver(const Noc& noc) const {
    if (!can_receive()) {
        return std::optional<decltype(receiver(noc))>{};
    }
    return std::optional<decltype(receiver(noc))>{receiver(noc)};
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
SenderRuntimeArgumentsFor<
    McastArgsImpl<true, CT_BASE, RT_BASE>::rectangle_capacity
        ? McastArgsImpl<true, CT_BASE, RT_BASE>::rectangle_capacity
        : 1>
McastArgsImpl<true, CT_BASE, RT_BASE>::sender_runtime_arguments() const {
    // Every group has destinations, so capacity one guarantees exactly one record.
    constexpr bool single_rectangle = rectangle_capacity == 1;
    SenderRuntimeArgumentsFor<rectangle_capacity ? rectangle_capacity : 1> prepared;
    prepared.num_rectangles = single_rectangle ? 1u : get_arg_val<uint32_t>(RT_BASE + mcast_wire::NUM_RECTANGLES);
    prepared.ack_count = ack_count != ACK_EQUALS_FANOUT ? ack_count : get_arg_val<uint32_t>(RT_BASE + mcast_wire::ACK);
    ASSERT(prepared.num_rectangles >= 1 && prepared.num_rectangles <= rectangle_capacity);
    for (uint32_t i = 0; i < prepared.num_rectangles; ++i) {
        const uint32_t base = RT_BASE + mcast_wire::rectangles_offset(rotating_span) + i * mcast_wire::RECT_WORDS;
        prepared.rectangles[i] = {
            {get_arg_val<uint32_t>(base + mcast_wire::SX),
             get_arg_val<uint32_t>(base + mcast_wire::SY),
             get_arg_val<uint32_t>(base + mcast_wire::EX),
             get_arg_val<uint32_t>(base + mcast_wire::EY)},
            // At capacity one, positive uniform group counts are also this rectangle's counts.
            single_rectangle && remote_count > 0 ? remote_count : get_arg_val<uint32_t>(base + mcast_wire::REMOTE),
            single_rectangle && loopback_count > 0 ? loopback_count
                                                   : get_arg_val<uint32_t>(base + mcast_wire::LOOPBACK),
            mcast_wire::concrete(transfer_mode)
                ? transfer_mode
                : static_cast<SenderTransferMode>(get_arg_val<uint32_t>(base + mcast_wire::MODE))};
    }
    return prepared;
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
ChainRuntimeArguments McastArgsImpl<true, CT_BASE, RT_BASE>::chain_runtime_arguments() const {
    constexpr uint32_t base = RT_BASE + mcast_wire::chain_offset(rotating_span, rectangle_capacity);
    return {
        get_arg_val<uint32_t>(base + mcast_wire::PREDECESSOR_X),
        get_arg_val<uint32_t>(base + mcast_wire::PREDECESSOR_Y),
        get_arg_val<uint32_t>(base + mcast_wire::SUCCESSOR_X),
        get_arg_val<uint32_t>(base + mcast_wire::SUCCESSOR_Y),
        get_arg_val<uint32_t>(base + mcast_wire::INCLUDES_SENDER) != 0u};
}

}  // namespace dataflow_kernel_lib::detail
