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

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
bool McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::should_send(
    uint32_t round) const {
    return can_send() &&
           sender_index(round) == Runtime::read(
                                      mcast_wire::roles_offset(rotating_span, rectangle_capacity, transfer_mode) +
                                      mcast_wire::SENDER_ROUND);
}

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
auto McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::sender(const Noc& noc) const {
    ASSERT(can_send());
    static_assert(sender_noc == noc_index, "Host multicast NoC does not match sending kernel NoC");
    if constexpr (transfer_mode == TransferMode::Multicast) {
        return dataflow_kernel_lib::SenderPipe<
            noc_index,
            data_ready,
            pre_handshake,
            consumer_ready,
            signal,
            rotating,
            sender_mcast_mode,
            rectangle_capacity>(noc, sender_runtime_arguments());
    } else {
        return dataflow_kernel_lib::ChainSenderPipe<noc_index, data_ready, consumer_ready, signal_source, signal>(
            noc, chain_runtime_arguments());
    }
}

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
auto McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::optional_sender(
    const Noc& noc) const {
    if (!can_send()) {
        return std::optional<decltype(sender(noc))>{};
    }
    return std::optional<decltype(sender(noc))>{sender(noc)};
}

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
auto McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::receiver(const Noc& noc) const {
    ASSERT(can_receive());
    if constexpr (transfer_mode != TransferMode::Multicast) {
        // Relays issue payload writes, so they need the family NoC like a sender does.
        static_assert(sender_noc == noc_index, "Host family NoC does not match forwarding receiver kernel NoC");
    }
    if constexpr (transfer_mode == TransferMode::ChainUnicast) {
        return dataflow_kernel_lib::ChainReceiverPipe<noc_index, data_ready, consumer_ready, signal_source, signal>(
            noc, chain_runtime_arguments());
    } else {
        auto coords = Runtime::coordinates(mcast_wire::sender_coords_offset(rotating_span));
        return dataflow_kernel_lib::
            ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders, decltype(coords)>(noc, coords);
    }
}

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
auto McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::optional_receiver(
    const Noc& noc) const {
    if (!can_receive()) {
        return std::optional<decltype(receiver(noc))>{};
    }
    return std::optional<decltype(receiver(noc))>{receiver(noc)};
}

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
SenderRuntimeArgumentsFor<
    McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::rectangle_capacity
        ? McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::rectangle_capacity
        : 1>
McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::sender_runtime_arguments() const {
    // Every group has destinations, so capacity one guarantees exactly one record.
    constexpr bool single_rectangle = rectangle_capacity == 1;
    SenderRuntimeArgumentsFor<rectangle_capacity ? rectangle_capacity : 1> prepared;
    prepared.num_rectangles = single_rectangle ? 1u : Runtime::read(mcast_wire::NUM_RECTANGLES);
    prepared.ack_count = ack_count != ACK_EQUALS_FANOUT ? ack_count : Runtime::read(mcast_wire::ACK);
    ASSERT(prepared.num_rectangles >= 1 && prepared.num_rectangles <= rectangle_capacity);
    for (uint32_t i = 0; i < prepared.num_rectangles; ++i) {
        const uint32_t base = mcast_wire::rectangles_offset(rotating_span) + i * mcast_wire::RECT_WORDS;
        prepared.rectangles[i] = {
            {Runtime::read(base + mcast_wire::SX),
             Runtime::read(base + mcast_wire::SY),
             Runtime::read(base + mcast_wire::EX),
             Runtime::read(base + mcast_wire::EY)},
            // At capacity one, positive uniform group counts are also this rectangle's counts.
            single_rectangle && remote_count > 0 ? remote_count : Runtime::read(base + mcast_wire::REMOTE),
            single_rectangle && loopback_count > 0 ? loopback_count : Runtime::read(base + mcast_wire::LOOPBACK),
            mcast_wire::concrete(sender_mcast_mode)
                ? sender_mcast_mode
                : static_cast<SenderMcastMode>(Runtime::read(base + mcast_wire::RECT_SENDER_MCAST_MODE))};
    }
    return prepared;
}

template <
    mcast_wire::FamilyMetadata METADATA,
    typename Runtime,
    auto DATA_READY,
    auto CONSUMER_READY,
    auto SIGNAL_SOURCE>
ChainRuntimeArguments
McastArgsImpl<true, METADATA, Runtime, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE>::chain_runtime_arguments() const {
    constexpr uint32_t base = mcast_wire::chain_offset(rotating_span, rectangle_capacity);
    return {
        Runtime::read(base + mcast_wire::PREDECESSOR_X),
        Runtime::read(base + mcast_wire::PREDECESSOR_Y),
        Runtime::read(base + mcast_wire::SUCCESSOR_X),
        Runtime::read(base + mcast_wire::SUCCESSOR_Y),
        Runtime::read(base + mcast_wire::INCLUDES_SENDER) != 0u};
}

}  // namespace dataflow_kernel_lib::detail
