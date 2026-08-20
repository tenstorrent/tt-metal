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
    mcast_wire::ArgumentMetadata METADATA,
    typename Runtime,
    typename DataReadyBinding,
    typename ConsumerReadyBinding,
    typename SignalSourceBinding>
bool McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::should_send(
    uint32_t round) const {
    if constexpr (!sender_available || !rotating) {
        return can_send();
    } else {
        return can_send() && sender_index(round) == Runtime::read(runtime_layout.sender_phase);
    }
}

template <
    mcast_wire::ArgumentMetadata METADATA,
    typename Runtime,
    typename DataReadyBinding,
    typename ConsumerReadyBinding,
    typename SignalSourceBinding>
typename McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::
    SenderPipeType
    McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::sender(
        const Noc& noc) const {
    static_assert(sender_available, "A sender pipe is unavailable on this placement");
    if constexpr (!sender_available) {
        return {};
    } else {
        ASSERT(can_send());
        static_assert(sender_noc == noc_index, "Host multicast NoC does not match sending kernel NoC");
        if constexpr (transfer_mode == TransferMode::Multicast) {
            return SenderPipeType(noc, sender_runtime_arguments());
        } else {
            return SenderPipeType(noc, chain_runtime_arguments());
        }
    }
}

template <
    mcast_wire::ArgumentMetadata METADATA,
    typename Runtime,
    typename DataReadyBinding,
    typename ConsumerReadyBinding,
    typename SignalSourceBinding>
std::optional<
    typename McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::
        SenderPipeType>
McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::optional_sender(
    const Noc& noc) const {
    if constexpr (!sender_available) {
        return std::nullopt;
    } else {
        if (!can_send()) {
            return std::nullopt;
        }
        return sender(noc);
    }
}

template <
    mcast_wire::ArgumentMetadata METADATA,
    typename Runtime,
    typename DataReadyBinding,
    typename ConsumerReadyBinding,
    typename SignalSourceBinding>
typename McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::
    ReceiverPipeType
    McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::receiver(
        const Noc& noc) const {
    static_assert(receiver_available, "A receiver pipe is unavailable on this placement");
    if constexpr (!receiver_available) {
        return {};
    } else {
        ASSERT(can_receive());
        if constexpr (transfer_mode == TransferMode::ChainUnicast) {
            static_assert(sender_noc == noc_index, "Host family NoC does not match forwarding receiver kernel NoC");
            return ReceiverPipeType(noc, chain_runtime_arguments());
        } else {
            return ReceiverPipeType(noc, Runtime::coordinates(runtime_layout.sender_coordinates));
        }
    }
}

template <
    mcast_wire::ArgumentMetadata METADATA,
    typename Runtime,
    typename DataReadyBinding,
    typename ConsumerReadyBinding,
    typename SignalSourceBinding>
std::optional<
    typename McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::
        ReceiverPipeType>
McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::optional_receiver(
    const Noc& noc) const {
    if constexpr (!receiver_available) {
        return std::nullopt;
    } else {
        if (!can_receive()) {
            return std::nullopt;
        }
        if constexpr (transfer_mode == TransferMode::ChainUnicast) {
            static_assert(sender_noc == noc_index, "Host family NoC does not match forwarding receiver kernel NoC");
            return std::optional<ReceiverPipeType>(std::in_place, noc, chain_runtime_arguments());
        } else {
            // Construct the owned coordinate table in the optional's final storage.
            return std::optional<ReceiverPipeType>(
                std::in_place, noc, Runtime::coordinates(runtime_layout.sender_coordinates));
        }
    }
}

template <
    mcast_wire::ArgumentMetadata METADATA,
    typename Runtime,
    typename DataReadyBinding,
    typename ConsumerReadyBinding,
    typename SignalSourceBinding>
SenderRuntimeArgumentsFor<
    McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::
            rectangle_capacity
        ? McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::
              rectangle_capacity
        : 1>
McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::
    sender_runtime_arguments() const {
    // Every group has destinations, so capacity one guarantees exactly one record.
    constexpr bool single_rectangle = rectangle_capacity == 1;
    SenderRuntimeArgumentsFor<rectangle_capacity ? rectangle_capacity : 1> prepared;
    prepared.num_rectangles = single_rectangle ? 1u : Runtime::read(runtime_layout.rectangle_count);
    if constexpr (pre_handshake) {
        prepared.ack_count = ack_count != ACK_EQUALS_FANOUT ? ack_count : Runtime::read(runtime_layout.ack);
    }
    ASSERT(prepared.num_rectangles >= 1 && prepared.num_rectangles <= rectangle_capacity);
    for (uint32_t i = 0; i < prepared.num_rectangles; ++i) {
        const uint32_t base = runtime_layout.rectangles + i * runtime_layout.rectangle_stride;
        auto& rectangle = prepared.rectangles[i];
        if constexpr (runtime_layout.rectangle_bounds != mcast_wire::OMITTED) {
            rectangle.bounds = {
                Runtime::read(base + runtime_layout.rectangle_bounds + mcast_wire::SX),
                Runtime::read(base + runtime_layout.rectangle_bounds + mcast_wire::SY),
                Runtime::read(base + runtime_layout.rectangle_bounds + mcast_wire::EX),
                Runtime::read(base + runtime_layout.rectangle_bounds + mcast_wire::EY)};
        }
        rectangle.remote_count = single_rectangle && METADATA.family.remote_count_known
                                     ? remote_count
                                     : Runtime::read(base + runtime_layout.rectangle_remote);
        rectangle.loopback_count = rectangle.remote_count + 1;
        rectangle.sender_mcast_mode =
            mcast_wire::concrete(sender_mcast_mode)
                ? sender_mcast_mode
                : static_cast<SenderMcastMode>(Runtime::read(base + runtime_layout.rectangle_mode));
    }
    return prepared;
}

template <
    mcast_wire::ArgumentMetadata METADATA,
    typename Runtime,
    typename DataReadyBinding,
    typename ConsumerReadyBinding,
    typename SignalSourceBinding>
ChainRuntimeArguments
McastArgsImpl<true, METADATA, Runtime, DataReadyBinding, ConsumerReadyBinding, SignalSourceBinding>::
    chain_runtime_arguments() const {
    constexpr uint32_t base = runtime_layout.chain_neighbors;
    return {
        Runtime::read(base + mcast_wire::PREDECESSOR_X),
        Runtime::read(base + mcast_wire::PREDECESSOR_Y),
        Runtime::read(base + mcast_wire::SUCCESSOR_X),
        Runtime::read(base + mcast_wire::SUCCESSOR_Y),
        Runtime::read(base + mcast_wire::INCLUDES_SENDER) != 0u};
}

}  // namespace dataflow_kernel_lib::detail
