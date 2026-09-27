// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace dataflow_kernel_lib {
namespace detail {
template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE ChainLink<N, D, C, V, S>::ChainLink(const Noc& noc, const ChainRuntimeArguments& args) :
    noc(noc),
    data_ready(make_mcast_semaphore<D>()),
    consumer_ready(make_mcast_semaphore<C>()),
    signal_source(make_mcast_semaphore<V>()),
    args(args) {
    ASSERT(noc.get_noc_id() == N);
    ASSERT((args.predecessor_x == NO_CHAIN_NEIGHBOR) == (args.predecessor_y == NO_CHAIN_NEIGHBOR));
    ASSERT((args.successor_x == NO_CHAIN_NEIGHBOR) == (args.successor_y == NO_CHAIN_NEIGHBOR));
    // Never reset consumer_ready or signal_source: a successor may already have acked,
    // and a reconstructed Counter injector must retain its sequence.
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE void ChainLink<N, D, C, V, S>::wait_for_successor() {
    consumer_ready.wait(1);  // Per-hop readiness, independent of the group fanout or encoded ACK.
    consumer_ready.set(0);
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE void ChainLink<N, D, C, V, S>::write_successor(uint32_t src, uint32_t dst, uint32_t bytes) {
    noc.async_write(
        CoreLocalMem<uint32_t>(src),
        UnicastEndpoint{},
        bytes,
        {},
        {.noc_x = args.successor_x, .noc_y = args.successor_y, .addr = dst});
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE uint32_t ChainLink<N, D, C, V, S>::sender_signal_value(uint32_t value) const {
    if constexpr (S == DataReadySignal::Counter) {
        return signal_source.value() + 1u;
    } else {
        return value;
    }
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE void ChainLink<N, D, C, V, S>::publish(uint32_t value) {
    // wait_for_successor() proved the previous relay has read this source. Keep it stable until
    // the next successor ack; incoming data_ready and pipe destruction do not modify it.
    signal_source.set(value);
    signal_source.relay_unicast(noc, data_ready, args.successor_x, args.successor_y);
}
}  // namespace detail

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE ChainSenderPipe<N, D, C, V, S>::ChainSenderPipe(const Noc& noc, const RuntimeArguments& args) :
    link_(noc, args) {
    ASSERT(args.predecessor_x == NO_CHAIN_NEIGHBOR);
    ASSERT(link_.has_successor() || args.includes_sender);
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void ChainSenderPipe<N, D, C, V, S>::send(uint32_t src_l1, uint32_t dst_l1, uint32_t size_bytes) {
    const bool forward = link_.has_successor();
    const bool local_copy = link_.args.includes_sender && src_l1 != dst_l1;
    if (forward) {
        link_.wait_for_successor();
        link_.write_successor(src_l1, dst_l1, size_bytes);
        link_.publish(link_.sender_signal_value(VALID));
    }
    if (local_copy) {
        link_.noc.async_write(
            CoreLocalMem<uint32_t>(src_l1),
            UnicastEndpoint{},
            size_bytes,
            {},
            {.noc_x = my_x[N], .noc_y = my_y[N], .addr = dst_l1});
    }
    if (local_copy) {
        // The caller can publish its local destination immediately after return, under either guard.
        link_.noc.async_write_barrier();
    } else if constexpr (SOURCE_GUARD == SourceL1Guard::Guard) {
        if (forward) {
            link_.noc.async_writes_flushed();
        }
    }
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void ChainSenderPipe<N, D, C, V, S>::send_signal(uint32_t value) {
    ASSERT(value >= VALID);
    if constexpr (S == DataReadySignal::Counter) {
        ASSERT(value == VALID);
    }
    if (link_.has_successor()) {
        link_.wait_for_successor();
        link_.publish(link_.sender_signal_value(value));
        if constexpr (SOURCE_GUARD == SourceL1Guard::Guard) {
            link_.noc.async_writes_flushed();
        }
    }
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE ChainReceiverPipe<N, D, C, V, S>::ChainReceiverPipe(const Noc& noc, const RuntimeArguments& args) :
    link_(noc, args) {
    ASSERT(args.predecessor_x != NO_CHAIN_NEIGHBOR);
    if constexpr (S == DataReadySignal::Flag) {
        // Clear before the first predecessor acknowledgment permits an incoming signal write.
        link_.data_ready.set(INVALID);
    }
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE ChainReceiverPipe<N, D, C, V, S>::~ChainReceiverPipe() {
    link_.noc.async_atomic_barrier();
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
template <bool PAYLOAD>
FORCE_INLINE uint32_t ChainReceiverPipe<N, D, C, V, S>::consume_(uint32_t round) {
    link_.consumer_ready.up(link_.noc, link_.args.predecessor_x, link_.args.predecessor_y, 1);
    uint32_t value = VALID;
    if constexpr (S == DataReadySignal::Counter) {
        link_.data_ready.wait_min(round + 1);
        value = round + 1;
    } else {
        if constexpr (PAYLOAD) {
            link_.data_ready.wait(VALID);
        } else {
            link_.data_ready.wait_min(VALID);
            value = link_.data_ready.value();
        }
        link_.data_ready.set(INVALID);
    }
    return value;
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void ChainReceiverPipe<N, D, C, V, S>::receive_and_forward(
    uint32_t dst_l1, uint32_t size_bytes, uint32_t round) {
    const uint32_t value = consume_<true>(round);
    if (link_.has_successor()) {
        link_.wait_for_successor();
        link_.write_successor(dst_l1, dst_l1, size_bytes);
        link_.publish(value);
        if constexpr (SOURCE_GUARD == SourceL1Guard::Guard) {
            link_.noc.async_writes_flushed();
        }
    }
}

template <uint8_t N, auto D, auto C, auto V, DataReadySignal S>
FORCE_INLINE uint32_t ChainReceiverPipe<N, D, C, V, S>::receive_signal(uint32_t round) {
    const uint32_t value = consume_<false>(round);
    if (link_.has_successor()) {
        link_.wait_for_successor();
        link_.publish(value);
        link_.noc.async_writes_flushed();
    }
    return value;
}
}  // namespace dataflow_kernel_lib
