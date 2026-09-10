// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace dataflow_kernel_lib {
namespace detail {
template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
ChainLink<N, D, C, S>::ChainLink(const Noc& noc, const ChainRuntimeArguments& args) :
    noc(noc), data_ready(D), consumer_ready(C), args(args) {
    ASSERT(noc.get_noc_id() == N);
    ASSERT((args.predecessor_x == NO_CHAIN_NEIGHBOR) == (args.predecessor_y == NO_CHAIN_NEIGHBOR));
    ASSERT((args.successor_x == NO_CHAIN_NEIGHBOR) == (args.successor_y == NO_CHAIN_NEIGHBOR));
    // Never reset consumer_ready: a successor may already have acknowledged readiness.
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
void ChainLink<N, D, C, S>::wait_for_successor() {
    consumer_ready.wait(1);  // Per-hop readiness, independent of the group fanout or encoded ACK.
    consumer_ready.set(0);
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
void ChainLink<N, D, C, S>::write_successor(uint32_t src, uint32_t dst, uint32_t bytes) {
    noc.async_write(
        CoreLocalMem<uint32_t>(src),
        UnicastEndpoint{},
        bytes,
        {},
        {.noc_x = args.successor_x, .noc_y = args.successor_y, .addr = dst});
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
void ChainLink<N, D, C, S>::publish(uint32_t value) {
    // Counter relays forward round + 1 from receive_signal; the cell only ever advances by one.
    data_ready.up(noc, args.successor_x, args.successor_y, S == DataReadySignal::Counter ? 1u : value);
    // Conservative: the protocol does not need this wait (the successor cannot ack before the
    // increment lands), but it keeps every hop's NoC accounting drained per round.
    noc.async_atomic_barrier();
}
}  // namespace detail

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
ChainSenderPipe<N, D, C, S>::ChainSenderPipe(const Noc& noc, const RuntimeArguments& args) : link_(noc, args) {
    ASSERT(args.predecessor_x == NO_CHAIN_NEIGHBOR);
    ASSERT(link_.has_successor() || args.includes_sender);
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void ChainSenderPipe<N, D, C, S>::send(uint32_t src_l1, uint32_t dst_l1, uint32_t size_bytes) {
    const bool forward = link_.has_successor();
    const bool local_copy = link_.args.includes_sender && src_l1 != dst_l1;
    if (forward) {
        link_.wait_for_successor();
        link_.write_successor(src_l1, dst_l1, size_bytes);
    }
    if (local_copy) {
        link_.noc.async_write(
            CoreLocalMem<uint32_t>(src_l1),
            UnicastEndpoint{},
            size_bytes,
            {},
            {.noc_x = my_x[N], .noc_y = my_y[N], .addr = dst_l1});
    }
    // Atomic readiness uses another path: complete both payload and local copy before publishing.
    // This ordering barrier also protects source L1, even with CallerManaged.
    if (forward || local_copy) {
        link_.noc.async_write_barrier();
    }
    if (forward) {
        link_.publish(VALID);
    }
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void ChainSenderPipe<N, D, C, S>::send_signal(uint32_t value) {
    ASSERT(value >= VALID);
    if constexpr (S == DataReadySignal::Counter) {
        ASSERT(value == VALID);
    }
    if (link_.has_successor()) {
        link_.wait_for_successor();
        link_.publish(value);
    }
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
ChainReceiverPipe<N, D, C, S>::ChainReceiverPipe(const Noc& noc, const RuntimeArguments& args) : link_(noc, args) {
    ASSERT(args.predecessor_x != NO_CHAIN_NEIGHBOR);
    if constexpr (S == DataReadySignal::Flag) {
        // The first predecessor acknowledgement orders this clear before the first atomic signal.
        link_.data_ready.set(INVALID);
    }
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
template <bool PAYLOAD>
uint32_t ChainReceiverPipe<N, D, C, S>::consume_(uint32_t round) {
    link_.consumer_ready.up(link_.noc, link_.args.predecessor_x, link_.args.predecessor_y, 1);
    if constexpr (S == DataReadySignal::Counter) {
        link_.data_ready.wait_min(round + 1);
        return round + 1;
    } else {
        uint32_t value = VALID;
        if constexpr (PAYLOAD) {
            link_.data_ready.wait(VALID);
        } else {
            link_.data_ready.wait_min(VALID);
            uintptr_t address = get_semaphore(D);
#ifdef ARCH_QUASAR
            address += MEM_L1_UNCACHED_BASE;
#endif
            value = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        }
        link_.data_ready.set(INVALID);
        return value;
    }
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
void ChainReceiverPipe<N, D, C, S>::receive_and_forward(uint32_t dst_l1, uint32_t size_bytes, uint32_t round) {
    consume_<true>(round);
    if (link_.has_successor()) {
        link_.wait_for_successor();
        link_.write_successor(dst_l1, dst_l1, size_bytes);
        link_.noc.async_write_barrier();
        link_.publish(VALID);
    }
}

template <uint8_t N, uint32_t D, uint32_t C, DataReadySignal S>
uint32_t ChainReceiverPipe<N, D, C, S>::receive_signal(uint32_t round) {
    const uint32_t value = consume_<false>(round);
    if (link_.has_successor()) {
        link_.wait_for_successor();
        link_.publish(value);
    }
    return value;
}
}  // namespace dataflow_kernel_lib
