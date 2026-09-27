// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_semaphore.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "hostdevcommon/common_values.hpp"

namespace dataflow_kernel_lib {
namespace detail {
// Chain-specific link plumbing shared by the injector and relays. No multicast ACK/count inputs.
//
// Each outgoing event consumes one successor ack before changing signal_source. The next ack
// proves the preceding relay read its source, even with CallerManaged. Incoming data_ready is
// separate, so a receiver can acknowledge its predecessor before waiting for its successor.
// Payload and readiness are ordinary writes on the same NoC/VC; no fence is needed between them.
template <uint8_t NOC_ID, auto DATA_READY, auto CONSUMER_READY, auto SIGNAL_SOURCE, DataReadySignal SIGNAL>
struct ChainLink {
    static_assert(
        mcast_semaphore_id(CONSUMER_READY) != UNUSED_SEM_ID, "Chain forwarding requires a readiness handshake");
    static_assert(
        mcast_semaphore_id(DATA_READY) != UNUSED_SEM_ID &&
            mcast_semaphore_id(DATA_READY) != mcast_semaphore_id(CONSUMER_READY),
        "Invalid chain semaphores");
    static_assert(
        mcast_semaphore_id(SIGNAL_SOURCE) != UNUSED_SEM_ID &&
            mcast_semaphore_id(SIGNAL_SOURCE) != mcast_semaphore_id(DATA_READY) &&
            mcast_semaphore_id(SIGNAL_SOURCE) != mcast_semaphore_id(CONSUMER_READY),
        "Chain forwarding requires a distinct signal-source semaphore");
    Noc noc;
    decltype(make_mcast_semaphore<DATA_READY>()) data_ready;
    decltype(make_mcast_semaphore<CONSUMER_READY>()) consumer_ready;
    decltype(make_mcast_semaphore<SIGNAL_SOURCE>()) signal_source;
    ChainRuntimeArguments args;

    FORCE_INLINE ChainLink(const Noc& noc, const ChainRuntimeArguments& args);
    FORCE_INLINE bool has_successor() const { return args.successor_x != NO_CHAIN_NEIGHBOR; }
    FORCE_INLINE void wait_for_successor();
    FORCE_INLINE void write_successor(uint32_t src, uint32_t dst, uint32_t bytes);
    FORCE_INLINE uint32_t sender_signal_value(uint32_t value) const;
    FORCE_INLINE void publish(uint32_t value);
};
}  // namespace detail

// Fixed-sender injector. Guard protects source departure; CallerManaged leaves source lifetime
// and final write draining to the caller. A distinct local-copy destination always completes before
// return. Neither guard waits for the whole chain to consume the event.
// Resource arguments accept numeric IDs or native SemaphoreBindingToken values; all three are required.
template <uint8_t NOC_ID, auto DATA_READY, auto CONSUMER_READY, auto SIGNAL_SOURCE, DataReadySignal SIGNAL>
class ChainSenderPipe {
public:
    using RuntimeArguments = ChainRuntimeArguments;
    FORCE_INLINE explicit ChainSenderPipe(const Noc& noc, const RuntimeArguments& args);
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size_bytes);
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void send_signal(uint32_t value = VALID);

private:
    detail::ChainLink<NOC_ID, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE, SIGNAL> link_;
};

// All hops use the same destination and byte count. Reserve the destination before receiving.
// A relay acknowledges its predecessor before waiting for successor readiness; no hop may skip
// an event. The default Guard protects the forwarding source before returning.
template <uint8_t NOC_ID, auto DATA_READY, auto CONSUMER_READY, auto SIGNAL_SOURCE, DataReadySignal SIGNAL>
class ChainReceiverPipe {
public:
    using RuntimeArguments = ChainRuntimeArguments;
    FORCE_INLINE explicit ChainReceiverPipe(const Noc& noc, const RuntimeArguments& args);
    // Drain outstanding NoC atomics, including predecessor acknowledgments, when the pipe leaves scope.
    FORCE_INLINE ~ChainReceiverPipe();
    // CallerManaged: flush before modifying/recycling dst_l1, including a later receive that reuses
    // it, and drain outstanding writes before kernel exit. Independent work may precede that flush.
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void receive_and_forward(uint32_t dst_l1, uint32_t size_bytes, uint32_t round = 0);
    FORCE_INLINE uint32_t receive_signal(uint32_t round = 0);

private:
    template <bool PAYLOAD>
    FORCE_INLINE uint32_t consume_(uint32_t round);
    detail::ChainLink<NOC_ID, DATA_READY, CONSUMER_READY, SIGNAL_SOURCE, SIGNAL> link_;
};
}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/chain_pipe.inl"
