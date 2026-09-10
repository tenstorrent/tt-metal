// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/mcast_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "hostdevcommon/common_values.hpp"

namespace dataflow_kernel_lib {
namespace detail {
// Chain-specific link plumbing shared by the injector and relays. No multicast ACK/count inputs.
//
// Readiness protocol. Every hop's consumer_ready cell counts exactly one outstanding successor ack,
// and every data_ready publish is an atomic increment onto a cell the successor cleared before it
// acked. The pairing is strict: a hop publishes only after consuming one ack, and a receiver acks
// only after clearing its own Flag. A cell therefore never exceeds the published value, so Flag
// receivers wait for VALID exactly and Counter cells stay monotonic without any constructor reset.
template <uint8_t NOC_ID, uint32_t DATA_READY, uint32_t CONSUMER_READY, DataReadySignal SIGNAL>
struct ChainLink {
    static_assert(CONSUMER_READY != UNUSED_SEM_ID, "Chain forwarding requires a readiness handshake");
    Noc noc;
    Semaphore<> data_ready;
    Semaphore<> consumer_ready;
    ChainRuntimeArguments args;

    ChainLink(const Noc& noc, const ChainRuntimeArguments& args);
    bool has_successor() const { return args.successor_x != NO_CHAIN_NEIGHBOR; }
    void wait_for_successor();
    void write_successor(uint32_t src, uint32_t dst, uint32_t bytes);
    void publish(uint32_t value);
};
}  // namespace detail

// Fixed-sender injector. Owns prepared coordinates; send protects both the source and any local copy.
// SOURCE_GUARD is accepted for signature parity with SenderPipe but does not change the barriers:
// the successor's data-before-ready ordering already needs write completion, which also protects
// source L1. send() returns once the first hop holds the payload, not when the whole chain does.
template <uint8_t NOC_ID, uint32_t DATA_READY, uint32_t CONSUMER_READY, DataReadySignal SIGNAL>
class ChainSenderPipe {
public:
    using RuntimeArguments = ChainRuntimeArguments;
    explicit ChainSenderPipe(const Noc& noc, const RuntimeArguments& args);
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size_bytes);
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void send_signal(uint32_t value = VALID);

private:
    detail::ChainLink<NOC_ID, DATA_READY, CONSUMER_READY, SIGNAL> link_;
};

// receive_and_forward() includes relay completion. All hops must use the same destination and byte count.
// The caller reserves that destination before calling, and only re-acknowledges after reuse is safe.
// A relay blocks until its successor is ready, so the chain advances one round in
// lockstep; no hop may skip a round or exit early.
template <uint8_t NOC_ID, uint32_t DATA_READY, uint32_t CONSUMER_READY, DataReadySignal SIGNAL>
class ChainReceiverPipe {
public:
    using RuntimeArguments = ChainRuntimeArguments;
    explicit ChainReceiverPipe(const Noc& noc, const RuntimeArguments& args);
    void receive_and_forward(uint32_t dst_l1, uint32_t size_bytes, uint32_t round = 0);
    uint32_t receive_signal(uint32_t round = 0);

private:
    template <bool PAYLOAD>
    uint32_t consume_(uint32_t round);
    detail::ChainLink<NOC_ID, DATA_READY, CONSUMER_READY, SIGNAL> link_;
};
}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/chain_pipe.inl"
