// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// R1b rotating A stream: a multicast receiver whose readiness ACK and data WAIT are separate calls
// (tournament 2, perf_experiments/t2p2_act_stream_rounds).
//
// RAW justification (bypasses dataflow_kernel_lib ReceiverPipe::receive(round), mcast_pipe.inl): receive() fuses
// "ack round r's sender" with "wait for round r's data", so an injector could not ack round r + 1 and THEN spend
// ~3 us of RISC time issuing its next turn's A reads: either the issue sat in front of its own send (HEAD: issued
// before the send whenever the next turn's slots were free -- every first-cycle round) or in front of its ack of
// round r + 1 (which gates round r + 1's send). With the two halves apart the issue leaves the critical path, and a
// core polls the data-ready flag itself, so an injector issues its next turn's reads one tile per poll while it
// waits for the other injectors' rounds. The mechanism is the pipe's own: consumer_ready.up(sender of round r), then
// wait / clear the data-ready flag, from the same semaphore bindings and sender coordinate table the pipe uses
// (template-parameter extraction from McastArgs::ReceiverPipeType: nothing is duplicated host-side). The protocol is
// unchanged -- a core acks round r only after it consumed round r - 1's flag -- so this is a pure reordering of the
// RISC's work. Helper follow-up: ReceiverPipe::receive_ack(round) / ready(round) / receive_wait(round).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace gm_rx {

template <typename T>
struct RxTraits;

template <
    typename DataReadyBinding,
    bool PRE_HANDSHAKE,
    typename ConsumerReadyBinding,
    dataflow_kernel_lib::DataReadySignal SIGNAL,
    uint32_t NUM_SENDERS,
    typename SenderCoordinates>
struct RxTraits<dataflow_kernel_lib::detail::ReceiverPipeImpl<
    DataReadyBinding,
    PRE_HANDSHAKE,
    ConsumerReadyBinding,
    SIGNAL,
    NUM_SENDERS,
    SenderCoordinates>> {
    using DataReady = DataReadyBinding;
    using ConsumerReady = ConsumerReadyBinding;
    using Coords = SenderCoordinates;
    static constexpr bool pre_handshake = PRE_HANDSHAKE;
    static constexpr dataflow_kernel_lib::DataReadySignal signal = SIGNAL;
    static constexpr uint32_t num_senders = NUM_SENDERS;
};

// Stand-in for writers without a rotating A stream (kernel_main is not a template: both faces must compile).
struct NoReceiver {
    FORCE_INLINE explicit NoReceiver(const Noc&) {}
    FORCE_INLINE void ack(uint32_t) {}
    FORCE_INLINE bool ready(uint32_t) const { return true; }
    FORCE_INLINE void wait(uint32_t) {}
};

// MC = the McastArgs type, RT_BASE its runtime-arg offset. receive(r) == ack(r); wait(r).
template <typename MC, uint32_t RT_BASE>
class SplitReceiver {
    using T = RxTraits<typename MC::ReceiverPipeType>;
    static_assert(T::pre_handshake, "the split ack needs the receiver handshake");
    static constexpr bool counter = T::signal == dataflow_kernel_lib::DataReadySignal::Counter;

public:
    FORCE_INLINE explicit SplitReceiver(const Noc& noc) :
        noc_(noc),
        data_ready_(dataflow_kernel_lib::detail::make_mcast_semaphore<typename T::DataReady{}>()),
        consumer_ready_(dataflow_kernel_lib::detail::make_mcast_semaphore<typename T::ConsumerReady{}>()),
        coords_(dataflow_kernel_lib::detail::PositionalMcastRuntime<RT_BASE>::coordinates(
            MC::runtime_layout.sender_coordinates)) {}

    // "My slot for round r is free": increment round r's sender's consumer-ready counter. Call it only after
    // wait(r - 1) (Flag: the sender may set this core's flag as soon as every core acked).
    FORCE_INLINE void ack(uint32_t round) {
        const uint32_t s = round % T::num_senders;
        consumer_ready_.up(
            noc_,
            coords_
                [dataflow_kernel_lib::mcast_wire::SENDER_COORD_WORDS * s + dataflow_kernel_lib::mcast_wire::SENDER_X],
            coords_
                [dataflow_kernel_lib::mcast_wire::SENDER_COORD_WORDS * s + dataflow_kernel_lib::mcast_wire::SENDER_Y],
            1);
    }
    // Non-blocking: has round r's data landed?
    FORCE_INLINE bool ready(uint32_t round) const {
        if constexpr (counter) {
            return data_ready_.value() >= round + 1;
        } else {
            return data_ready_.value() == VALID;
        }
    }
    // Wait for round r's data and consume its signal.
    FORCE_INLINE void wait(uint32_t round) {
        if constexpr (counter) {
            data_ready_.wait_min(round + 1);
        } else {
            data_ready_.wait(VALID);
            data_ready_.set(INVALID);
        }
    }

private:
    Noc noc_;
    decltype(dataflow_kernel_lib::detail::make_mcast_semaphore<typename T::DataReady{}>()) data_ready_;
    decltype(dataflow_kernel_lib::detail::make_mcast_semaphore<typename T::ConsumerReady{}>()) consumer_ready_;
    typename T::Coords coords_;
};

}  // namespace gm_rx
