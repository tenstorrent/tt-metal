// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// t2p2_act_stream_rounds: a rotating-sender multicast receiver whose readiness ACK and data WAIT are separate calls.
//
// HELPER BYPASSED: dataflow_kernel_lib ReceiverPipe::receive(round) (mcast_pipe.inl) fuses "ack round r's sender"
// with "wait for round r's data". The R1b A stream needs them apart: a core must ack a LATER round (its slot is free)
// before it waits for the current one, and an injector must ack round r + 1 BEFORE it spends ~3 us of RISC time
// issuing its next turn's A reads -- with the fused call, the ack of r + 1 waited behind the issue, and every round's
// send waited behind that ack (A alone 47 us, 5 rounds of 216 KB at ~5 us each = send-bound ~30 us). The mechanism
// is identical to receive(): consumer_ready.up(sender of round r) and data_ready.wait_min(r + 1) (Counter signal),
// built from the same semaphore bindings and sender coordinate table the pipe would use (template-param extraction
// from McastArgs::ReceiverPipeType, so nothing is duplicated host-side). Graduation option: add
// receive_ack(round) / receive_wait(round) to ReceiverPipeImpl and drop this header.
//
// Ack-ahead SAFETY (Counter signal only; Flag: see `counter`) (rotating senders, consumer_ready reset by the sender
// with set(0) before its data): ack round q only after round q - SPAN's sender reset its counter, i.e. after this core
// has seen round q - SPAN land (or sent it). The caller keeps q <= (last landed or sent round) + SPAN.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace asr {

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
    FORCE_INLINE void wait(uint32_t) {}
    FORCE_INLINE bool ready(uint32_t) const { return true; }
    static constexpr bool counter = false;
};

// MC = the McastArgs type, RT_BASE its runtime-arg offset.
template <typename MC, uint32_t RT_BASE>
class SplitReceiver {
    using T = RxTraits<typename MC::ReceiverPipeType>;
    static_assert(T::pre_handshake, "split ack needs the receiver handshake");

public:
    // Counter: data_ready counts completed rounds, so a core may ack rounds AHEAD of the one it waits for. Flag (the
    // R1b channel's default): one VALID cell per core, cleared by wait(); a sender may only signal it once this core
    // has cleared the previous round's flag, so a core acks round q only after wait(q - 1) -- ack-ahead 0.
    static constexpr bool counter = T::signal == dataflow_kernel_lib::DataReadySignal::Counter;

    FORCE_INLINE explicit SplitReceiver(const Noc& noc) :
        noc_(noc),
        data_ready_(dataflow_kernel_lib::detail::make_mcast_semaphore<typename T::DataReady{}>()),
        consumer_ready_(dataflow_kernel_lib::detail::make_mcast_semaphore<typename T::ConsumerReady{}>()),
        coords_(dataflow_kernel_lib::detail::PositionalMcastRuntime<RT_BASE>::coordinates(
            MC::runtime_layout.sender_coordinates)) {}

    // "My slot for round r is free": increment round r's sender's consumer-ready counter.
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
    // Non-blocking: has round r's data landed? (wait(r) then consumes it.)
    FORCE_INLINE bool ready(uint32_t round) const {
        if constexpr (counter) {
            return data_ready_.value() >= round + 1;
        } else {
            return data_ready_.value() == VALID;
        }
    }
    // Round r's data has landed (the Counter has counted r + 1 completed rounds).
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

}  // namespace asr
