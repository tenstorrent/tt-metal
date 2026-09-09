// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file mcast_pipe.inl
 * @brief Out-of-line definitions for SenderPipe, ReceiverPipe, and McastArgs.
 *
 * NoC-multicast + semaphore-handshake helper. This file should only be included
 * by mcast_pipe.hpp.
 */

namespace dataflow_kernel_lib {

// =============================================================================
// SenderPipe
// =============================================================================

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderTransferMode TRANSFER_MODE,
    uint32_t MAX_RECTS>
SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    TRANSFER_MODE,
    MAX_RECTS>::SenderPipe(const Noc& noc, const RuntimeArguments& runtime_args) :
    noc_(noc), data_ready_(DATA_READY_SEM_ID), consumer_ready_(CONSUMER_READY_SEM_ID), args_(runtime_args) {
    ASSERT(noc_.get_noc_id() == NOC_ID);
    ASSERT(args_.num_rectangles >= 1 && args_.num_rectangles <= MAX_RECTS);
    for (uint32_t i = 0; i < args_.num_rectangles; ++i) {
        const auto& rectangle = args_.rectangles[i];
        ASSERT(mcast_wire::concrete(rectangle.transfer_mode));
        ASSERT(TRANSFER_MODE == SenderTransferMode::TransferModeUnknown || TRANSFER_MODE == rectangle.transfer_mode);
        ASSERT((rectangle.transfer_mode == SenderTransferMode::LocalCopy) == (rectangle.remote_count == 0));
        loopback_ |= rectangle.transfer_mode == SenderTransferMode::LocalCopy ||
                     rectangle.transfer_mode == SenderTransferMode::MulticastIncludeSource;
    }
    // Never initialize sender cells: receivers may already have acknowledged readiness.
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderTransferMode TRANSFER_MODE,
    uint32_t MAX_RECTS>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    TRANSFER_MODE,
    MAX_RECTS>::send(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
    if constexpr (PRE_HANDSHAKE) {
        consumer_ready_.wait(args_.ack_count);
        consumer_ready_.set(0);
    }
    if constexpr (MAX_RECTS == 1) {
        send_rectangle_<TRANSFER_MODE>(args_.rectangles[0], src_l1, dst_l1, size);
    } else {
        // Bound unrolling by the compile-time capacity while ignoring padded records.
#pragma GCC unroll 3
        for (uint32_t i = 0; i < MAX_RECTS; ++i) {
            if (i < args_.num_rectangles) {
                send_rectangle_<TRANSFER_MODE>(args_.rectangles[i], src_l1, dst_l1, size);
            }
        }
    }
    // Equal addresses skip the local copy, so only remote source-lifetime
    // guards apply. Wait for ACKED completion only when we wrote a local destination.
    fence_<SOURCE_GUARD>(loopback_ && src_l1 != dst_l1);
    // Atomic multicasts exclude their source. Count this completed sending round locally too.
    if constexpr (ROTATING_SENDER && DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.up(1);
    }
    if constexpr (ROTATING_SENDER && DATA_READY_SIGNAL == DataReadySignal::Flag) {
        data_ready_.set(INVALID);
    }
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderTransferMode TRANSFER_MODE,
    uint32_t MAX_RECTS>
template <SenderTransferMode MODE>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    TRANSFER_MODE,
    MAX_RECTS>::
    send_rectangle_(const RectangleRuntimeArguments& rectangle, uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
    if constexpr (MODE == SenderTransferMode::TransferModeUnknown) {
        switch (rectangle.transfer_mode) {
            case SenderTransferMode::LocalCopy:
                return send_rectangle_<SenderTransferMode::LocalCopy>(rectangle, src_l1, dst_l1, size);
            case SenderTransferMode::MulticastExcludeSource:
                return send_rectangle_<SenderTransferMode::MulticastExcludeSource>(rectangle, src_l1, dst_l1, size);
            case SenderTransferMode::MulticastIncludeSource:
                return send_rectangle_<SenderTransferMode::MulticastIncludeSource>(rectangle, src_l1, dst_l1, size);
            default: ASSERT(false); return;
        }
    } else if constexpr (MODE == SenderTransferMode::LocalCopy) {
        ASSERT(rectangle.transfer_mode == MODE);
        // A singleton containing the sender must use unicast, including inside a larger group.
        if (src_l1 != dst_l1) {
            local_copy_(src_l1, dst_l1, size);
        }
    } else {
        ASSERT(rectangle.transfer_mode == MODE);
        const bool loopback = MODE == SenderTransferMode::MulticastIncludeSource && src_l1 != dst_l1;
        const uint32_t destinations = loopback ? rectangle.loopback_count : rectangle.remote_count;
        send_data_(rectangle, loopback, src_l1, dst_l1, size, destinations);
        signal_ready_(rectangle, loopback, destinations);
    }
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderTransferMode TRANSFER_MODE,
    uint32_t MAX_RECTS>
template <SourceL1Guard SOURCE_GUARD>
void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    TRANSFER_MODE,
    MAX_RECTS>::send_signal(uint32_t value) {
    if constexpr (PRE_HANDSHAKE) {
        consumer_ready_.wait(args_.ack_count);
        consumer_ready_.set(0);
    }
    if constexpr (MAX_RECTS == 1) {
        const auto& rectangle = args_.rectangles[0];
        if (rectangle.remote_count > 0) {
            ASSERT(value >= VALID);
            if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
                ASSERT(value == VALID);
            }
            signal_ready_(rectangle, false, rectangle.remote_count, value);
        }
    } else {
#pragma GCC unroll 3
        for (uint32_t i = 0; i < MAX_RECTS; ++i) {
            if (i >= args_.num_rectangles) {
                break;
            }
            const auto& rectangle = args_.rectangles[i];
            if (rectangle.remote_count > 0) {
                ASSERT(value >= VALID);
                if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
                    ASSERT(value == VALID);
                }
                signal_ready_(rectangle, false, rectangle.remote_count, value);
            }
        }
    }
    fence_<SOURCE_GUARD>(false);
    // Count this completed sending round locally once, including local-only groups.
    if constexpr (ROTATING_SENDER && DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.up(1);
    }
    if constexpr (ROTATING_SENDER && DATA_READY_SIGNAL == DataReadySignal::Flag) {
        data_ready_.set(INVALID);
    }
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderTransferMode TRANSFER_MODE,
    uint32_t MAX_RECTS>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    TRANSFER_MODE,
    MAX_RECTS>::
    send_data_(
        const RectangleRuntimeArguments& rectangle,
        bool loopback,
        uint32_t src_l1,
        uint32_t dst_l1,
        uint32_t size,
        uint32_t mcast_dests) {
    const auto& r = rectangle.bounds;  // Already in routing order, prepared by the host.
    UnicastEndpoint src_ep;
    MulticastEndpoint dst_ep;
    const typename noc_traits_t<UnicastEndpoint>::src_args_type src_args{.addr = src_l1};
    const typename noc_traits_t<MulticastEndpoint>::dst_args_mcast_type dst_args{r.sx, r.sy, r.ex, r.ey, dst_l1};
    // Data is always linked to the following signal mcast (signal_ready_ issues the signal with
    // linked=false to terminate the chain). The linked pair enforces data-before-signal without a
    // barrier.
    if (loopback) {
        noc_.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
            src_ep, dst_ep, size, mcast_dests, src_args, dst_args, /*linked=*/true);
    } else {
        noc_.async_write_multicast<NocOptions::DEFAULT>(
            src_ep, dst_ep, size, mcast_dests, src_args, dst_args, /*linked=*/true);
    }
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderTransferMode TRANSFER_MODE,
    uint32_t MAX_RECTS>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    TRANSFER_MODE,
    MAX_RECTS>::
    signal_ready_(const RectangleRuntimeArguments& rectangle, bool loopback, uint32_t mcast_dests, uint32_t value) {
    const auto& r = rectangle.bounds;  // Already in routing order, prepared by the host.
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.inc_multicast(noc_, r.sx, r.sy, r.ex, r.ey, /*value=*/1, rectangle.remote_count);  // monotone +1

    } else {
        // set_multicast broadcasts this core's own cell as the source, so write this round's value
        // first. A core that also receives on this cell leaves it INVALID after a receive, and a
        // once-only set would go stale and stall the receivers. Rewriting VALID is a redundant no-op
        // for a send-only core; a typed control signal instead writes its caller-supplied value.
        data_ready_.set(value);
        if (loopback) {
            data_ready_.set_multicast<NocOptions::MCAST_INCL_SRC>(
                noc_, r.sx, r.sy, r.ex, r.ey, mcast_dests, /*linked=*/false);
        } else {
            data_ready_.set_multicast<NocOptions::DEFAULT>(noc_, r.sx, r.sy, r.ex, r.ey, mcast_dests, /*linked=*/false);
        }
    }
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderTransferMode TRANSFER_MODE,
    uint32_t MAX_RECTS>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    TRANSFER_MODE,
    MAX_RECTS>::fence_(bool loopback) {
    // Loopback and local unicast copies require ACKED completion to protect the locally published
    // destination as well as the source lifetime, independent of SOURCE_GUARD.
    if (loopback) {
        // The sender never calls receive() on itself, so it has no data-ready wait that proves its
        // destination arrived before a same-core consumer observes the caller's publication.
        noc_.async_write_barrier();
    } else if constexpr (
        SOURCE_GUARD == SourceL1Guard::Guard || (ROTATING_SENDER && DATA_READY_SIGNAL == DataReadySignal::Flag)) {
        // Guard waits for the remote-only payload source to depart. A rotating Flag sender also
        // needs this wait before send() resets the local semaphore cell used as the signal source.
        noc_.async_writes_flushed();
    }
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        // inc_multicast is a NON-POSTED multicast atomic: it expects num_dests acks that the flush
        // or write barrier above does not drain, so the Counter path additionally waits the atomic barrier.
        noc_.async_atomic_barrier();
    }
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderTransferMode TRANSFER_MODE,
    uint32_t MAX_RECTS>
void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    TRANSFER_MODE,
    MAX_RECTS>::local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
    // PRECONDITION: src_l1 != dst_l1 (send() owns that test, so it can pair the copy with its fence).
    // Issued on the write channel so it settles under the caller's / the pipe's own write accounting;
    // the completion wait belongs to send().
    ASSERT(src_l1 != dst_l1);
    UnicastEndpoint dst_ep;
    const uint32_t mx = my_x[NOC_ID];
    const uint32_t my = my_y[NOC_ID];
    noc_.async_write(
        CoreLocalMem<uint32_t>(src_l1),
        dst_ep,
        size,
        {},
        typename noc_traits_t<UnicastEndpoint>::dst_args_type{mx, my, dst_l1});
}

// =============================================================================
// ReceiverPipe
// =============================================================================

template <
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    uint32_t NUM_SENDERS>
ReceiverPipe<DATA_READY_SEM_ID, PRE_HANDSHAKE, CONSUMER_READY_SEM_ID, DATA_READY_SIGNAL, NUM_SENDERS>::ReceiverPipe(
    const Noc& noc, const uint32_t* sender_coords) :
    noc_(noc), data_ready_(DATA_READY_SEM_ID), consumer_ready_(CONSUMER_READY_SEM_ID), coords_(sender_coords) {
    // Init the flag THIS side waits on. The Counter signal needs no reset/init (monotone).
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Flag) {
        data_ready_.set(INVALID);
    }
}

template <
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    uint32_t NUM_SENDERS>
void ReceiverPipe<DATA_READY_SEM_ID, PRE_HANDSHAKE, CONSUMER_READY_SEM_ID, DATA_READY_SIGNAL, NUM_SENDERS>::receive(
    uint32_t round) {
    // `round` is the caller's ABSOLUTE work round, not an index into the coord table: sender
    // selection wraps every NUM_SENDERS rounds, so a rotating receiver just forwards its loop
    // counter and never has to reduce it at the call site.
    const uint32_t sender_index = round % NUM_SENDERS;
    const uint32_t sender_x = coords_[2 * sender_index + 0];
    const uint32_t sender_y = coords_[2 * sender_index + 1];
    if constexpr (PRE_HANDSHAKE) {
        // tell the sender "my dest is free / I am ready" (remote atomic inc on its counter)
        consumer_ready_.up(noc_, sender_x, sender_y, 1);
    }
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.wait_min(round + 1);
    } else {
        data_ready_.wait(VALID);
        data_ready_.set(INVALID);  // clear this round's flag; next receive()'s ack follows
    }
}

template <
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    uint32_t NUM_SENDERS>
uint32_t
ReceiverPipe<DATA_READY_SEM_ID, PRE_HANDSHAKE, CONSUMER_READY_SEM_ID, DATA_READY_SIGNAL, NUM_SENDERS>::receive_signal(
    uint32_t round) {
    if constexpr (PRE_HANDSHAKE) {
        // tell the round-th sender "I am ready" (remote atomic inc on its counter)
        const uint32_t sender_index = round % NUM_SENDERS;
        consumer_ready_.up(noc_, coords_[2 * sender_index + 0], coords_[2 * sender_index + 1], 1);
    }
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.wait_min(round + 1);
        return round + 1;
    } else {
        // Flag control signals may carry any non-zero value. Capture it before the single clear so
        // the caller can distinguish the ordinary VALID doorbell from a typed control state.
        data_ready_.wait_min(VALID);
        uintptr_t data_ready_addr = get_semaphore(DATA_READY_SEM_ID);
#ifdef ARCH_QUASAR
        data_ready_addr += MEM_L1_UNCACHED_BASE;
#endif
        const uint32_t value = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_addr);
        data_ready_.set(INVALID);
        return value;
    }
}

// =============================================================================
// McastArgs
// =============================================================================

namespace detail {

template <uint32_t CT_BASE, uint32_t RT_BASE>
bool McastArgsImpl<true, CT_BASE, RT_BASE>::should_send(uint32_t round) const {
    return can_send() &&
           sender_index(round) == get_arg_val<uint32_t>(next_runtime_args_offset() - mcast_wire::PHASE_FROM_END);
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
typename McastArgsImpl<true, CT_BASE, RT_BASE>::SenderPipe McastArgsImpl<true, CT_BASE, RT_BASE>::sender(
    const Noc& noc) const {
    ASSERT(can_send());
    static_assert(sender_noc == noc_index, "Host multicast NoC does not match sending kernel NoC");
    return SenderPipe(noc, sender_runtime_arguments());
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
std::optional<typename McastArgsImpl<true, CT_BASE, RT_BASE>::SenderPipe>
McastArgsImpl<true, CT_BASE, RT_BASE>::optional_sender(const Noc& noc) const {
    if (!can_send()) {
        return std::nullopt;
    }
    return sender(noc);
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
typename McastArgsImpl<true, CT_BASE, RT_BASE>::ReceiverPipe McastArgsImpl<true, CT_BASE, RT_BASE>::receiver(
    const Noc& noc) const {
    ASSERT(can_receive());
    const uint32_t* coords =
        reinterpret_cast<const uint32_t*>(get_arg_addr(RT_BASE + mcast_wire::sender_coords_offset(rotating_span)));
    return ReceiverPipe(noc, coords);
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
std::optional<typename McastArgsImpl<true, CT_BASE, RT_BASE>::ReceiverPipe>
McastArgsImpl<true, CT_BASE, RT_BASE>::optional_receiver(const Noc& noc) const {
    if (!can_receive()) {
        return std::nullopt;
    }
    return receiver(noc);
}

template <uint32_t CT_BASE, uint32_t RT_BASE>
typename McastArgsImpl<true, CT_BASE, RT_BASE>::SenderPipe::RuntimeArguments
McastArgsImpl<true, CT_BASE, RT_BASE>::sender_runtime_arguments() const {
    // Every group has destinations, so capacity one guarantees exactly one record.
    constexpr bool single_rectangle = rectangle_capacity == 1;
    typename SenderPipe::RuntimeArguments prepared;
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

}  // namespace detail

}  // namespace dataflow_kernel_lib
