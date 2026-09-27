// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file mcast_pipe.inl
 * @brief Out-of-line definitions for SenderPipe and ReceiverPipe.
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
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderMcastMode SENDER_MCAST_MODE,
    uint32_t MAX_RECTS>
FORCE_INLINE SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    SENDER_MCAST_MODE,
    MAX_RECTS>::SenderPipe(const Noc& noc, const SenderRuntimeArgumentsFor<MAX_RECTS>& runtime_args) :
    noc_(noc),
    data_ready_(detail::make_mcast_semaphore<DATA_READY_SEM_ID>()),
    consumer_ready_(detail::make_mcast_semaphore<CONSUMER_READY_SEM_ID>()),
    args_(runtime_args) {
    ASSERT(noc_.get_noc_id() == NOC_ID);
    ASSERT(args_.num_rectangles >= 1 && args_.num_rectangles <= MAX_RECTS);
    for (uint32_t i = 0; i < args_.num_rectangles; ++i) {
        const auto& rectangle = args_.rectangles[i];
        ASSERT(mcast_wire::concrete(rectangle.sender_mcast_mode));
        ASSERT(SENDER_MCAST_MODE == SenderMcastMode::Unknown || SENDER_MCAST_MODE == rectangle.sender_mcast_mode);
        ASSERT((rectangle.sender_mcast_mode == SenderMcastMode::LocalCopy) == (rectangle.remote_count == 0));
        loopback_ |= rectangle.sender_mcast_mode == SenderMcastMode::LocalCopy ||
                     rectangle.sender_mcast_mode == SenderMcastMode::MulticastIncludeSource;
    }
    // Never initialize sender cells: receivers may already have acknowledged readiness.
}

template <
    uint8_t NOC_ID,
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderMcastMode SENDER_MCAST_MODE,
    uint32_t MAX_RECTS>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    SENDER_MCAST_MODE,
    MAX_RECTS>::send(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
    if constexpr (PRE_HANDSHAKE) {
        consumer_ready_.wait(args_.ack_count);
        consumer_ready_.set(0);
    }
    if constexpr (MAX_RECTS == 1) {
        send_rectangle_<SENDER_MCAST_MODE>(args_.rectangles[0], src_l1, dst_l1, size);
    } else {
        // Bound unrolling by the compile-time capacity while ignoring padded records.
#pragma GCC unroll 3
        for (uint32_t i = 0; i < MAX_RECTS; ++i) {
            if (i < args_.num_rectangles) {
                send_rectangle_<SENDER_MCAST_MODE>(args_.rectangles[i], src_l1, dst_l1, size);
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
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderMcastMode SENDER_MCAST_MODE,
    uint32_t MAX_RECTS>
template <SenderMcastMode RECTANGLE_MCAST_MODE>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    SENDER_MCAST_MODE,
    MAX_RECTS>::
    send_rectangle_(const RectangleRuntimeArguments& rectangle, uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
    if constexpr (RECTANGLE_MCAST_MODE == SenderMcastMode::Unknown) {
        switch (rectangle.sender_mcast_mode) {
            case SenderMcastMode::LocalCopy:
                return send_rectangle_<SenderMcastMode::LocalCopy>(rectangle, src_l1, dst_l1, size);
            case SenderMcastMode::MulticastExcludeSource:
                return send_rectangle_<SenderMcastMode::MulticastExcludeSource>(rectangle, src_l1, dst_l1, size);
            case SenderMcastMode::MulticastIncludeSource:
                return send_rectangle_<SenderMcastMode::MulticastIncludeSource>(rectangle, src_l1, dst_l1, size);
            default: ASSERT(false); return;
        }
    } else if constexpr (RECTANGLE_MCAST_MODE == SenderMcastMode::LocalCopy) {
        ASSERT(rectangle.sender_mcast_mode == RECTANGLE_MCAST_MODE);
        // A singleton containing the sender must use unicast, including inside a larger group.
        if (src_l1 != dst_l1) {
            local_copy_(src_l1, dst_l1, size);
        }
    } else {
        ASSERT(rectangle.sender_mcast_mode == RECTANGLE_MCAST_MODE);
        const bool loopback = RECTANGLE_MCAST_MODE == SenderMcastMode::MulticastIncludeSource && src_l1 != dst_l1;
        const uint32_t destinations = loopback ? rectangle.loopback_count : rectangle.remote_count;
        send_data_(rectangle, loopback, src_l1, dst_l1, size, destinations);
        signal_ready_(rectangle, loopback, destinations);
    }
}

template <
    uint8_t NOC_ID,
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderMcastMode SENDER_MCAST_MODE,
    uint32_t MAX_RECTS>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    SENDER_MCAST_MODE,
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
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderMcastMode SENDER_MCAST_MODE,
    uint32_t MAX_RECTS>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    SENDER_MCAST_MODE,
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
    // Flag readiness is an unlinked write that terminates the linked payload sequence.
    // Counter readiness is an atomic: do not leave a write path reserved for it, and ensure
    // the payload has landed before publishing the counter on the separate atomic path.
    constexpr bool linked = DATA_READY_SIGNAL == DataReadySignal::Flag;
    if (loopback) {
        noc_.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
            src_ep, dst_ep, size, mcast_dests, src_args, dst_args, linked);
    } else {
        noc_.async_write_multicast<NocOptions::DEFAULT>(src_ep, dst_ep, size, mcast_dests, src_args, dst_args, linked);
    }
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        noc_.async_write_barrier();
    }
}

template <
    uint8_t NOC_ID,
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderMcastMode SENDER_MCAST_MODE,
    uint32_t MAX_RECTS>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    SENDER_MCAST_MODE,
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
            data_ready_.template set_multicast<NocOptions::MCAST_INCL_SRC>(
                noc_, r.sx, r.sy, r.ex, r.ey, mcast_dests, /*linked=*/false);
        } else {
            data_ready_.template set_multicast<NocOptions::DEFAULT>(
                noc_, r.sx, r.sy, r.ex, r.ey, mcast_dests, /*linked=*/false);
        }
    }
}

template <
    uint8_t NOC_ID,
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderMcastMode SENDER_MCAST_MODE,
    uint32_t MAX_RECTS>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    SENDER_MCAST_MODE,
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
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    SenderMcastMode SENDER_MCAST_MODE,
    uint32_t MAX_RECTS>
FORCE_INLINE void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    SENDER_MCAST_MODE,
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
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    uint32_t NUM_SENDERS,
    typename SenderCoordinates>
FORCE_INLINE ReceiverPipe<
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    NUM_SENDERS,
    SenderCoordinates>::ReceiverPipe(const Noc& noc, SenderCoordinates sender_coords) :
    noc_(noc),
    data_ready_(detail::make_mcast_semaphore<DATA_READY_SEM_ID>()),
    consumer_ready_(detail::make_mcast_semaphore<CONSUMER_READY_SEM_ID>()),
    coords_(sender_coords) {
    // Init the flag THIS side waits on. The Counter signal needs no reset/init (monotone).
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Flag) {
        data_ready_.set(INVALID);
    }
}

template <
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    uint32_t NUM_SENDERS,
    typename SenderCoordinates>
FORCE_INLINE void ReceiverPipe<
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    NUM_SENDERS,
    SenderCoordinates>::receive(uint32_t round) {
    // `round` is the caller's ABSOLUTE work round, not an index into the coord table: sender
    // selection wraps every NUM_SENDERS rounds, so a rotating receiver just forwards its loop
    // counter and never has to reduce it at the call site.
    const uint32_t sender_index = round % NUM_SENDERS;
    const uint32_t sender_x = coords_[mcast_wire::SENDER_COORD_WORDS * sender_index + mcast_wire::SENDER_X];
    const uint32_t sender_y = coords_[mcast_wire::SENDER_COORD_WORDS * sender_index + mcast_wire::SENDER_Y];
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
    auto DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    auto CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    uint32_t NUM_SENDERS,
    typename SenderCoordinates>
FORCE_INLINE uint32_t ReceiverPipe<
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    NUM_SENDERS,
    SenderCoordinates>::receive_signal(uint32_t round) {
    if constexpr (PRE_HANDSHAKE) {
        // tell the round-th sender "I am ready" (remote atomic inc on its counter)
        const uint32_t sender_index = round % NUM_SENDERS;
        consumer_ready_.up(
            noc_,
            coords_[mcast_wire::SENDER_COORD_WORDS * sender_index + mcast_wire::SENDER_X],
            coords_[mcast_wire::SENDER_COORD_WORDS * sender_index + mcast_wire::SENDER_Y],
            1);
    }
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.wait_min(round + 1);
        return round + 1;
    } else {
        // Flag control signals may carry any non-zero value. Capture it before the single clear so
        // the caller can distinguish the ordinary VALID doorbell from a typed control state.
        data_ready_.wait_min(VALID);
        const uint32_t value = data_ready_.value();
        data_ready_.set(INVALID);
        return value;
    }
}

}  // namespace dataflow_kernel_lib
