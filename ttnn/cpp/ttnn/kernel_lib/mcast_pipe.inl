// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file mcast_pipe.inl
 * @brief Out-of-line definitions for SenderPipe / ReceiverPipe.
 *
 * NoC-multicast + semaphore-handshake helper. This file should only be included
 * by mcast_pipe.hpp.
 */

namespace dataflow_kernel_lib {

// =============================================================================
// SenderPipe
// =============================================================================

// Semaphore storage is initialized by the host: consumer_ready starts at 0 and data_ready at INVALID.
// Do not reinitialize it here, because receivers may have already incremented consumer_ready.
template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    uint32_t MAX_RECTS>
SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::
    SenderPipe(const Noc& noc, const McastRect<NOC_ID>& dest, uint32_t consumer_ack_count) :
    noc_(noc), data_ready_(DATA_READY_SEM_ID), consumer_ready_(CONSUMER_READY_SEM_ID) {
    ASSERT(noc_.get_noc_id() == NOC_ID);
    dests_[0] = dest;
    num_rects_ = 1;
    initialize_geometry_(consumer_ack_count);
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    uint32_t MAX_RECTS>
SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::SenderPipe(
    const Noc& noc, const uint32_t* dest_coords, uint32_t num_rects, uint32_t consumer_ack_count) :
    noc_(noc), data_ready_(DATA_READY_SEM_ID), consumer_ready_(CONSUMER_READY_SEM_ID), num_rects_(num_rects) {
    ASSERT(noc_.get_noc_id() == NOC_ID);
    ASSERT(num_rects_ <= MAX_RECTS);
    for (uint32_t i = 0; i < num_rects_; ++i) {
        dests_[i] = McastRect<NOC_ID>(
            dest_coords[4u * i + 0u],
            dest_coords[4u * i + 1u],
            dest_coords[4u * i + 2u],
            dest_coords[4u * i + 3u]);
    }
    initialize_geometry_(consumer_ack_count);
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    uint32_t MAX_RECTS>
void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::initialize_geometry_(uint32_t consumer_ack_count) {
    total_num_dests_excl_ = 0;
    for (uint32_t i = 0; i < num_rects_; ++i) {
        const auto& dest = dests_[i];
        in_rect_[i] = my_x[NOC_ID] >= dest.xlo() && my_x[NOC_ID] <= dest.xhi() && my_y[NOC_ID] >= dest.ylo() &&
                      my_y[NOC_ID] <= dest.yhi();
        num_dests_excl_[i] = dest.area() - (in_rect_[i] ? 1u : 0u);
        num_dests_incl_[i] = num_dests_excl_[i] + 1u;
        total_num_dests_excl_ += num_dests_excl_[i];
    }
    degenerate_ = total_num_dests_excl_ == 0u;
    ack_count_ = consumer_ack_count == ACK_EQUALS_FANOUT ? total_num_dests_excl_ : consumer_ack_count;
}

template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE,
    uint32_t CONSUMER_READY_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL,
    bool ROTATING_SENDER,
    uint32_t MAX_RECTS>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void
SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::send(
    uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
    // Degenerate: no receiver cores. If the sender is in its own box and lands a copy elsewhere, do
    // a local copy (a loopback to just self may hang); else nothing.
    if (degenerate_) {
        bool self_destination = false;
        for (uint32_t i = 0; i < num_rects_; ++i) {
            self_destination = self_destination || in_rect_[i];
        }
        if (self_destination && src_l1 != dst_l1) {
            local_copy_(src_l1, dst_l1, size);
            // The sender does not call receive() on itself, so ensure the local copy lands before
            // send() returns and the data can be consumed.
            noc_.async_write_barrier();
        }
        return;
    }
    if constexpr (PRE_HANDSHAKE) {
        consumer_ready_.wait(ack_count_);
        consumer_ready_.set(0);
    }
    // Loopback iff the sender is in the box AND lands its own copy somewhere other than its source
    // (src == dst means the copy is already in place; never self-overwrite in place). The
    // in-box test is precomputed in the ctor; only the src/dst aliasing varies per send.
    bool any_loopback = false;
    for (uint32_t i = 0; i < num_rects_; ++i) {
        const bool loopback = in_rect_[i] && src_l1 != dst_l1;
        const uint32_t mcast_dests = loopback ? num_dests_incl_[i] : num_dests_excl_[i];
        if (mcast_dests == 0u) {
            continue;
        }
        send_data_(i, src_l1, dst_l1, size, loopback, mcast_dests);
        signal_ready_(i, loopback, mcast_dests);
        any_loopback = any_loopback || loopback;
    }
    // The default post-signal fence is also the source-L1 lifetime guard. CallerManaged may omit the
    // remote-only SENT wait, but cannot omit completion required by loopback, rotating-Flag reset, or
    // Counter atomic acknowledgement semantics.
    fence_<SOURCE_GUARD>(any_loopback);
    // A rotating sender receives on this same flag cell in another round. Clear the local multicast
    // source after the fence proves it is no longer in use, or that later receive() can consume this
    // core's own stale value instead of waiting for the next sender.
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
    uint32_t MAX_RECTS>
void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::
    send_signal(uint32_t value) {
    if (degenerate_) {
        return;  // nobody to signal
    }
    if constexpr (PRE_HANDSHAKE) {
        consumer_ready_.wait(ack_count_);
        consumer_ready_.set(0);
    }
    // Typed values are intentionally Flag-only. Counter stays a monotone +1 event channel.
    ASSERT(value >= VALID);
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        ASSERT(value == VALID);
    }
    for (uint32_t i = 0; i < num_rects_; ++i) {
        if (num_dests_excl_[i] > 0u) {
            signal_ready_(i, /*loopback=*/false, num_dests_excl_[i], value);
        }
    }
    fence_<SourceL1Guard::Guard>(/*loopback=*/false);
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
    uint32_t MAX_RECTS>
FORCE_INLINE void
SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::send_data_(
    uint32_t rect_index, uint32_t src_l1, uint32_t dst_l1, uint32_t size, bool loopback, uint32_t mcast_dests) {
    const auto& r = dests_[rect_index].bounds();
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
    uint32_t MAX_RECTS>
FORCE_INLINE void
SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::signal_ready_(uint32_t rect_index, bool loopback, uint32_t mcast_dests, uint32_t value) {
    const auto& r = dests_[rect_index].bounds();
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.inc_multicast(noc_, r.sx, r.sy, r.ex, r.ey, /*value=*/1, mcast_dests);  // monotone +1
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
    uint32_t MAX_RECTS>
template <SourceL1Guard SOURCE_GUARD>
FORCE_INLINE void
SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::fence_(
    bool loopback) {
    // A real sender loopback always needs ACKED completion: this protects the locally published
    // destination as well as the source lifetime, independent of SOURCE_GUARD.
    if (loopback) {
        // The sender does not call receive() on itself, so ensure the loopback copy lands before
        // send() returns and the data can be consumed.
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
    uint32_t MAX_RECTS>
void SenderPipe<
    NOC_ID,
    DATA_READY_SEM_ID,
    PRE_HANDSHAKE,
    CONSUMER_READY_SEM_ID,
    DATA_READY_SIGNAL,
    ROTATING_SENDER,
    MAX_RECTS>::
    local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
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
    uint32_t dst_l1, uint32_t size_bytes, uint32_t round) {
    (void)dst_l1;
    (void)size_bytes;
    const uint32_t sender_index = round % NUM_SENDERS;
    const uint32_t sender_x = coords_[2 * sender_index + 0];
    const uint32_t sender_y = coords_[2 * sender_index + 1];
    if constexpr (PRE_HANDSHAKE) {
        // tell the sender "my dest is free / I am ready" (remote atomic inc on its counter)
        consumer_ready_.up(noc_, sender_x, sender_y, 1);
    }
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.wait_min(++round_);
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
uint32_t ReceiverPipe<DATA_READY_SEM_ID, PRE_HANDSHAKE, CONSUMER_READY_SEM_ID, DATA_READY_SIGNAL, NUM_SENDERS>::
    receive_signal() {
    if constexpr (PRE_HANDSHAKE) {
        const uint32_t sender_index = round_ % NUM_SENDERS;
        consumer_ready_.up(noc_, coords_[2 * sender_index + 0], coords_[2 * sender_index + 1], 1);
    }
    if constexpr (DATA_READY_SIGNAL == DataReadySignal::Counter) {
        data_ready_.wait_min(++round_);
        return round_;
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
        ++round_;
        return value;
    }
}

}  // namespace dataflow_kernel_lib
