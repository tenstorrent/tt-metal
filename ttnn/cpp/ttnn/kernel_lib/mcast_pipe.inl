// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

/**
 * @file mcast_pipe.inl
 * @brief Implementation of the SenderPipe / ReceiverPipe member functions.
 *
 * Included at the bottom of mcast_pipe.hpp; see that header for the full design rationale
 * (geometry + recipient count, the inferred EXCLUDE/INCLUDE-SRC loopback dual-path, the
 * bake-off-decided style choices, and the semaphore-init ownership rules).
 */

namespace dataflow_kernel_lib {

// =============================================================================
// SenderPipe
// =============================================================================

template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING,
    bool PRE_HANDSHAKE,
    uint32_t INITIAL_READY>
SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE, INITIAL_READY>::
    SenderPipe(const Noc& noc, const McastRect& dest) :
    noc_(noc), dest_(dest), data_ready_(DATA_READY_SEM_ID), consumed_(CONSUMED_SEM_ID) {
    // NOTE: the `consumed` ACK counter is NOT kernel-initialized here. Receivers increment it
    // remotely (`up()`) with NO happens-before relative to this ctor — a receiver can ack before
    // the sender core even runs, so a ctor `set(0)` would clobber an early ack and hang. Its
    // initial value (0) MUST come from host `CreateSemaphore(..., 0)` (every call site does this).
    // The sender DOES own its own local data-ready cell (only it writes it before the first send),
    // so pre-setting that to the broadcast "ready" value is race-free (Flag staging only).
    if constexpr (STAGING == Staging::Flag) {
        data_ready_.set(INITIAL_READY);
    }
}

// send() is atomic and absorbs ALL FOUR guards (callers cannot reorder or skip them):
//   [if PRE_HANDSHAKE] wait(consumed)  — gate the mcast on receivers having drained (H2)
//   mcast data                          — object API auto-chunks a ready block > burst
//   raise flag                          — data-before-flag, same VC (INV4); reset owned (H11)
//   fence                               — flush (F1); atomic-barrier on the counter path
template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING,
    bool PRE_HANDSHAKE,
    uint32_t INITIAL_READY>
void SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE, INITIAL_READY>::
    send(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
    // Self-only degenerate: no receiver cores at all. If the sender is in its own box and lands
    // a copy elsewhere, do a local copy (loopback to just self may hang, H5); else nothing.
    if (NUM_ACTIVE_RECEIVER_CORES == 0) {
        if (sender_in_rect_()) {
            local_copy_(src_l1, dst_l1, size);
        }
        return;
    }
    if constexpr (PRE_HANDSHAKE) {
        consumed_.wait(NUM_ACTIVE_RECEIVER_CORES);
        consumed_.set(0);
    }
    // Loopback iff the sender is in the box AND lands its own copy somewhere else than its
    // source (src == dst means the copy is already in place; never self-overwrite in place).
    const bool loopback = sender_in_rect_() && src_l1 != dst_l1;
    // NUM_ACTIVE_RECEIVER_CORES is the RECIPIENT count (the EXCLUDE_SRC NoC num_dests = the ACK
    // count). The loopback (INCLUDE_SRC) path adds +1 for the sender's own self-copy. The sender's
    // own copy never acks, so the consumed wait above is on NUM_ACTIVE_RECEIVER_CORES.
    const uint32_t mcast_dests = loopback ? NUM_ACTIVE_RECEIVER_CORES + 1 : NUM_ACTIVE_RECEIVER_CORES;
    send_data_(src_l1, dst_l1, size, loopback, mcast_dests);
    raise_flag_(VALID, loopback, mcast_dests);  // flag rides the same mode as the data (INV4)
    fence_();
}

// Broadcast a control flag. `value` carries a payload for value-carrying flags
// (e.g. moe_gpt token counts); defaults to VALID for a plain doorbell. Always EXCLUDE_SRC
// (no data accompanies it), so the destination population is the recipient count.
template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING,
    bool PRE_HANDSHAKE,
    uint32_t INITIAL_READY>
void SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE, INITIAL_READY>::
    send_signal(uint32_t value) {
    if (NUM_ACTIVE_RECEIVER_CORES == 0) {
        return;  // nobody to signal
    }
    raise_flag_(value, /*loopback=*/false, NUM_ACTIVE_RECEIVER_CORES);
    fence_();
}

// ---- is the sender's own core inside the receiver rect? (IR1: compare in noc_'s space) ----
template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING,
    bool PRE_HANDSHAKE,
    uint32_t INITIAL_READY>
bool SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE, INITIAL_READY>::
    sender_in_rect_() const {
    const uint32_t mx = my_x[noc_.get_noc_id()];
    const uint32_t my = my_y[noc_.get_noc_id()];
    return mx >= dest_.xlo() && mx <= dest_.xhi() && my >= dest_.ylo() && my <= dest_.yhi();
}

// ---- data multicast via the Noc object (noc 2.0) ----
template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING,
    bool PRE_HANDSHAKE,
    uint32_t INITIAL_READY>
void SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE, INITIAL_READY>::
    send_data_(uint32_t src_l1, uint32_t dst_l1, uint32_t size, bool loopback, uint32_t mcast_dests) {
    const auto r = dest_.start_end_for_noc(noc_.get_noc_id());  // routing-correct start/end
    UnicastEndpoint src_ep;
    MulticastEndpoint dst_ep;
    const typename noc_traits_t<UnicastEndpoint>::src_args_type src_args{.addr = src_l1};
    const typename noc_traits_t<MulticastEndpoint>::dst_args_mcast_type dst_args{r.sx, r.sy, r.ex, r.ey, dst_l1};
    // Data is ALWAYS linked to the following flag mcast (raise_flag_ issues the flag with
    // linked=false to terminate the chain). The linked pair enforces data-before-flag without
    // a barrier (F4, −36%). No unlinked arm — see the header note.
    if (loopback) {
        noc_.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
            src_ep, dst_ep, size, mcast_dests, src_args, dst_args, /*linked=*/true);
    } else {
        noc_.async_write_multicast<NocOptions::DEFAULT>(
            src_ep, dst_ep, size, mcast_dests, src_args, dst_args, /*linked=*/true);
    }
}

// ---- raise the data-ready flag (or a control flag with a payload) ----
// `loopback` mirrors the data mcast of the same send() (INV4: same path); send_signal()
// has no data, so its flag is always EXCLUDE_SRC.
template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING,
    bool PRE_HANDSHAKE,
    uint32_t INITIAL_READY>
void SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE, INITIAL_READY>::
    raise_flag_(uint32_t value, bool loopback, uint32_t mcast_dests) {
    const auto r = dest_.start_end_for_noc(noc_.get_noc_id());  // routing-correct start/end
    if constexpr (STAGING == Staging::Counter) {
        data_ready_.inc_multicast(noc_, r.sx, r.sy, r.ex, r.ey, value, mcast_dests);
    } else {
        data_ready_.set(value);
        if (loopback) {
            data_ready_.set_multicast<NocOptions::MCAST_INCL_SRC>(
                noc_, r.sx, r.sy, r.ex, r.ey, mcast_dests, /*linked=*/false);
        } else {
            data_ready_.set_multicast<NocOptions::DEFAULT>(
                noc_, r.sx, r.sy, r.ex, r.ey, mcast_dests, /*linked=*/false);
        }
    }
}

// ---- post-send fence (F1) ----
template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING,
    bool PRE_HANDSHAKE,
    uint32_t INITIAL_READY>
void SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE, INITIAL_READY>::
    fence_() {
    if constexpr (STAGING == Staging::Counter) {
        // inc_multicast is a NON-POSTED multicast atomic: it expects num_dests ACKs that
        // must be drained, so a write flush is NOT sufficient — an atomic barrier is forced.
        noc_.async_writes_flushed();
        noc_.async_atomic_barrier();
    } else {
        noc_.async_writes_flushed();  // SENT — source L1 safe to reuse (H1). Flag proves arrival (INV4).
    }
}

// ---- local L1 self-copy (degenerate self-only guard) via the Noc object (noc 2.0) ----
template <
    uint32_t NUM_ACTIVE_RECEIVER_CORES,
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMED_SEM_ID,
    Staging STAGING,
    bool PRE_HANDSHAKE,
    uint32_t INITIAL_READY>
void SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE, INITIAL_READY>::
    local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size) {
    if (src_l1 == dst_l1) {
        return;  // src == dst: nothing to copy (source polymorphism, R4)
    }
    UnicastEndpoint src_ep, dst_ep;
    const uint32_t mx = my_x[noc_.get_noc_id()];
    const uint32_t my = my_y[noc_.get_noc_id()];
    noc_.async_read(
        src_ep,
        dst_ep,
        size,
        typename noc_traits_t<UnicastEndpoint>::src_args_type{mx, my, src_l1},
        typename noc_traits_t<UnicastEndpoint>::dst_args_type{0, 0, dst_l1});
    noc_.async_read_barrier();
}

// =============================================================================
// ReceiverPipe
// =============================================================================

template <uint32_t DATA_READY_SEM_ID, uint32_t CONSUMED_SEM_ID, Staging STAGING, bool PRE_HANDSHAKE>
ReceiverPipe<DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE>::ReceiverPipe(const Noc& noc) :
    noc_(noc), data_ready_(DATA_READY_SEM_ID), consumed_(CONSUMED_SEM_ID) {
    // Init the flag THIS side waits on. Counter staging needs no reset/init (monotone).
    if constexpr (STAGING == Staging::Flag) {
        data_ready_.set(INVALID);
    }
}

// receive(): [ack the sender], wait data-ready, clear the flag (clear-before-ack, H11).
template <uint32_t DATA_READY_SEM_ID, uint32_t CONSUMED_SEM_ID, Staging STAGING, bool PRE_HANDSHAKE>
void ReceiverPipe<DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE>::receive(
    uint32_t sender_x, uint32_t sender_y) {
    if constexpr (PRE_HANDSHAKE) {
        // tell the sender "my dest is free / I am ready" (remote atomic inc on its counter)
        consumed_.up(noc_, sender_x, sender_y, 1);
    }
    if constexpr (STAGING == Staging::Counter) {
        data_ready_.wait_min(++round_);
    } else {
        data_ready_.wait(VALID);
        data_ready_.set(INVALID);  // clear this round's flag; next receive()'s ack follows (H11)
    }
}

// Wait the control flag and RETURN its value. Symmetric with SenderPipe::send_signal().
template <uint32_t DATA_READY_SEM_ID, uint32_t CONSUMED_SEM_ID, Staging STAGING, bool PRE_HANDSHAKE>
uint32_t ReceiverPipe<DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING, PRE_HANDSHAKE>::receive_signal() {
    if constexpr (STAGING == Staging::Counter) {
        data_ready_.wait_min(++round_);
        return round_;
    } else {
        data_ready_.wait(VALID);
        data_ready_.set(INVALID);
        return VALID;
    }
}

}  // namespace dataflow_kernel_lib
