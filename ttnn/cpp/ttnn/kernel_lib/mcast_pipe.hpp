// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// mcast_pipe — `SenderPipe` / `ReceiverPipe`: a NoC-multicast + semaphore-handshake helper.
// =============================================================================
//
// Wraps the recurring dataflow block:
//
//   stage a source L1 region -> multicast a block to an exact receiver set ->
//   signal the receivers that the data is ready.
//
// Sender cores use `SenderPipe`; receiver cores use `ReceiverPipe`.
//
// Preconditions: one active sender per round; semaphores are initialized to INVALID on every
// participating core; the landing address `dst_l1` is identical across all receivers.
// =============================================================================

#pragma once

// Caller-facing API version — the staleness key for the apply-dm-helper rollout ledger
// (helper_design/mcast_pipe/migration/ledger.json). BUMP THIS (and only this) whenever a
// re-materialization changes the caller-facing API (renamed/removed type, moved param, changed
// count/flag semantics — anything that forces a call site rewrite); leave it for internal-only
// changes.
#define MCAST_PIPE_API_VERSION 22

#include "ttnn/cpp/ttnn/kernel_lib/mcast_common.hpp"

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "hostdevcommon/common_values.hpp"

namespace dataflow_kernel_lib {

// =============================================================================
// SenderPipe — the broadcasting face of the channel.
// =============================================================================
//   * NOC_ID                     — compile-time NoC id; must match the `noc` argument.
//   * DATA_READY_SEM_ID          — sender-to-receiver data-ready semaphore id.
//   * PRE_HANDSHAKE              — wait for receiver readiness before sending data or a signal.
//   * CONSUMER_READY_SEM_ID      — receiver-to-sender readiness semaphore id; required with PRE_HANDSHAKE.
//   * DATA_READY_SIGNAL          — Flag (default) or Counter.
//   * ROTATING_SENDER            — whether this core sends on some rounds and receives on others.
template <
    uint8_t NOC_ID,
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE = true,
    uint32_t CONSUMER_READY_SEM_ID = UNUSED_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag,
    bool ROTATING_SENDER = false,
    SenderTransferMode TRANSFER_MODE = SenderTransferMode::TransferModeUnknown,
    uint32_t MAX_RECTS = 1>
class SenderPipe {
    static_assert(MAX_RECTS >= 1 && MAX_RECTS <= MAX_MCAST_RECTANGLES, "Multicast supports one to three rectangles");
    static_assert(
        mcast_wire::concrete(TRANSFER_MODE) || TRANSFER_MODE == SenderTransferMode::TransferModeUnknown,
        "SenderPipe requires a concrete transfer mode or TransferModeUnknown");
    static_assert(
        !PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID,
        "PRE_HANDSHAKE=true requires a real CONSUMER_READY_SEM_ID (the receiver->sender readiness ack). "
        "Pass it, or set PRE_HANDSHAKE=false for a fire-and-forget broadcast.");

public:
    // Capture prepared values once. Argument storage may be changed or destroyed after construction.
    explicit SenderPipe(const Noc& noc, const SenderRuntimeArgumentsFor<MAX_RECTS>& runtime_args);

    // ===== DATA channel (a block + a ready signal) =====
    // send() handles receiver readiness when enabled, data multicast, ready signaling, and source L1 protection.
    // With SOURCE_GUARD=CallerManaged, the caller provides that protection, so send() may return before the NoC
    // finishes reading source L1.
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    FORCE_INLINE void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    // ===== CONTROL channel (a signal with no data block) =====
    // Handle receiver readiness when enabled, then broadcast a control signal.
    // Flag sends `value`; Counter records one event. Pairs with ReceiverPipe::receive_signal(round).
    // CallerManaged requires the caller to preserve the local Flag semaphore value until the NoC
    // has read it. Rotating Flag cleanup and Counter atomic completion remain protected in either policy.
    template <SourceL1Guard SOURCE_GUARD = SourceL1Guard::Guard>
    void send_signal(uint32_t value = VALID);

private:
    template <SenderTransferMode MODE>
    FORCE_INLINE void send_rectangle_(
        const RectangleRuntimeArguments& rectangle, uint32_t src_l1, uint32_t dst_l1, uint32_t size);
    FORCE_INLINE void send_data_(
        const RectangleRuntimeArguments& rectangle,
        bool loopback,
        uint32_t src_l1,
        uint32_t dst_l1,
        uint32_t size,
        uint32_t mcast_dests);
    FORCE_INLINE void signal_ready_(
        const RectangleRuntimeArguments& rectangle, bool loopback, uint32_t mcast_dests, uint32_t value = VALID);
    template <SourceL1Guard SOURCE_GUARD>
    FORCE_INLINE void fence_(bool loopback);
    void local_copy_(uint32_t src_l1, uint32_t dst_l1, uint32_t size);

    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    SenderRuntimeArgumentsFor<MAX_RECTS> args_;
    // Sender membership selects the payload fence even when src == dst skips a local write.
    bool loopback_ = false;
};

// =============================================================================
// ReceiverPipe — the listening face of the channel.
// =============================================================================
//   * DATA_READY_SEM_ID      — sender-to-receiver data-ready semaphore id.
//   * PRE_HANDSHAKE          — signal receiver readiness before waiting; must match the SenderPipe's.
//   * CONSUMER_READY_SEM_ID  — receiver-to-sender readiness semaphore id; required with PRE_HANDSHAKE.
//   * DATA_READY_SIGNAL      — must match the SenderPipe's.
//   * NUM_SENDERS            — number of stored sender coordinate pairs.
//
template <
    uint32_t DATA_READY_SEM_ID,
    bool PRE_HANDSHAKE = true,
    uint32_t CONSUMER_READY_SEM_ID = UNUSED_SEM_ID,
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag,
    uint32_t NUM_SENDERS = 1>
class ReceiverPipe {
    static_assert(
        !PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID,
        "PRE_HANDSHAKE=true requires a real CONSUMER_READY_SEM_ID (the receiver->sender readiness ack). "
        "Pass it, or set PRE_HANDSHAKE=false to wait the data-ready signal without acking.");
    static_assert(NUM_SENDERS >= 1, "ReceiverPipe needs at least one sender coord pair.");

public:
    // `sender_coords` contains NUM_SENDERS virtual NoC coordinate pairs and must outlive the pipe.
    explicit ReceiverPipe(const Noc& noc, const uint32_t* sender_coords);

    // Handle receiver readiness, then wait for data from the sender selected by the absolute work round.
    void receive(uint32_t round = 0);

    // Transport-compatible receive used by kernels that may also be configured for chain forwarding.
    // Hardware multicast lands the payload directly, so dst_l1 and size_bytes are intentionally ignored.
    FORCE_INLINE void receive_and_forward(uint32_t /*dst_l1*/, uint32_t /*size_bytes*/, uint32_t round = 0) {
        receive(round);
    }

    // Handle receiver readiness when enabled, then wait for a control signal.
    // Returns the Flag value or round + 1 for Counter. Pairs with SenderPipe::send_signal().
    uint32_t receive_signal(uint32_t round = 0);

private:
    Noc noc_;
    Semaphore<> data_ready_;
    Semaphore<> consumer_ready_;
    const uint32_t* coords_;  // non-owning sender coord pairs [x0,y0,...]; storage outlives this pipe
};

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl"
