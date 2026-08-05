// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"

/**
 * ChainConfig: Runtime args for store-and-forward chain configuration.
 * Mirrors the append_to_args() layout in ring_joint_sdpa_program_factory.cpp.
 */
struct ChainConfig {
    static constexpr uint32_t kRuntimeArgCount =
        ttnn::operations::transformer::sdpa::ring_joint::kChainConfigRuntimeArgCount;
    static_assert(kRuntimeArgCount == 16, "ChainConfig::read_from_args must match the shared runtime arg layout");

    bool participates = false;
    bool is_injector = false;
    bool is_sink = false;
    uint32_t head = 0;
    uint32_t prev_physical_x = 0;
    uint32_t prev_physical_y = 0;
    uint32_t next_physical_x = 0;
    uint32_t next_physical_y = 0;
    uint32_t next_core_q_chunks = 0;
    uint32_t mcast_start_x = 0;
    uint32_t mcast_start_y = 0;
    uint32_t mcast_end_x = 0;
    uint32_t mcast_end_y = 0;
    uint32_t injector_physical_x = 0;
    uint32_t injector_physical_y = 0;
    uint32_t mcast_num_dests = 0;

    // Read kRuntimeArgCount args in canonical order matching append_to_args().
    static ChainConfig read_from_args(uint32_t& argidx) {
        ChainConfig cfg;
        cfg.participates = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.is_injector = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.is_sink = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.head = get_arg_val<uint32_t>(argidx++);
        cfg.prev_physical_x = get_arg_val<uint32_t>(argidx++);
        cfg.prev_physical_y = get_arg_val<uint32_t>(argidx++);
        cfg.next_physical_x = get_arg_val<uint32_t>(argidx++);
        cfg.next_physical_y = get_arg_val<uint32_t>(argidx++);
        cfg.next_core_q_chunks = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_start_x = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_start_y = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_end_x = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_end_y = get_arg_val<uint32_t>(argidx++);
        cfg.injector_physical_x = get_arg_val<uint32_t>(argidx++);
        cfg.injector_physical_y = get_arg_val<uint32_t>(argidx++);
        cfg.mcast_num_dests = get_arg_val<uint32_t>(argidx++);
        return cfg;
    }

    // Compute signal target based on mcast mode
    template <bool mcast_enabled>
    uint32_t signal_target_x() const {
        return mcast_enabled ? injector_physical_x : prev_physical_x;
    }

    template <bool mcast_enabled>
    uint32_t signal_target_y() const {
        return mcast_enabled ? injector_physical_y : prev_physical_y;
    }
};

/**
 * ChainLink: Unified abstraction for store-and-forward chaining.
 *
 * Each core in a chain is a "link" that can receive from upstream and forward downstream.
 *
 * Template parameters:
 * - mcast_enabled: selects multicast vs unicast forwarding at compile time
 * - is_head_level: true = head chain (matches head), false = shared-K chain
 *
 * Constructor stores semaphore IDs and target coords.
 *
 * For non-participating cores the semaphores are not used; the valid semaphore is only
 * initialized when is_participant=true.
 */
template <bool mcast_enabled, bool is_head_level>
class ChainLink {
public:
    // Participation flags
    const bool is_participant;
    const bool is_injector;
    const bool is_sink;

    // Chain scope matching
    const uint32_t chain_head;  // Only checked when is_head_level
    const uint32_t next_core_q_chunks;

    ChainLink(
        bool is_participant,
        bool is_injector,
        bool is_sink,
        uint32_t sender_sem_id,
        uint32_t receiver_sem_id,
        uint32_t valid_sem_id,
        uint32_t signal_target_x,
        uint32_t signal_target_y,
        uint32_t next_core_x,
        uint32_t next_core_y,
        uint32_t mcast_start_x,
        uint32_t mcast_start_y,
        uint32_t mcast_end_x,
        uint32_t mcast_end_y,
        uint32_t mcast_num_dests,
        uint32_t chunk_tiles,
        uint32_t tile_bytes,
        uint32_t chain_head,
        uint32_t next_core_q_chunks) :
        is_participant(is_participant),
        is_injector(is_injector),
        is_sink(is_sink),
        chain_head(chain_head),
        next_core_q_chunks(next_core_q_chunks),
        sender_sem_id_(sender_sem_id),
        receiver_sem_id_(receiver_sem_id),
        valid_sem_id_(valid_sem_id),
        signal_target_x_(signal_target_x),
        signal_target_y_(signal_target_y),
        next_core_x_(next_core_x),
        next_core_y_(next_core_y),
        mcast_start_x_(mcast_start_x),
        mcast_start_y_(mcast_start_y),
        mcast_end_x_(mcast_end_x),
        mcast_end_y_(mcast_end_y),
        sender_wait_count_((mcast_enabled && is_injector) ? mcast_num_dests : 1),
        mcast_num_dests_(mcast_num_dests),
        chunk_tiles_(chunk_tiles),
        tile_bytes_(tile_bytes) {
        // Initialize valid semaphore (only meaningful for participants; non-participants leave it alone)
        if (is_participant) {
            Semaphore<>(valid_sem_id_).set(VALID);
        }
    }

    /**
     * Check if this core should receive data from upstream.
     * Head-level chains match the head; shared-K chains apply to every iteration.
     */
    bool should_receive(uint32_t nq) const {
        if (!is_participant || is_injector) {
            return false;
        }
        if constexpr (is_head_level) {
            if (nq != chain_head) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if this core should forward data to downstream.
     * Also checks iteration count against next core's expected reads.
     */
    bool should_forward(uint32_t nq, uint32_t q_iter_local) const {
        if (!is_participant || is_sink) {
            return false;
        }
        if constexpr (is_head_level) {
            if (nq != chain_head) {
                return false;
            }
        }
        if (q_iter_local >= next_core_q_chunks) {
            return false;
        }
        return true;
    }

    /**
     * Receive data from upstream link (called by non-injector participants).
     * Protocol: signal sender that we're ready, then wait for data.
     */
    void receive(Noc noc) const {
        Semaphore<> receiver_sem(receiver_sem_id_);
        receiver_sem.set(INVALID);
        Semaphore<>(sender_sem_id_).up(noc, signal_target_x_, signal_target_y_, 1);
        receiver_sem.wait(VALID);
    }

    /**
     * Forward data to downstream link(s) using default chunk size.
     * In mcast mode: wait for all receivers, then broadcast.
     * In unicast mode: wait for next receiver, then point-to-point transfer.
     */
    void forward(Noc noc, uint32_t cb_addr) const { forward(noc, cb_addr, chunk_tiles_, tile_bytes_); }

    /**
     * Forward data to downstream link(s) with explicit size.
     * Use this when the data size differs from the default (e.g., K using head chain).
     */
    void forward(Noc noc, uint32_t cb_addr, uint32_t num_tiles, uint32_t tile_bytes) const {
        Semaphore<> sender_sem(sender_sem_id_);
        if constexpr (mcast_enabled) {
            sender_sem.wait(sender_wait_count_);
            sender_sem.set(0);
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(cb_addr),
                MulticastEndpoint{},
                num_tiles * tile_bytes,
                mcast_num_dests_,
                {},
                {.noc_x_start = mcast_start_x_,
                 .noc_y_start = mcast_start_y_,
                 .noc_x_end = mcast_end_x_,
                 .noc_y_end = mcast_end_y_,
                 .addr = cb_addr},
                /*linked=*/true);
            // Must be back-to-back after the linked data write — any flush/barrier between them
            // deadlocks the linked transaction.
            Semaphore<>(valid_sem_id_)
                .relay_multicast(
                    noc,
                    Semaphore<>(receiver_sem_id_),
                    mcast_start_x_,
                    mcast_start_y_,
                    mcast_end_x_,
                    mcast_end_y_,
                    mcast_num_dests_,
                    /*linked=*/false);
            noc.async_writes_flushed();
        } else {
            sender_sem.wait(1);
            sender_sem.set(0);
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_addr),
                UnicastEndpoint{},
                num_tiles * tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = next_core_x_, .noc_y = next_core_y_, .addr = cb_addr});
            noc.async_writes_flushed();
            Semaphore<>(valid_sem_id_).relay_unicast(noc, Semaphore<>(receiver_sem_id_), next_core_x_, next_core_y_);
        }
    }

private:
    // Semaphore IDs (resolved to L1 addresses on use via Semaphore<>).
    uint32_t sender_sem_id_;
    uint32_t receiver_sem_id_;
    uint32_t valid_sem_id_;

    // Remote coordinates, NoC arithmetic happens per-call
    uint32_t signal_target_x_;  // Upstream sender semaphore target
    uint32_t signal_target_y_;
    uint32_t next_core_x_;  // Downstream unicast target
    uint32_t next_core_y_;

    // Multicast rectangle
    uint32_t mcast_start_x_;
    uint32_t mcast_start_y_;
    uint32_t mcast_end_x_;
    uint32_t mcast_end_y_;

    // Configuration
    uint32_t sender_wait_count_;
    uint32_t mcast_num_dests_;
    uint32_t chunk_tiles_;
    uint32_t tile_bytes_;
};
