// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

/**
 * ChainConfig: Runtime args for store-and-forward chain configuration.
 * Mirrors the append_to_args() layout in ring_joint_sdpa_program_factory.cpp.
 */
struct ChainConfig {
    bool participates = false;
    bool is_injector = false;
    bool is_sink = false;
    uint32_t batch = 0;
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
    uint32_t mcast_sender_wait = 0;

    // Read 18 args in canonical order matching append_to_args()
    static ChainConfig read_from_args(uint32_t& argidx) {
        ChainConfig cfg;
        cfg.participates = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.is_injector = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.is_sink = static_cast<bool>(get_arg_val<uint32_t>(argidx++));
        cfg.batch = get_arg_val<uint32_t>(argidx++);
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
        cfg.mcast_sender_wait = get_arg_val<uint32_t>(argidx++);
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
 * - is_head_level: true = head chain (matches batch AND head), false = batch chain (matches batch only)
 *
 * @param signal_target_x/y: Where receivers send "ready" signal
 *   - Unicast mode: previous core's sender semaphore
 *   - Mcast mode: injector's sender semaphore
 * @param next_core_x/y: Where to forward data in unicast mode
 * @param mcast_start_x/y, mcast_end_x/y: Multicast rectangle bounds (injector only)
 */
template <bool mcast_enabled, bool is_head_level>
class ChainLink {
public:
    // Participation flags
    const bool is_participant;
    const bool is_injector;
    const bool is_sink;

    // Chain scope matching
    const uint32_t chain_batch;
    const uint32_t chain_head;  // Only checked when is_head_level
    const uint32_t next_core_q_chunks;

    ChainLink(
        bool is_participant,
        bool is_injector,
        bool is_sink,
        uint32_t sender_sem_addr,
        uint32_t receiver_sem_addr,
        uint32_t valid_sem_addr,
        uint32_t signal_target_x,
        uint32_t signal_target_y,
        uint32_t next_core_x,
        uint32_t next_core_y,
        uint32_t mcast_start_x,
        uint32_t mcast_start_y,
        uint32_t mcast_end_x,
        uint32_t mcast_end_y,
        uint32_t mcast_num_dests,
        uint32_t mcast_sender_wait,
        uint32_t chunk_tiles,
        uint32_t tile_bytes,
        uint32_t chain_batch,
        uint32_t chain_head,
        uint32_t next_core_q_chunks) :
        is_participant(is_participant),
        is_injector(is_injector),
        is_sink(is_sink),
        chain_batch(chain_batch),
        chain_head(chain_head),
        next_core_q_chunks(next_core_q_chunks),
        sender_sem_ptr_(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr)),
        receiver_sem_ptr_(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr)),
        valid_sem_addr_(valid_sem_addr),
        sender_sem_noc_addr_(get_noc_addr(signal_target_x, signal_target_y, sender_sem_addr)),
        receiver_sem_noc_addr_(0),
        mcast_base_noc_addr_(0),
        mcast_sem_noc_addr_(0),
        sender_wait_count_(1),
        mcast_num_dests_(mcast_num_dests),
        chunk_tiles_(chunk_tiles),
        tile_bytes_(tile_bytes),
        next_core_x_(next_core_x),
        next_core_y_(next_core_y) {
        // Initialize valid semaphore (skip if address is 0 to avoid corrupting memory)
        if (valid_sem_addr != 0) {
            volatile tt_l1_ptr uint32_t* valid_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_sem_addr);
            *valid_sem_ptr = VALID;
        }

        if constexpr (mcast_enabled) {
            if (is_injector) {
                mcast_base_noc_addr_ =
                    get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, 0);
                mcast_sem_noc_addr_ = mcast_base_noc_addr_ | receiver_sem_addr;
                sender_wait_count_ = mcast_sender_wait;
            }
        } else {
            receiver_sem_noc_addr_ = get_noc_addr(next_core_x, next_core_y, receiver_sem_addr);
        }
    }

    /**
     * Check if this core should receive data from upstream.
     * Head-level chains match (batch, head), batch-level chains match batch only.
     * In mcast mode, skip batch check: mcast is only enabled for B=1, and padded
     * iterations may have garbage nb values from out-of-bounds global_q_chunk.
     */
    bool should_receive(uint32_t nb, uint32_t nq) const {
        if (!is_participant || is_injector) {
            return false;
        }
        if constexpr (!mcast_enabled) {
            if (nb != chain_batch) {
                return false;
            }
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
     * In mcast mode, skip batch check (see should_receive comment).
     */
    bool should_forward(uint32_t nb, uint32_t nq, uint32_t q_iter_local) const {
        if (!is_participant || is_sink) {
            return false;
        }
        if constexpr (!mcast_enabled) {
            if (nb != chain_batch) {
                return false;
            }
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
    void receive() const {
        noc_semaphore_set(receiver_sem_ptr_, INVALID);
        noc_semaphore_inc(sender_sem_noc_addr_, 1);
        noc_semaphore_wait(receiver_sem_ptr_, VALID);
    }

    /**
     * Forward data to downstream link(s) using default chunk size.
     * In mcast mode: wait for all receivers, then broadcast.
     * In unicast mode: wait for next receiver, then point-to-point transfer.
     */
    void forward(uint32_t cb_addr) const { forward(cb_addr, chunk_tiles_, tile_bytes_); }

    /**
     * Forward data to downstream link(s) with explicit size.
     * Use this when the data size differs from the default (e.g., K using head chain).
     */
    void forward(uint32_t cb_addr, uint32_t num_tiles, uint32_t tile_bytes) const {
        if constexpr (mcast_enabled) {
            noc_semaphore_wait(sender_sem_ptr_, sender_wait_count_);
            noc_semaphore_set(sender_sem_ptr_, 0);
            uint64_t mcast_addr = mcast_base_noc_addr_ | cb_addr;
            noc_async_write_multicast(cb_addr, mcast_addr, num_tiles * tile_bytes, mcast_num_dests_, true);
            noc_semaphore_set_multicast(valid_sem_addr_, mcast_sem_noc_addr_, mcast_num_dests_);
            noc_async_writes_flushed();
        } else {
            noc_semaphore_wait(sender_sem_ptr_, 1);
            noc_semaphore_set(sender_sem_ptr_, 0);
            uint64_t unicast_addr = get_noc_addr(next_core_x_, next_core_y_, cb_addr);
            noc_async_write(cb_addr, unicast_addr, num_tiles * tile_bytes);
            noc_async_writes_flushed();
            noc_semaphore_set_remote(valid_sem_addr_, receiver_sem_noc_addr_);
        }
    }

private:
    // Semaphore pointers (derived from addresses in constructor)
    volatile tt_l1_ptr uint32_t* sender_sem_ptr_;
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr_;

    // Semaphore L1 address (needed for mcast set_multicast)
    uint32_t valid_sem_addr_;

    // NOC addresses (computed in constructor)
    uint64_t sender_sem_noc_addr_;
    uint64_t receiver_sem_noc_addr_;
    uint64_t mcast_base_noc_addr_;
    uint64_t mcast_sem_noc_addr_;

    // Configuration
    uint32_t sender_wait_count_;
    uint32_t mcast_num_dests_;
    uint32_t chunk_tiles_;
    uint32_t tile_bytes_;
    uint32_t next_core_x_;
    uint32_t next_core_y_;
};
