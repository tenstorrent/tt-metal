// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// K split of the sliding ring wrappers (ring_joint_ksplit.hpp, compute: recipe/recipe_ring_ksplit.hpp), the writer's
// side. A sender band's compute pushes a handoff page once its unnormalized state is in L1; the writer signals the
// reducer. The reducer's writer pulls each sender's state, in band order, into the receive CBs for compute's merge. The
// state CBs sit at the same L1 offsets on every core: O rows (plane 0 of CB 9's rows), the maxima (CB max_a or max_b,
// named by the signal) and l (CB 13).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "../recipe_ring_cbs.hpp"

// The reducer's ready semaphore: each sender band adds its own bit (bit band for maxima in CB max_a, bit band + 8 for
// CB max_b), so atomic addition is an OR: no carries, no ordering between senders.
class RingKSplitHandoff {
public:
    explicit RingKSplitHandoff(uint32_t semaphore_id) : semaphore_(semaphore_id) {}
    static constexpr uint32_t kMaxBShift = 8;

    void signal(const Noc& noc, uint32_t x, uint32_t y, uint32_t band, uint32_t max_cb) {
        semaphore_.up(noc, x, y, 1u << (band + (max_cb == recipe_ring_cb::max_a ? 0 : kMaxBShift)));
    }
    // Wait for sender `band`; returns the CB holding its maxima.
    uint32_t wait(uint32_t band) const {
        WAYPOINT("KSHW");
        uint32_t value = 0;
        while (((value = semaphore_.value()) & ((1u << band) | (1u << (band + kMaxBShift)))) == 0) {
            invalidate_l1_cache();
        }
        WAYPOINT("KSHD");
        return (value & (1u << band)) != 0 ? recipe_ring_cb::max_a : recipe_ring_cb::max_b;
    }
    // Only after every sender's signal was observed; program completion separates cached runs.
    void reset_after_run() { semaphore_.set(0); }

private:
    Semaphore<> semaphore_;
};

// A K-split core's writer before its output: a sender signals its reducer (runtime args at peer_args: the reducer's
// physical x, y) and returns true (it writes no output); the reducer pulls the senders' states (peer_args: their x, y
// in band order) and returns false. O rows are o_planes * vd_tiles tiles apart in CB 9 (the BF16-state recipes' second
// plane, unless the chunk PV has its own scratch CB).
template <uint32_t Sq_chunk_t, uint32_t vd_tiles, uint32_t ksplit_count, uint32_t o_planes>
bool ring_ksplit_writer_handoff(Noc noc, uint32_t semaphore_id, uint32_t band, uint32_t peer_args) {
    RingKSplitHandoff handoff(semaphore_id);
    if (band + 1 < ksplit_count) {
        CircularBuffer staged(recipe_ring_cb::ksplit_signal);
        staged.wait_front(1);
        const auto* words = reinterpret_cast<volatile uint32_t*>(staged.get_read_ptr());
        handoff.signal(
            noc,
            get_arg_val<uint32_t>(peer_args),
            get_arg_val<uint32_t>(peer_args + 1),
            band,
            words[recipe_ring_cb::ksplit_word_max_cb]);
        noc.async_atomic_barrier();
        staged.pop_front(1);
        return true;
    }
    constexpr uint32_t o_tile_bytes = get_tile_size(recipe_ring_cb::out_b);
    constexpr uint32_t o_row_stride = o_planes * vd_tiles * o_tile_bytes;
    UnicastEndpoint sender_l1;
    CircularBuffer in_out(recipe_ring_cb::ksplit_recv_out), in_max(recipe_ring_cb::ksplit_recv_max),
        in_sum(recipe_ring_cb::ksplit_recv_sum);
    for (uint32_t sender = 0; sender + 1 < ksplit_count; ++sender) {
        const uint32_t x = get_arg_val<uint32_t>(peer_args + 2 * sender);
        const uint32_t y = get_arg_val<uint32_t>(peer_args + 2 * sender + 1);
        const uint32_t max_cb = handoff.wait(sender);
        auto pull = [&](uint32_t dst, uint32_t src, uint32_t bytes) {
            noc.async_read(sender_l1, CoreLocalMem<uint32_t>(dst), bytes, {.noc_x = x, .noc_y = y, .addr = src}, {});
        };
        in_max.reserve_back(Sq_chunk_t);
        in_sum.reserve_back(Sq_chunk_t);
        in_out.reserve_back(Sq_chunk_t * vd_tiles);
        pull(in_max.get_write_ptr(), CircularBuffer(max_cb).get_read_ptr(), Sq_chunk_t * get_tile_size(max_cb));
        pull(
            in_sum.get_write_ptr(),
            CircularBuffer(recipe_ring_cb::sum_b).get_read_ptr(),
            Sq_chunk_t * get_tile_size(recipe_ring_cb::sum_b));
        const uint32_t o_base = CircularBuffer(recipe_ring_cb::out_b).get_read_ptr();
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            pull(
                in_out.get_write_ptr() + r * vd_tiles * o_tile_bytes,
                o_base + r * o_row_stride,
                vd_tiles * o_tile_bytes);
        }
        noc.async_read_barrier();
        in_max.push_back(Sq_chunk_t);
        in_sum.push_back(Sq_chunk_t);
        in_out.push_back(Sq_chunk_t * vd_tiles);
    }
    handoff.reset_after_run();
    return false;
}
