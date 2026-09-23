// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../../recipe_state_layout.hpp"

// The dataflow owner copies raw tile bytes; no unpack, arithmetic, or narrowing
// is allowed here. Each recurrent CB spans exactly one complete Q-block state.
template <uint32_t q_tiles, uint32_t request_cb, uint32_t ack_cb>
void recipe_checkpoint(RecipeAccumulatorState& state, uint32_t slot, bool restore) {
    using Transfer = sdpa::streaming::StateTransfer;
    CircularBuffer request(request_cb), ack(ack_cb);
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK({
        t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
        t6_semaphore_get<>(semaphore::PACK_DONE);
        mailbox_write(ckernel::ThreadId::MathThreadId, 1);
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })
    MATH((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    request.reserve_back(1);
    PACK({
        auto* words = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(request_cb).fifo_wr_ptr << 4);
        words[Transfer::Operation] = restore ? Transfer::Restore : Transfer::Save;
        words[Transfer::Slot] = slot;
        words[Transfer::Numerator] = state.prev.out;
        words[Transfer::Maximum] = state.prev.max;
        words[Transfer::Denominator] = state.prev.sum;
        words[Transfer::Local] = state.cur.out;
        words[Transfer::Chunks] = state.processed_chunks;
        uint32_t flags = 0;
#ifndef SDPA_RECIPE_FP32
        for (uint32_t i = 0; i < kRecipeMaxRowGroups; ++i) {
            flags |= uint32_t(state.group_local_valid[i]) << i;
        }
#endif
        words[Transfer::ValidGroups] = flags;
    })
    request.push_back(1);
    ack.wait_front(1);
    if (restore) {
        state.processed_chunks = ack.read_tile_value(0, Transfer::Chunks);
#ifndef SDPA_RECIPE_FP32
        const uint32_t flags = ack.read_tile_value(0, Transfer::ValidGroups);
        for (uint32_t i = 0; i < kRecipeMaxRowGroups; ++i) {
            state.group_local_valid[i] = (flags >> i) & 1;
        }
#endif
        CircularBuffer(state.prev.out).reserve_back(q_tiles * 4 * sdpa_out_stride);
        CircularBuffer(state.prev.max).reserve_back(q_tiles);
        CircularBuffer(state.prev.sum).reserve_back(q_tiles * sdpa_sum_stride);
        CircularBuffer(state.prev.out).push_back(q_tiles * 4 * sdpa_out_stride);
        CircularBuffer(state.prev.max).push_back(q_tiles);
        CircularBuffer(state.prev.sum).push_back(q_tiles * sdpa_sum_stride);
    } else {
        CircularBuffer(state.prev.out).pop_front(q_tiles * 4 * sdpa_out_stride);
        CircularBuffer(state.prev.max).pop_front(q_tiles);
        CircularBuffer(state.prev.sum).pop_front(q_tiles * sdpa_sum_stride);
    }
    ack.pop_front(1);
    UNPACK({
        mailbox_write(ckernel::ThreadId::MathThreadId, 1);
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })
    MATH((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
}
