// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../../recipe_state_layout.hpp"

// The dataflow owner copies raw tile bytes; no unpack, arithmetic, or narrowing
// is allowed here. Each recurrent CB spans exactly one complete Q-block state.
// FP32 recipes keep O and l in the pushed prev banks. The BF16 recipes keep them in the
// cur banks' L1 (CB 9 and CB 13, popped between chunks), so only their maxima are pushed.
template <uint32_t q_tiles, uint32_t request_cb, uint32_t ack_cb, uint32_t d_tiles = 4>
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
#ifdef SDPA_RECIPE_FP32
        words[Transfer::Numerator] = state.prev.out;
        words[Transfer::Denominator] = state.prev.sum;
#else
        words[Transfer::Numerator] = state.cur.out;
        words[Transfer::Denominator] = state.cur.sum;
#endif
        words[Transfer::Maximum] = state.prev.max;
        words[Transfer::Chunks] = state.processed_chunks;
    })
    request.push_back(1);
    ack.wait_front(1);
    if (restore) {
        state.processed_chunks = ack.read_tile_value(0, Transfer::Chunks);
        CircularBuffer(state.prev.max).reserve_back(q_tiles);
        CircularBuffer(state.prev.max).push_back(q_tiles);
#ifdef SDPA_RECIPE_FP32
        CircularBuffer(state.prev.out).reserve_back(q_tiles * d_tiles);
        CircularBuffer(state.prev.sum).reserve_back(q_tiles);
        CircularBuffer(state.prev.out).push_back(q_tiles * d_tiles);
        CircularBuffer(state.prev.sum).push_back(q_tiles);
#endif
    } else {
        CircularBuffer(state.prev.max).pop_front(q_tiles);
#ifdef SDPA_RECIPE_FP32
        CircularBuffer(state.prev.out).pop_front(q_tiles * d_tiles);
        CircularBuffer(state.prev.sum).pop_front(q_tiles);
#endif
    }
    ack.pop_front(1);
    UNPACK({
        mailbox_write(ckernel::ThreadId::MathThreadId, 1);
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })
    MATH((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
}
