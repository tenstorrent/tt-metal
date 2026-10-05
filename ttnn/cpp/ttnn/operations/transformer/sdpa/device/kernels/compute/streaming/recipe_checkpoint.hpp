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

#ifdef SDPA_RING_STREAM_STATE
// Streamed checkpoints (BF16 recipes with fused chunks; see StateTransfer and recipe_fused_chunk.hpp).
// Restore: waits only for header, maxima and sums; the fused chunk waits for each group's O rows before its PV.
template <uint32_t q_tiles, uint32_t request_cb, uint32_t ack_cb>
void recipe_checkpoint_restore_stream(RecipeAccumulatorState& state, uint32_t slot) {
    using Transfer = sdpa::streaming::StateTransfer;
    CircularBuffer request(request_cb), ack(ack_cb);
    request.reserve_back(1);
    PACK({
        auto* words = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(request_cb).fifo_wr_ptr << 4);
        words[Transfer::Operation] = Transfer::RestoreStream;
        words[Transfer::Slot] = slot;
        words[Transfer::Numerator] = state.cur.out;
        words[Transfer::Denominator] = state.cur.sum;
        words[Transfer::Maximum] = state.prev.max;
    })
    request.push_back(1);
    ack.wait_front(1);
    state.processed_chunks = ack.read_tile_value(0, Transfer::Chunks);
    CircularBuffer(state.prev.max).reserve_back(q_tiles);
    CircularBuffer(state.prev.max).push_back(q_tiles);
    ack.pop_front(1);
    recipe_stream_sync_threads();
    recipe_stream_restored_rows = 0;
}

// Save tail: O rows not yet handed over, maxima, sums and header. Before a restore compute does not wait for it
// (the writer flushes it out of the banks before restoring into them); before a block that starts without a
// restore (the first ring iteration) it waits until the bytes have left L1 (wait_flushed).
template <uint32_t q_tiles, uint32_t request_cb, uint32_t ack_cb>
void recipe_checkpoint_save_tail(RecipeAccumulatorState& state, uint32_t slot, bool wait_flushed) {
    recipe_stream_wait_rows(q_tiles);  // a restore's acks never outlive its block
    CircularBuffer request(request_cb);
    request.reserve_back(1);
    PACK({
        using Transfer = sdpa::streaming::StateTransfer;
        auto* words = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(request_cb).fifo_wr_ptr << 4);
        words[Transfer::Operation] = Transfer::SaveTail;
        words[Transfer::Slot] = slot;
        words[Transfer::Numerator] = state.cur.out;
        words[Transfer::Denominator] = state.cur.sum;
        words[Transfer::Maximum] = state.prev.max;
        words[Transfer::Chunks] = state.processed_chunks;
        words[Transfer::Row0] = recipe_stream_saved_rows;
        words[Transfer::Rows] = wait_flushed ? 1 : 0;
    })
    request.push_back(1);  // lands after every pack of this block (maxima, folded sums, O)
    if (wait_flushed) {
        CircularBuffer ack(ack_cb);
        ack.wait_front(1);
        ack.pop_front(1);
        recipe_stream_sync_threads();
    }
    CircularBuffer(state.prev.max).pop_front(q_tiles);
    recipe_stream_save_slot = kNoStreamSlot;
    recipe_stream_saved_rows = 0;
}
#endif
