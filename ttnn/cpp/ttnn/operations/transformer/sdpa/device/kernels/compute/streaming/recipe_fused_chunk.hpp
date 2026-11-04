// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// STANDARD's and FAST's fused K chunk (sdpa_fused_chunk): this header holds its state; the chunk itself is
// recipe_fused_chunk.inc, which recipe_step.inc builds after the helpers it uses (reduce_c_row_group,
// sub_exp_block_bcast_cols, normalize_row_streaming, blocked_matmul_and_pack) and before sdpa_inner_loop_step, which
// dispatches to it.
//
// Code size. Every recipe kernel must fit the 70656 B kernel config buffer, and the fused chunk is the largest
// addition. What keeps the ring and exp ring builds inside it:
//   - recipe_cold_step (recipe_step.inc): with fused chunks, the reduce path (a Q chunk's first K chunk, redone
//     groups), normalization and sdpa_inner_loop_step are out of line and -Os.
//   - fused_os (recipe_step.inc): the ring, key-range and Wormhole kernels' unpack copy of this chunk is -Os (unpack
//     only issues and waits here, and its image is the one nearest the limit).
//   - The ring and exp ring recipe factories build fused kernels at -O2 (ring_joint_sdpa_recipe.cpp /
//     exp_ring_joint_sdpa_recipe.cpp build math at -O2 everywhere).
//   - The ring and exp ring reader/writer dataflow kernels are -Os.
// The dense kernel fits at its default -O3.

#pragma once

#include "../../recipe_state_layout.hpp"
#include "recipe_config.hpp"

/**
 * Fused STANDARD / FAST K chunk (every chunk after a Q chunk's first). The reference max m_ref is already
 * known, so the row max is not needed before the exp:
 *   - QK accumulates s - m_ref straight into DEST through one extra inner step, [Q | M] x [K^T ; -e0]
 *     (M: the reference-max tile, m_ref in column 0; -e0 in K's format). The pack thread takes the fast exp
 *     in place and packs P once, L1-accumulating it onto QK-subblock-width partial row sums.
 *   - Row groups are software-pipelined: QK/exp of group g, the saturation check of group g - 1 and the
 *     PV of group g - 2 are interleaved piece by piece, so the FPU's PV overlaps the pack thread's exp.
 *   - Saturation check: a group whose chunk row sums reach kFusedRedoSum may have saturated the exp (P
 *     saturates near 1.66 at this scale). Rows are independent, so only that group is redone on the
 *     reduce path (real max, theta select, rescale of its O and l rows); every other group keeps m_ref.
 * Same CB protocol as sdpa_inner_loop_step: scores, chunk sums, O and l banks, the max ping-pong.
 * Built only with recipe::fused_active.
 */
// Wormhole unpack/pack: the chunk's steps are inlined. Called out of line, each step's closure (one pointer per
// captured variable, ~160 B in all) and the captured variables themselves sit on the 2 KiB TRISC stack. (Lambda
// attributes for one architecture's TRISCs: only the preprocessor selects them.)
#if defined(ARCH_WORMHOLE) && (defined(TRISC_UNPACK) || defined(TRISC_PACK))
#define SDPA_FUSED_STEP __attribute__((always_inline))
// The rare redo step too: it has one call site, and out of line its closure captures almost every local.
#define SDPA_FUSED_REDO __attribute__((always_inline))
#else
#define SDPA_FUSED_STEP
#define SDPA_FUSED_REDO SDPA_RECIPE_COLD
#endif
static bool fused_neg_unit_ready = false;
// Streamed ring checkpoints (recipe_ring.hpp, StateTransfer): a restored Q block's O rows land row by row while
// its first fused chunk runs, and the last fused chunk of a saved block hands its O rows to the writer group by
// group. kNoStreamSlot: no save in flight; restored rows >= every row end: no restore in flight.
constexpr uint32_t kStreamRequestCb = 17;
constexpr uint32_t kStreamAckCb = 18;
constexpr uint32_t kNoStreamSlot = 0xffffffffu;
static uint32_t recipe_stream_save_slot = kNoStreamSlot;
static uint32_t recipe_stream_saved_rows = 0;
static uint32_t recipe_stream_restored_rows = 0xffffffffu;
// Out of line throughout: the ring kernels sit at the kernel config buffer limit.
// UNPACK tells MATH and PACK it has passed a handshake (the threads leave it together).
static __attribute__((noinline)) void recipe_stream_sync_threads() {
    UNPACK({
        mailbox_write(ckernel::ThreadId::MathThreadId, 1);
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })
    MATH((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
}
static __attribute__((noinline)) void recipe_stream_wait_rows_slow(uint32_t row_end) {
    UNPACK({
        for (uint32_t r = recipe_stream_restored_rows; r < row_end; ++r) {
            llk_wait_tiles(kStreamAckCb, 1);
            llk_pop_tiles(kStreamAckCb, 1);
        }
        mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    })
    PACK((void)mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    recipe_stream_restored_rows = row_end;
}
// O rows [0, row_end) must have landed before any thread touches them (PACK accumulates PV onto them, UNPACK
// reads them in a redo). Every thread runs this with the same counters, so the mailbox calls pair up.
ALWI void recipe_stream_wait_rows(uint32_t row_end) {
    if (recipe_stream_restored_rows < row_end) {
        recipe_stream_wait_rows_slow(row_end);
    }
}
// Hand O rows [saved, row_end) of the block being saved to the writer (the push lands after their packs).
static __attribute__((noinline)) void recipe_stream_save_rows(uint32_t numerator_cb, uint32_t row_end) {
    CircularBuffer request(kStreamRequestCb);
    request.reserve_back(1);
    PACK({
        using Transfer = sdpa::streaming::StateTransfer;
        auto* words = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(kStreamRequestCb).fifo_wr_ptr << 4);
        words[Transfer::Operation] = Transfer::SaveRows;
        words[Transfer::Slot] = recipe_stream_save_slot;
        words[Transfer::Numerator] = numerator_cb;
        words[Transfer::Row0] = recipe_stream_saved_rows;
        words[Transfer::Rows] = row_end - recipe_stream_saved_rows;
    })
    request.push_back(1);
    recipe_stream_saved_rows = row_end;
}
