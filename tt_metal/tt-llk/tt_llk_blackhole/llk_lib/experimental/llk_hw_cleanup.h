// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_assert.h"

namespace ckernel
{

constexpr std::uint32_t HW_CLEANUP_CANONICAL_TILE_SIZE_BYTES = 2048;
constexpr std::uint32_t HW_CLEANUP_CANONICAL_TILE_SIZE_16B   = HW_CLEANUP_CANONICAL_TILE_SIZE_BYTES / 16;

namespace hw_cleanup
{

// Software-only mailbox payloads. Cleanup uses T0/T1/T2 mailboxes and does not
// borrow a Tensix protocol semaphore.
//
// Protocol (Math coordinates; Config RAM is thread-shared so configure is
// serialized T0 → T1 → T2). CFG_STATE_ID itself is per-thread, so bank flips do
// not need a separate lock:
//
//   Unpack/Pack: drain → READY → wait CONFIGURE → <caller configures> →
//                drain → CONFIGURED → wait DONE
//   Math:        drain → wait both READY → grant Unpack and wait CONFIGURED →
//                <caller configures> → grant Pack and wait CONFIGURED → DONE
constexpr std::uint32_t UNPACK_READY      = 0x434C4E01;
constexpr std::uint32_t PACK_READY        = 0x434C4E02;
constexpr std::uint32_t UNPACK_CONFIGURE  = 0x434C4E21;
constexpr std::uint32_t UNPACK_CONFIGURED = 0x434C4E22;
constexpr std::uint32_t PACK_CONFIGURE    = 0x434C4E31;
constexpr std::uint32_t PACK_CONFIGURED   = 0x434C4E32;
constexpr std::uint32_t CLEANUP_DONE      = 0x434C4E40;

inline void select_cfg_state(const std::uint32_t state_id)
{
    LLK_ASSERT(state_id <= 1, "Blackhole cleanup configuration state must be zero or one.");

    if (cfg_state_id != state_id)
    {
        flip_cfg_state_id();
        tensix_sync();
    }
}

// Mailbox reads are bare volatile loads. Fence so later config / CLEANUP_DONE
// writes cannot race ahead when the payload is only checked under LLK_ASSERT.
inline void mailbox_fence()
{
    asm volatile("fence" ::: "memory");
}

/**
 * Enter cleanup and wait until this thread owns the configure turn.
 *
 * @note Drain every operation-owned message from the incoming mailboxes before calling this.
 * @note On return the pipelines are drained and Unpack/Pack may configure; Math has already
 *       waited for Unpack to finish configuring.
 */
template <ThreadId thread_id>
inline void start()
{
    static_assert(IS_TRISC_THREAD<thread_id>, "Hardware cleanup requires a TRISC thread.");

    mop_sync();
    tensix_sync();

    if constexpr (thread_id == UnpackThreadId)
    {
        mailbox_write(MathThreadId, UNPACK_READY);
        const std::uint32_t configure = mailbox_read(MathThreadId);
        mailbox_fence();
        LLK_ASSERT(configure == UNPACK_CONFIGURE, "Unexpected unpack cleanup configuration grant.");
    }
    else if constexpr (thread_id == PackThreadId)
    {
        mailbox_write(MathThreadId, PACK_READY);
        const std::uint32_t configure = mailbox_read(MathThreadId);
        mailbox_fence();
        LLK_ASSERT(configure == PACK_CONFIGURE, "Unexpected pack cleanup configuration grant.");
    }
    else
    {
        const std::uint32_t unpack_ready = mailbox_read(UnpackThreadId);
        const std::uint32_t pack_ready   = mailbox_read(PackThreadId);
        mailbox_fence();
        LLK_ASSERT(unpack_ready == UNPACK_READY, "Unexpected cleanup message from unpack thread.");
        LLK_ASSERT(pack_ready == PACK_READY, "Unexpected cleanup message from pack thread.");

        // Unpack configures while Math blocks on CONFIGURED.
        mailbox_write(UnpackThreadId, UNPACK_CONFIGURE);
        const std::uint32_t configured = mailbox_read(UnpackThreadId);
        mailbox_fence();
        LLK_ASSERT(configured == UNPACK_CONFIGURED, "Unexpected unpack cleanup configuration completion.");
    }
}

/**
 * Leave the configure turn and wait until every TRISC has finished cleanup.
 *
 * @note Finish this thread's configuration (including any mop/tensix side effects) before
 *       calling this.
 * @note On return cfg bank 0 is selected, and no thread returns before all three have
 *       configured.
 */
template <ThreadId thread_id>
inline void finish()
{
    static_assert(IS_TRISC_THREAD<thread_id>, "Hardware cleanup requires a TRISC thread.");
    LLK_ASSERT(cfg_state_id == 0, "Blackhole cleanup must exit with configuration state zero selected.");

    // Do not hand off while buffered MOP / Tensix config work could overlap the
    // next owner (or the caller's return into the next MicroOp).
    mop_sync();
    tensix_sync();

    if constexpr (thread_id == UnpackThreadId)
    {
        mailbox_write(MathThreadId, UNPACK_CONFIGURED);
        const std::uint32_t done = mailbox_read(MathThreadId);
        mailbox_fence();
        LLK_ASSERT(done == CLEANUP_DONE, "Unexpected unpack cleanup completion.");
    }
    else if constexpr (thread_id == PackThreadId)
    {
        mailbox_write(MathThreadId, PACK_CONFIGURED);
        const std::uint32_t done = mailbox_read(MathThreadId);
        mailbox_fence();
        LLK_ASSERT(done == CLEANUP_DONE, "Unexpected pack cleanup completion.");
    }
    else
    {
        // Pack configures while Math blocks on CONFIGURED.
        mailbox_write(PackThreadId, PACK_CONFIGURE);
        const std::uint32_t configured = mailbox_read(PackThreadId);
        mailbox_fence();
        LLK_ASSERT(configured == PACK_CONFIGURED, "Unexpected pack cleanup configuration completion.");

        mailbox_write(UnpackThreadId, CLEANUP_DONE);
        mailbox_write(PackThreadId, CLEANUP_DONE);
    }
}

} // namespace hw_cleanup

} // namespace ckernel
