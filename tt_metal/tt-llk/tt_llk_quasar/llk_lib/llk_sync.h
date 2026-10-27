// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_trisc_common.h"
using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @file llk_sync.h
 * @brief Thread-agnostic semaphore primitives for LLK-level producer/consumer
 *        handshakes.
 *
 * Each helper is a thin wrapper over the underlying TT instruction
 * (SEMINIT / SEMWAIT / SEMGET / SEMPOST) with a single semaphore-index
 * argument in the in range [0, 31] range. The producer/consumer role is assigned at the
 * call site.
 *
 * Usage notes:
 *  - `sem_index` is the integer semaphore id (e.g. `semaphore::UNPACK_MATH = 4`),
 *    not the one-hot bitmask form (e.g. `p_stall::SEMAPHORE_4 = 0x10`).
 *  - `_llk_sync_wait_` blocks until the semaphore satisfies the given condition
 *    (typically `p_stall::STALL_ON_ZERO` or `p_stall::STALL_ON_MAX`). The stall
 *    resource (e.g. `STALL_UNPACK`) is a template parameter.
 *  - `_llk_sync_get_` and `_llk_sync_post_` accept optional pre-stall resources
 *    via template parameters, matching `t6_semaphore_get` / `t6_semaphore_post`.
 */

/**
 * @brief Initialize semaphore `sem_index` with the given max and initial values.
 *
 * @param sem_index Semaphore id in range [0, 7].
 * @param max       Maximum value the semaphore can hold.
 * @param init      Initial value of the semaphore.
 */
inline void _llk_sync_init_(std::uint8_t sem_index, std::uint32_t max, std::uint32_t init)
{
    TT_SEMINIT(max, init, 0, semaphore::t6_sem(sem_index));
}

/**
 * @brief Block until every listed semaphore satisfies `Condition`.
 *
 * One or more semaphore indices are combined into a single SEMWAIT mask, so a
 * multi-index wait costs one instruction (and one stall) instead of one per
 * semaphore. The shared `Condition` gates all of them: the thread is released
 * only once every selected semaphore satisfies it.
 *
 * @tparam StallRes   Resource the calling thread holds while blocked
 *                    (e.g. p_stall::STALL_UNPACK).
 * @tparam Condition  Wait predicate, typically p_stall::STALL_ON_ZERO
 *                    (wait > 0) or p_stall::STALL_ON_MAX (wait < max).
 * @param  sem_index  One or more semaphore ids. They must all live in the same
 *                    bank of 8 (sem_bank_sel = 0, i.e. indices in [0, 7]).
 */
template <std::uint32_t StallRes, std::uint32_t Condition, typename... Idx>
inline void _llk_sync_wait_(Idx... sem_index)
{
    static_assert(sizeof...(Idx) > 0, "at least one semaphore index required");
    static_assert((std::is_integral_v<Idx> && ...), "semaphore indices must be integral");
    TT_SEMWAIT(StallRes, Condition, 0, (semaphore::t6_sem(sem_index) | ...));
}

/**
 * @brief Decrement semaphore `sem_index`. Optionally stall on up to three
 *        resources first (matches `t6_semaphore_get`).
 *
 * @tparam WaitRes0  Optional first pre-stall resource (default: p_stall::NOTHING).
 * @tparam WaitRes1  Optional second pre-stall resource.
 * @tparam WaitRes2  Optional third pre-stall resource.
 * @param sem_index  Semaphore id in range [0, 31].
 */
template <std::uint32_t WaitRes0 = p_stall::NOTHING, std::uint32_t WaitRes1 = p_stall::NOTHING, std::uint32_t WaitRes2 = p_stall::NOTHING>
inline void _llk_sync_get_(std::uint8_t sem_index)
{
    t6_semaphore_get<WaitRes0, WaitRes1, WaitRes2>(sem_index);
}

/**
 * @brief Increment semaphore `sem_index`. Optionally stall on up to three
 *        resources first (matches `t6_semaphore_post`).
 *
 * @tparam WaitRes0  Optional first pre-stall resource (default: p_stall::NOTHING).
 * @tparam WaitRes1  Optional second pre-stall resource.
 * @tparam WaitRes2  Optional third pre-stall resource.
 * @param sem_index  Semaphore id in range [0, 31].
 */
template <std::uint32_t WaitRes0 = p_stall::NOTHING, std::uint32_t WaitRes1 = p_stall::NOTHING, std::uint32_t WaitRes2 = p_stall::NOTHING>
inline void _llk_sync_post_(std::uint8_t sem_index)
{
    t6_semaphore_post<WaitRes0, WaitRes1, WaitRes2>(sem_index);
}

/**
 * @brief Stall the next CFG write until the listed resources have drained.
 *
 * Use before reprogramming CFG state that in-flight PACR/UNPACR/MATH
 * instructions still depend on.
 *
 * @tparam DrainRes0  Primary resource to drain (e.g. p_stall::PACK0,
 *                    p_stall::UNPACK0, p_stall::MATH).
 * @tparam DrainRes1  Optional second drain resource.
 * @tparam DrainRes2  Optional third drain resource.
 */
template <std::uint32_t DrainRes0, std::uint32_t DrainRes1 = p_stall::NOTHING, std::uint32_t DrainRes2 = p_stall::NOTHING>
inline void _llk_stall_cfg_on_()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, DrainRes2, DrainRes1, DrainRes0);
}

/**
 * @brief Advance the calling thread's dest bank section in SyncHalf mode.
 *
 * The math thread's bank flip: after it has finished with one bank and handed
 * off via a semaphore, it must (1) flip its local dest-register offset to the
 * other bank, (2) wait for any in-flight instructions that still address the
 * old base to drain, and (3) reprogram its per-TRISC dest section base before
 * the next op issues. This helper performs those three steps in order.
 *
 * Only math-issued instructions consult the per-TRISC section base registers
 * (DEST_TARGET_REG_CFG_MATH_SEC0..3), while other threads carry their bank
 * where their engine reads it: the packer in THCON SRC_ADDR_OFFSET
 * The unpacker in UNP_DEST tile index.
 *
 * @tparam TRISC_ID       The calling thread's id (ckernel::TRISC_ID = COMPILE_FOR_TRISC:
 *                        0=unpack, 1=math, 2=pack, 3=isolate-SFPU).
 * @tparam EN_32BIT_DEST  True if the math destination register is in
 *                        Float32/Int32 (32-bit) mode, false for 16-bit.
 * @tparam DrainRes0      Primary stall resource to drain before reprogramming
 *                        CFG (e.g. p_stall::PACK0, p_stall::UNPACK0, p_stall::MATH).
 * @tparam DrainRes1      Optional second drain resource.
 * @tparam DrainRes2      Optional third drain resource.
 *
 * @note The caller must gate this on DST_SYNC_MODE == DstSync::SyncHalf
 *       the bank-toggle is a no-op in SyncFull mode.
 */
template <
    std::uint8_t TRISC_ID,
    bool EN_32BIT_DEST,
    std::uint32_t DrainRes0,
    std::uint32_t DrainRes1 = p_stall::NOTHING,
    std::uint32_t DrainRes2 = p_stall::NOTHING>
inline void _llk_sync_advance_dest_section_()
{
    _update_dest_register_offset_<EN_32BIT_DEST>();
    const std::uint32_t base_addr = _get_dest_buffer_base_();
    _llk_stall_cfg_on_<DrainRes0, DrainRes1, DrainRes2>();
    _set_dest_section_base_<TRISC_ID>(base_addr);
}

/**
 * @brief Raise the startup barrier on semaphore `sem_index` (max 1, count 0 -> 1).
 *
 * Call it on the thread that seeds the shared semaphores, after the seeds: the SEMINIT / SEMPOST pair sits behind
 * them in the same instruction stream, so a count of 1 proves the seeds have executed. The rest state is 0, so the
 * first program after reset blocks the passing thread until this runs, and @ref _llk_sync_barrier_pass_ restores 0.
 *
 * @param sem_index Semaphore id in range [0, 7].
 * @note Call once per program on the seeding thread, after the seeds, paired with exactly one
 *       @ref _llk_sync_barrier_pass_ on the passing thread. Neither helper waits for the other's previous instruction
 *       to retire, so a second pair with no DEST section between them races the first pass's SEMGET.
 */
inline void _llk_sync_barrier_raise_(std::uint8_t sem_index)
{
    _llk_sync_init_(sem_index, 1 /*max*/, 0 /*init*/);
    _llk_sync_post_(sem_index);
}

/**
 * @brief Pass the startup barrier on semaphore `sem_index`: spin until it reads 1, then consume it (back to 0).
 *
 * The spin is a RISC-side register read, not a SEMWAIT: no Tensix waiter is armed on this thread, so the raiser's
 * SEMINIT can never land on a pending wait, and every wait or post this thread issues afterwards on the seeded
 * semaphores is ordered behind their SEMINITs.
 *
 * @param sem_index Semaphore id in range [0, 7].
 * @note Call once per program on the passing thread, before its first wait or post on the seeded semaphores, paired
 *       with exactly one @ref _llk_sync_barrier_raise_ on the seeding thread.
 */
inline void _llk_sync_barrier_pass_(std::uint8_t sem_index)
{
    while (semaphore_read(sem_index) == 0)
    {
    }
    _llk_sync_get_(sem_index);
}
