// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_defs.h"

using namespace ckernel;
using namespace ckernel::trisc;

// =============================================================================
// llk_sync.h — Quasar DEST-bank semaphore primitives (Layer 1)
//
// Replaces the DEST DVALID synchronization scheme on Quasar with a three-
// semaphore SW protocol:
//
//   PACK -- DEST_FREE --> UNPACK -- UNPACK_MATH --> MATH -- MATH_PACK --> PACK
//
// Each primitive issues exactly one Tensix SEM instruction (or a tightly-
// coupled STALLWAIT + SEM pair). Single-producer / single-consumer per
// semaphore; UNPACK bootstraps DEST_FREE at init as the one documented
// exception (PACK is the steady-state producer thereafter).
//
// HW access_id auto-rotation (UNP_DEST_DVALID_CTRL / MATH_DEST_DVALID_CTRL
// toggle_mask = 1, wait_mask = 0) is programmed elsewhere
// (_llk_unpack_to_dest_hw_configure_ — Task 2.3); this header is pure SW sync.
//
// See docs/superpowers/specs/2026-05-26-unpack-to-dest-semaphores-design.md
// =============================================================================

// =============================================================================
// Init
// =============================================================================

/**
 * @brief UNPACK: initialize the DEST-bank semaphores owned/bootstrapped by UNPACK.
 *
 * UNPACK SEMINITs UNPACK_MATH (which UNPACK produces) and bootstraps DEST_FREE so
 * the first UNPACK acquire does not block on PACK. After bootstrap, PACK becomes
 * the steady-state producer of DEST_FREE via SEMPOST in _llk_sync_pack_release_dest_.
 *
 * DEST_FREE / UNPACK_MATH max values:
 *   - DstSync::SyncFull -> 1 (single bank in use)
 *   - DstSync::SyncHalf -> 2 (two banks alternating)
 *
 * @tparam DST_SYNC_MODE Dest-bank synchronization mode (SyncFull or SyncHalf).
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_sync_unpack_init_dest_sems_()
{
    static_assert(
        DST_SYNC_MODE == DstSync::SyncFull || DST_SYNC_MODE == DstSync::SyncHalf, "DST_SYNC_MODE must be DstSync::SyncFull or DstSync::SyncHalf");

    constexpr std::uint32_t MAX = (DST_SYNC_MODE == DstSync::SyncFull) ? 1u : 2u;

    // UNPACK bootstrap: PACK is the steady-state producer of DEST_FREE, but
    // we seed DEST_FREE = MAX here so UNPACK's first SEMWAIT passes.
    TTI_SEMINIT(MAX, MAX, 0, semaphore::t6_sem(semaphore::DEST_FREE));
    TTI_SEMINIT(MAX, 0, 0, semaphore::t6_sem(semaphore::UNPACK_MATH));
}

/**
 * @brief MATH: initialize the DEST-bank semaphores owned by MATH.
 *
 * MATH SEMINITs MATH_PACK (which MATH produces).
 *
 * @tparam DST_SYNC_MODE Dest-bank synchronization mode (SyncFull or SyncHalf).
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_sync_math_init_dest_sems_()
{
    static_assert(
        DST_SYNC_MODE == DstSync::SyncFull || DST_SYNC_MODE == DstSync::SyncHalf, "DST_SYNC_MODE must be DstSync::SyncFull or DstSync::SyncHalf");

    constexpr std::uint32_t MAX = (DST_SYNC_MODE == DstSync::SyncFull) ? 1u : 2u;

    TTI_SEMINIT(MAX, 0, 0, semaphore::t6_sem(semaphore::MATH_PACK));
}

// PACK has no SEMINITs to issue (pure consumer of MATH_PACK; steady-state
// producer of DEST_FREE — which UNPACK bootstrap-inits).
// No init function exposed; calling site simply omits the PACK init call.

// =============================================================================
// UNPACK side
// =============================================================================

/**
 * @brief UNPACK: wait for a free DEST bank and claim it.
 *
 * SEMWAIT DEST_FREE >= 1, then SEMGET (decrement). Stalls the UNPACK thread
 * (STALL_TDMA — UNPACK is on the TDMA pipe) until PACK (or the UNPACK bootstrap)
 * has posted DEST_FREE.
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_sync_unpack_acquire_dest_()
{
    static_assert(
        DST_SYNC_MODE == DstSync::SyncFull || DST_SYNC_MODE == DstSync::SyncHalf, "DST_SYNC_MODE must be DstSync::SyncFull or DstSync::SyncHalf");

    TTI_SEMWAIT(p_stall::STALL_TDMA, p_stall::STALL_ON_ZERO, 0, semaphore::t6_sem(semaphore::DEST_FREE));
    TTI_SEMGET(0, semaphore::t6_sem(semaphore::DEST_FREE));
}

/**
 * @brief UNPACK: signal MATH that the just-claimed bank is filled.
 *
 * SEMPOST UNPACK_MATH. Pair with _llk_sync_math_acquire_dest_ on the MATH side.
 */
inline void _llk_sync_unpack_commit_dest_()
{
    TTI_SEMPOST(0, semaphore::t6_sem(semaphore::UNPACK_MATH));
}

// =============================================================================
// MATH side
// =============================================================================

/**
 * @brief MATH: wait for UNPACK to fill a bank, claim it.
 *
 * SEMWAIT UNPACK_MATH >= 1, then SEMGET. Stalls on STALL_MATH | STALL_SFPU
 * | STALL_SYNC so any outstanding matrix-unit / SFPU work drains before MATH
 * starts operating on the newly-filled bank.
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_sync_math_acquire_dest_()
{
    static_assert(
        DST_SYNC_MODE == DstSync::SyncFull || DST_SYNC_MODE == DstSync::SyncHalf, "DST_SYNC_MODE must be DstSync::SyncFull or DstSync::SyncHalf");

    TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO, 0, semaphore::t6_sem(semaphore::UNPACK_MATH));
    TTI_SEMGET(0, semaphore::t6_sem(semaphore::UNPACK_MATH));
}

/**
 * @brief MATH: signal PACK that the bank is ready to be consumed.
 *
 * SEMPOST MATH_PACK. Pair with _llk_sync_pack_wait_for_dest_filled_ on PACK.
 *
 * Flow-A caveat (unpack-to-dest with no matrix-unit MOPs): the HW
 * MATH_DEST_access_id auto-rotation (toggle_mask=1, programmed in Task 2.3)
 * relies on an instruction tagged last=1 to fire. When MATH is a pure forwarder
 * (no MOPs this section), MATH's view of access_id will not advance, and on the
 * second section MATH will read the wrong bank.
 *
 * TODO(unpack-to-dest semaphores): explicit MATH_DEST_access_id toggle to keep
 * the HW auto-rotation aligned in Flow A (no matrix-unit MOPs). Resolve at
 * implementation time by reading MATH_DEST_DVALID_CTRL bit layout in
 * cfg_defines.h (addr 237) or Confluence. The MATH_DEST_DVALID_CTRL register at
 * ADDR32 = 237 has bit fields:
 *   - wait_mask     [3:0]   = MATH_DEST_DVALID_CTRL_wait_mask
 *   - wait_polarity [7:4]   = MATH_DEST_DVALID_CTRL_wait_polarity
 *   - toggle_mask   [11:8]  = MATH_DEST_DVALID_CTRL_toggle_mask
 *   - disable_auto_bank_id_toggle [12] = MATH_DEST_DVALID_CTRL_disable_auto_bank_id_toggle
 * The access_id state register (separate from this CTRL) is not yet identified
 * — the bit layout for a manual write needs HW docs. Until resolved,
 * transpose-only (Flow B) kernels will work but datacopy-only (Flow A) kernels
 * may desync on the second section.
 */
inline void _llk_sync_math_commit_dest_()
{
    TTI_SEMPOST(0, semaphore::t6_sem(semaphore::MATH_PACK));
}

// =============================================================================
// PACK side
// =============================================================================

/**
 * @brief PACK: wait for MATH to fill a bank, claim it.
 *
 * SEMWAIT MATH_PACK >= 1, then SEMGET. Stalls the TDMA pipe so any outstanding
 * pack operation drains before claiming the next bank.
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_sync_pack_wait_for_dest_filled_()
{
    static_assert(
        DST_SYNC_MODE == DstSync::SyncFull || DST_SYNC_MODE == DstSync::SyncHalf, "DST_SYNC_MODE must be DstSync::SyncFull or DstSync::SyncHalf");

    TTI_SEMWAIT(p_stall::STALL_TDMA, p_stall::STALL_ON_ZERO, 0, semaphore::t6_sem(semaphore::MATH_PACK));
    TTI_SEMGET(0, semaphore::t6_sem(semaphore::MATH_PACK));
}

/**
 * @brief PACK: release the just-consumed DEST bank back to UNPACK.
 *
 * Bitmode-dispatched: the 16-bit path is the legacy ZEROACC + SEMPOST sequence
 * (replacing the old DVALID clear); the 32-bit path skips STALLWAIT/ADDR_MOD_7/
 * ZEROACC (Approach G: those are HW-incompatible with the 32-bit dual-row dest
 * path) and just SEMPOSTs.
 *
 * Steady-state producer of DEST_FREE.
 *
 * @tparam PACK_SEL             Which PACK to drive (p_pacr::PACK0 or PACK1).
 *                              Forward-compat: PACK1 isn't used yet, but the
 *                              parameter matches the sibling helper.
 * @tparam is_fp32_dest_acc_en  true  -> 32-bit unpack-to-dest path (Approach G).
 *                              false -> 16-bit legacy path.
 * @tparam DST_SYNC_MODE        Dest-bank synchronization mode.
 */
template <std::uint32_t PACK_SEL, bool is_fp32_dest_acc_en, DstSync DST_SYNC_MODE>
inline void _llk_sync_pack_release_dest_()
{
    static_assert(
        DST_SYNC_MODE == DstSync::SyncFull || DST_SYNC_MODE == DstSync::SyncHalf, "DST_SYNC_MODE must be DstSync::SyncFull or DstSync::SyncHalf");

    if constexpr (is_fp32_dest_acc_en)
    {
        // 32-bit unpack-to-dest path. Approach G (per prior debug memory):
        // STALLWAIT(STALL_MATH,...,PACK), ADDR_MOD_7, and ZEROACC are
        // HW-incompatible with the 32-bit dual-row dest path. Skip them and
        // SEMPOST DEST_FREE directly; advance the PACK-side bank pointer via
        // DEST_TARGET_REG_CFG_MATH_SEC2_Offset in SyncHalf mode.
        //
        // SEC2 is the PACK TRISC's dest section register and is specific to
        // PACK0 in the current architecture, so the 32-bit path is PACK0-only
        // until a PACK1 equivalent is wired up.
        static_assert(PACK_SEL == p_pacr::PACK0, "32-bit pack-release currently only supports PACK0 (SEC2 is PACK0-specific)");

        TTI_SEMPOST(0, semaphore::t6_sem(semaphore::DEST_FREE));

        if constexpr (DST_SYNC_MODE == DstSync::SyncHalf)
        {
            _update_dest_register_offset_<true /*EN_32BIT_DEST*/>();
            _set_dest_section_base_<2 /*PACK TRISC*/>(_get_dest_buffer_base_());
        }
    }
    else
    {
        // 16-bit path: same STALLWAIT + ADDR_MOD_7 + ZEROACC + dest-offset update as
        // the legacy 16-bit _llk_pack_dest_section_done_ path, but the trailing
        // SEMGET(MATH_PACK) is replaced by SEMPOST(DEST_FREE). MATH_PACK is already
        // consumed by _llk_sync_pack_wait_for_dest_filled_; PACK's job at release is
        // to credit UNPACK with a free bank.
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::NOTHING, p_stall::PACK); // wait for pack to finish

        // TODO: (RT) Addrmod here is dangerous, can be overwritten by other pack operations
        //  Need to pick a addrmod, and assert no other math uses it
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_7);

        if constexpr (DST_SYNC_MODE == DstSync::SyncFull)
        {
            TTI_ZEROACC(p_zeroacc::CLR_ALL, false /*EN_32BIT_DEST*/, 0, ADDR_MOD_7, 0);
        }
        else
        {
            static_assert(DST_SYNC_MODE == DstSync::SyncHalf);
            TT_ZEROACC(p_zeroacc::CLR_HALF, false /*EN_32BIT_DEST*/, 0, ADDR_MOD_7, dest_register_offset != 0);
        }

        // Tell UNPACK that the bank is free.
        TTI_SEMPOST(0, semaphore::t6_sem(semaphore::DEST_FREE));

        if constexpr (DST_SYNC_MODE == DstSync::SyncHalf)
        {
            _update_dest_register_offset_<false /*EN_32BIT_DEST*/>();
            _set_packer_dest_registers_<PACK_SEL, DST_SYNC_MODE>();
        }
    }
}
