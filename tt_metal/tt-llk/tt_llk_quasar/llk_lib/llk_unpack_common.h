// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cunpack_common.h"
#include "llk_defs.h"
#include "llk_sync.h"
using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief UNPACK-side init for the 32-bit unpack-to-dest semaphore path.
 *
 * Called from the UNPACK thread at compute-kernel init when UnpackToDestEn is
 * true. Programs the HW state that lets each matrix-unit client (UNPACK, MATH,
 * PACK) maintain its own DEST bank pointer (`access_id`), so the three-
 * semaphore SW protocol in llk_sync.h does not race on shared SEC<N>:
 *
 *   1. ALU_FORMAT_SPEC_REG (ALU_ACC_CTRL_*) — selects 32-bit dest mode bits
 *      from EN_FP32 / EN_INT32 template args (mutually exclusive).
 *   2. UNPACK_TO_DEST_DVALID_CTRL — toggle_mask=1, wait_mask=0: HW auto-
 *      rotates UNPACK's access_id each unpack-to-dest commit, no waiting.
 *   3. MATH_DEST_DVALID_CTRL       — toggle_mask=1, wait_mask=0: HW auto-
 *      rotates MATH's access_id each matrix-unit "last" instruction.
 *   4. PACK_DEST_DVALID_CTRL = 0   — PACK gets no HW auto-rotation; SW THCON
 *      flip in _llk_sync_pack_release_dest_ moves PACK's bank pointer instead.
 *   5. SEC0 (UNPACK base) = 0; SEC1 (MATH base) = 0x100 in SyncHalf (256-row
 *      half-bank offset for 32-bit dest), 0 in SyncFull (full bank from 0).
 *
 * @tparam EN_FP32        Enable 32-bit float dest (mutually exclusive with EN_INT32).
 * @tparam EN_INT32       Enable 32-bit integer dest (mutually exclusive with EN_FP32).
 * @tparam DST_SYNC_MODE  DstSync::SyncFull or DstSync::SyncHalf.
 */
template <bool EN_FP32, bool EN_INT32, DstSync DST_SYNC_MODE>
inline void _llk_unpack_to_dest_hw_configure_()
{
    static_assert(!(EN_FP32 && EN_INT32), "Cannot have Int32 dest & Float32 dest at the same time");
    static_assert(
        DST_SYNC_MODE == DstSync::SyncFull || DST_SYNC_MODE == DstSync::SyncHalf, "DST_SYNC_MODE must be DstSync::SyncFull or DstSync::SyncHalf");

    // 1. ALU_FORMAT_SPEC_REG: 32-bit dest mode bits.
    //    Fp32_enabled / SFPU_Fp32_enabled track EN_FP32; INT8_math_enabled tracks EN_INT32.
    cfg_rmw(ALU_ACC_CTRL_Fp32_enabled_RMW, EN_FP32 ? 1u : 0u);
    cfg_rmw(ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW, EN_FP32 ? 1u : 0u);
    cfg_rmw(ALU_ACC_CTRL_INT8_math_enabled_RMW, EN_INT32 ? 1u : 0u);

    // 2. UNPACK_TO_DEST_DVALID_CTRL: toggle_mask=1, wait_mask=0 -> HW auto-rotates
    //    UNPACK's access_id each unpack-to-dest commit, no SW waiting.
    cfg_rmw(UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_RMW, 1u);
    cfg_rmw(UNPACK_TO_DEST_DVALID_CTRL_wait_mask_RMW, 0u);

    // 3. MATH_DEST_DVALID_CTRL: toggle_mask=1, wait_mask=0 -> HW auto-rotates
    //    MATH's access_id each matrix-unit last=1 instruction.
    cfg_rmw(MATH_DEST_DVALID_CTRL_toggle_mask_RMW, 1u);
    cfg_rmw(MATH_DEST_DVALID_CTRL_wait_mask_RMW, 0u);

    // 4. PACK_DEST_DVALID_CTRL = 0: no HW auto-rotation for PACK; the SW THCON
    //    flip in _llk_sync_pack_release_dest_ moves PACK's bank pointer.
    cfg_rmw(PACK_DEST_DVALID_CTRL_toggle_mask_RMW, 0u);
    cfg_rmw(PACK_DEST_DVALID_CTRL_wait_mask_RMW, 0u);

    // 5. SEC<N> base addresses for UNPACK (SEC0) and MATH (SEC1).
    //    SyncHalf: SEC1 = 0x100 (256-row offset for 32-bit dest half-bank).
    //    SyncFull: SEC1 = 0     (single bank, both clients start at 0).
    cfg[DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32] = 0u;
    if constexpr (DST_SYNC_MODE == DstSync::SyncHalf)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC1_Offset_ADDR32] = 0x100u;
    }
    else
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC1_Offset_ADDR32] = 0u;
    }
}

/**
 * @brief UNPACK-side per-section acquire. Wait for a free DEST bank.
 *
 * Part of the three-semaphore protocol replacing dvalid sync. Called from
 * tile_regs_acquire on the UNPACK trisc.
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_unpack_wait_for_dest_available_()
{
    _llk_sync_unpack_acquire_dest_<DST_SYNC_MODE>();
}

/**
 * @brief UNPACK-side per-section commit. Signal MATH the bank is filled.
 *
 * Called from tile_regs_commit on the UNPACK trisc.
 */
inline void _llk_unpack_dest_section_done_()
{
    _llk_sync_unpack_commit_dest_();
}

/**
 * @brief UNPACK-side compute-kernel init for the three-semaphore protocol.
 *
 * Bootstraps DEST_FREE and inits UNPACK_MATH. Called once at compute-kernel
 * init from the UNPACK trisc, alongside _llk_unpack_to_dest_hw_configure_.
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_unpack_pack_sync_init_()
{
    _llk_sync_unpack_init_dest_sems_<DST_SYNC_MODE>();
}

/**
 * @brief Programs unpacker l1 info & source register format
 * @tparam UNP_SEL: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src: Contains source reg format
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_hw_configure_(const tdma_descriptor_t& tdma_desc_src)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_S) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_S/UNP_DEST");

    // RT: make defines to aggregate the source format address, to make the below a single function
    // Program src formats
    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        cfg_rmw(THCON_UNPACKER1_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_S)
    {
        cfg_rmw(THCON_UNPACKER2_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(tdma_desc_src.reg_data_format));
    }
}

// RT: make defines to aggregate _llk_unpack_hw_configure_ calls into one
/**
 * @brief Programs unpacker l1 info & source register format for unary operation
 * @tparam UNP_SEL: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src: Contains L1 buffer descriptor information & source reg format for Src Reg
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_configure_unary_(const tdma_descriptor_t& tdma_desc_src)
{
    _llk_unpack_hw_configure_<UNP_SEL>(tdma_desc_src);
}

/**
 * @brief Programs unpacker l1 info & source register format for binary operation
 * @tparam UNP_SEL0/1: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src0/1: Contains L1 buffer descriptor information & source reg format for Src Reg
 */
template <std::uint32_t UNP_SEL_0, std::uint32_t UNP_SEL_1>
inline void _llk_unpack_configure_binary_(const tdma_descriptor_t& tdma_desc_src0, const tdma_descriptor_t& tdma_desc_src1)
{
    _llk_unpack_hw_configure_<UNP_SEL_0>(tdma_desc_src0);
    _llk_unpack_hw_configure_<UNP_SEL_1>(tdma_desc_src1);
}

/**
 * @brief Sets dummy SrcB dvalid
 */
inline void _llk_unpack_set_srcB_dummy_valid_()
{
    TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Set_Dvalid*/, 0, 0, 0, p_unpacr::UNP_NOP);
}
