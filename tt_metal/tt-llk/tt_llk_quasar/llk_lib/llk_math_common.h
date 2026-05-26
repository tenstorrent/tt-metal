// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_sync.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @brief Sets up ALU formats for math destination register
 * @tparam EN_IMPLIED_MATH_FORMAT: If set to true, will imply math dest format
 * from SrcA reg format
 * @tparam EN_FP32_MATH_FORMAT: Set to true to use math dest in Float32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 * @tparam EN_INT32_MATH_FORMAT: Set to true to use math dest in Int32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 * @param srcA_format: Input srcA format, used to set ALU configs if not implied math format
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 * @param srcB_format: Input srcB format, used to set ALU configs if not implied math format
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 */
template <bool EN_IMPLIED_MATH_FORMAT, bool EN_FP32_MATH_FORMAT, bool EN_INT32_MATH_FORMAT>
inline void _llk_math_srcAB_hw_configure_(DataFormat srcA_format, DataFormat srcB_format)
{
    // Turn on automatic Tensix-TRISC synchronization
    // RT: This is turned on by default by HW, this should be removed
    set_ttsync_enables<TRACK_ALL>(TRISC_ID);

    static_assert(!(EN_FP32_MATH_FORMAT && EN_INT32_MATH_FORMAT), "Cannot have Int32 dest & Float32 dest at the same time");

    // Set implied math dest format mode
    cfg[DISABLE_IMPLIED_SRCA_FMT_SEC0_Base_ADDR32 + TRISC_ID] = !EN_IMPLIED_MATH_FORMAT;
    cfg[DISABLE_IMPLIED_SRCB_FMT_SEC0_Base_ADDR32 + TRISC_ID] = !EN_IMPLIED_MATH_FORMAT;

    std::uint8_t SRCA_FORMAT_MASKED = static_cast<std::uint8_t>(srcA_format) & 0xFF;
    std::uint8_t SRCB_FORMAT_MASKED = static_cast<std::uint8_t>(srcB_format) & 0xFF;

    alu_config_u alu_config;
    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        alu_config.val[i] = 0;
    }

    if constexpr (!EN_IMPLIED_MATH_FORMAT)
    {
        // Set ALU SrcA format since it is not implied
        // If input format has exp_width == 5, the math dest set to Float16
        // else input format has exp_width == 8, the math dest set to Float16_b
        alu_config.f.ALU_FORMAT_SPEC_REG_SrcA_val      = SRCA_FORMAT_MASKED;
        alu_config.f.ALU_FORMAT_SPEC_REG_SrcA_override = 0x1;
        alu_config.f.ALU_FORMAT_SPEC_REG_SrcB_val      = SRCB_FORMAT_MASKED;
        alu_config.f.ALU_FORMAT_SPEC_REG_SrcB_override = 0x1;

        // RT: Since SrcA & SrcB need to match exponent widths, can set them the same for now
        // Check with HW team if different mixes between Src registers are allowed
        alu_config.f.ALU_FORMAT_SPEC_REG0_SrcA = SRCA_FORMAT_MASKED;
        alu_config.f.ALU_FORMAT_SPEC_REG1_SrcB = SRCB_FORMAT_MASKED;
    }

    alu_config.f.ALU_ACC_CTRL_Fp32_enabled      = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_INT8_math_enabled = EN_INT32_MATH_FORMAT;

    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32 + i] = alu_config.val[i];
    }
}

/**
 * @brief Sets up ALU formats for math destination register, specifically for upk to dest
 * @tparam EN_IMPLIED_MATH_FORMAT: If set to true, will imply math dest format
 * from SrcA reg format
 * @tparam EN_FP32_MATH_FORMAT: Set to true to use math dest in Float32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 * @tparam EN_INT32_MATH_FORMAT: Set to true to use math dest in Int32
 * otherwise default behaviour is Float16/Float16_b depending on input
 * format exponent width
 */
template <bool EN_IMPLIED_MATH_FORMAT, bool EN_FP32_MATH_FORMAT, bool EN_INT32_MATH_FORMAT>
inline void _llk_math_upk_to_dest_hw_configure_()
{
    // Set implied math dest format mode
    cfg[DISABLE_IMPLIED_SRCA_FMT_SEC0_Base_ADDR32 + TRISC_ID] = !EN_IMPLIED_MATH_FORMAT;

    alu_config_u alu_config;
    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        alu_config.val[i] = 0;
    }

    // Program DEST fmt
    alu_config.f.ALU_ACC_CTRL_Fp32_enabled      = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = EN_FP32_MATH_FORMAT;
    alu_config.f.ALU_ACC_CTRL_INT8_math_enabled = EN_INT32_MATH_FORMAT;

    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32 + i] = alu_config.val[i];
    }
}

/**
 * @brief MATH: initialize the three-semaphore DEST-bank protocol (math side).
 *
 * Forwards to _llk_sync_math_init_dest_sems_<DST_SYNC_MODE>(), which SEMINITs
 * MATH_PACK (the semaphore MATH produces). UNPACK separately initializes
 * UNPACK_MATH and bootstraps DEST_FREE in _llk_sync_unpack_init_dest_sems_().
 *
 * Replaces the legacy dvalid-based math/pack sync.
 *
 * @tparam DST_SYNC_MODE Dest-bank synchronization mode (SyncFull or SyncHalf).
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_math_pack_sync_init_()
{
    _llk_sync_math_init_dest_sems_<DST_SYNC_MODE>();
}

/**
 * @brief MATH: wait until UNPACK has filled the next DEST bank.
 *
 * Forwards to _llk_sync_math_acquire_dest_<DST_SYNC_MODE>(), which SEMWAITs on
 * UNPACK_MATH (produced by UNPACK in _llk_sync_unpack_commit_dest_) and SEMGETs
 * to claim the bank.
 *
 * @tparam DST_SYNC_MODE Dest-bank synchronization mode (SyncFull or SyncHalf).
 */
template <DstSync DST_SYNC_MODE>
inline void _llk_math_wait_for_dest_available_()
{
    _llk_sync_math_acquire_dest_<DST_SYNC_MODE>();
}

/**
 * @brief MATH: signal PACK that the current DEST bank is filled with math results.
 *
 * Forwards to _llk_sync_math_commit_dest_(), which SEMPOSTs MATH_PACK.
 *
 * The HW MATH_DEST_access_id auto-rotation (programmed by
 * _llk_unpack_to_dest_hw_configure_) fires on the last matrix-unit MOP of the
 * section. For Flow A (no FPU MOPs) there's a TODO in
 * _llk_sync_math_commit_dest_ about explicitly toggling access_id.
 *
 * @tparam EN_32BIT_DEST  Whether the dest is in 32-bit mode (Float32/Int32).
 *                       Currently informational only; HW manages bank rotation
 *                       via MATH_DEST_access_id when 32-bit unpack-to-dest is
 *                       enabled, and _llk_sync_math_commit_dest_ handles the
 *                       SW protocol uniformly for the 16-bit path.
 * @tparam DST_SYNC_MODE  Dest-bank synchronization mode (SyncFull or SyncHalf).
 */
template <bool EN_32BIT_DEST, DstSync DST_SYNC_MODE>
inline void _llk_math_dest_section_done_()
{
    _llk_sync_math_commit_dest_();
    // The HW MATH_DEST_access_id auto-rotation (programmed by
    // _llk_unpack_to_dest_hw_configure_) fires on the last matrix-unit MOP of
    // the section. For Flow A (no FPU MOPs) there's a TODO in
    // _llk_sync_math_commit_dest_ about explicitly toggling access_id.
}
