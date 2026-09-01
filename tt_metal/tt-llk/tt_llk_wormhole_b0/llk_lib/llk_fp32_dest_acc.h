// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"

using namespace ckernel;

/**
 * @brief Coordinate a mid-kernel FP32 dest-acc reconfiguration across Unpack, Math, and Pack.
 *
 * Dest-acc CFG is MATH-owned. tensix_sync drains each thread's Tensix FIFO (including FPU / SFPU /
 * packer); the mailbox then holds RISC so no new work is issued until MATH has written dest-acc:
 *   1. UNPACK/PACK tensix_sync, signal MATH, and wait.
 *   2. MATH tensix_syncs, waits for both, programs ALU_ACC_CTRL and PCK_DEST_RD_CTRL, and releases
 *      UNPACK/PACK.
 *   3. Every thread STALLWAITs on TRISC_CFG, blocking unpacker / packer / FPU / SFPU until those
 *      writes are visible.
 *
 * @tparam thread_id: TRISC thread compiling this specialization, values = <UnpackThreadId/MathThreadId/PackThreadId>
 * @param enable: MATH only. True to enable FP32 dest accumulation, false to disable.
 * @note All three TRISC threads must call their specialization together. Not supported on Quasar.
 */
template <ThreadId thread_id>
inline void _llk_set_fp32_dest_acc_(bool enable = false)
{
    static_assert(
        (thread_id == ThreadId::MathThreadId) || (thread_id == ThreadId::UnpackThreadId) || (thread_id == ThreadId::PackThreadId),
        "_llk_set_fp32_dest_acc_ requires a TRISC thread");

    constexpr std::uint32_t dest_acc_stall = p_stall::STALL_UNPACK | p_stall::STALL_PACK | p_stall::STALL_MATH | p_stall::STALL_SFPU;

    tensix_sync();

    if constexpr (thread_id == ThreadId::UnpackThreadId || thread_id == ThreadId::PackThreadId)
    {
        mailbox_write(ThreadId::MathThreadId, 1);
        mailbox_read(ThreadId::MathThreadId);
        TTI_STALLWAIT(dest_acc_stall, p_stall::TRISC_CFG);
    }
    else
    {
        mailbox_read(ThreadId::UnpackThreadId);
        mailbox_read(ThreadId::PackThreadId);

        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(enable);
        TTI_STALLWAIT(dest_acc_stall, p_stall::TRISC_CFG);

        mailbox_write(ThreadId::UnpackThreadId, 1);
        mailbox_write(ThreadId::PackThreadId, 1);
    }
}
