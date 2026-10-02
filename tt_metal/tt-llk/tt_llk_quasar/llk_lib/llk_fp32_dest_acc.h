// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "llk_assert.h"

using namespace ckernel;

namespace fp32_dest_acc
{
constexpr std::uint32_t UNPACK_READY = 0x46504101; // 'FPA' | 0x01
constexpr std::uint32_t PACK_READY   = 0x46504102; // 'FPA' | 0x02
constexpr std::uint32_t MATH_DONE    = 0x46504110; // 'FPA' | 0x10
} // namespace fp32_dest_acc

/**
 * @brief Coordinate a mid-kernel FP32 dest-acc reconfiguration across Unpack, Math, and Pack.
 *
 * Dest-acc CFG is MATH-owned. tensix_sync drains each thread's Tensix FIFO (including FPU / SFPU /
 * packer); the mailbox then holds RISC so no new work is issued until MATH has written dest-acc:
 *   1. UNPACK/PACK tensix_sync, signal MATH, and wait.
 *   2. MATH tensix_syncs, waits for both, programs ALU_ACC_CTRL Fp32_enabled / SFPU_Fp32_enabled, and
 *      releases UNPACK/PACK.
 *   3. Every thread STALLWAITs on TRISC_CFG, blocking unpacker / packer / FPU / SFPU until those
 *      writes are visible.
 *
 * Quasar differences from WH/BH:
 *   - There is no PCK_DEST_RD_CTRL Read_32b_data bit. The packer reads dest as 32-bit when its
 *     IN_DATA_FORMAT is Float32/Int32, so the pack side must reprogram it explicitly
 *     (@ref _llk_pack_reconfig_data_format_) after this call.
 *   - The two ALU_ACC_CTRL bits share ALU_FORMAT_SPEC word 2 with fields the math init helpers own, so
 *     they are updated with a RISC-side cfg_rmw (never a full-word write). After tensix_sync MATH has no
 *     Tensix work in flight, so TTSync lets the write go straight to the live bank.
 *   - The math ALU format latch must be invalidated after this call; @ref _llk_math_set_fp32_dest_acc_
 *     does both.
 *
 * @tparam thread_id: TRISC thread compiling this specialization, values = <UnpackThreadId/MathThreadId/PackThreadId>
 * @param enable: MATH only. True to enable FP32 dest accumulation, false to disable.
 * @note All three TRISC threads must call their specialization together, at an op boundary with dest
 * empty. The isolated-SFPU TRISC (thread 3) is not part of the handshake: SFPU_Fp32_enabled is global,
 * and TTSync only orders this write against MATH's own instructions, so a kernel that drives the SFPU
 * on dest from TRISC3 must not use this API.
 */
template <ThreadId thread_id>
inline void _llk_set_fp32_dest_acc_(bool enable = false)
{
    static_assert(
        thread_id == ThreadId::UnpackThreadId || thread_id == ThreadId::MathThreadId || thread_id == ThreadId::PackThreadId,
        "_llk_set_fp32_dest_acc_ requires the Unpack, Math or Pack TRISC");

    constexpr std::uint32_t dest_acc_stall = p_stall::STALL_UNPACK | p_stall::STALL_PACK | p_stall::STALL_MATH | p_stall::STALL_SFPU;

    tensix_sync();

    if constexpr (thread_id == ThreadId::UnpackThreadId)
    {
        mailbox_write(ThreadId::MathThreadId, fp32_dest_acc::UNPACK_READY);
        const std::uint32_t math_done = mailbox_read(ThreadId::MathThreadId);
        LLK_ASSERT(math_done == fp32_dest_acc::MATH_DONE, "Unexpected dest-acc message from math thread.");
        TTI_STALLWAIT(dest_acc_stall, 0, 0, p_stall::TRISC_CFG);
    }
    else if constexpr (thread_id == ThreadId::PackThreadId)
    {
        mailbox_write(ThreadId::MathThreadId, fp32_dest_acc::PACK_READY);
        const std::uint32_t math_done = mailbox_read(ThreadId::MathThreadId);
        LLK_ASSERT(math_done == fp32_dest_acc::MATH_DONE, "Unexpected dest-acc message from math thread.");
        TTI_STALLWAIT(dest_acc_stall, 0, 0, p_stall::TRISC_CFG);
    }
    else
    {
        const std::uint32_t unpack_ready = mailbox_read(ThreadId::UnpackThreadId);
        const std::uint32_t pack_ready   = mailbox_read(ThreadId::PackThreadId);
        LLK_ASSERT(unpack_ready == fp32_dest_acc::UNPACK_READY, "Unexpected dest-acc message from unpack thread.");
        LLK_ASSERT(pack_ready == fp32_dest_acc::PACK_READY, "Unexpected dest-acc message from pack thread.");

        cfg_rmw(ALU_ACC_CTRL_Fp32_enabled_RMW, enable);
        cfg_rmw(ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW, enable);
        TTI_STALLWAIT(dest_acc_stall, 0, 0, p_stall::TRISC_CFG);

        mailbox_write(ThreadId::UnpackThreadId, fp32_dest_acc::MATH_DONE);
        mailbox_write(ThreadId::PackThreadId, fp32_dest_acc::MATH_DONE);
    }
}
