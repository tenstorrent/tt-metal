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
    static_assert(IS_TRISC_THREAD<thread_id>, "_llk_set_fp32_dest_acc_ requires a TRISC thread");
#ifdef RS_SEM_SWITCH
    // RS test: tt-metal PR #57747's Tensix-semaphore handshake (Blackhole body, verbatim).
    {
        constexpr std::uint8_t RS_UNPACK_SEM       = semaphore::UNPACK_TO_DEST;
        constexpr std::uint8_t RS_ARRIVE_SEM       = semaphore::MATH_PACK;
        constexpr std::uint8_t RS_PACK_SEM         = semaphore::MATH_DONE;
        constexpr std::uint32_t RS_THREAD_IDLE     = p_stall::UNPACK | p_stall::PACK | p_stall::MATH | p_stall::WAIT_SFPU | p_stall::TRISC_CFG;
        if constexpr (thread_id == ThreadId::MathThreadId)
        {
            constexpr std::uint32_t consumer_sems = semaphore::t6_sem(RS_UNPACK_SEM) | semaphore::t6_sem(RS_PACK_SEM);
            tensix_sync();
            t6_semaphore_post(RS_ARRIVE_SEM);
            TTI_SEMWAIT(p_stall::STALL_CFG | p_stall::STALL_SYNC, consumer_sems, p_stall::STALL_ON_ZERO);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::TRISC_CFG);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
            cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(enable);
            TTI_SEMGET(consumer_sems | semaphore::t6_sem(RS_ARRIVE_SEM));
            tensix_sync();
        }
        else
        {
            constexpr std::uint8_t sem = (thread_id == ThreadId::UnpackThreadId) ? RS_UNPACK_SEM : RS_PACK_SEM;
            if constexpr (thread_id == ThreadId::PackThreadId)
            {
                t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(RS_ARRIVE_SEM);
            }
            t6_semaphore_post<RS_THREAD_IDLE>(sem);
            t6_semaphore_wait_on_max<p_stall::STALL_THREAD>(sem);
            TTI_NOP;
            tensix_sync();
        }
        return;
    }
#endif

    constexpr std::uint32_t dest_acc_stall = p_stall::STALL_UNPACK | p_stall::STALL_PACK | p_stall::STALL_MATH | p_stall::STALL_SFPU;

    tensix_sync();

    if constexpr (thread_id == ThreadId::UnpackThreadId)
    {
        mailbox_write(ThreadId::MathThreadId, fp32_dest_acc::UNPACK_READY);
        const std::uint32_t math_done = mailbox_read(ThreadId::MathThreadId);
        LLK_ASSERT(math_done == fp32_dest_acc::MATH_DONE, "Unexpected dest-acc message from math thread.");
        TTI_STALLWAIT(dest_acc_stall, p_stall::TRISC_CFG);
#ifdef RS_SAFE_FP32_SWITCH
        tensix_sync();  // RS probe: no new work until this thread's view of the switch has drained
#endif
    }
    else if constexpr (thread_id == ThreadId::PackThreadId)
    {
        mailbox_write(ThreadId::MathThreadId, fp32_dest_acc::PACK_READY);
        const std::uint32_t math_done = mailbox_read(ThreadId::MathThreadId);
        LLK_ASSERT(math_done == fp32_dest_acc::MATH_DONE, "Unexpected dest-acc message from math thread.");
        TTI_STALLWAIT(dest_acc_stall, p_stall::TRISC_CFG);
#ifdef RS_SAFE_FP32_SWITCH
        tensix_sync();  // RS probe: no new work until this thread's view of the switch has drained
#endif
    }
    else
    {
        const std::uint32_t unpack_ready = mailbox_read(ThreadId::UnpackThreadId);
        const std::uint32_t pack_ready   = mailbox_read(ThreadId::PackThreadId);
        LLK_ASSERT(unpack_ready == fp32_dest_acc::UNPACK_READY, "Unexpected dest-acc message from unpack thread.");
        LLK_ASSERT(pack_ready == fp32_dest_acc::PACK_READY, "Unexpected dest-acc message from pack thread.");

        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(enable);
        TTI_STALLWAIT(dest_acc_stall, p_stall::TRISC_CFG);
#if defined(RS_SAFE_FP32_SWITCH) || defined(RS_DRAIN_BEFORE_RELEASE)
        // RS probe / backport of tt-metal #55807: drain MATH's config RMWs before releasing UNPACK/PACK.
        tensix_sync();
#endif

        mailbox_write(ThreadId::UnpackThreadId, fp32_dest_acc::MATH_DONE);
        mailbox_write(ThreadId::PackThreadId, fp32_dest_acc::MATH_DONE);
    }
}
