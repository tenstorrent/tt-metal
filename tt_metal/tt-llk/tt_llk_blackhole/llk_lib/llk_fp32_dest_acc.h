// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_structs.h"

using namespace ckernel;

namespace fp32_dest_acc
{
// One 0/1 ping-pong semaphore per consumer. No semaphore is free, so these borrow the unpack-to-dest
// pair: both are Max=1, start at 0, and are balanced at op boundaries. Do not switch mid unpack-to-dest.
constexpr std::uint8_t UNPACK_SEM = semaphore::UNPACK_TO_DEST;
constexpr std::uint8_t PACK_SEM   = semaphore::MATH_DONE;

// This thread has nothing in flight on any engine that reads the dest-acc fields (FPU, SFPU, packer) or
// the unpacker. Any thread can drive any engine (e.g. SFPU from PACK), so each thread drains all of them.
constexpr std::uint32_t THREAD_IDLE = p_stall::UNPACK | p_stall::PACK | p_stall::MATH | p_stall::WAIT_SFPU;
} // namespace fp32_dest_acc

/**
 * @brief Coordinate a mid-kernel FP32 dest-acc reconfiguration across Unpack, Math, and Pack.
 *
 * Dest-acc CFG is MATH-owned. The handshake is Tensix-only, so ordering is enforced at each thread's
 * Wait Gate and no RISC blocks:
 *   1. UNPACK/PACK wait until none of their own work is in flight on any engine, SEMPOST their
 *      semaphore, then SEMWAIT with every instruction class blocked until MATH takes it back.
 *   2. MATH waits for both semaphores, waits until its own work has drained, programs ALU_ACC_CTRL and
 *      PCK_DEST_RD_CTRL, then SEMGETs both semaphores to release UNPACK/PACK. The Wait Gate issues in
 *      order and RMWCIB executes in the cycle it leaves the gate, so the SEMGET cannot take effect
 *      before the config writes have.
 * Old-mode work finishes before the config changes, and no Tensix instruction issued after the call
 * on any thread sees the old config.
 *
 * @tparam thread_id: TRISC thread compiling this specialization, values = <UnpackThreadId/MathThreadId/PackThreadId>
 * @param enable: MATH only. True to enable FP32 dest accumulation, false to disable.
 * @note All three TRISC threads must call their specialization together, between ops. Only Tensix
 *       instructions are ordered: RISC code after the call is not held back. Not supported
 *       on Quasar.
 */
template <ThreadId thread_id>
inline void _llk_set_fp32_dest_acc_(bool enable = false)
{
    static_assert(IS_TRISC_THREAD<thread_id>, "_llk_set_fp32_dest_acc_ requires a TRISC thread");

    if constexpr (thread_id == ThreadId::MathThreadId)
    {
        constexpr std::uint32_t both_sems = semaphore::t6_sem(fp32_dest_acc::UNPACK_SEM) | semaphore::t6_sem(fp32_dest_acc::PACK_SEM);

        TTI_SEMWAIT(p_stall::STALL_CFG | p_stall::STALL_SYNC, both_sems, p_stall::STALL_ON_ZERO);
        TTI_STALLWAIT(p_stall::STALL_CFG, fp32_dest_acc::THREAD_IDLE);

        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(enable);

        TTI_SEMGET(both_sems);
    }
    else
    {
        constexpr std::uint8_t sem = (thread_id == ThreadId::UnpackThreadId) ? fp32_dest_acc::UNPACK_SEM : fp32_dest_acc::PACK_SEM;

        t6_semaphore_post<fp32_dest_acc::THREAD_IDLE>(sem);
        t6_semaphore_wait_on_max<p_stall::STALL_THREAD>(sem);
    }
}
