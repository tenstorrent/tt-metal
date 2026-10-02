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
// No semaphore is free, so the handshake borrows three. Each carries tokens in the direction its owner
// already uses it, and every token is a Tensix instruction:
//   UNPACK_TO_DEST  UNPACK -> MATH  "UNPACK is drained and here". Unpack-to-dest uses it the same way
//                                   (UNPACK posts, MATH SEMGETs in tile_ready, both in program order), so a
//                                   switch token posted while MATH is still taking tile tokens just adds one
//                                   to the count.
//   MATH_PACK       MATH -> PACK    "MATH is here", i.e. past every math_unpack_to_dest_math_ready. PACK has
//                                   released every dest section MATH committed before the switch, so the
//                                   only token PACK's switch can see is this one. MATH takes it back itself.
//   MATH_DONE       PACK -> MATH    "PACK is drained and here". math_unpack_to_dest_math_ready polls and
//                                   decrements MATH_DONE from the math RISC, so PACK must not post it until
//                                   MATH has arrived; otherwise the math RISC takes PACK's token.
// All three are back to their pre-switch values when the call returns on every thread.
constexpr std::uint8_t UNPACK_SEM = semaphore::UNPACK_TO_DEST;
constexpr std::uint8_t ARRIVE_SEM = semaphore::MATH_PACK;
constexpr std::uint8_t PACK_SEM   = semaphore::MATH_DONE;

// This thread has nothing in flight on any engine that reads the dest-acc fields (FPU, SFPU, packer) or
// the unpacker, and no pending RISC config write. Any thread can drive any engine (e.g. SFPU from
// PACK), so each consumer drains all of them.
constexpr std::uint32_t THREAD_IDLE = p_stall::UNPACK | p_stall::PACK | p_stall::MATH | p_stall::WAIT_SFPU | p_stall::TRISC_CFG;
} // namespace fp32_dest_acc

/**
 * @brief Coordinate a mid-kernel FP32 dest-acc reconfiguration across Unpack, Math, and Pack.
 *
 * Dest-acc CFG is MATH-owned. The handshake is Tensix-only, so ordering is enforced at each thread's
 * Wait Gate:
 *   1. MATH tensix_syncs, draining its own work and landing the RISC semaphore store that ends a
 *      preceding math_unpack_to_dest_math_ready, then posts MATH_PACK to announce its arrival.
 *   2. UNPACK, and PACK once it sees MATH's arrival, wait until none of their own work is in flight,
 *      post their token, then SEMWAIT with every instruction class blocked until MATH takes it back.
 *   3. MATH waits for both consumer tokens, programs ALU_ACC_CTRL and PCK_DEST_RD_CTRL, then SEMGETs
 *      both tokens and its arrival to release UNPACK/PACK. The Wait Gate issues in order and RMWCIB
 *      executes in the cycle it leaves the gate, so the SEMGET cannot take effect before the config
 *      writes have.
 *   4. Every thread holds its RISC until its part has executed: MATH until its SEMGET, UNPACK/PACK
 *      until MATH's release. A SEMWAIT is complete as soon as it latches, so each consumer queues a
 *      NOP behind it (blocked, as all block bits are set) for tensix_sync to wait on.
 * Old-mode work finishes before the config changes, and nothing issued after the call on any thread,
 * Tensix or RISC, sees the old config. A thread that arrives early blocks at its first wait, whatever
 * the other threads are doing.
 *
 * @tparam thread_id: TRISC thread compiling this specialization, values = <UnpackThreadId/MathThreadId/PackThreadId>
 * @param enable: MATH only. True to enable FP32 dest accumulation, false to disable.
 * @note All three TRISC threads must call their specialization together, between ops: PACK must have
 *       released every dest section MATH committed before the switch, and every unpack-to-dest tile
 *       UNPACK posted must have been consumed by MATH. All threads must use config state 0.
 */
template <ThreadId thread_id>
inline void _llk_set_fp32_dest_acc_(bool enable = false)
{
    static_assert(IS_TRISC_THREAD<thread_id>, "_llk_set_fp32_dest_acc_ requires a TRISC thread");

    if constexpr (thread_id == ThreadId::MathThreadId)
    {
        constexpr std::uint32_t consumer_sems = semaphore::t6_sem(fp32_dest_acc::UNPACK_SEM) | semaphore::t6_sem(fp32_dest_acc::PACK_SEM);

        tensix_sync();
        t6_semaphore_post(fp32_dest_acc::ARRIVE_SEM);

        TTI_SEMWAIT(p_stall::STALL_CFG | p_stall::STALL_SYNC, consumer_sems, p_stall::STALL_ON_ZERO);
        // The following stallwait is not needed if Auto TTSYNC is enabled to order the write
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::TRISC_CFG);

        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(enable);

        TTI_SEMGET(consumer_sems | semaphore::t6_sem(fp32_dest_acc::ARRIVE_SEM));
        tensix_sync();
    }
    else
    {
        constexpr std::uint8_t sem = (thread_id == ThreadId::UnpackThreadId) ? fp32_dest_acc::UNPACK_SEM : fp32_dest_acc::PACK_SEM;

        if constexpr (thread_id == ThreadId::PackThreadId)
        {
            t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(fp32_dest_acc::ARRIVE_SEM);
        }
        t6_semaphore_post<fp32_dest_acc::THREAD_IDLE>(sem);
        t6_semaphore_wait_on_max<p_stall::STALL_THREAD>(sem);
        TTI_NOP;
        tensix_sync();
    }
}
