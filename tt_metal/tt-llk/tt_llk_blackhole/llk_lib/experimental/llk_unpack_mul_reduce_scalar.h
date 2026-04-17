// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "llk_defs.h"
#include "llk_operands.h"

using namespace ckernel;

/*************************************************************************
 * LLK MUL REDUCE SCALAR UNPACK - Unpacker operations for fused mul+reduce
 *************************************************************************/

/**
 * @brief Switch UNPACK state for mul_reduce_scalar reduce phase
 *
 * Prepares for the reduce phase where the math thread reuses destination
 * registers as source operands. Resets UNPACK counters, signals context switch,
 * and sets DVALID flags for srcA and srcB.
 *
 * Must be called after multiply phase and before reduce phase.
 */
inline void _llk_unpack_mul_reduce_scalar_switch_to_reduce_()
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
}
