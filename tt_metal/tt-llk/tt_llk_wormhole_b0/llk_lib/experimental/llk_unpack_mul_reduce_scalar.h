// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "llk_defs.h"

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
    // WH requires ZEROSRC and SET_DVALID to be sequenced as separate UNPACR_NOPs.
    // ZEROSRC carries the wait-like-UNPACR bit so it gates on Unpackers[i].SrcBank -- the bank it
    // actually clears -- rather than MatrixUnit.Src?Bank; SET_DVALID inherits that wait by sequencing.
    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC_STALL_RESET_WR_RDY);
    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_SET_DVALID);
    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC_STALL_RESET_WR_RDY);
    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
}
