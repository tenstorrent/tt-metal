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
 * registers as source operands. Resets UNPACK counters and marks SrcA/SrcB
 * data-valid (handing bank ownership to MatrixUnit) without unpacking real data.
 *
 * @note Call after the multiply-phase unpack loop and before the reduce phase.
 *       @ref _llk_math_mul_reduce_scalar_move_dest_to_src_ on the math thread
 *       waits on this handoff (via SRCA_VLD/SRCB_VLD) before reusing DEST as SrcA/SrcB.
 */
inline void _llk_unpack_mul_reduce_scalar_switch_to_reduce_()
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // UNPACR_NOP SET_DVALID does not itself wait at the hardware Wait Gate for
    // AllowedClient == Unpackers (unlike a real UNPACR), so drain the still-in-flight
    // multiply-phase unpack before flipping SrcA/SrcB bank ownership to MatrixUnit below,
    // or the flip races the last real UNPACR. Matches the only other "mark Src valid for a
    // MATH DEST-reuse sequence" helper in this codebase, @ref _llk_unpack_set_srcb_dummy_valid_.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}
