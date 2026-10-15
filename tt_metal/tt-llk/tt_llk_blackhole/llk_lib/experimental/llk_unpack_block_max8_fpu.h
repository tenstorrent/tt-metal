// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"

namespace ckernel
{

/**
 * @brief Publish SrcA cleared to negative infinity and SrcB cleared to 1.0 for one FPU block-max call.
 *
 * SrcB row 0 is the GMPOOL scaler (1.0 leaves scores unchanged); SrcA rows 8..15
 * are the negative-infinity padding that turns the 16-row GMPOOL into a max of eight.
 * No L1 data is read.
 * @note Call once per @ref _llk_math_block_max8_fpu_ call, after any unpack that
 *       the math thread consumes before it.
 */
inline void _llk_unpack_block_max8_fpu_()
{
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_1, p_unpacr_nop::CLR_SRC);
}

} // namespace ckernel
