// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "llk_defs.h"

using namespace ckernel;

/*************************************************************************
 * LLK MUL REDUCE SCALAR UNPACK (Quasar)
 *************************************************************************/

/**
 * @brief Hand SrcA and SrcB to the math thread for the mul_reduce_scalar reduce tail.
 *
 * The reduce tail fills both source registers from DEST with MOVD2A/MOVD2B, so the unpacker only
 * sets a dummy dvalid on each. Each NOP waits for the bank the FPU is reading to be released.
 *
 * @note Call after the multiply (or copy) phase. On the math thread, pair with
 *       @ref _llk_math_mul_reduce_scalar_init_ (T1), which runs the reduce tail.
 */
inline void _llk_unpack_mul_reduce_scalar_switch_to_reduce_()
{
    TTI_UNPACR_NOP(p_unpacr::UNP_A, 0 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, 0 /*Src_ClrVal_Ctrl*/, p_unpacr::UNP_SET_DVALID);
    TTI_UNPACR_NOP(p_unpacr::UNP_B, 0 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, 0 /*Src_ClrVal_Ctrl*/, p_unpacr::UNP_SET_DVALID);
}
