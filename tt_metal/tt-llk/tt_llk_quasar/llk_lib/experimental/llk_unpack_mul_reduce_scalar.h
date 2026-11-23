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
 * @brief Hand cleared SrcA and SrcB banks to the math thread for the mul_reduce_scalar reduce tail.
 *
 * The reduce tail fills both source registers from DEST with MOVD2A/MOVD2B. Each bank is zeroed
 * first, so rows the moves do not write (past a tiny tile's face rows) pool as zeros, and its
 * dvalid is set.
 *
 * @note Call after the multiply (or copy) phase. On the math thread, pair with
 *       @ref _llk_math_mul_reduce_scalar_init_ (T1), which runs the reduce tail.
 */
inline void _llk_unpack_mul_reduce_scalar_switch_to_reduce_()
{
    TTI_UNPACR_NOP(p_unpacr::UNP_A, 1 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
    TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
}
