// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "llk_defs.h"

using namespace ckernel;

/**
 * @brief Give math a zeroed SrcA and SrcB bank for the mul_reduce_scalar reduce tail.
 *
 * Math fills them from DEST, and the zeros cover rows that short faces leave empty.
 *
 * @note Call after the multiply or copy phase. Pairs with @ref _llk_math_mul_reduce_scalar_init_ (T1).
 */
inline void _llk_unpack_mul_reduce_scalar_switch_to_reduce_()
{
    TTI_UNPACR_NOP(p_unpacr::UNP_A, 1 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
    TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Set_Dvalid*/, 0 /*Stall_Cntrl*/, 0 /*Bank_Clr_Ctrl*/, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
}
