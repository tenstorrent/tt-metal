// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_defs.h"
#include "tensor_shape.h"

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
    // Stall_Clr_Cntrl=1: wait on Unpackers[i].SrcBank -- the bank actually cleared and published --
    // not MatrixUnit.Src?Bank, which is a different bank once double-buffering reaches steady state.
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
}

/*************************************************************************
 * Compiled row of 32x32 tiles: each UNPACR writes two faces, so a SrcA/SrcB bank holds faces 2h and 2h + 1 in rows
 * 0 to 31 and the math thread takes two bank hand-offs per tile instead of four. _llk_unpack_AB_ runs each tile.
 *************************************************************************/

inline bool _llk_unpack_mul_reduce_scalar_two_face_banks_(const ckernel::TensorShape tensor_shape)
{
    return tensor_shape.face_r_dim == ckernel::MAX_FACE_R_DIM && tensor_shape.total_num_faces() == 4;
}

// _llk_unpack_AB_init_ for this row: no transpose, two faces per UNPACR.
inline void _llk_unpack_mul_reduce_scalar_two_face_banks_init_()
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCXX(p_setadc::UNP_AB, 2 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    // Ch0 Z steps two faces per UNPACR.
    static constexpr std::uint32_t unpack_srca = TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb = TT_OP_UNPACR(SrcB, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    ckernel_template tmp(1, 2, unpack_srca, unpack_srcb);
    tmp.program();
}
