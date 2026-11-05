// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// Unpack side of DEST_SRCA_MM (llk_math_dest_srca_mm.h): in0 from L1 into SrcB, and the SrcA bank handed to the math
// thread, which fills it from DEST.

/**
 * @brief Configure unpacker B to move all of an in0 tile's faces with one UNPACR.
 *
 * @param in0_num_faces: Faces of an in0 tile in L1, values = <2/4> (2: its top 16 rows)
 * @note Call after the unpacker inits of the kernel's other operands, and again after any init that sets unpacker B.
 */
inline void _llk_unpack_dest_srca_mm_init_(const std::uint32_t in0_num_faces)
{
    TT_SETADCXX(p_setadc::UNP_B, in0_num_faces * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

/**
 * @brief One step of dest_srca_mm: hand SrcA to the math thread, then unpack in0 into SrcB.
 *
 * @param in0_address: L1 address of the in0 tile, in 16-byte units minus one (as for @ref _llk_unpack_A_).
 * @note On the math thread, pair with @ref _llk_math_dest_srca_mm_.
 */
inline void _llk_unpack_dest_srca_mm_(const std::uint32_t in0_address)
{
    // SrcA zeroed and marked valid without an unpack: MOVD2A fills it on the math thread.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    // in0 -> SrcB in its own config context, as _llk_unpack_AB_matmul_ does for in0.
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[unp_cfg_context == 0 ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = in0_address;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
