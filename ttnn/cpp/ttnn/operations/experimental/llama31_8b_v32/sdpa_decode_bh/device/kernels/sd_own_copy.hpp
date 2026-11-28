// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sd_own_dest.hpp"

#if defined(ARCH_BLACKHOLE)

#ifdef TRISC_UNPACK
ALWI void sdo_unpack_copy_setup() {
    using namespace ckernel;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

ALWI void sdo_unpack_copy(std::uint32_t cb, std::uint32_t tile) {
    using namespace ckernel;
    const auto& ci = get_local_cb_interface(cb);
    const std::uint32_t address = ci.fifo_rd_ptr - 1 + ci.fifo_page_size * tile;
    volatile std::uint32_t tt_reg_ptr* sem = pc_buf_base + PC_BUF_SEMAPHORE_BASE + semaphore::UNPACK_SYNC;
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    while (static_cast<std::uint8_t>(*sem) >= 2) {
    }
    if (unp_cfg_context == 0) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    }
    *sem = 0;
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
#pragma GCC unroll 4
    for (int f = 0; f < 4; f++) {
        TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR_NOP(
            SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    }
    TTI_SEMGET(semaphore::t6_sem(semaphore::UNPACK_SYNC));
    if (unp_cfg_context == 0) {
        unp_cfg_context = 1;
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
    } else {
        unp_cfg_context = 0;
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    }
}
#endif

#ifdef TRISC_MATH
ALWI void sdo_math_copy_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_SETC16(ADDR_MOD_AB_SEC3_SrcAIncr_ADDR32, 0);
    TTI_SETC16(ADDR_MOD_DST_SEC3_DestIncr_ADDR32, 0);
    TTI_SETC16(ADDR_MOD_BIAS_SEC3_BiasIncr_ADDR32, 0);
    TTI_SETC16(ADDR_MOD_AB_SEC0_SrcAIncr_ADDR32, 1);
    TTI_SETC16(ADDR_MOD_DST_SEC0_DestIncr_ADDR32, 1);
    TTI_SETC16(ADDR_MOD_BIAS_SEC0_BiasIncr_ADDR32, 0);
    TTI_SETC16(ADDR_MOD_AB_SEC2_SrcAIncr_ADDR32, 8);
    TTI_SETC16(ADDR_MOD_DST_SEC2_DestIncr_ADDR32, 8);
    TTI_SETC16(ADDR_MOD_BIAS_SEC2_BiasIncr_ADDR32, 0);
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    if (src_zero_flag_hw != 1) {
        src_zero_flag_hw = 1;
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    }
}

ALWI void sdo_math_copy(std::uint32_t idst) {
    using namespace ckernel;
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (idst << 6) + sdo_dest_base());
#pragma GCC unroll 4
    for (int f = 0; f < 4; f++) {
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}
#endif

ALWI void sdo_copy_init([[maybe_unused]] std::uint32_t cb) {
    UNPACK((sdo_unpack_copy_setup()));
    MATH((sdo_math_copy_setup()));
}

ALWI void sdo_copy_tile(std::uint32_t cb, std::uint32_t itile, std::uint32_t idst) {
    UNPACK((sdo_unpack_copy(cb, itile)));
    MATH((sdo_math_copy(idst)));
}

#endif
