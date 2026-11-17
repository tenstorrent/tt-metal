// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(ARCH_BLACKHOLE)

static_assert(DST_ACCUM_MODE);

#ifdef TRISC_MATH
ALWI void sdo_math_hold() {
    using namespace ckernel;
    TTI_SEMWAIT(
        p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
        semaphore::t6_sem(semaphore::MATH_PACK),
        p_stall::STALL_ON_MAX);
}

ALWI void sdo_math_hand_over() {
    using namespace ckernel;
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::MATH | p_stall::WAIT_SFPU);
    TTI_SEMPOST(semaphore::t6_sem(semaphore::MATH_PACK));
    math_sync_tile_dst_index = 0;
    dest_offset_id = 1 - dest_offset_id;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    if (dest_offset_id) {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, DEST_REGISTER_HALF_SIZE);
    } else {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
    }
}
#endif

#ifdef TRISC_PACK
ALWI void sdo_pack_hold() {
    using namespace ckernel;
    TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_SYNC, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
}

ALWI void sdo_pack_hand_back() {
    using namespace ckernel;
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    if (dest_offset_id) {
        TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 1);
        TTI_SEMGET(semaphore::t6_sem(semaphore::MATH_PACK));
        dest_offset_id = 0;
        TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO, p_cfg::WRCFG_32b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    } else {
        TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0);
        TTI_SEMGET(semaphore::t6_sem(semaphore::MATH_PACK));
        dest_offset_id = 1;
        TTI_WRCFG(p_gpr_pack::DEST_OFFSET_HI, p_cfg::WRCFG_32b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    TTI_DMANOP;
    TTI_DMANOP;
}
#endif

ALWI std::uint32_t sdo_dest_base() { return ckernel::dest_offset_id ? DEST_REGISTER_HALF_SIZE : 0; }

ALWI void sdo_regs_acquire() { MATH((sdo_math_hold())); }

ALWI void sdo_regs_commit() { MATH((sdo_math_hand_over())); }

ALWI void sdo_regs_wait() { PACK((sdo_pack_hold())); }

ALWI void sdo_regs_release() { PACK((sdo_pack_hand_back())); }

#endif
