// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TopK-XL copy: single UNPACR (unpack_srca_to_dest encoding) per tile run for FP32 and 16-bit
// unpack-to-dest — no MOP wrapper — instead of face-by-face loops from stock _llk_unpack_A_mop_config_.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel::unpacker;

namespace ckernel
{

inline void _llk_unpack_topk_xl_copy_init_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b0 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(SrcA, 0b0 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t clear_srca_to_neginf = TT_OP_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
    static constexpr std::uint32_t clear_srcb_dvalid    = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    if (is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        ckernel_template tmp(1, 1, unpack_srca_to_dest);
        tmp.program();
    }
    else
    {
        ckernel_template tmp(1, 1, clear_srcb_dvalid, unpack_srca);
        tmp.set_start_op(clear_srca_to_neginf);
        tmp.program();
    }
}

// Minimal unpack-to-dest execute path for TopK-XL copy (one TTI_UNPACR per tile).
inline void _llk_unpack_topk_xl_copy_(
    const std::uint32_t address, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t elements_this_tile)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    wait_for_next_context(2);

    const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    cfg[upk0_reg]                = address;

    semaphore_post(semaphore::UNPACK_SYNC);

    if (elements_this_tile == 0)
    {
        // Zero-element tile: clear SrcA to -inf so math datacopy writes full -inf padding.
        // This avoids SETADC underflow behavior and avoids reading source tile data.
        TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

        t6_semaphore_get(semaphore::UNPACK_SYNC);
        switch_config_context(unp_cfg_context);
        return;
    }

    if (is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        if (elements_this_tile < 1024)
        {
            // clear to -inf first for padding
            TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
            TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        }
        set_dst_write_addr(unp_cfg_context, unpack_dst_format);
        wait_for_dest_available();
    }

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run();

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    if (is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        unpack_to_dest_tile_done(unp_cfg_context);
    }

    switch_config_context(unp_cfg_context);
}

} // namespace ckernel
