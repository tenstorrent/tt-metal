// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "lltt.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * Src-safe variant of unpack_to_dest_tile_done (cunpack_common.h:920).
 *
 * The production function issues an extra TT_UNPACR(SrcA, ...) to work around HW
 * bug TEN-3868. That extra UNPACR inherits the surrounding in/out data formats,
 * which is UndefinedBehavior when the unpack-to-dest path is in INT32 / UInt32
 * (formats invalid for SrcA). This function temporarily swaps the in/out formats
 * to UInt16 around the extra UNPACR, then restores them. Matching the fix in
 * tt-llk PR #1357 but exposed only via the experimental opt-in API.
 */
inline void unpack_to_dest_tile_done_src_safe(std::uint32_t& context_id)
{
    t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
    TTI_WRCFG(p_gpr_unpack::UNPACK_STRIDE, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32); // Restore unpack stride
    // Restore config context
    if (context_id == 0)
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(4 * 16);
    }
    else
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(4 * 16);
    }
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4); // re-enable address bit swizzle

    // Due to a hardware bug (TEN-3868), we need to have one unpack-to-srcA instruction after the last unpack-to-dest instruction.
    TTI_SETADCXX(p_setadc::UNP_A, FACE_C_DIM - 1, 0x0);

    // Save current in/out formats and temporarily set both to UInt16 so the
    // dummy UNPACR below is a contract-valid SrcA unpack regardless of what the
    // surrounding unpack-to-dest was doing (e.g. INT32 / UInt32).
    TTI_RDCFG(p_gpr_unpack::TMP_LO, THCON_SEC0_REG2_Out_data_format_ADDR32);
    TTI_RDCFG(p_gpr_unpack::TMP_HI, THCON_SEC0_REG0_TileDescriptor_ADDR32);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(to_underlying(DataFormat::UInt16));
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0xF>(to_underlying(DataFormat::UInt16));

    TT_UNPACR(SrcA, 0, 0, context_id, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 1, 0, 0, 0, 1);

    TTI_WRCFG(p_gpr_unpack::TMP_HI, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32);
    TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32);

    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

/**
 * Src-safe variant of _llk_unpack_A_ (llk_unpack_A.h:242). Identical to the
 * production function except it calls unpack_to_dest_tile_done_src_safe at the
 * post-unpack cleanup site so the TEN-3868 workaround does not corrupt SrcA
 * formats. The MOP/init path is unchanged, only the tail of the function
 * differs, so the existing _llk_unpack_A_init_ can be reused as-is.
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_src_safe_custom_(const std::uint32_t address, const std::uint32_t unpack_src_format = 0, const std::uint32_t unpack_dst_format = 0)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    // Wait for free context
    wait_for_next_context(2);

    // Set upk0/1 L1 read addr
    if constexpr (((BType == BroadcastType::NONE) && (!acc_to_dest)) || binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB || unpack_to_dest)
    {
        const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
        cfg[upk0_reg]                = address;
    }
    else
    {
        const std::uint32_t upk1_reg = (unp_cfg_context == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32;
        cfg[upk1_reg]                = address;
    }

    if constexpr (unpack_to_dest)
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            set_dst_write_addr(unp_cfg_context, unpack_dst_format);
            wait_for_dest_available();
        }
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    if (unpack_to_dest)
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            unpack_to_dest_tile_done_src_safe(unp_cfg_context);
        }
    }

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
