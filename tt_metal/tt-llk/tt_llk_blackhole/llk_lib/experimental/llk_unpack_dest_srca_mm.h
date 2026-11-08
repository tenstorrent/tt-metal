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

/**
 * @brief Unpack a page whose codes are 4-bit block-float datums: the codes (BFP4, the exponent forced to exp, no
 * exponent section) as FP16 to Dst16b rows from 2 m + 4 for the first address m from the math thread, then the page's
 * 32-bit words from byte words_offset (the scale fields) to the DEST tile of the second address, as many rows in. An
 * unpack-to-dest with its handshake (@ref _llk_math_dest_srca_mm_codes_post_ on the math thread). Unpacker A's formats
 * and X range are put back afterwards.
 *
 * @param address: L1 address of the page, in 16-byte units minus one.
 * @param codes: Codes at the start of the page, values = <16..1024, a multiple of 16>
 * @param exp: The shared exponent of every code (FP16 bias).
 * @param words_offset: Byte offset of the words in the page, a multiple of 64.
 * @param words: 32-bit words to unpack, a multiple of 16.
 * @param x_end: Unpacker A's X end to put back.
 * @note The first call reads unpacker A's formats (the page's), which the words and later unpacks use.
 */
inline void _llk_unpack_dest_srca_mm_codes_page_(
    const std::uint32_t address,
    const std::uint32_t codes,
    const std::uint32_t exp,
    const std::uint32_t words_offset,
    const std::uint32_t words,
    const std::uint32_t x_end)
{
    constexpr std::uint32_t bfp4 = to_underlying(DataFormat::Bfp4), fp16 = to_underlying(DataFormat::Float16);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    // The page's formats as configured, read once (after the configuring instructions are done).
    static std::uint32_t in_format = 0xff, out_format = 0xff;
    if (in_format == 0xff)
    {
        tensix_sync();
        in_format  = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & 0x0f;
        out_format = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] & 0x0f;
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(bfp4);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(fp16);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1);
    cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(exp);
    TT_SETADCXX(p_setadc::UNP_A, codes - 1, 0x0);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    cfg[unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    semaphore_post(semaphore::UNPACK_SYNC);
    set_dst_write_addr(unp_cfg_context, fp16);
    wait_for_dest_available();
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*no Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    // unpack_to_dest_tile_done's restores of this context, once the unpack is done, without its post to the math
    // thread.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    if (unp_cfg_context == 0)
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(4 * 16);
    }
    else
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(4 * 16);
    }
    switch_config_context(unp_cfg_context);

    // The words, 32-bit, in the formats the page is configured for.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(in_format);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(out_format);
    TT_SETADCXX(p_setadc::UNP_A, words - 1, 0x0);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    cfg[unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address + words_offset / 16;
    semaphore_post(semaphore::UNPACK_SYNC);
    {
        // set_dst_write_addr with the rows past the tile.
        const std::uint32_t dst_byte_addr = 16 * (4 + mailbox_read(ThreadId::MathThreadId) + words_offset / 64);
        TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0);
        TT_SETDMAREG(0, LOWER_HALFWORD(canonical_unpA_z_stride(out_format) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT), 0, LO_16(p_gpr_unpack::TMP_LO));
        TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
        if (unp_cfg_context == 0)
        {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);
        }
        else
        {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dst_byte_addr);
        }
    }
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*no Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    unpack_to_dest_tile_done(unp_cfg_context, out_format);
    switch_config_context(unp_cfg_context);
    TT_SETADCXX(p_setadc::UNP_A, x_end, 0x0);
}
