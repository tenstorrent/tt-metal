// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_eltwise_unary_datacopy.h"

// ============================================================================
// BH Fast-Tilize Math
//
// Modified A2D datacopy for fast-tilize's SrcA layout (8 data rows + 8 gap rows).
// MOVA2D MOV_8_ROWS with srca_incr=16, dest_incr=16 copies data rows while
// preserving the gap layout. 4 MOVA2D per dvalid = 32 data rows.
//
// Accumulates 8 dvalids into one DEST half-bank (512 rows for bf16):
//   dvalid 0: DEST rows 0-7, 16-23, 32-39, 48-55
//   dvalid 1: DEST rows 64-71, 80-87, 96-103, 112-119
//   ...
//   dvalid 7: DEST rows 448-455, 464-471, 480-487, 496-503
// Total: 32 data groups x 8 rows x 16 cols = 4096 datums = 4 tiles.
// Pack then reads 4 tiles from the DEST half using DST_ACCESS_STRIDED_MODE.
// ============================================================================

template <bool is_fp32_dest_acc_en = false, bool configure_remap = true>
inline void _llk_math_fast_tilize_init_([[maybe_unused]] const std::uint32_t unpack_dst_format)
{
    if constexpr (configure_remap)
    {
        // DEST remap (remap_addrs + swizzle_32b) is enabled here to mirror
        // pack_untilize_dest_init's BH workaround (see tt-metal#17132 / tt-llk#989).
        // No uninit - leaving the bits set is benign for subsequent ops.
        _llk_math_reconfig_remap_(true);
    }

    // Compat fp32-dest: MOVA2D does not correctly handle 32-bit DEST rows (BH HW quirk,
    // same as WH). Temporarily clear Fp32_enabled so MOVA2D treats DEST as 16-bit.
    // Restored in uninit. This is safe for bf16/fp16 outputs where 32-bit DEST is only
    // used for accumulation precision, not for the data path through fast-tilize.
    if constexpr (is_fp32_dest_acc_en)
    {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    }

    // Enable SrcA bank return in SETRWC(CLR_A). Without this, SETRWC flips the
    // bank pointer but does NOT return the bank to Unpackers, causing a livelock
    // when the Unpacker runs out of free banks.
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    // srca incr=8 (contiguous SrcA), dest incr=16 (8 data + 8 gap in DEST for pack stride-16)
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_2);

    // outerloop=4 (4 dvalids per unit), innerloop=8 (8 MOVA2D per dvalid).
    // SETRWC in end_op clears/flips SrcA between dvalids; DEST RWC keeps
    // advancing across outer loops. Single run() per unit.
    ckernel_template tmp(4, 8, TT_OP_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_A));
    tmp.program();
}

// One call = one row-chunk (one MOP run). num_units loop removed; block
// height is always 1 and chunk iteration is in the caller.
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_fast_tilize_block_(
    const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t unpack_dst_format,
    [[maybe_unused]] const std::uint32_t unit_dim,
    [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    // Use SrcRegs (not DestReg) - DestReg sends a hardware mailbox to the unpack
    // thread, but fast-tilize unpack uses counter-based addressing and never reads
    // the mailbox. After ~4 unread writes the mailbox FIFO fills and math RISC-V
    // deadlocks. SrcRegs uses TT_SETC16 to set DEST offset directly.
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // 1 MOP = 4 dvalids x 8 MOVA2D + 4 SETRWC(CLR_A)
    ckernel_template::run();

    math::clear_dst_reg_addr();
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_tilize_uninit_([[maybe_unused]] const std::uint32_t unpack_dst_format)
{
    // Restore fp32 dest mode if it was cleared in init
    if constexpr (is_fp32_dest_acc_en)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
    }

    // DEST remap is cleared by pack uninit.

    // Restore standard addr_mod for A2D
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
}
