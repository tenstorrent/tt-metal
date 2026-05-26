// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_tilize.h"

// ============================================================================
// BH Fast-Tilize Unpack
//
// Software tilization via UNPACR address modes. Processes 4 tiles at a time
// (unit_dim=4). Each UNPACR reads 128 datums (4 tile widths) into 8 SrcA rows.
// CH1_Z stride = 256 bytes (8 contiguous SrcA rows per read).
// MASK_LOOP MOP with zmask=0x80808080 fires dvalid every 8th read
// (32 reads total, 4 dvalids per unit).
// ============================================================================

inline void _llk_unpack_fast_tilize_mop_config_()
{
    // addr_mode: CH0_Z+=1 (next L1 row), CH1_Z+=1 (next SrcA dest with gap)
    constexpr std::uint8_t ADDRMOD = 0b00'01'00'01;

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                 // unpackB
        false,                                 // unpackHalo
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD, 0), // A0: read, no dvalid
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD, 1), // skipA: read WITH dvalid
        TT_OP_NOP,
        TT_OP_NOP);
    tmp.program();
}

// BH fast-tilize: block height is always 1 (one row of tiles per call).
// Multiple rows are handled by the caller, looping over rows and calling
// the block function once per chunk per row.
//
// init_unit_dim: unit_dim of the first chunk (= decompose_row(ct_dim)[0]).
// Initialising X to the first chunk's width avoids one reinit_xdim call per row.
// Formula: ct_dim > 5 ? 4 : ct_dim == 5 ? 2 : ct_dim.
inline void _llk_unpack_fast_tilize_init_(const std::uint32_t unpack_dst_format, const std::uint32_t ct_dim, const std::uint32_t init_unit_dim)
{
    // Context-safe writes only: Tile_x_dim (WRCFG below writes the full 32-bit word,
    // covering cntx0 low-16 and cntx1 high-16), TileDescriptor (shared across contexts),
    // Zstride (RMW on shared reg), and SETADCXX (thread-scoped counter, not per
    // cfg context - see ISA SETADCXX). Per-call context switching happens in
    // _llk_unpack_fast_tilize_block_.

    // Save state
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    // Save the unpacker Out_data_format word so we can restore it in uninit if the
    // fp32/tf32 -> bf16 downgrade below modifies it.
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, THCON_SEC0_REG2_Out_data_format_ADDR32);

    // BH fast-tilize forces a 16-bit DEST view in the math thread (MOVA2D cannot
    // safely write Dst32b in this flow - see _llk_math_fast_tilize_init_). That makes
    // TF32/Float32 SrcA incompatible: MOVA2D(TF32) + Dst16b is ISA UB, and MOVA2D(bf16)
    // + Dst32b is also UB. The only consistent combination is bf16 SrcA + Dst16b, so
    // downgrade SrcA output to Float16_b here when the caller requested fp32/tf32.
    // The unpacker performs the fp32 -> bf16 conversion on the L1 -> SrcA path. This
    // matches the precision Metal previously consumed from the fp32 fast-tilize path.
    const std::uint32_t effective_dst_format =
        (unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Float32) || unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Tf32))
            ? static_cast<std::uint32_t>(DataFormat::Float16_b)
            : unpack_dst_format;
    if (effective_dst_format != unpack_dst_format)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(effective_dst_format);
    }

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    // Tile_x_dim = 32, Tile_y_dim = ct_dim, Tile_z_dim = 16
    TT_SETDMAREG(0, TILE_C_DIM, 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, TILE_C_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);

    TT_SETDMAREG(0, ct_dim, 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, FACE_R_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    // BH HW bug: TileDescriptor words 1-3 (YDim, ZDim) are not tracked as unpacker
    // resources. Explicit stall ensures the write completes.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    // X counter end = first chunk's tile width. CH1_Z stride stays at 4-wide
    // (8 SrcA rows) regardless of unit_dim - natural gaps in SrcA for unit_dim < 4.
    // Matching init_unit_dim to the first chunk eliminates one reinit_xdim per row.
    TT_SETADCXX(p_setadc::UNP_A, init_unit_dim * TILE_C_DIM - 1, 0x0);

    // CH1 Z stride: controls SrcA dest address gap between reads.
    // Uses effective_dst_format because Float32/Tf32 are downgraded to bf16 above.
    const std::uint32_t ch1_x_stride =
        (effective_dst_format == static_cast<std::uint32_t>(DataFormat::Float32) || effective_dst_format == static_cast<std::uint32_t>(DataFormat::Int32) ||
         effective_dst_format == static_cast<std::uint32_t>(DataFormat::Tf32))
            ? 4
            : 2;
    // stride = 4 * 32 * 2 = 256 bytes = 8 contiguous SrcA rows per read
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(4 * TILE_C_DIM * ch1_x_stride);

    _llk_unpack_fast_tilize_mop_config_();
}

// Reconfigure X counter for a different unit_dim without full reinit.
// CH1_Z stride and MOP stay unchanged - only the read width changes.
inline void _llk_unpack_fast_tilize_reinit_xdim_(const std::uint32_t unit_dim)
{
    TT_SETADCXX(p_setadc::UNP_A, unit_dim * TILE_C_DIM - 1, 0x0);
}

// One call = one row-chunk (one unit_dim, one MOP run).
// Block height is always 1; multiple rows and chunks are loops in the caller.
inline void _llk_unpack_fast_tilize_block_(
    const std::uint32_t base_address,
    [[maybe_unused]] const std::uint32_t tile_index,
    [[maybe_unused]] const std::uint32_t unpack_src_format,
    [[maybe_unused]] const std::uint32_t unit_dim,
    [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    // Standard BH unpacker context dance (see _llk_unpack_untilize_pass_).
    // Programs REG3 Base_address for the current cfg context via cfg[] write,
    // then synchronises Trisc<->T6 so the MOP runs with that base in place.
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_single_address_(base_address, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // L1 addressing via CH0 counters. The caller folds any tile/column offset
    // into base_address, so Y starts at zero for this chunk and Z selects tensor
    // row (0..31).
    TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b0011); // reset CH0_X=0, CH0_Y=0
    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111); // reset all Z,W = 0

    // Hoist zmask high 16 bits - they persist in mop_zmask_hi16 until changed.
    constexpr std::uint32_t ZMASK = 0x80808080;
    TT_MOP_CFG(ZMASK >> 16);
    TT_MOP(0, 32 - 1, ZMASK & 0xFFFF);

    // Release the unpacker context acquired above and advance the software
    // tracker so the next call targets the other cfg context slot.
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_tilize_uninit_()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, 0, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, 0, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    // Restore Out_data_format (init may have downgraded it from fp32/tf32 to bf16).
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, 0, THCON_SEC0_REG2_Out_data_format_ADDR32);

    TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1010);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111);
}
