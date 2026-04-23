// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Tilize Pack — Multi-tile MOP with per-tile Last=1, zero L1 overflow.
//
// Replay buffer layout (BH = 32 entries total):
//   [0..15]:  Tile replay (16 PACRs, Last=1 on final PACR)
//   [16..19]: Address-update replay (ADDDMAREG + STALLWAIT + WRCFG + NOP)
//
// MOP: outerloop=unit_dim, innerloop=1.
//   Each outer iteration packs one tile via tile replay, then advances L1
//   address via address-update replay in end_ops. No per-tile RISC-V overhead.

#pragma once

#include <cstdint>

#include "llk_pack.h"

constexpr std::uint32_t REPLAY_TILE_OFFSET = 0;
constexpr std::uint32_t REPLAY_TILE_LEN    = 16;

constexpr std::uint32_t REPLAY_ADDR_UPDATE_OFFSET = ckernel::packer::replay_buf_offset;
constexpr std::uint32_t REPLAY_ADDR_UPDATE_LEN    = 4;

__attribute__((noinline)) void _llk_pack_fast_tilize_configure_addrmod_()
{
    addr_mod_pack_t {.y_src = {.incr = 1}}.set(ADDR_MOD_0);
    addr_mod_pack_t {.y_src = {.clr = 1, .cr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_1);
    addr_mod_pack_t {.y_src = {.incr = 4, .cr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_2);
    addr_mod_pack_t {.y_src = {.cr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_3);
}

#define EMIT_FACE_PACRS(boundary_am, last) \
    TTI_PACR(                              \
        p_pacr::CFG_CTXT_0,                \
        p_pacr::NO_ROW_PAD_ZERO,           \
        p_pacr::DST_ACCESS_STRIDED_MODE,   \
        ADDR_MOD_0,                        \
        p_pacr::ADDR_CNT_CTXT_0,           \
        0,                                 \
        p_pacr::ALL_INTF_ACTIVE,           \
        0,                                 \
        0,                                 \
        p_pacr::NO_CTXT_CTRL,              \
        0,                                 \
        0);                                \
    TTI_PACR(                              \
        p_pacr::CFG_CTXT_0,                \
        p_pacr::NO_ROW_PAD_ZERO,           \
        p_pacr::DST_ACCESS_STRIDED_MODE,   \
        ADDR_MOD_0,                        \
        p_pacr::ADDR_CNT_CTXT_0,           \
        0,                                 \
        p_pacr::ALL_INTF_ACTIVE,           \
        0,                                 \
        0,                                 \
        p_pacr::NO_CTXT_CTRL,              \
        0,                                 \
        0);                                \
    TTI_PACR(                              \
        p_pacr::CFG_CTXT_0,                \
        p_pacr::NO_ROW_PAD_ZERO,           \
        p_pacr::DST_ACCESS_STRIDED_MODE,   \
        ADDR_MOD_0,                        \
        p_pacr::ADDR_CNT_CTXT_0,           \
        0,                                 \
        p_pacr::ALL_INTF_ACTIVE,           \
        0,                                 \
        0,                                 \
        p_pacr::NO_CTXT_CTRL,              \
        0,                                 \
        0);                                \
    TTI_PACR(                              \
        p_pacr::CFG_CTXT_0,                \
        p_pacr::NO_ROW_PAD_ZERO,           \
        p_pacr::DST_ACCESS_STRIDED_MODE,   \
        boundary_am,                       \
        p_pacr::ADDR_CNT_CTXT_0,           \
        0,                                 \
        p_pacr::ALL_INTF_ACTIVE,           \
        0,                                 \
        0,                                 \
        p_pacr::NO_CTXT_CTRL,              \
        0,                                 \
        last)

__attribute__((noinline)) void _llk_pack_fast_tilize_load_replay_()
{
    // Tile replay [0..15]: 16 PACRs, Last=1 on final.
    TTI_REPLAY(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN, 0, 1);
    EMIT_FACE_PACRS(ADDR_MOD_3, 0);
    EMIT_FACE_PACRS(ADDR_MOD_2, 0);
    EMIT_FACE_PACRS(ADDR_MOD_3, 0);
    EMIT_FACE_PACRS(ADDR_MOD_1, 1); // face3: Last=1

    // Address-update replay [16..19]: advance L1_Dest_addr for next tile.
    // Same pattern as BH pack-untilize (llk_pack_untilize.h:94-108).
    TTI_REPLAY(REPLAY_ADDR_UPDATE_OFFSET, REPLAY_ADDR_UPDATE_LEN, 0, 1);
    TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    TTI_NOP;
}

#undef EMIT_FACE_PACRS

// Multi-tile MOP: outerloop=unit_dim tiles, each with tile replay + addr update.
// end_ops fire per outer iteration = per tile.
inline void _llk_pack_fast_tilize_mop_config_(const std::uint32_t unit_dim)
{
    ckernel::ckernel_template tmp(
        unit_dim, // outerloop: one iteration per tile
        1,        // innerloop: single tile body
        lltt::replay_insn(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN),
        TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010 /* CH0_W */));

    // start_op: restore Z/W from CR shadows. W_Cr accumulates across tiles.
    tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011));

    // With innerloop=1: every inner iter is "last inner".
    // Tiles 0..N-2: last_inner fires (advance W).
    // Tile N-1: last_outer fires (advance W — same op, no PACR_FLUSH).
    tmp.set_last_inner_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_outer_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));

    // end_ops: L1 address advance via replay (runs per outer iteration = per tile).
    tmp.set_end_ops(lltt::replay_insn(REPLAY_ADDR_UPDATE_OFFSET, REPLAY_ADDR_UPDATE_LEN), TT_OP_NOP);

    tmp.program();
}

template <DstSync Dst, bool is_fp32_dest_acc_en = false>
__attribute__((noinline)) void _llk_pack_fast_tilize_init_(
    [[maybe_unused]] const std::uint32_t use_32bit_dest,
    const std::uint32_t pack_dst_format,
    [[maybe_unused]] const std::uint32_t unit_dim,
    [[maybe_unused]] const std::uint32_t num_faces = 4,
    const std::uint32_t pack_src_format            = (std::uint32_t)DataFormat::Float16_b)
{
    // DEST remap (remap_addrs + swizzle_32b) is set by _llk_math_fast_tilize_init_
    // on the math thread (mirrors pack_untilize_dest_init; tracked by
    // tt-metal#17132 / tt-llk#989). No action needed here, no uninit.

    if constexpr (is_fp32_dest_acc_en)
    {
        // Packer borrow for fast-tilize: the caller's pack_src_format may be fp32
        // (e.g. after mm_init for fp32 matmul output, or infer_pack_in picking fp32
        // for Bfp4_b+dest_acc). Fast-tilize needs bf16-stride stepping through DEST
        // (Read_32b=0), but per ISA fp32 src requires Read_32b=1. Reconfigure the
        // packer to a bf16-compat src — this coherently updates in_data_format
        // across SEC0/SEC1 REG1/REG8, exp_section_size for BFP dst, TILE_HEADER,
        // and dest_rd_ctrl. Then force Read_32b=0 (reconfig sets it to 1 when
        // is_fp32_dest_acc_en). Uninit mirrors by restoring caller's pack_src.
        constexpr std::uint32_t compat_src = ckernel::to_underlying(DataFormat::Float16_b);
        const std::uint32_t tile_size      = SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM * TILE_R_DIM);
        reconfig_packer_data_format<is_fp32_dest_acc_en>(compat_src, pack_dst_format, tile_size, FACE_R_DIM, TILE_C_DIM, num_faces, /*partial_face=*/false);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(0);
    }

    // Per-tile L1 advance (used by addr-update replay's ADDDMAREG).
    const std::uint32_t tile_l1_size = GET_L1_HEADERLESS_TILE_SIZE(pack_dst_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_l1_size), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));

    TTI_SETDMAREG(0, 0x000, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
    select_packer_dest_registers<Dst>();

    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);

    // Stride registers are in bytes, divided by pack_src datum size at PACR execution
    // (ttsim: src_addr = byte_offset / src_element_align).
    // x_stride = datum_bytes(effective pack_src). When is_fp32_dest_acc_en we
    // overrode pack_src cfg to Float16_b above, so use bf16 datum size (2) here
    // regardless of the caller's original pack_src_format param.
    const std::uint32_t effective_src = is_fp32_dest_acc_en ? ckernel::to_underlying(DataFormat::Float16_b) : pack_src_format;
    const std::uint32_t x_stride      = (effective_src & 0x3) == ckernel::to_underlying(DataFormat::Float32)   ? 4
                                        : (effective_src & 0x3) == ckernel::to_underlying(DataFormat::Float16) ? 2
                                                                                                               : 1;
    std::uint32_t y_stride       = 64 * FACE_C_DIM * x_stride;
    std::uint32_t z_stride       = FACE_C_DIM * x_stride;
    std::uint32_t w_stride       = 2 * FACE_C_DIM * x_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);

    _llk_pack_fast_tilize_configure_addrmod_();
    _llk_pack_fast_tilize_load_replay_();
    _llk_pack_fast_tilize_mop_config_(unit_dim);
}

// ===========================================================================
// Row-scoped pack helpers.
//
// Hoist program_packer_destination() and the bulk of counter resets out of
// the per-chunk loop so they are paid once per row instead of once per chunk.
//
// The fast-tilize replay already advances OUTPUT_ADDR per tile via end_ops
// (ADDDMAREG + STALLWAIT + WRCFG + NOP), so after each chunk MOP OUTPUT_ADDR
// is already at the correct position for the next chunk.
//
// Counter state after every complete tile (confirmed via ISA docs + ttsim):
//   X: wraps to 0 (SETADCXX end=FACE_C_DIM-1, 16 PACRs per tile)
//   Y: 0 (ADDR_MOD_1 last PACR: y_src={clr:1,cr:1})
//   Z: 0 (ADDR_MOD_1 last PACR: z_src={clr:1})
//   W: accumulates (last_inner/last_outer ADDRCRZW W+=1 per tile)
//
// row_begin: reset X/Y, program destination — once per row.
// row_chunk: reset W only (X/Y/Z are naturally 0 at chunk boundaries) + MOP.
// row_end:   no-op (placeholder for future cleanup hooks).
// ===========================================================================

inline void _llk_pack_fast_tilize_row_begin_(const std::uint32_t address)
{
    // Reset X and Y once for the whole row. After each tile replay:
    //   X wraps to 0 (SETADCXX end=FACE_C_DIM-1, 16 PACRs per tile → X=0).
    //   Y=0 (ADDR_MOD_1 fires last PACR of every tile: y_src={clr:1,cr:1}).
    // So these counters are naturally 0 at every chunk boundary after row_begin.
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    program_packer_destination(address);
}

inline void _llk_pack_fast_tilize_row_chunk_(
    [[maybe_unused]] const std::uint32_t tile_index, [[maybe_unused]] const std::uint32_t unit_dim, [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    // Only W needs resetting per chunk — it accumulates via W_Cr across tiles
    // (last_inner/last_outer each fire ADDRCRZW W+=1, and start_op restores W=W_Cr).
    // Z=0 naturally after every tile (ADDR_MOD_1: z_src={clr:1}); no reset needed.
    // X=0 and Y=0 naturally at chunk boundaries (see row_begin comment).
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010); // reset W only (mask bit1=W)
    ckernel::ckernel_template::run();
}

inline void _llk_pack_fast_tilize_row_end_()
{
    // No-op.
}

// Reprogram MOP outerloop for a different unit_dim.
inline void _llk_pack_fast_tilize_reinit_unit_dim_([[maybe_unused]] const std::uint32_t pack_dst_format, const std::uint32_t new_unit_dim)
{
    _llk_pack_fast_tilize_mop_config_(new_unit_dim);
}

// One call = one row-chunk (one MOP run). num_units loop removed; block
// height is always 1 and chunk iteration is in the caller.
inline void _llk_pack_fast_tilize_block_(
    [[maybe_unused]] const std::uint32_t tile_index,
    const std::uint32_t address,
    [[maybe_unused]] const std::uint32_t unit_dim,
    [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    program_packer_destination(address);
    // Reset Z/W counters. MOP start_op restores from CR shadows.
    // W_Cr accumulates across tiles within the unit via ADDRCRZW in loop_op1.
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    ckernel::ckernel_template::run();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_fast_tilize_uninit_(
    const std::uint32_t pack_dst_format,
    [[maybe_unused]] const std::uint32_t face_r_dim,
    [[maybe_unused]] const std::uint32_t num_faces,
    const std::uint32_t pack_src_format = (std::uint32_t)DataFormat::Float16_b)
{
    if constexpr (is_fp32_dest_acc_en)
    {
        // Mirror of init: restore caller's pack_src_format via reconfig, which also
        // sets Read_32b correctly (=1 for fp32/dest_acc per cpack_common logic).
        const std::uint32_t tile_size = SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM * TILE_R_DIM);
        reconfig_packer_data_format<is_fp32_dest_acc_en>(
            pack_src_format, pack_dst_format, tile_size, FACE_R_DIM, TILE_C_DIM, num_faces, /*partial_face=*/false);
    }
    // DEST remap is NOT cleared here — set/owned by the math thread (see init comment).
    // BH-specific: restore strides modified by fast-tilize init (WH doesn't modify them).
    // Note: set_packer_strides's first param is semantically pack_src_format.
    set_packer_strides(pack_src_format, TILE_C_DIM);
    // Restore X counter, addr_mods, and MOP via _llk_pack_init_ (aligned with WH approach)
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    _llk_pack_init_<false, false, false>(pack_dst_format, FACE_R_DIM, TILE_C_DIM, 4);
}
