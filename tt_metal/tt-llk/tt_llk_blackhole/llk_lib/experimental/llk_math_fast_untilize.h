// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_fast_untilize.h. Same Tensix instruction sequence and
// access kinds as production, expressed through the HAL.
//
// BH Fast-Untilize Math.
//
// Consumes 2/3/4 fast_untilize unpack tiles (four real SrcA dvalids per tile,
// plus zero SrcB dvalids only for native fp32 DEST) and writes a Dst layout
// matching llk_pack_fast_untilize.h:
//
//   Dst rows   0..63:  t0.F0 | t0.F1 | t1.F0 | t1.F1
//   Dst rows  64..127: t2.F0 | t2.F1 | t3.F0 | t3.F1
//   Dst rows 128..191: t0.F2 | t0.F3 | t1.F2 | t1.F3
//   Dst rows 192..255: t2.F2 | t2.F3 | t3.F2 | t3.F3
//
// Each Dst row is one 16-datum face row. The packer then uses
// ALL_INTF_ACTIVE + DST_ACCESS_STRIDED_MODE to read rows R, R+16, R+32, R+48
// and emit a contiguous 64-datum chunk.
//
// Math is the layout bridge: unpack delivers faces one at a time through SrcA,
// while pack wants four interface rows arranged so one full-width PACR can emit
// a contiguous row-major strip. These copies do not reduce input payload reads;
// they make the pack-side access pattern wider and regular.

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "cmath_common.h"
#include "experimental/llk_fast_untilize_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/fpu.h"
#include "hal/move.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

namespace hal_port
{
namespace fast_untilize_math
{

namespace move = hal::move;
namespace rwc  = hal::counters::rwc;

// Dest rows the copies address: the 256-row fast-untilize layout plus a Dest-section base offset fit in 10 bits.
inline constexpr std::uint32_t MAX_DEST_ROW = 1023;

// TT_OP_MOVA2D(0, src_row, ADDR_MOD_4, MOV_8_ROWS, dest_row). Only the Dest row is a runtime operand: the rest of the
// word is encoded at compile time and the row added, as production's TTI_MOVA2D(..., dst_row) folds it.
template <std::uint32_t SrcRow>
constexpr std::uint32_t mova2d_operation(const std::uint32_t dest_row)
{
    return move::Transfer<move::SrcA, move::Dst> {
        .source_row = SrcRow, .destination_row = dest_row, .address_mode = ADDR_MOD_4, .number_of_rows = move::rows::Eight}
        .get_operation();
}

template <std::uint32_t SrcRow>
inline __attribute__((always_inline)) void mova2d_8_rows(const std::uint32_t dest_row)
{
    constexpr std::uint32_t base = mova2d_operation<SrcRow>(0);
    static_assert(base == TT_OP_MOVA2D(0, SrcRow, ADDR_MOD_4, p_mova2d::MOV_8_ROWS, 0), "MOVA2D encoding");
    static_assert(
        mova2d_operation<SrcRow>(1) == base + 1 && mova2d_operation<SrcRow>(MAX_DEST_ROW - 7) == base + MAX_DEST_ROW - 7, "Dest row is the low field");
    LLK_ASSERT(dest_row + 8 <= MAX_DEST_ROW + 1, "MOVA2D Dest row out of range");
    TTI_INSN(base + dest_row);
}

// TT_OP_ELWADD(0, DEST_ACCUM_DIS, SRCB_NO_BCAST, ADDR_MOD_5, dest_row), the Dest row added at runtime as above.
constexpr std::uint32_t elwadd_operation(const std::uint32_t dest_row)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = hal::fpu::ElementwiseOperation::Add,
        .broadcast        = hal::fpu::SrcBBroadcast::None,
        .dest_write       = hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = ADDR_MOD_5,
        .dest_row_offset  = static_cast<std::uint16_t>(dest_row),
        .release          = hal::fpu::SourceRelease::None});
}

inline __attribute__((always_inline)) void elwadd_copy(const std::uint32_t dest_row)
{
    constexpr std::uint32_t base = elwadd_operation(0);
    static_assert(base == TT_OP_ELWADD(0, p_elwise::DEST_ACCUM_DIS, p_elwise::SRCB_NO_BCAST, ADDR_MOD_5, 0), "ELWADD encoding");
    static_assert(elwadd_operation(1) == base + 1 && elwadd_operation(MAX_DEST_ROW - 7) == base + MAX_DEST_ROW - 7, "Dest row is the low field");
    LLK_ASSERT(dest_row + 8 <= MAX_DEST_ROW + 1, "ELWADD Dest row out of range");
    TTI_INSN(base + dest_row);
}

// TT_OP_SETRWC(CLR_A, 0, 0, 0, 0, SET_AB)
inline constexpr rwc::MathCounter CLR_A_SET_AB = rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .flip = rwc::Src::A};
static_assert(CLR_A_SET_AB.operation() == TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB), "SETRWC encoding");
// TT_OP_SETRWC(CLR_AB, 0, 0, 0, 0, SET_AB)
inline constexpr rwc::MathCounter CLR_AB_SET_AB = rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B};
static_assert(CLR_AB_SET_AB.operation() == TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB), "SETRWC encoding");

} // namespace fast_untilize_math
} // namespace hal_port

namespace ckernel
{

inline void _llk_math_fast_untilize_configure_addrmod_()
{
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;
    using Sec = hal::cfg::Sec;

    // Explicit MOVA2D source/destination row immediates drive all placement.
    // Keep RWCs unchanged so each face copy starts from SrcA rows 0/8 and
    // writes to the immediate Dst row selected below.
    set_math_addr_mod<Sec::S4, MathAddrMod {}>(); // ADDR_MOD_4: srca/srcb/dest incr 0

    // Native fp32 DEST uses ELWADD as SrcA + zero-SrcB -> DEST. ELWADD has no
    // SrcA row immediate, so the first copy advances SrcA to row 8 and the
    // second copy consumes that row. DEST placement still uses immediates.
    set_math_addr_mod<Sec::S5, MathAddrMod {.srca_incr = 8}>(); // ADDR_MOD_5: srca incr 8
}

template <bool configure_remap = true>
inline void _llk_math_fast_untilize_init_()
{
    if constexpr (configure_remap)
    {
        // Same Dst read remap required by pack_untilize/fast_tilize paths so packer
        // STRIDED_MODE sees a 16-row stride through Dst.
        _llk_math_reconfig_remap_(true);
    }

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::ClrDvalid::SrcA_Disable, hal::cfg::Sec::S0, 0>(), hal::cfg::set<hal::cfg::ClrDvalid::SrcB_Disable, hal::cfg::Sec::S0, 0>());
    hal_port::reset_counters_abd_f(); // math::reset_counters(p_setrwc::SET_ABD_F)
    _llk_math_fast_untilize_configure_addrmod_();
}

inline void _llk_math_fast_untilize_copy_face_(const std::uint32_t dst_row)
{
    namespace fum = hal_port::fast_untilize_math;
    // MOVA2D moves eight rows at a time, so one 16-row face is placed with two
    // copies. The source row immediates pick SrcA rows 0 and 8; ADDR_MOD_4 keeps
    // SrcA/DEST RWCs fixed so the destination row immediate controls placement.
    fum::mova2d_8_rows<0>(dst_row);
    fum::mova2d_8_rows<8>(dst_row + 8);

    // SrcA has been consumed for this face. Clear its dvalid so unpack can
    // reuse the bank for the next face without waiting on stale validity state.
    hal::counters::rwc::set<fum::CLR_A_SET_AB>();
}

inline void _llk_math_fast_untilize_copy_face_fp32_(const std::uint32_t dst_row)
{
    namespace fum = hal_port::fast_untilize_math;
    // Native fp32 DEST cannot use MOVA2D here; use ELWADD with zero SrcB as a
    // copy. ADDR_MOD_5 advances SrcA between the two half-face operations.
    fum::elwadd_copy(dst_row);
    fum::elwadd_copy(dst_row + 8);

    // ELWADD consumes both source operands, including the dummy zero SrcB face.
    hal::counters::rwc::set<fum::CLR_AB_SET_AB>();
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_untilize_copy_tile_(const std::uint32_t tile_index)
{
    const std::uint32_t top_row    = FAST_UNTILIZE_TOP_STRIP_ROW_OFFSET + tile_index * FAST_UNTILIZE_TILE_STRIDE_ROWS;
    const std::uint32_t bottom_row = FAST_UNTILIZE_BOTTOM_STRIP_ROW_OFFSET + tile_index * FAST_UNTILIZE_TILE_STRIDE_ROWS;

    // The unpacker presents each tile as F2, F3, F0, F1. Copy bottom faces
    // first, then top faces, so SrcA is consumed in arrival order while DEST is
    // arranged into the shared fast-untilize two-strip layout.
    if constexpr (is_fp32_dest_acc_en)
    {
        _llk_math_fast_untilize_copy_face_fp32_(bottom_row);
        _llk_math_fast_untilize_copy_face_fp32_(bottom_row + FAST_UNTILIZE_PHASE_ROWS);
        _llk_math_fast_untilize_copy_face_fp32_(top_row);
        _llk_math_fast_untilize_copy_face_fp32_(top_row + FAST_UNTILIZE_PHASE_ROWS);
    }
    else
    {
        _llk_math_fast_untilize_copy_face_(bottom_row);
        _llk_math_fast_untilize_copy_face_(bottom_row + FAST_UNTILIZE_PHASE_ROWS);
        _llk_math_fast_untilize_copy_face_(top_row);
        _llk_math_fast_untilize_copy_face_(top_row + FAST_UNTILIZE_PHASE_ROWS);
    }
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_untilize_block_(const std::uint32_t dst_index, const std::uint32_t block_ct_dim)
{
    LLK_ASSERT(block_ct_dim >= 2 && block_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM, "BH fast-untilize supports block_ct_dim 2, 3, or 4");

    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    hal_port::reset_counters_abd_f(); // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_ABD_F)

    _llk_math_fast_untilize_copy_tile_<is_fp32_dest_acc_en>(0);
    _llk_math_fast_untilize_copy_tile_<is_fp32_dest_acc_en>(1);
    if (block_ct_dim >= 3)
    {
        _llk_math_fast_untilize_copy_tile_<is_fp32_dest_acc_en>(2);
    }
    if (block_ct_dim >= 4)
    {
        _llk_math_fast_untilize_copy_tile_<is_fp32_dest_acc_en>(3);
    }

    hal_port::clear_dst_reg_addr();
}

inline void _llk_math_fast_untilize_uninit_()
{
}

} // namespace ckernel
